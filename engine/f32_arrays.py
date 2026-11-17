"""Array search for float32 blocks (cm=3) for block sizes N = 8, 16, 32: the same results as
engine.refine.refine_dct_array for clean blocks (no NoData, no zeros, all values finite) with the
default options (fast_eval_basis, incremental_backscan, verify_with_full_idct, decoder rounding to
float32).

The four stages of refine_dct_array run on the whole batch at once:
1. binary search for the smallest L (numba prange on the kernel of _eval_error_for_len for N <= 16,
   vectorised numpy on the tensordot path for N = 32, which the reference takes there),
2. backscan in float64 on the float32 original, up to int(backscan * (N/8) * sqrt(N/8)) steps, stop
   after brk consecutive failures,
3. verification with the classic IDCT and the error of the float32 stored value (L up to +5),
4. quantisation check of round(c * scaling_factor) (L up to +16).

The reconstructions stay with the libraries the reference calls (BLAS tensordot, scipy IDCT), so their
values are the same on every machine. The per-pixel work around them (differences, the float32
stored value, the backscan steps) runs in numba prange kernels without fastmath: every pixel sees
the same IEEE operations in the same order as in the numpy expressions of the reference.
The kernels use the OpenMP threads of numba, so they must run in the main process only (a forked
Pool worker cannot start OpenMP threads once the parent has used them).
"""

import math

import numba as nb
import numpy as np
from scipy.fft import idct as _sidct

from core.dct import _get_idct_basis, _get_zigzag_indices, fft_workers
from engine.refine import _nb_eval_fast, _nb_tail_l2

MAX_VERIFY = 5
MAX_QUANT = 16
_THREADS_MIN_SIZE = 1 << 15  # elements; below this waking the numba threads costs more than the loop


def _run(kernel, size, *args):
    """Call a prange kernel on `size` elements. A small batch (the last rows of the verification and
    quantisation loops, a small file) stays on the calling thread; the values are the same."""
    threads = nb.get_num_threads()
    if size >= _THREADS_MIN_SIZE or threads == 1:
        kernel(*args)
        return
    nb.set_num_threads(1)
    try:
        kernel(*args)
    finally:
        nb.set_num_threads(threads)


@nb.njit(parallel=True, cache=True)
def _bsearch_kernel(zz, orig, acc, basis2, thr, lf):
    """Binary search for the smallest L with error <= acc on the kernel of _eval_error_for_len:
    L2 precheck (tail > acc * N -> failure), then the fast-basis error."""
    m = zz.shape[0]
    total = zz.shape[1]
    for i in nb.prange(m):
        lo = 1
        hi = total
        ans = total
        a = acc[i]
        while lo <= hi:
            mid = (lo + hi) // 2
            ok = _nb_tail_l2(zz[i], mid) <= a * thr
            if ok:
                ok = _nb_eval_fast(zz[i], basis2, orig[i], mid, a, False, False, 0.0, 0.0, 0.0) <= a
            if ok:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        lf[i] = ans


def _binary_numpy(zz32, z64, org32, acc, basis, n):
    """Binary search on the tensordot path (N = 32 in the reference): float32 L2 tail and the
    float64 error of the fast basis without the float32 store."""
    m, nn = zz32.shape
    lo = np.ones(m, dtype=np.int64)
    hi = np.full(m, nn, dtype=np.int64)
    ans = np.full(m, nn, dtype=np.int64)
    k = np.arange(nn)
    org64 = org32.astype(np.float64)
    while True:
        act = np.flatnonzero(lo <= hi)
        if act.size == 0:
            break
        mid = (lo[act] + hi[act]) // 2
        z32 = zz32[act].astype(np.float32)
        tail_sq = np.where(k[None, :] >= mid[:, None], z32 * z32, np.float32(0))
        tail = np.sqrt(tail_sq.sum(axis=1, dtype=np.float32).astype(np.float64))
        ok = tail <= acc[act] * float(n)
        err = np.full(act.size, np.inf)
        ev = np.flatnonzero(ok)
        if ev.size:
            rec = _grouped_recon(z64, act[ev], mid[ev], basis)
            d = org64[act[ev]] - rec
            err[ev] = np.maximum(d.max(axis=(1, 2)), -d.min(axis=(1, 2)))
        good = err <= acc[act]
        rows = act[good]
        ans[rows] = mid[good]
        hi[rows] = mid[good] - 1
        bad = act[~good]
        lo[bad] = mid[~good] + 1
    return ans


def _grouped_recon(coeffs, rows, lengths, basis):
    """tensordot(coeffs[row, :L], basis[:L]) for the given rows of coeffs (m, N*N) float64, grouped
    by equal L (bitwise equal to the per-row call). Returns (len(rows), N, N)."""
    n = basis.shape[1]
    rec = np.empty((rows.size, n, n))
    for lv in np.unique(lengths):
        sel = np.flatnonzero(lengths == lv)
        rec[sel] = np.tensordot(coeffs[rows[sel], :lv], basis[:lv], axes=(1, 0))
    return rec


@nb.njit(parallel=True, cache=True)
def _backscan_kernel(zb, org, rec, rows, lf, acc, basis2, brk, steps, out):
    """Backscan steps of one row at a time. rec (len(rows), N*N) holds the reconstruction of the
    first L coefficients on entry and is used as the running difference. Step j drops coefficient
    L - j: diff += basis[L - j] * c, error = max(diff.max(), -diff.min())."""
    nn = rec.shape[1]
    for i in nb.prange(rows.size):
        r = rows[i]
        L = lf[i]
        a = acc[r]
        for p in range(nn):
            rec[i, p] = np.float64(np.float32(org[r, p])) - rec[i, p]
        best = L
        streak = 0
        for j in range(1, steps + 1):
            k = L - j
            if k < 1:
                break
            c = zb[r, k]
            mx = -np.inf
            mn = np.inf
            nan = False
            for p in range(nn):
                v = rec[i, p] + basis2[k, p] * c
                rec[i, p] = v
                if v > mx:
                    mx = v
                if v < mn:
                    mn = v
                if v != v:
                    nan = True
            if not nan and (mx if mx > -mn else -mn) <= a:
                best = k
                streak = 0
            else:
                streak += 1
                if brk > 0 and streak >= brk:
                    break
        out[r] = best


def _backscan(zb, org2, lf, acc, basis, basis2, brk, steps):
    """Backscan of refine_dct_array (incremental, float64 difference on the float32 original).
    zb: the coefficients rounded to float32 (refine_dct_array keeps them so), as float64."""
    out = lf.copy()
    rows = np.flatnonzero(lf > 1)
    if rows.size == 0:
        return out
    lr = lf[rows]
    rec = _grouped_recon(zb, rows, lr, basis).reshape(rows.size, -1)
    _backscan_kernel(zb, org2, rec, rows, lr, acc, basis2, brk, steps, out)
    return out


@nb.njit(parallel=True, cache=True)
def _scatter_prefix(z64, rows, lf, zz_idx, flat):
    """flat[i, zz_idx[k]] = z64[rows[i], k] for k < lf[i], 0 for the dropped coefficients."""
    nn = z64.shape[1]
    for i in nb.prange(rows.size):
        r = rows[i]
        L = lf[i]
        for k in range(nn):
            flat[i, zz_idx[k]] = z64[r, k] if k < L else 0.0


@nb.njit(parallel=True, cache=True)
def _store_err_kernel(org, rec, rows, err):
    """err[i] = max(diff.max(), -diff.min()) of refine's _stored_float32_diff(org[rows[i]], rec[i]):
    original - stored value, where stored is the float64 reconstruction rounded to float32. A pixel
    whose reconstruction lies within a relative 1e-9 of a float32 rounding midpoint takes the worse
    of the two neighbours."""
    nn = rec.shape[1]
    for i in nb.prange(rows.size):
        row = rows[i]
        mx = -np.inf
        mn = np.inf
        nan = False
        for p in range(nn):
            r = rec[i, p]
            o = np.float64(org[row, p])
            r32 = np.float32(r)
            r32d = np.float64(r32)
            d = o - r32d
            if r > r32d:
                alt = np.nextafter(r32, np.float32(np.inf))
            else:
                alt = np.nextafter(r32, np.float32(-np.inf))
            altd = np.float64(alt)
            scale = abs(r)
            if scale < 1.0:
                scale = 1.0
            if abs(abs(r - r32d) - 0.5 * abs(altd - r32d)) <= 1e-9 * scale:
                d_alt = o - altd
                if abs(d_alt) > abs(d):
                    d = d_alt
            if d > mx:
                mx = d
            if d < mn:
                mn = d
            if d != d:
                nan = True
        # np.maximum(mx, -mn): the second operand when they are equal (the sign of a zero error)
        err[i] = np.nan if nan else (mx if mx > -mn else -mn)


def _store_err(org2, rec, rows):
    err = np.empty(rows.size)
    _run(_store_err_kernel, rec.size, org2, rec.reshape(rows.size, -1), rows, err)
    return err


def _classic_err(z64, org2, rows, lf, zz_idx, n):
    """Error of the classic verification for the given rows: IDCT of the first L coefficients (axis 0
    then axis 1 of the block, as core.dct.idct2 on 2-D input), then the float32 stored value (refine's
    _eval_error_for_len with use_fast_basis=False and float32_store=True)."""
    flat = np.empty((rows.size, n * n))
    _run(_scatter_prefix, flat.size, z64, rows, lf, zz_idx, flat)
    w = fft_workers(flat.size)
    rec = _sidct(flat.reshape(rows.size, n, n), axis=1, norm="ortho", type=2, overwrite_x=True, workers=w)
    rec = _sidct(rec, axis=2, norm="ortho", type=2, overwrite_x=True, workers=w)
    return _store_err(org2, rec, rows)


def _verify(z64, org2, lf, acc, zz_idx, n):
    lf = lf.copy()
    err = _classic_err(z64, org2, np.arange(lf.size), lf, zz_idx, n)
    steps = np.zeros(lf.size, dtype=np.int64)
    for _ in range(MAX_VERIFY):
        act = np.flatnonzero((err > acc) & (lf < z64.shape[1]) & (steps < MAX_VERIFY))
        if act.size == 0:
            break
        lf[act] += 1
        steps[act] += 1
        err[act] = _classic_err(z64, org2, act, lf[act], zz_idx, n)
    return lf, err


def _quant_err(dq, org2, rows, lf, basis):
    """Error of round(c * sf) / sf (dq) as the decoder restores it (float32 stored)."""
    return _store_err(org2, _grouped_recon(dq, rows, lf, basis), rows)


def _quantise(dq, org2, lf, acc, basis):
    lf = lf.copy()
    err = _quant_err(dq, org2, np.arange(lf.size), lf, basis)
    steps = np.zeros(lf.size, dtype=np.int64)
    for _ in range(MAX_QUANT):
        act = np.flatnonzero((err > acc) & (lf < dq.shape[1]) & (steps < MAX_QUANT))
        if act.size == 0:
            break
        lf[act] += 1
        steps[act] += 1
        err[act] = _quant_err(dq, org2, act, lf[act], basis)
    return lf, err


@nb.njit(parallel=True, cache=True)
def _pack_prefixes(q, lf, starts, out):
    """out[starts[i] : starts[i] + lf[i]] = q[i, : lf[i]]: the kept coefficients of all rows in a row."""
    for i in nb.prange(lf.size):
        s = starts[i]
        for k in range(lf[i]):
            out[s + k] = q[i, k]


def search_f32(zz32, org32, acc, sf, dct_type=2, backscan=10, brk=3):
    """zz32 (m, N*N) float32 zigzag DCT rows, org32 (m, N, N) float32 blocks, acc (m,) accuracy,
    sf the scaling factor. Returns (LF, ERR, COEFFS): length and error per row and the int64
    coefficients of all rows one after another (LF[i] of them for row i), matching
    refine_dct_array(zz, acc, org, N, dct_type, sf, cfg, src_dtype="float32")."""
    zz32 = np.ascontiguousarray(zz32)  # dtype kept: float32 for float data, float64 for uint8 sampling
    org32 = np.ascontiguousarray(org32)
    acc = np.ascontiguousarray(acc, dtype=np.float64)
    m, nn = zz32.shape
    if m == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=np.int64)
    n = int(round(math.sqrt(nn)))
    zz_idx = _get_zigzag_indices(n)
    basis = _get_idct_basis(n, dct_type)
    basis2 = np.ascontiguousarray(basis.reshape(nn, nn))
    steps = int(backscan * (n / 8) * math.sqrt(n / 8))  # refine: backscan_base * scale_factor
    org2 = org32.reshape(m, nn)
    z64 = zz32.astype(np.float64)
    if n <= 16:
        lf = np.zeros(m, dtype=np.int64)
        _bsearch_kernel(z64, org2.astype(np.float64), acc, basis2, float(n), lf)
    else:
        lf = _binary_numpy(zz32, z64, org32, acc, basis, n)
    if backscan > 0:
        zb = z64 if zz32.dtype == np.float32 else zz32.astype(np.float32).astype(np.float64)
        lf = _backscan(zb, org2, lf, acc, basis, basis2, brk, steps)
        del zb
    lf, _ = _verify(z64, org2, lf, acc, zz_idx, n)
    del z64
    q = np.round(zz32 * sf)
    lf, err = _quantise(q.astype(np.float64) / sf, org2, lf, acc, basis)
    ends = np.cumsum(lf)
    coeffs = np.empty(int(ends[-1]), dtype=np.int64)
    _run(_pack_prefixes, coeffs.size, q.astype(np.int64), lf, ends - lf, coeffs)
    return lf, err, coeffs
