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
"""

import math

import numba as nb
import numpy as np
from scipy.fft import idct as _sidct

from core.dct import _get_idct_basis, _get_zigzag_indices
from engine.refine import _nb_eval_fast, _nb_tail_l2, _stored_float32_diff

MAX_VERIFY = 5
MAX_QUANT = 16


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


def _binary_numpy(zz32, org32, acc, basis, n):
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
            rec = _grouped_recon(zz32[act[ev]].astype(np.float64), mid[ev], basis)
            d = org64[act[ev]] - rec
            err[ev] = np.maximum(d.max(axis=(1, 2)), -d.min(axis=(1, 2)))
        good = err <= acc[act]
        rows = act[good]
        ans[rows] = mid[good]
        hi[rows] = mid[good] - 1
        bad = act[~good]
        lo[bad] = mid[~good] + 1
    return ans


def _grouped_recon(coeffs, lengths, basis):
    """tensordot(coeffs[:L], basis[:L]) per row, grouped by equal L (bitwise equal to the per-row call)."""
    n = basis.shape[1]
    rec = np.empty((coeffs.shape[0], n, n))
    for lv in np.unique(lengths):
        sel = np.flatnonzero(lengths == lv)
        rec[sel] = np.tensordot(coeffs[sel, :lv], basis[:lv], axes=(1, 0))
    return rec


def _backscan(zz32, org32, lf, acc, basis, brk, steps):
    """Backscan of refine_dct_array (incremental, float64 difference on the float32 original)."""
    out = lf.copy()
    rows = np.flatnonzero(lf > 1)
    if rows.size == 0:
        return out
    lr = lf[rows]
    ar = acc[rows]
    coeffs32 = zz32.astype(np.float32)  # refine_dct_array keeps the backscan coefficients in float32
    diff = org32[rows].astype(np.float32).astype(np.float64) - _grouped_recon(
        coeffs32[rows].astype(np.float64), lr, basis)
    best = lr.copy()
    streak = np.zeros(rows.size, dtype=np.int64)
    live = np.ones(rows.size, dtype=bool)
    for j in range(1, steps + 1):
        cur = np.flatnonzero(live & (lr - j >= 1))
        if cur.size == 0:
            break
        k = lr[cur] - j
        diff[cur] = diff[cur] + basis[k] * coeffs32[rows[cur], k][:, None, None].astype(np.float64)
        err = np.maximum(diff[cur].max(axis=(1, 2)), -diff[cur].min(axis=(1, 2)))
        ok = err <= ar[cur]
        best[cur[ok]] = k[ok]
        streak[cur[ok]] = 0
        bad = cur[~ok]
        streak[bad] += 1
        if brk > 0:
            live[bad[streak[bad] >= brk]] = False
    out[rows] = best
    return out


def _classic_err(zz32, org32, lf, zz_idx, n):
    """Error of the classic verification: IDCT of the first L coefficients (axis 0 then axis 1 of the
    block, as core.dct.idct2 on 2-D input), then the float32 stored value (refine's
    _eval_error_for_len with use_fast_basis=False and float32_store=True)."""
    m, nn = zz32.shape
    keep = np.arange(nn)[None, :] < lf[:, None]
    coeffs = np.where(keep, zz32.astype(np.float64), 0.0)
    flat = np.zeros((m, nn))
    flat[:, zz_idx] = coeffs
    rec = _sidct(_sidct(flat.reshape(m, n, n), axis=1, norm="ortho", type=2),
                 axis=2, norm="ortho", type=2)
    diff = _stored_float32_diff(org32, rec)
    return np.maximum(diff.max(axis=(1, 2)), -diff.min(axis=(1, 2)))


def _verify(zz32, org32, lf, acc, zz_idx, n):
    lf = lf.copy()
    err = _classic_err(zz32, org32, lf, zz_idx, n)
    steps = np.zeros(lf.size, dtype=np.int64)
    for _ in range(MAX_VERIFY):
        act = np.flatnonzero((err > acc) & (lf < zz32.shape[1]) & (steps < MAX_VERIFY))
        if act.size == 0:
            break
        lf[act] += 1
        steps[act] += 1
        err[act] = _classic_err(zz32[act], org32[act], lf[act], zz_idx, n)
    return lf, err


def _quant_err(zz32, org32, lf, sf, basis):
    """Error of round(c * sf) / sf as the decoder restores it (float32 stored)."""
    nn = zz32.shape[1]
    keep = np.arange(nn)[None, :] < lf[:, None]
    dq = np.where(keep, np.round(zz32 * sf).astype(np.float64) / sf, 0.0)
    rec = _grouped_recon(dq, lf, basis)
    diff = _stored_float32_diff(org32, rec)
    return np.maximum(diff.max(axis=(1, 2)), -diff.min(axis=(1, 2)))


def _quantise(zz32, org32, lf, acc, sf, basis):
    lf = lf.copy()
    err = _quant_err(zz32, org32, lf, sf, basis)
    steps = np.zeros(lf.size, dtype=np.int64)
    for _ in range(MAX_QUANT):
        act = np.flatnonzero((err > acc) & (lf < zz32.shape[1]) & (steps < MAX_QUANT))
        if act.size == 0:
            break
        lf[act] += 1
        steps[act] += 1
        err[act] = _quant_err(zz32[act], org32[act], lf[act], sf, basis)
    return lf, err


def search_f32(zz32, org32, acc, sf, dct_type=2, backscan=10, brk=3):
    """zz32 (m, N*N) float32 zigzag DCT rows, org32 (m, N, N) float32 blocks, acc (m,) accuracy,
    sf the scaling factor. Returns (LF, ERR, RES): length, error and int coefficients per row,
    matching refine_dct_array(zz, acc, org, N, dct_type, sf, cfg, src_dtype="float32")."""
    zz32 = np.ascontiguousarray(zz32)  # dtype kept: float32 for float data, float64 for uint8 sampling
    org32 = np.ascontiguousarray(org32)
    acc = np.ascontiguousarray(acc, dtype=np.float64)
    m, nn = zz32.shape
    n = int(round(math.sqrt(nn)))
    zz_idx = _get_zigzag_indices(n)
    basis = _get_idct_basis(n, dct_type)
    basis2 = np.ascontiguousarray(basis.reshape(nn, nn))
    steps = int(backscan * (n / 8) * math.sqrt(n / 8))  # refine: backscan_base * scale_factor
    if m:
        if n <= 16:
            lf = np.zeros(m, dtype=np.int64)
            _bsearch_kernel(zz32.astype(np.float64), org32.reshape(m, nn).astype(np.float64), acc,
                            basis2, float(n), lf)
        else:
            lf = _binary_numpy(zz32, org32, acc, basis, n)
        if backscan > 0:
            lf = _backscan(zz32, org32, lf, acc, basis, brk, steps)
        lf, _ = _verify(zz32, org32, lf, acc, zz_idx, n)
        lf, err = _quantise(zz32, org32, lf, acc, sf, basis)
    else:
        err = np.zeros(0)
    keep = np.arange(nn)[None, :] < lf[:, None]
    res = np.where(keep, np.round(zz32 * sf).astype(np.int64), 0)
    return lf, err, res
