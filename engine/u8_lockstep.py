"""Batched uint8 search (block edge n = 8, 16 or 32) in three phases, bit-identical to refine_dct_array (cm=5/cm=6).

  phase 1  binary search of L (float criterion) + backscan (integer criterion)  -> LPRE
           GPU: search_u8_phases.cu (pre_kernel), or CPU: numba prange
  phase 2  full IDCT verification (integer criterion) with the L2 precheck      -> LV
           CPU, vectorised, scipy.fft (the same IDCT as the reference, so the same rounding ties)
  phase 3  quantisation and the check after quantisation                         -> LF, ERR, RES
           GPU: quant_kernel, or CPU: numba prange
"""


import numba as nb
import numpy as np
from scipy.fft import idct as _sidct

from core.dct import _get_zigzag_indices
from engine import gpu_u8
from engine.gpu_u8 import phases as gpu_phases
from engine.refine import _l2_tail, _nb_eval_fast, _nb_tail_l2

CU8 = 128.0
LO8 = 0.0
HI8 = 255.0
MAX_VERIFY = 5
MAX_QUANT = 16
GPU_MIN_ROWS = 256  # below this, kernel launch and host-device copies do not pay off


def _edge(nn):
    """Block edge of a block with nn = n * n coefficients."""
    return int(round(nn ** 0.5))


@nb.njit(cache=True)
def _eval_float(zz, orig, L, acc, basis2):
    if _nb_tail_l2(zz, L) > acc * np.sqrt(np.float64(zz.shape[0])):
        return acc + 1.0
    return _nb_eval_fast(zz, basis2, orig, L, acc, False, False, 0.0, 0.0, 0.0)


@nb.njit(cache=True)
def _eval_quant(dqz, orig, L, acc, basis2):
    return _nb_eval_fast(dqz, basis2, orig, L, acc, False, True, CU8, LO8, HI8)


@nb.njit(cache=True)
def _int_err_from_diff(diff, orig):
    mx = 0.0
    mn = 0.0
    for p in range(diff.shape[0]):
        t = diff[p] + orig[p] + CU8
        v = np.rint(t)
        if v < LO8:
            v = LO8
        elif v > HI8:
            v = HI8
        e = (orig[p] + CU8) - v
        if e > mx:
            mx = e
        if e < mn:
            mn = e
    return mx if mx > -mn else -mn


@nb.njit(cache=True)
def _pre_row(zz, orig, acc, bs, basis2, brk):
    """Phase 1 for one row: binary search and backscan. Returns L."""
    nn = zz.shape[0]
    lo, hi, ans = 1, nn, nn
    while lo <= hi:
        mid = (lo + hi) // 2
        err = _eval_float(zz, orig, mid, acc, basis2)
        if err <= acc:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    L = ans

    if bs > 0 and L > 1:
        start = max(1, L - bs)
        diff = np.empty(nn)
        for p in range(nn):
            rec = 0.0
            for k in range(L):
                rec += np.float64(np.float32(zz[k])) * basis2[k, p]
            diff[p] = orig[p] - rec
        new_L = L
        fail = 0
        for k in range(L - 1, start - 1, -1):
            ck = np.float64(np.float32(zz[k]))
            for p in range(nn):
                diff[p] += basis2[k, p] * ck
            e2 = _int_err_from_diff(diff, orig)
            if e2 <= acc:
                new_L = k
                fail = 0
            else:
                fail += 1
                if brk > 0 and fail >= brk:
                    break
        L = new_L
    return L


@nb.njit(cache=True, parallel=True)
def pre_par(ZZ, ORIG, ACC, BS, basis2, LPRE, brk):
    for i in nb.prange(ZZ.shape[0]):  # pylint: disable=not-an-iterable
        LPRE[i] = _pre_row(ZZ[i], ORIG[i], ACC[i], BS[i], basis2, brk)


@nb.njit(cache=True)
def _quant_row(zz, orig, acc, sf, basis2, L0, LF, ERR, RES, i):
    """Phase 3 for one row: quantise from L0 and check the quantised error."""
    L = L0
    nn = zz.shape[0]
    dqz = np.zeros(nn)
    for k in range(L):
        dqz[k] = np.rint(zz[k] * sf) / sf
    max_err = _eval_quant(dqz, orig, L, acc, basis2)
    q = 0
    while max_err > acc and L < nn and q < MAX_QUANT:
        L += 1
        q += 1
        dqz = np.zeros(nn)
        for k in range(L):
            dqz[k] = np.rint(zz[k] * sf) / sf
        max_err = _eval_quant(dqz, orig, L, acc, basis2)
    LF[i] = L
    ERR[i] = max_err
    for k in range(L):
        RES[i, k] = np.int64(np.rint(zz[k] * sf))


@nb.njit(cache=True, parallel=True)
def quant_par(ZZ, ORIG, ACC, SF, basis2, LIN, LF, ERR, RES):
    for i in nb.prange(ZZ.shape[0]):  # pylint: disable=not-an-iterable
        _quant_row(ZZ[i], ORIG[i], ACC[i], SF[i], basis2, LIN[i], LF, ERR, RES, i)


# phase 2 - full IDCT verification (scipy, vectorised over rows)
def _classic_err(ZZ, ORIG, L):
    """max |orig + 128 - clip(round(idct(coeffs[:L]) + 128))| per row (classic path)."""
    m, nn = ZZ.shape
    n = _edge(nn)
    keep = np.arange(nn)[None, :] < L[:, None]
    coeffs = np.where(keep, ZZ, 0.0)
    flat = np.zeros((m, nn))
    flat[:, _get_zigzag_indices(n)] = coeffs
    # scipy idct2 of a block: axis 0 first, then axis 1 (as core.dct.idct2)
    rec = _sidct(_sidct(flat.reshape(m, n, n), axis=1, norm="ortho", type=2),
                 axis=2, norm="ortho", type=2).reshape(m, nn)
    rec_u8 = np.clip(np.rint(rec + CU8), LO8, HI8)
    diff = (ORIG + CU8) - rec_u8
    return np.maximum(diff.max(axis=1), -diff.min(axis=1))


def _classic_full(ZZ, ORIG, ACC, L):
    """Classic error with the L2 precheck of _eval_error_for_len (numpy path, use_fast_basis=False)."""
    m, nn = ZZ.shape
    t32 = ZZ.astype(np.float32)
    sq = (t32 * t32).astype(np.float64)
    tail = np.zeros(m)
    for k in range(nn):  # ascending k, as the reference sum over the dropped coefficients
        tail += np.where(k >= L, sq[:, k], 0.0)
    thr = ACC * float(_edge(nn))
    tl = np.sqrt(tail)
    fail = tl > thr
    # the vectorised tail may differ from numpy's pairwise float32 sum in the last bits, so rows close
    # to the threshold are decided with the reference helper
    amb = np.flatnonzero(np.abs(tl - thr) <= 1e-5 * np.maximum(1.0, thr))
    for i in amb:
        fail[i] = _l2_tail(ZZ[i], int(L[i]), False) > ACC[i] * float(_edge(nn))
    err = _classic_err(ZZ, ORIG, L)
    return np.where(fail, ACC + 1.0, err)


def classic_verify(ZZ, ORIG, ACC, LPRE):
    """Phase 2: the verify loop (up to MAX_VERIFY increments), vectorised per row."""
    L = np.array(LPRE, dtype=np.int64, copy=True)
    err = _classic_full(ZZ, ORIG, ACC, L)
    steps = np.zeros(L.shape[0], dtype=np.int64)
    for _ in range(MAX_VERIFY):
        act = (err > ACC) & (L < ZZ.shape[1]) & (steps < MAX_VERIFY)
        if not act.any():
            break
        idx = np.flatnonzero(act)
        L[idx] += 1
        steps[idx] += 1
        err[idx] = _classic_full(ZZ[idx], ORIG[idx], ACC[idx], L[idx])
    return L


def _run_pre(ZZ, ORIG, ACC, BS, basis2, LPRE, use_gpu, brk):
    if use_gpu:
        try:
            gpu_phases.pre_gpu(ZZ, ORIG, ACC, BS, basis2, LPRE, brk)
            return
        except RuntimeError as exc:
            gpu_u8.warn_once(f"GPU pre phase failed, using CPU: {exc}")
    pre_par(ZZ, ORIG, ACC, BS, basis2, LPRE, brk)


def _run_quant(ZZ, ORIG, ACC, SF, basis2, LIN, LF, ERR, RES, use_gpu):
    if use_gpu:
        try:
            gpu_phases.quant_gpu(ZZ, ORIG, ACC, SF, basis2, LIN, LF, ERR, RES)
            return
        except RuntimeError as exc:
            gpu_u8.warn_once(f"GPU quant phase failed, using CPU: {exc}")
    quant_par(ZZ, ORIG, ACC, SF, basis2, LIN, LF, ERR, RES)


def search_u8(ZZ, ORIG, ACC, SF, BS, basis2, LF, ERR, RES, brk=3):
    """Entry point used by blocks.py: phase 1 (GPU/CPU), phase 2 (scipy), phase 3 (GPU/CPU)."""
    m = ZZ.shape[0]
    LPRE = np.zeros(m, dtype=np.int64)
    use_gpu = m >= GPU_MIN_ROWS and gpu_phases.available(_edge(ZZ.shape[1]))
    _run_pre(ZZ, ORIG, ACC, BS, basis2, LPRE, use_gpu, brk)
    LV = classic_verify(ZZ, ORIG, ACC, LPRE)
    _run_quant(ZZ, ORIG, ACC, SF, basis2, LV, LF, ERR, RES, use_gpu)


def search_u8_cpu(ZZ, ORIG, ACC, SF, BS, basis2, LF, ERR, RES, brk=3):
    """The three phases on the CPU only (numba prange and scipy verification)."""
    LPRE = np.zeros(ZZ.shape[0], dtype=np.int64)
    pre_par(ZZ, ORIG, ACC, BS, basis2, LPRE, brk)
    LV = classic_verify(ZZ, ORIG, ACC, LPRE)
    quant_par(ZZ, ORIG, ACC, SF, basis2, LV, LF, ERR, RES)


def _dc_only_rows(ZZr, ORGr, ACCr, SFr, basis2):
    """Flagged rows whose final length is 1 (constant blocks, only the DC coefficient is coded).

    The flag is a rounding tie of the classic verification. The verification is repeated with the
    scipy IDCT (_classic_full), which decides the tie the way the decoder does; the quantisation check
    at L = 1 uses the arithmetic of the numba kernel. Returns (ok, error, RES): ok rows are final with
    L = 1, the others go to the full CPU search."""
    m = ZZr.shape[0]
    e1 = _classic_full(ZZr, ORGr, ACCr, np.ones(m, dtype=np.int64))
    ok = e1 <= ACCr
    dq = np.rint(ZZr[:, 0] * SFr) / SFr
    rec = dq[:, None] * basis2[0][None, :]
    v = np.clip(np.rint(rec + CU8), LO8, HI8)
    err = np.max(np.abs((ORGr + CU8) - v), axis=1)
    ok &= err <= ACCr
    RES = np.zeros((m, ZZr.shape[1]), dtype=np.int64)
    RES[:, 0] = np.rint(ZZr[:, 0] * SFr).astype(np.int64)
    return ok, err, RES


ACCEPT_CHUNK = 32768  # rows per scipy batch of the accept check (bounds memory on big images)


def accept_predicted(ZZ, ORIG, U, ACC, SF, LP, scales, BS=None, brk=3):
    """Predicted-L acceptance for rows U (accept_prediction_if_within_accuracy, as in refine_dct_array).

    Candidates round(L_hat * s) from the uint8 scale cascade are tried in ascending order; the first
    one whose decoder-equivalent error (quantised coefficients, scipy IDCT) is within ACC is taken.
    The largest candidate is checked first: if it fails, no smaller one is tried. The accepted length
    then gets the backscan (BS steps per row; brk consecutive failures stop it, brk=None disables it).

    ZZ/ORIG are the full batch arrays, U the row indices, ACC/SF/LP/BS aligned with U.
    Returns (LF, ERR, RES, OK) aligned with U; rows with OK False need the search."""
    u = U.size
    NN = TOTAL = ZZ.shape[1]
    LF = np.zeros(u, dtype=np.int64)
    ERR = np.zeros(u)
    RES = np.zeros((u, NN), dtype=np.int64)
    OK = np.zeros(u, dtype=bool)
    S = np.asarray(scales, dtype=np.float64)
    if u == 0 or S.size == 0:
        return LF, ERR, RES, OK
    # candidate list: round(L_hat * s) clamped to [1, TOTAL], ascending; L == TOTAL dropped
    cand = np.clip(np.rint(LP.astype(np.float64)[:, None] * S[None, :]), 1, TOTAL).astype(np.int64)
    cand.sort(axis=1)
    nval = (cand < TOTAL).sum(axis=1)
    for c0 in range(0, u, ACCEPT_CHUNK):
        c1 = min(u, c0 + ACCEPT_CHUNK)
        rows = U[c0:c1]
        orig = ORIG[rows]
        sf = SF[c0:c1]
        acc = ACC[c0:c1]
        cc = cand[c0:c1]
        nv = nval[c0:c1]
        zz = ZZ[rows]
        q = np.rint(zz * sf[:, None]) / sf[:, None]  # dequantised as the decoder sees it
        qi = np.rint(zz * sf[:, None]).astype(np.int64)
        has = np.flatnonzero(nv > 0)
        if has.size == 0:
            continue
        e_big = _classic_err(q[has], orig[has], cc[has, nv[has] - 1])
        alive = np.zeros(c1 - c0, dtype=bool)
        alive[has[e_big <= acc[has]]] = True
        done = np.zeros(c1 - c0, dtype=bool)
        for j in range(S.size):
            act = np.flatnonzero(alive & ~done & (nv > j))
            if act.size == 0:
                break
            e = _classic_err(q[act], orig[act], cc[act, j])
            hit = e <= acc[act]
            ih = act[hit]
            LF[c0 + ih] = cc[ih, j]
            ERR[c0 + ih] = e[hit]
            OK[c0 + ih] = True
            done[ih] = True
        # backscan of the accepted length: try L-1 ... L-minimize_backscan per row, keep every shorter
        # length that passes, stop after brk failures in a row
        if done.any() and brk is not None and BS is not None:
            L0 = LF[c0:c1].copy()
            bsv = BS[c0:c1]
            streak = np.zeros(c1 - c0, dtype=np.int64)
            live = done.copy()
            for step in range(1, int(bsv.max()) + 1):
                cur = np.flatnonzero(live & (bsv >= step) & (L0 - step >= 1))
                if cur.size == 0:
                    break
                kk = L0[cur] - step
                e = _classic_err(q[cur], orig[cur], kk)
                ok = e <= acc[cur]
                good = cur[ok]
                LF[c0 + good] = kk[ok]
                ERR[c0 + good] = e[ok]
                streak[good] = 0
                bad = cur[~ok]
                streak[bad] += 1
                if brk > 0:
                    live[bad[streak[bad] >= brk]] = False
        if done.any():
            ih = np.flatnonzero(done)
            keep = np.arange(NN)[None, :] < LF[c0 + ih][:, None]
            RES[c0 + ih] = np.where(keep, qi[ih], 0)
    return LF, ERR, RES, OK


def search_rows_pred(ZZ, ORG, U, ACC, SF, BS, basis2, LP, scales, brk=3):
    """Predicted-L search for rows U of one channel: accept_predicted (with backscan), then the CPU
    search for the rows it did not accept (no GPU session on this path). Returns (LF, ERR, RES)
    aligned with U."""
    LF, ERR, RES, OK = accept_predicted(ZZ, ORG, U, ACC, SF, LP, scales, BS, brk)
    rest = np.flatnonzero(~OK)
    if rest.size:
        lf, er, rs, _ = search_rows(None, 0, U[rest], ZZ, ORG, ACC[rest], SF[rest], BS[rest], basis2,
                                    brk)
        LF[rest] = lf
        ERR[rest] = er
        RES[rest] = rs
    return LF, ERR, RES


def search_rows(sess, ch, U, ZZ, ORG, ACC, SF, BS, basis2, brk=3):
    """Search rows U of one channel (ZZ/ORG hold all rows of the batch).

    With a GPU session the batch is already in VRAM: only U and the per-row parameters are sent. The
    kernel decides every row it can; the flagged rows (IDCT rounding tie or L2 precheck boundary) are
    recomputed here with scipy and written back into VRAM, so the session stays consistent for the
    cm=6 RGB check. Returns (LF, ERR, RES, used_session)."""
    u = U.size
    if sess is not None:
        try:
            LF, ERR, RES, FLAG = sess.search(ch, U, ACC, SF, BS, brk=brk)
            fl = np.flatnonzero(FLAG)
            if fl.size:
                rows = U[fl]
                lf2 = np.zeros(fl.size, dtype=np.int64)
                er2 = np.zeros(fl.size)
                rs2 = np.zeros((fl.size, ZZ.shape[1]), dtype=np.int64)
                # flagged constant blocks (final L = 1) are decided here, vectorised; the rest goes
                # through the CPU search
                done = np.zeros(fl.size, dtype=bool)
                one = np.flatnonzero(LF[fl] == 1)
                if one.size:
                    ok, e1, r1 = _dc_only_rows(ZZ[rows[one]], ORG[rows[one]], ACC[fl[one]],
                                               SF[fl[one]], basis2)
                    k = one[ok]
                    lf2[k] = 1
                    er2[k] = e1[ok]
                    rs2[k] = r1[ok]
                    done[k] = True
                rest = np.flatnonzero(~done)
                if rest.size:
                    lf3 = np.zeros(rest.size, dtype=np.int64)
                    er3 = np.zeros(rest.size)
                    rs3 = np.zeros((rest.size, ZZ.shape[1]), dtype=np.int64)
                    search_u8_cpu(np.ascontiguousarray(ZZ[rows[rest]]),
                                  np.ascontiguousarray(ORG[rows[rest]]),
                                  np.ascontiguousarray(ACC[fl[rest]]),
                                  np.ascontiguousarray(SF[fl[rest]]),
                                  np.ascontiguousarray(BS[fl[rest]]), basis2, lf3, er3, rs3, brk)
                    lf2[rest] = lf3
                    er2[rest] = er3
                    rs2[rest] = rs3
                LF[fl] = lf2
                ERR[fl] = er2
                RES[fl] = rs2
                sess.patch(ch, fl, rs2)
            return LF, ERR, RES, True
        except RuntimeError as exc:  # fallback on CPU, same result
            gpu_u8.warn_once(f"GPU session search failed, using CPU: {exc}")
    LF = np.zeros(u, dtype=np.int64)
    ERR = np.zeros(u)
    RES = np.zeros((u, ZZ.shape[1]), dtype=np.int64)
    search_u8(np.ascontiguousarray(ZZ[U]), np.ascontiguousarray(ORG[U]), ACC, SF, BS, basis2,
              LF, ERR, RES, brk)
    return LF, ERR, RES, False
