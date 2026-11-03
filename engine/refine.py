"""
FIDWAC v2 - DCT Refinement (Binary Search + L Prediction)
===========================================================

Optimized implementation with:
- Incremental binary search (tensordot instead of full IDCT)
- L2 precheck for fast rejection
- Backscan for coefficient minimization
"""

import os
import math
from typing import Tuple, Optional

import numpy as np
import numba as nb

from core.dct import from_zigzag, idct2, _get_idct_basis
from config import Config, load_config
from engine.utils import integer_range

# ---------------------------------------------------------------------------
# Numba JIT helpers for hot-path binary search (8×8 blocks)
# ---------------------------------------------------------------------------
# The numba cache must live on a Linux filesystem. Reads through the Windows
# mounts (/mnt/...) add about 500 ms of latency. Prefer tmpfs (/dev/shm) and
# fall back to /tmp when it is unavailable.
_NB_CACHE_DIR = (
    "/dev/shm/fidwac_numba_cache" if os.path.isdir("/dev/shm") else "/tmp/fidwac_numba_cache"
)
os.makedirs(_NB_CACHE_DIR, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", _NB_CACHE_DIR)

@nb.njit(cache=True, fastmath=True)
def _nb_step_add(diff: np.ndarray, basis_k: np.ndarray, ck: float) -> float:
    """diff += basis_k * ck  (removing coefficient from recon → diff grows)"""
    n = diff.shape[0]
    mx, mn = diff[0, 0], diff[0, 0]
    for i in range(n):
        for j in range(n):
            v = diff[i, j] + basis_k[i, j] * ck
            diff[i, j] = v
            if v > mx:
                mx = v
            if v < mn:
                mn = v
    return mx if mx > -mn else -mn


@nb.njit(cache=True, fastmath=True)
def _nb_step_sub(diff: np.ndarray, basis_k: np.ndarray, ck: float) -> float:
    """diff -= basis_k * ck  (adding coefficient to recon → diff shrinks)"""
    n = diff.shape[0]
    mx, mn = diff[0, 0], diff[0, 0]
    for i in range(n):
        for j in range(n):
            v = diff[i, j] - basis_k[i, j] * ck
            diff[i, j] = v
            if v > mx:
                mx = v
            if v < mn:
                mn = v
    return mx if mx > -mn else -mn


@nb.njit(cache=True, fastmath=True)
def _nb_jump_diff(
    diff: np.ndarray, basis: np.ndarray, coeffs: np.ndarray, from_L: int, to_L: int
) -> float:
    """Incrementally shift diff from position from_L to to_L (arbitrary jump)."""
    n = diff.shape[0]
    if from_L < to_L:
        for k in range(from_L, to_L):
            ck = coeffs[k]
            for i in range(n):
                for j in range(n):
                    diff[i, j] -= basis[k, i, j] * ck
    elif from_L > to_L:
        for k in range(to_L, from_L):
            ck = coeffs[k]
            for i in range(n):
                for j in range(n):
                    diff[i, j] += basis[k, i, j] * ck
    mx, mn = diff[0, 0], diff[0, 0]
    for i in range(n):
        for j in range(n):
            v = diff[i, j]
            if v > mx:
                mx = v
            if v < mn:
                mn = v
    return mx if mx > -mn else -mn


@nb.njit(cache=True, fastmath=True)
def _nb_max_err(diff: np.ndarray) -> float:
    n = diff.shape[0]
    mx, mn = diff[0, 0], diff[0, 0]
    for i in range(n):
        for j in range(n):
            v = diff[i, j]
            if v > mx:
                mx = v
            if v < mn:
                mn = v
    return mx if mx > -mn else -mn


# Compile on import (8x8 dummy block) so the first real block does not wait
_d = np.zeros((8, 8), dtype=np.float64)
_b = np.zeros((64, 8, 8), dtype=np.float64)
_c = np.zeros(64, dtype=np.float64)
_nb_max_err(_d)
_nb_step_add(_d, _b[0], 0.0)
_nb_step_sub(_d, _b[0], 0.0)
_nb_jump_diff(_d, _b, _c, 0, 1)
del _d, _b, _c


@nb.njit(cache=True)
def _nb_eval_fast(coeffs, basis2, orig, L, accuracy, use_l2, int_mode, c, lo, hi):
    """Fast-basis error for L coefficients in one compiled pass (same maths as the numpy path).

    coeffs: (N*N,) float64 zigzag coefficients; basis2: (N*N, N*N) float64 zigzag IDCT basis;
    orig: (N*N,) float64 centred block. int_mode: add c, round, clip to [lo, hi] (integer accuracy).
    The L2 precheck returns accuracy + 1 when the tail energy proves the error exceeds accuracy."""
    total = coeffs.shape[0]
    nn = orig.shape[0]
    if use_l2:
        s = 0.0
        for k in range(L, total):
            t = np.float32(coeffs[k])
            s += np.float64(t * t)
        if np.sqrt(s) > accuracy * np.sqrt(nn):
            return accuracy + 1.0
    rec = np.zeros(nn)
    for k in range(L):
        ck = coeffs[k]
        for p in range(nn):
            rec[p] += ck * basis2[k, p]
    mx = 0.0
    for p in range(nn):
        if int_mode:
            v = np.round(rec[p] + c)
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            d = orig[p] + c - v
        else:
            d = orig[p] - rec[p]
        if d < 0.0:
            d = -d
        if d > mx:
            mx = d
    return mx


_nb_eval_fast(np.zeros(64), np.zeros((64, 64)), np.zeros(64), 1, 1.0, True, True, 128.0, 0.0, 255.0)
_BASIS2 = {}  # (n, dct_type) -> (N*N, N*N) zigzag IDCT basis for _nb_eval_fast


@nb.njit(cache=True)
def _nb_tail_l2(coeffs, L):
    """L2 norm of the coefficients dropped at length L (float32 squares, float64 sum)."""
    s = 0.0
    for k in range(L, coeffs.shape[0]):
        t = np.float32(coeffs[k])
        s += np.float64(t * t)
    return np.sqrt(s)


_nb_tail_l2(np.zeros(64), 1)


def _uses_kernel(use_fast_basis: bool, float32_store: bool, n: int) -> bool:
    """True when _eval_error_for_len takes the compiled fast path."""
    return use_fast_basis and not float32_store and n <= 16


def _l2_tail(org_dct_zigzag: np.ndarray, L: int, kernel: bool) -> float:
    """Tail energy for the L2 precheck, with the arithmetic of the path that evaluates the error."""
    if kernel:
        return float(_nb_tail_l2(np.ascontiguousarray(org_dct_zigzag, dtype=np.float64), L))
    tail = org_dct_zigzag[L:].astype(np.float32, copy=False)
    return float(np.sqrt(np.sum(tail * tail)))


def _stored_float32_diff(original_matrix: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """original - stored value, where stored is the float64 reconstruction rounded to float32, as the
    cm=3 decoder writes it. A pixel whose reconstruction lies within a relative 1e-9 of a float32
    rounding midpoint takes the worse of the two neighbours, which covers the tiny float64 differences
    between this reconstruction and the decoder's own IDCT."""
    orig = original_matrix.astype(np.float64, copy=False)
    r32 = recon.astype(np.float32)
    diff = orig - r32
    away = np.where(recon > r32, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
    alt = np.nextafter(r32, away)
    near = np.abs(np.abs(recon - r32) - 0.5 * np.abs(alt.astype(np.float64) - r32)) <= 1e-9 * np.maximum(
        np.abs(recon), 1.0
    )
    if np.any(near):
        diff_alt = orig - alt
        diff = np.where(near & (np.abs(diff_alt) > np.abs(diff)), diff_alt, diff)
    return diff


def _eval_error_for_len(
    org_dct_zigzag: np.ndarray,
    original_matrix: np.ndarray,
    L: int,
    n: int,
    dct_type: int,
    accuracy: float,
    use_fast_basis: bool = True,
    use_l2_precheck: bool = True,
    src_dtype: Optional[str] = None,
    float32_store: bool = False,
) -> float:
    """
    Compute max reconstruction error for L coefficients.

    float32_store: float data (cm=3) is decoded to a float32 raster - measure the error of the value
    rounded to float32, i.e. the stored one (final checks only; the searches keep the float64 error).

    OPTIMIZATIONS:
    - L2 precheck: fast rejection without reconstruction
    - Fast basis: tensordot instead of full IDCT
    """
    L = int(max(1, min(L, len(org_dct_zigzag))))
    _rng = integer_range(src_dtype)
    kernel = _uses_kernel(use_fast_basis, float32_store, n)

    # L2 precheck - only for fast rejection (error definitely too large)
    if use_l2_precheck and _l2_tail(org_dct_zigzag, L, kernel) > accuracy * np.sqrt(n * n):
        return accuracy + 1.0

    # Compiled fast path (N <= 16; for N = 32 the BLAS tensordot below is faster)
    if kernel:
        basis2 = _BASIS2.get((n, dct_type))
        if basis2 is None:
            basis2 = np.ascontiguousarray(_get_idct_basis(n, dct_type).reshape(n * n, n * n))
            _BASIS2[(n, dct_type)] = basis2
        c, lo, hi = _rng if _rng is not None else (0.0, 0.0, 0.0)
        return float(
            _nb_eval_fast(
                np.ascontiguousarray(org_dct_zigzag, dtype=np.float64),
                basis2,
                np.ascontiguousarray(original_matrix, dtype=np.float64).reshape(-1),
                L,
                float(accuracy),
                False,
                _rng is not None,
                c,
                lo,
                hi,
            )
        )


    # Fast path with precomputed basis
    if use_fast_basis:
        coeffs = org_dct_zigzag[:L].astype(np.float64, copy=False)
        basis = _get_idct_basis(n, dct_type)
        recon = np.tensordot(coeffs, basis[:L], axes=(0, 0))
        if _rng is not None:
            # original_matrix is centred; restore the centre, clip to the dtype range and round,
            # then measure the integer error the decoder will produce
            _c, _lo, _hi = _rng
            orig_u8 = original_matrix + _c
            recon_u8 = np.clip(np.round(recon + _c), _lo, _hi)
            diff = orig_u8 - recon_u8
        else:
            if float32_store:
                diff = _stored_float32_diff(original_matrix, recon)
            else:
                diff = original_matrix.astype(np.float64, copy=False) - recon
        return max(float(diff.max()), float(-diff.min()))

    # Classic path
    array = np.zeros(len(org_dct_zigzag), dtype=np.float64)
    array[:L] = org_dct_zigzag[:L]
    reconstructed = from_zigzag(array, n)
    idct_reconstructed = idct2(reconstructed, dct_type)
    if _rng is not None:
        _c, _lo, _hi = _rng
        orig_u8 = original_matrix + _c
        recon_u8 = np.clip(np.round(idct_reconstructed + _c), _lo, _hi)
        diff = orig_u8 - recon_u8
    else:
        if float32_store:
            diff = _stored_float32_diff(original_matrix, idct_reconstructed)
        else:
            diff = original_matrix - idct_reconstructed
    return max(float(diff.max()), float(-diff.min()))


def _binary_search_min_len_from_start(
    org_dct_zigzag: np.ndarray,
    original_matrix: np.ndarray,
    accuracy: float,
    start_L: int,
    lo: int,
    hi: int,
    n: int,
    dct_type: int,
    skip_queue: Optional[list] = None,
    src_dtype: Optional[str] = None,
    eval_fn=None,
) -> Tuple[int, int]:
    ev = eval_fn if eval_fn is not None else _eval_error_for_len
    lo = max(1, int(lo))
    hi = min(len(org_dct_zigzag), int(hi))
    if lo > hi:
        return hi, 0
    steps, ans = 0, hi
    if skip_queue is not None and len(skip_queue) > 0:
        for L_test in skip_queue:
            if L_test < lo or L_test > hi or L_test == 0:
                continue
            steps += 1
            err = ev(
                org_dct_zigzag,
                original_matrix,
                L_test,
                n,
                dct_type,
                accuracy,
                use_fast_basis=True,
                src_dtype=src_dtype,
            )
            if err <= accuracy:
                ans = L_test
                hi = L_test - 1
            else:
                lo = L_test + 1
            if lo > hi:
                return ans, steps
    elif lo <= start_L <= hi:
        steps += 1
        err = ev(
            org_dct_zigzag,
            original_matrix,
            start_L,
            n,
            dct_type,
            accuracy,
            use_fast_basis=True,
            src_dtype=src_dtype,
        )
        if err <= accuracy:
            ans = start_L
            hi = start_L - 1
        else:
            lo = start_L + 1
    if lo > hi:
        return ans, steps
    l, r = lo, hi
    while l <= r:
        mid = (l + r) // 2
        steps += 1
        err = ev(
            org_dct_zigzag,
            original_matrix,
            mid,
            n,
            dct_type,
            accuracy,
            use_fast_basis=True,
            src_dtype=src_dtype,
        )
        if err <= accuracy:
            ans = mid
            r = mid - 1
        else:
            l = mid + 1
    return l, steps


def _binary_search_min_len_incremental(
    org_dct_zigzag: np.ndarray,
    original_matrix: np.ndarray,
    accuracy: float,
    start_L: int,
    lo: int,
    hi: int,
    n: int,
    dct_type: int,
) -> Tuple[int, int]:
    """
    Binary search with incremental reconstruction, starting from prediction.

    Same strategy as float path (_binary_search_min_len_from_start):
    1. Check start_L → narrow [lo, hi] range
    2. Standard binary search on narrowed range using _nb_jump_diff

    Returns
    -------
    Tuple[int, int]
        (L_found, steps)
    """
    lo = max(1, int(lo))
    hi = min(len(org_dct_zigzag), int(hi))

    if lo > hi:
        return hi, 0

    basis = _get_idct_basis(n, dct_type)
    coeffs = org_dct_zigzag.astype(np.float64, copy=False)

    # Initialize at start_L
    cur_L = max(lo, min(start_L, hi))
    if cur_L > 0:
        recon = np.tensordot(coeffs[:cur_L], basis[:cur_L], axes=(0, 0))
    else:
        recon = np.zeros((n, n), dtype=np.float64)

    diff = (original_matrix.astype(np.float64, copy=False) - recon).copy()
    steps = 1

    err_start = float(_nb_max_err(diff))

    # Step 1: Check start_L and narrow range
    if err_start <= accuracy:
        ans = cur_L
        hi = cur_L - 1
    else:
        ans = hi
        lo = cur_L + 1

    if lo > hi:
        return ans, steps

    # Step 2: Standard binary search on [lo, hi]
    l, r = lo, hi
    while l <= r:
        mid = (l + r) // 2
        steps += 1
        err = float(_nb_jump_diff(diff, basis, coeffs, cur_L, mid))
        cur_L = mid
        if err <= accuracy:
            ans = mid
            r = mid - 1
        else:
            l = mid + 1

    return ans, steps


def _binary_search_min_len(
    org_dct_zigzag: np.ndarray,
    original_matrix: np.ndarray,
    accuracy: float,
    n: int,
    dct_type: int,
    lo: Optional[int] = None,
    hi: Optional[int] = None,
    use_fast_basis: bool = True,
    eval_fn=None,
) -> Tuple[int, int]:
    """
    Standard binary search (non-incremental).
    """
    ev = eval_fn if eval_fn is not None else _eval_error_for_len
    if lo is None:
        lo = 1
    if hi is None:
        hi = len(org_dct_zigzag)

    lo = max(1, int(lo))
    hi = min(len(org_dct_zigzag), int(hi))

    if lo > hi:
        return hi, 0

    steps = 0
    ans = hi

    while lo <= hi:
        mid = (lo + hi) // 2
        steps += 1

        try:
            err = ev(
                org_dct_zigzag,
                original_matrix,
                mid,
                n,
                dct_type,
                accuracy,
                use_fast_basis=use_fast_basis,
            )
        except (ValueError, np.linalg.LinAlgError):
            err = float("inf")

        if err <= accuracy:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return ans, steps


def refine_dct_array(
    org_dct_zigzag: np.ndarray,
    accuracy: float,
    original_matrix: np.ndarray,
    n: int,
    dct_type: int = 2,
    scaling_factor: int = 100,
    config: Optional[Config] = None,
    predicted_L: Optional[int] = None,
    skip_queue: Optional[list] = None,
    src_dtype: Optional[str] = None,
    err_cache: Optional[dict] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Optimize the number of DCT coefficients for required accuracy.

    OPTIMIZED VERSION with:
    - Incremental binary search
    - L prediction as starting point
    - Backscan for minimization
    - Verification with full IDCT

    Parameters
    ----------
    org_dct_zigzag : np.ndarray
        DCT coefficients in zigzag order
    accuracy : float
        Required accuracy (max error)
    original_matrix : np.ndarray
        Original matrix before DCT
    n : int
        Block size
    dct_type : int
        DCT type
    scaling_factor : int
        Scaling factor for integer conversion
    config : Config, optional
        Configuration
    predicted_L : int, optional
        Predicted L value as starting point for binary search

    Returns
    -------
    Tuple[np.ndarray, float, np.ndarray]
        (compressed_array, max_error, reconstructed_matrix)
    """
    if config is None:
        config = load_config()

    total_len = len(org_dct_zigzag)

    # Options from config
    use_fast_basis = config.performance.fast_eval_basis
    use_incremental = config.performance.incremental_backscan
    verify_full = config.performance.verify_with_full_idct
    backscan_base = config.model.minimize_backscan
    backscan_break = config.model.backscan_break_after
    accept_if_within = config.model.accept_prediction_if_within_accuracy

    # Starting point
    if predicted_L is not None and predicted_L > 0:
        start_L = max(1, min(int(predicted_L), total_len))
    else:
        # Default starting point - middle of range
        start_L = total_len // 2

    _is_uint8 = src_dtype in ("uint8", "int8", "uint16", "int16")
    _store32 = integer_range(src_dtype) is None  # float data: decoded to a float32 raster (cm=3)


    def _eval(zz, orig, L, n_, dct_, acc, use_fast_basis=True, use_l2_precheck=True,
              src_dtype=None, float32_store=False):
        """err(L) memoised per (block, channel): the multiplier cascade calls this refine with the
        same zigzag and other accuracies, so repeated evaluations hit the cache instead of re-running
        the IDCT. The cache holds accuracy-independent values only - the tail energy of the L2
        precheck and the error computed without the precheck - and the precheck is applied again at
        the current accuracy, so every call returns exactly what _eval_error_for_len would.
        Raw and quantized zigzags get separate keys (the quantised zigzag is fixed by L)."""
        if err_cache is None:
            return _eval_error_for_len(
                zz, orig, L, n_, dct_, acc,
                use_fast_basis=use_fast_basis, use_l2_precheck=use_l2_precheck,
                src_dtype=src_dtype, float32_store=float32_store,
            )
        tag = "r" if zz is org_dct_zigzag else "q"
        L = int(max(1, min(L, len(zz))))
        if use_l2_precheck:
            kernel = _uses_kernel(use_fast_basis, float32_store, n_)
            tkey = (tag, L, kernel)
            tail = err_cache.get(tkey)
            if tail is None:
                tail = err_cache[tkey] = _l2_tail(zz, L, kernel)
            if tail > acc * np.sqrt(n_ * n_):
                return acc + 1.0
        key = (tag, L, bool(use_fast_basis), bool(float32_store), src_dtype)
        v = err_cache.get(key)
        if v is None:
            v = err_cache[key] = _eval_error_for_len(
                zz, orig, L, n_, dct_, acc,
                use_fast_basis=use_fast_basis, use_l2_precheck=False,
                src_dtype=src_dtype, float32_store=float32_store,
            )
        return v

    def _decoder_error(L):
        """Max error of the stored (quantized) coefficients reconstructed with the full IDCT,
        i.e. exactly what the decoder will produce after unpacking."""
        dq = np.round(org_dct_zigzag[:L] * scaling_factor).astype(np.float64) / scaling_factor
        dq_zigzag = np.zeros_like(org_dct_zigzag, dtype=np.float64)
        dq_zigzag[:L] = dq
        return _eval(
            dq_zigzag,
            original_matrix,
            L,
            n,
            dct_type,
            accuracy,
            use_fast_basis=False,
            use_l2_precheck=False,
            src_dtype=src_dtype,
            float32_store=_store32,
        )

    # accept_prediction_if_within_accuracy: if the prediction meets the accuracy, accept it without the
    # binary search. For uint8 an increasing cascade of L scales is tried first. An accepted candidate is
    # always re-verified with the decoder-equivalent reconstruction.
    _check_accept = accept_if_within
    if _check_accept and predicted_L is not None and predicted_L > 0:
        try:
            if _is_uint8:
                raw_scales = getattr(config.model, "uint8_L_prediction_scales", [1.5])
                if not isinstance(raw_scales, (list, tuple)):
                    raw_scales = [raw_scales]
                candidate_lengths = []
                for scale in raw_scales:
                    try:
                        candidate_lengths.append(int(round(int(predicted_L) * float(scale))))
                    except (TypeError, ValueError):
                        continue
            else:
                candidate_lengths = [start_L]

            # Convert to a sorted list of unique candidate lengths
            candidates = sorted(
                list(set(max(1, min(int(l), total_len)) for l in candidate_lengths))
            )
            if _is_uint8:
                candidates = [l for l in candidates if l < total_len]

            if candidates:
                # pre-check: evaluate the largest candidate first. If it fails, no smaller candidate can meet
                # the accuracy, so one check replaces the sequential checks.
                max_cand = candidates[-1]
                if _is_uint8:
                    _c_dq = (
                        np.round(org_dct_zigzag[:max_cand] * scaling_factor).astype(np.float64)
                        / scaling_factor
                    )
                    _dq_z = np.zeros_like(org_dct_zigzag, dtype=np.float64)
                    _dq_z[:max_cand] = _c_dq
                    err_max = _eval(
                        _dq_z,
                        original_matrix,
                        max_cand,
                        n,
                        dct_type,
                        accuracy,
                        use_fast_basis=use_fast_basis,
                        use_l2_precheck=False,
                        src_dtype=src_dtype,
                    )
                else:
                    err_max = _eval(
                        org_dct_zigzag,
                        original_matrix,
                        max_cand,
                        n,
                        dct_type,
                        accuracy,
                        use_fast_basis=use_fast_basis,
                        src_dtype=src_dtype,
                    )

                best_L = None
                best_err = None
                if err_max <= accuracy:
                    # At least one candidate works.
                    if accept_if_within:
                        # iterate the candidates in ascending order and accept the first one that meets the accuracy
                        for candidate_L in candidates:
                            if _is_uint8:
                                _c_dq = (
                                    np.round(org_dct_zigzag[:candidate_L] * scaling_factor).astype(
                                        np.float64
                                    )
                                    / scaling_factor
                                )
                                _dq_z = np.zeros_like(org_dct_zigzag, dtype=np.float64)
                                _dq_z[:candidate_L] = _c_dq
                                err_candidate = _eval(
                                    _dq_z,
                                    original_matrix,
                                    candidate_L,
                                    n,
                                    dct_type,
                                    accuracy,
                                    use_fast_basis=use_fast_basis,
                                    use_l2_precheck=False,
                                    src_dtype=src_dtype,
                                )
                            else:
                                err_candidate = _eval(
                                    org_dct_zigzag,
                                    original_matrix,
                                    candidate_L,
                                    n,
                                    dct_type,
                                    accuracy,
                                    use_fast_basis=use_fast_basis,
                                    src_dtype=src_dtype,
                                )
                            if err_candidate <= accuracy:
                                err_decoded = _decoder_error(candidate_L)
                                if err_decoded > accuracy:
                                    continue
                                best_L = candidate_L
                                best_err = err_decoded
                                coeffs = org_dct_zigzag[:best_L]
                                result = np.round(coeffs * scaling_factor).astype(int)
                                return result, best_err, None
                    else:
                        # binary search over the candidate list for the smallest working candidate
                        lo_idx, hi_idx = 0, len(candidates) - 1
                        best_idx = hi_idx  # we know candidates[-1] works
                        best_err = err_max

                        while lo_idx <= hi_idx:
                            mid_idx = (lo_idx + hi_idx) // 2
                            candidate_L = candidates[mid_idx]

                            if mid_idx == len(candidates) - 1:
                                err_mid = err_max
                            else:
                                if _is_uint8:
                                    _c_dq = (
                                        np.round(org_dct_zigzag[:candidate_L] * scaling_factor).astype(
                                            np.float64
                                        )
                                        / scaling_factor
                                    )
                                    _dq_z = np.zeros_like(org_dct_zigzag, dtype=np.float64)
                                    _dq_z[:candidate_L] = _c_dq
                                    err_mid = _eval(
                                        _dq_z,
                                        original_matrix,
                                        candidate_L,
                                        n,
                                        dct_type,
                                        accuracy,
                                        use_fast_basis=use_fast_basis,
                                        use_l2_precheck=False,
                                        src_dtype=src_dtype,
                                    )
                                else:
                                    err_mid = _eval(
                                        org_dct_zigzag,
                                        original_matrix,
                                        candidate_L,
                                        n,
                                        dct_type,
                                        accuracy,
                                        use_fast_basis=use_fast_basis,
                                        src_dtype=src_dtype,
                                    )

                            if err_mid <= accuracy:
                                best_idx = mid_idx
                                best_err = err_mid
                                hi_idx = mid_idx - 1  # try to find a smaller working candidate
                            else:
                                lo_idx = mid_idx + 1  # try larger candidates

                        best_L = candidates[best_idx]
                        err_decoded = _decoder_error(best_L)
                        if err_decoded <= accuracy:
                            best_err = err_decoded
                            coeffs = org_dct_zigzag[:best_L]
                            result = np.round(coeffs * scaling_factor).astype(int)
                            return result, best_err, None
        except (ValueError, IndexError):
            pass

    # Binary search
    if predicted_L is not None and predicted_L > 0 and use_fast_basis and use_incremental:
        # incremental search with the prediction as the starting point (a linear window around it)
        L_found, _ = _binary_search_min_len_incremental(
            org_dct_zigzag,
            original_matrix,
            accuracy,
            start_L=int(predicted_L),
            lo=1,
            hi=total_len,
            n=n,
            dct_type=dct_type,
        )
    elif predicted_L is not None and predicted_L > 0:
        # binary search with the prediction as the starting point (fallback when the incremental path is off)
        L_found, _ = _binary_search_min_len_from_start(
            org_dct_zigzag,
            original_matrix,
            accuracy,
            start_L=predicted_L,
            lo=1,
            hi=total_len,
            n=n,
            dct_type=dct_type,
            skip_queue=skip_queue,
            src_dtype=src_dtype,
            eval_fn=_eval,
        )
    else:
        # no prediction: plain binary search (finds the global minimum L)
        L_found, _ = _binary_search_min_len(
            org_dct_zigzag,
            original_matrix,
            accuracy,
            n,
            dct_type,
            lo=1,
            hi=total_len,
            use_fast_basis=use_fast_basis,
            eval_fn=_eval,
        )

    # Backscan for minimization
    if backscan_base > 0 and L_found > 1:
        # backscan scales with the block size
        scale_factor = (n / 8) * math.sqrt(n / 8)
        backscan = int(backscan_base * scale_factor)
        start = max(1, L_found - backscan)

        _rng = integer_range(src_dtype)

        if use_fast_basis and use_incremental:
            basis = _get_idct_basis(n, dct_type)
            coeffs = org_dct_zigzag.astype(np.float32, copy=False)
            recon = np.tensordot(coeffs[:L_found], basis[:L_found], axes=(0, 0))
            diff = original_matrix.astype(np.float32, copy=False) - recon
            _tmp = np.empty((n, n), dtype=np.float64)  # pre-alloc buffer

            new_L = L_found
            fail_streak = 0
            for k in range(L_found - 1, start - 1, -1):
                np.multiply(basis[k], coeffs[k], out=_tmp)
                np.add(diff, _tmp, out=diff)
                if _rng is not None:
                    # reference formula (keeps uint8 archives byte-identical); the full IDCT and the
                    # post-quantisation checks below measure the true decoder error
                    _c, _lo, _hi = _rng
                    orig_u8 = original_matrix + _c
                    recon_u8 = np.clip(np.round(diff + original_matrix + _c), _lo, _hi)
                    err = max(
                        float((orig_u8 - recon_u8).max()),
                        float(-(orig_u8 - recon_u8).min()),
                    )
                else:
                    err = max(float(diff.max()), float(-diff.min()))
                if err <= accuracy:
                    new_L = k
                    fail_streak = 0
                else:
                    fail_streak += 1
                    # stop after N consecutive failures (0 = never stop)
                    if backscan_break > 0 and fail_streak >= backscan_break:
                        break
            L_found = new_L
        else:
            for test_L in range(start, L_found):
                try:
                    test_error = _eval(
                        org_dct_zigzag,
                        original_matrix,
                        test_L,
                        n,
                        dct_type,
                        accuracy,
                        use_fast_basis=use_fast_basis,
                        src_dtype=src_dtype,
                    )
                    if test_error <= accuracy:
                        L_found = test_L
                        break
                except (ValueError, IndexError):
                    continue

    # Verification with full IDCT
    if verify_full:
        err_full = _eval(
            org_dct_zigzag,
            original_matrix,
            L_found,
            n,
            dct_type,
            accuracy,
            use_fast_basis=False,  # Full IDCT
            src_dtype=src_dtype,
            float32_store=_store32,
        )

        # error too large: increase L (at most 5 steps)
        refine_count = 0
        while err_full > accuracy and L_found < total_len and refine_count < 5:
            L_found += 1
            refine_count += 1
            err_full = _eval(
                org_dct_zigzag,
                original_matrix,
                L_found,
                n,
                dct_type,
                accuracy,
                use_fast_basis=False,
                src_dtype=src_dtype,
                float32_store=_store32,
            )

    # Build result
    coeffs = org_dct_zigzag[:L_found]
    result = np.round(coeffs * scaling_factor).astype(int)

    # Verify the error AFTER quantisation: the binary search does not account for the rounding error,
    # and the dequantised coefficients differ from org_dct_zigzag[:L]
    def _eval_quantized_error(L):
        """Reconstruction error from the dequantised coefficients."""
        c = org_dct_zigzag[:L]
        dq = np.round(c * scaling_factor).astype(np.float64) / scaling_factor
        dq_zigzag = np.zeros_like(org_dct_zigzag, dtype=np.float64)
        dq_zigzag[:L] = dq
        return _eval(
            dq_zigzag,
            original_matrix,
            L,
            n,
            dct_type,
            accuracy,
            use_fast_basis=use_fast_basis,
            use_l2_precheck=False,
            src_dtype=src_dtype,
            float32_store=_store32,
        )

    max_error = _eval_quantized_error(L_found)

    # rounding gave too large an error: increase L until the accuracy is met
    if max_error > accuracy:
        quant_refine = 0
        while max_error > accuracy and L_found < total_len and quant_refine < 16:
            L_found += 1
            quant_refine += 1
            result = np.round(org_dct_zigzag[:L_found] * scaling_factor).astype(int)
            max_error = _eval_quantized_error(L_found)

    return result, max_error, None  # Don't return reconstruction - not needed


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "refine_dct_array",
]
