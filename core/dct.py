"""
FIDWAC v2 - Core DCT/IDCT/Zigzag functions (cached, vectorized)
"""

import multiprocessing
import os

import numpy as np
from scipy.fft import dct, idct
from typing import Dict, Tuple

_ZIGZAG_IDX_CACHE: Dict[int, np.ndarray] = {}

_FFT_THREADS_MIN_SIZE = 1 << 19  # elements; below this the threads cost more than they save


def fft_workers(size: int) -> int:
    """Threads for one scipy.fft call over `size` elements. The transform runs along one axis, each
    1-D line on its own, so the values do not depend on the thread count. Small arrays and the
    worker processes of a Pool (their siblings hold the other cores) use one thread."""
    if size < _FFT_THREADS_MIN_SIZE or multiprocessing.parent_process() is not None:
        return 1
    return os.cpu_count() or 1


_FORKED = False


def _mark_forked() -> None:
    global _FORKED  # pylint: disable=global-statement
    _FORKED = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_mark_forked)


def in_forked_process() -> bool:
    """True in a process created by fork(), such as a worker of the block Pool. numba's OpenMP layer
    terminates such a process when it starts a parallel kernel after its parent has run one, and the
    Pool then waits for the lost worker forever, so the callers use a serial kernel there."""
    return _FORKED


def _get_zigzag_indices(n: int) -> np.ndarray:
    """Return cached zigzag scan indices for n×n matrix (computed once)."""
    if n in _ZIGZAG_IDX_CACHE:
        return _ZIGZAG_IDX_CACHE[n]

    order = []
    for i in range(2 * n - 1):
        if i % 2 == 0:
            row = min(i, n - 1)
            col = i - row
            while row >= 0 and col < n:
                order.append(row * n + col)
                row -= 1
                col += 1
        else:
            col = min(i, n - 1)
            row = i - col
            while col >= 0 and row < n:
                order.append(row * n + col)
                row += 1
                col -= 1

    idx = np.asarray(order, dtype=np.int64)
    _ZIGZAG_IDX_CACHE[n] = idx
    return idx


def to_zigzag(matrix: np.ndarray) -> np.ndarray:
    """Flatten N×N matrix to 1D vector in zigzag order."""
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    assert matrix.shape[0] == matrix.shape[1], "to_zigzag requires a square matrix"
    idx = _get_zigzag_indices(n)
    return matrix.reshape(-1)[idx]


def from_zigzag(vector: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct n×n matrix from zigzag vector (shorter vectors zero-padded)."""
    flat = np.zeros(n * n, dtype=np.float64)
    idx = _get_zigzag_indices(n)
    v = np.asarray(vector)
    flat[idx[: len(v)]] = v
    return flat.reshape(n, n)


def dct2(a: np.ndarray, dct_type: int = 2) -> np.ndarray:
    """2D DCT: accepts 2D (N×N) or batched 3D (B×N×N) arrays."""
    if a.ndim == 2:
        return dct(dct(a.T, norm="ortho", type=dct_type).T, norm="ortho", type=dct_type)
    if a.ndim == 3:
        w = fft_workers(a.size)
        return dct(
            dct(a, axis=2, norm="ortho", type=dct_type, workers=w),
            axis=1, norm="ortho", type=dct_type, workers=w,
        )
    else:
        raise ValueError(f"dct2 expects 2D or 3D input, got {a.shape}")


def idct2(a: np.ndarray, dct_type: int = 2) -> np.ndarray:
    """2D IDCT: accepts 2D (N×N) or batched 3D (B×N×N) arrays."""
    if a.ndim == 2:
        return idct(idct(a.T, norm="ortho", type=dct_type).T, norm="ortho", type=dct_type)
    if a.ndim == 3:
        w = fft_workers(a.size)
        return idct(
            idct(a, axis=2, norm="ortho", type=dct_type, workers=w),
            axis=1, norm="ortho", type=dct_type, workers=w,
        )
    else:
        raise ValueError(f"idct2 expects 2D or 3D input, got {a.shape}")


_IDCT_BASIS_CACHE: Dict[Tuple[int, int, str], np.ndarray] = {}


def _get_idct_basis(n: int, dct_type: int = 2, dtype=np.float64) -> np.ndarray:
    """Return precomputed IDCT basis [n*n, n, n] — cached per (n, dct_type, dtype)."""
    key = (n, dct_type, str(dtype))
    if key in _IDCT_BASIS_CACHE:
        return _IDCT_BASIS_CACHE[key]

    basis = np.zeros((n * n, n, n), dtype=dtype)
    for k in range(n * n):
        v = np.zeros(n * n, dtype=np.float32)
        v[k] = 1.0
        m = from_zigzag(v, n)
        basis[k] = idct2(m, dct_type).astype(dtype)

    _IDCT_BASIS_CACHE[key] = basis
    return basis


__all__ = [
    "dct2",
    "idct2",
    "to_zigzag",
    "from_zigzag",
    "_get_zigzag_indices",
    "_get_idct_basis",
    "fft_workers",
]
