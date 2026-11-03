"""ctypes wrapper for the cm=6 pre-verification error (rgb_pre.cu).

pre_err returns, per row, the max RGB error of the planes built from the quantised coefficients,
and a flag for rows whose YCbCr->RGB rounding is not robust (caller recomputes those exactly).
"""

import ctypes
import os

import numpy as np

from engine.gpu_u8 import phases as _phases

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS = {}


def _load(n=8):
    if n in _LIBS:
        return _LIBS[n]
    _LIBS[n] = None
    if not _phases.available(n):  # loads the CUDA runtime and checks for a device
        return None
    try:
        lib = ctypes.CDLL(os.path.join(_HERE, f"libfidwac_rgb_n{n}.so"))
    except OSError:
        return None
    p = ctypes.c_void_p
    lib.rgb_pre_gpu.restype = ctypes.c_int
    lib.rgb_pre_gpu.argtypes = [p, p, p, p, p, p, p, ctypes.c_double, ctypes.c_int, p, p]
    _LIBS[n] = lib
    return lib


def available(n=8) -> bool:
    return _load(n) is not None


def pre_err(RY, RCB, RCR, RF, GF, BF, basis2, sf):
    """RY/RCB/RCR int64 (B,n*n) quantised coefficients; RF/GF/BF float32 (B,n*n) originals."""
    lib = _load(_phases.edge_of(basis2))
    if lib is None:
        raise RuntimeError("GPU RGB backend unavailable")
    B = RY.shape[0]
    for name, arr, dt in (("RY", RY, np.int64), ("RCB", RCB, np.int64), ("RCR", RCR, np.int64),
                          ("RF", RF, np.float32), ("GF", GF, np.float32), ("BF", BF, np.float32),
                          ("basis2", basis2, np.float64)):
        if arr.dtype != dt or not arr.flags["C_CONTIGUOUS"]:
            raise RuntimeError(f"{name}: expected contiguous {np.dtype(dt)}")
    err = np.zeros(B, dtype=np.float64)
    flag = np.zeros(B, dtype=np.int32)
    c = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    rc = lib.rgb_pre_gpu(c(RY), c(RCB), c(RCR), c(RF), c(GF), c(BF), c(basis2), float(sf), int(B),
                         c(err), c(flag))
    if rc != 0:
        raise RuntimeError(f"CUDA error code {rc} in rgb pre phase")
    return err, flag.astype(bool)
