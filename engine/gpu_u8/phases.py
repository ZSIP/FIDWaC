"""ctypes wrapper for the phase-split uint8 search (search_u8_phases.cu).

pre_gpu:   phase 1, binary search + integer backscan  -> LPRE
quant_gpu: phase 3, quantisation + post-quant check   -> LF, ERR, RES
The integer verification (phase 2) runs on the CPU with scipy (engine/u8_lockstep.classic_verify).

Both calls raise RuntimeError on failure; the caller then uses the CPU kernels for that phase.
Use performance.gpu = off in config.json to disable.
"""

import ctypes
import glob
import os
import sys

import numpy as np

from config import gpu_mode

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS = {}  # block edge -> loaded library (None when it cannot be loaded)
_STATE = {"reason": ""}
BLOCK_EDGES = (8, 16, 32)  # the kernels are built for these block edges (build.sh)


def _cudart_candidates():
    yield "libcudart.so.13"
    yield "libcudart.so"
    pattern = os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "cu*", "lib",
                           "libcudart.so*")
    yield from sorted(glob.glob(pattern), reverse=True)


def edge_of(basis2) -> int:
    """Block edge n of a zigzag IDCT basis (n*n, n*n)."""
    return int(round(basis2.shape[0] ** 0.5))


def _load_lib(n=8):
    """Load the phase library for block edge n and the CUDA runtime; None with _STATE['reason'] when
    unavailable."""
    if n in _LIBS:
        return _LIBS[n]
    _LIBS[n] = None
    if n not in BLOCK_EDGES:
        _STATE["reason"] = f"no GPU kernels for block size {n}"
        return None
    so_path = os.path.join(_HERE, f"libfidwac_u8_phases_n{n}.so")
    if not os.path.exists(so_path):
        _STATE["reason"] = f"missing {so_path}"
        return None
    for cand in _cudart_candidates():
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    else:
        _STATE["reason"] = "CUDA runtime library not found"
        return None
    try:
        lib = ctypes.CDLL(so_path)
    except OSError as exc:
        _STATE["reason"] = f"cannot load {so_path}: {exc}"
        return None
    p = ctypes.c_void_p
    lib.pre_u8_gpu.restype = ctypes.c_int
    lib.pre_u8_gpu.argtypes = [p, p, p, p, p, p, ctypes.c_int, ctypes.c_int]
    lib.quant_u8_gpu.restype = ctypes.c_int
    lib.quant_u8_gpu.argtypes = [p, p, p, p, p, p, p, p, p, ctypes.c_int]
    _LIBS[n] = lib
    return lib


def _load(n=8):
    """The GPU library (block edge n) for the mode of performance.gpu: off -> None; auto -> None with a note when the
    GPU cannot be used (the caller runs on the CPU); on -> RuntimeError when it cannot be used."""
    mode = gpu_mode()
    if mode == "off":
        _STATE["reason"] = "performance.gpu = off"
        return None
    lib = _load_lib(n)
    if lib is None and mode == "on":
        raise RuntimeError(f"performance.gpu = on, but no GPU is usable: {_STATE['reason']}")
    if lib is None and not _STATE.get("noted"):
        _STATE["noted"] = True
        print(f"  note: GPU not found ({_STATE['reason']}), using the CPU (performance.gpu = auto)")
    return lib


def available(n=8) -> bool:
    """True when the library for block edge n loads and a CUDA device answers."""
    lib = _load(n)
    if lib is None:
        return False
    try:
        cnt = ctypes.c_int(0)
        rt = ctypes.CDLL("libcudart.so.13", mode=ctypes.RTLD_GLOBAL)
        rt.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        if rt.cudaGetDeviceCount(ctypes.byref(cnt)) != 0 or cnt.value < 1:
            _STATE["reason"] = "no CUDA device"
            return False
    except OSError:
        _STATE["reason"] = "cudart not loadable for device query"
        return False
    return True


def reason() -> str:
    return _STATE["reason"]


def _check(arr, dt, name):
    if arr.dtype != dt or not arr.flags["C_CONTIGUOUS"]:
        raise RuntimeError(f"{name}: expected contiguous {np.dtype(dt)}")


def _c(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)


def pre_gpu(ZZ, ORIG, ACC, BS, basis2, LPRE, brk=3) -> None:
    lib = _load(edge_of(basis2))
    if lib is None:
        raise RuntimeError(_STATE["reason"] or "GPU backend unavailable")
    B = ZZ.shape[0]
    _check(ZZ, np.float64, "ZZ")
    _check(ORIG, np.float64, "ORIG")
    _check(ACC, np.float64, "ACC")
    _check(BS, np.int64, "BS")
    _check(basis2, np.float64, "basis2")
    _check(LPRE, np.int64, "LPRE")
    rc = lib.pre_u8_gpu(_c(ZZ), _c(ORIG), _c(ACC), _c(BS), _c(basis2), _c(LPRE), int(B), int(brk))
    if rc != 0:
        raise RuntimeError(f"CUDA error code {rc} in pre phase")


def quant_gpu(ZZ, ORIG, ACC, SF, basis2, LIN, LF, ERR, RES) -> None:
    lib = _load(edge_of(basis2))
    if lib is None:
        raise RuntimeError(_STATE["reason"] or "GPU backend unavailable")
    B = ZZ.shape[0]
    _check(ZZ, np.float64, "ZZ")
    _check(ORIG, np.float64, "ORIG")
    _check(ACC, np.float64, "ACC")
    _check(SF, np.float64, "SF")
    _check(basis2, np.float64, "basis2")
    _check(LIN, np.int64, "LIN")
    _check(LF, np.int64, "LF")
    _check(ERR, np.float64, "ERR")
    _check(RES, np.int64, "RES")
    rc = lib.quant_u8_gpu(_c(ZZ), _c(ORIG), _c(ACC), _c(SF), _c(basis2), _c(LIN), _c(LF), _c(ERR),
                          _c(RES), int(B))
    if rc != 0:
        raise RuntimeError(f"CUDA error code {rc} in quant phase")
