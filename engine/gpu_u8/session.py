"""Device-resident GPU session (session.cu): the batch is uploaded to VRAM once.

    with GpuSession([zz_Y, zz_Cb, zz_Cr], [org_Y, org_Cb, org_Cr], basis2, rgb=(rf, gf, bf)) as s:
        LF, ERR, RES, FLAG = s.search(ch, idx, acc, sf, bs)   # fused pre + verify + quant
        s.patch(ch, positions, rows)                          # CPU-fixed rows back into VRAM
        err, tie = s.rgb(idx, sf)                             # cm=6 pre-verify RGB error

FLAG marks rows the GPU could not decide exactly (IDCT rounding tie or L2 precheck boundary); the
caller recomputes them on the CPU (engine/u8_lockstep.search_rows does this).
"""

import ctypes
import os

import numpy as np

from engine.gpu_u8 import phases as _phases

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS = {}
_P = ctypes.c_void_p


def _load(n=8):
    if n in _LIBS:
        return _LIBS[n]
    _LIBS[n] = None
    if not _phases.available(n):  # loads the CUDA runtime, checks for a device, honours performance.gpu
        return None
    try:
        lib = ctypes.CDLL(os.path.join(_HERE, f"libfidwac_session_n{n}.so"))
    except OSError:
        return None
    lib.sess_create.restype = _P
    lib.sess_create.argtypes = [ctypes.c_int, ctypes.c_int] + [_P] * 6 + [_P, _P, _P, _P,
                                                                          ctypes.POINTER(ctypes.c_int)]
    lib.sess_free.restype = None
    lib.sess_free.argtypes = [_P]
    lib.sess_search.restype = ctypes.c_int
    lib.sess_search.argtypes = [_P, ctypes.c_int, _P, ctypes.c_int, _P, _P, _P, _P, _P, _P, _P, ctypes.c_int]
    lib.sess_patch.restype = ctypes.c_int
    lib.sess_patch.argtypes = [_P, ctypes.c_int, _P, ctypes.c_int, _P]
    lib.sess_rgb.restype = ctypes.c_int
    lib.sess_rgb.argtypes = [_P, _P, ctypes.c_int, ctypes.c_double, _P, _P]
    _LIBS[n] = lib
    return lib


def available(n=8) -> bool:
    return _load(n) is not None


def _c(a):
    return None if a is None else a.ctypes.data_as(_P)


def _need(a, dt, name, shape=None):
    if a.dtype != dt or not a.flags["C_CONTIGUOUS"]:
        raise RuntimeError(f"{name}: expected contiguous {np.dtype(dt)}")
    if shape is not None and a.shape != shape:
        raise RuntimeError(f"{name}: expected shape {shape}, got {a.shape}")


class GpuSession:
    def __init__(self, zz_list, org_list, basis2, rgb=None):
        self.nn = int(basis2.shape[0])  # coefficients per block: n * n
        lib = _load(_phases.edge_of(basis2))
        if lib is None:
            raise RuntimeError("GPU session backend unavailable")
        self._lib = lib
        self.B = int(zz_list[0].shape[0])
        self.nch = len(zz_list)
        if not 1 <= self.nch <= 3 or len(org_list) != self.nch:
            raise RuntimeError("1..3 channels with matching originals expected")
        self._keep = []  # host arrays stay alive while the call runs
        ptrs = []
        for c in range(3):
            if c < self.nch:
                zz = np.ascontiguousarray(zz_list[c], dtype=np.float64)
                org = np.ascontiguousarray(org_list[c], dtype=np.float64)
                _need(zz, np.float64, "zz", (self.B, self.nn))
                _need(org, np.float64, "org", (self.B, self.nn))
                self._keep += [zz, org]
                ptrs += [_c(zz), _c(org)]
            else:
                ptrs += [None, None]
        b2 = np.ascontiguousarray(basis2, dtype=np.float64)
        _need(b2, np.float64, "basis2", (self.nn, self.nn))
        rgbp = [None, None, None]
        if rgb is not None:
            rgb_arrs = [np.ascontiguousarray(a, dtype=np.float32).reshape(self.B, self.nn) for a in rgb]
            self._keep += rgb_arrs
            rgbp = [_c(a) for a in rgb_arrs]
        rc = ctypes.c_int(0)
        self._h = lib.sess_create(self.B, self.nch, *ptrs, _c(b2), *rgbp, ctypes.byref(rc))
        self._keep = []  # data is in VRAM now
        if not self._h:
            raise RuntimeError(f"sess_create failed (CUDA error {rc.value})")
        self.has_rgb = rgb is not None

    def search(self, ch, idx, acc, sf, bs, brk=3):
        """Search rows idx of channel ch (one block of NN threads per row). Returns LF, ERR, RES, FLAG;
        FLAG marks the rows the CPU must decide (IDCT rounding tie or precheck boundary)."""
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        n = idx.size
        acc = np.ascontiguousarray(acc, dtype=np.float64)
        sf = np.ascontiguousarray(sf, dtype=np.float64)
        bs = np.ascontiguousarray(bs, dtype=np.int64)
        for name, a in (("acc", acc), ("sf", sf), ("bs", bs)):
            if a.shape != (n,):
                raise RuntimeError(f"{name}: expected ({n},)")
        LF = np.zeros(n, dtype=np.int64)
        ERR = np.zeros(n, dtype=np.float64)
        RES = np.zeros((n, self.nn), dtype=np.int64)
        FLAG = np.zeros(n, dtype=np.int32)
        rc = self._lib.sess_search(self._h, int(ch), _c(idx), int(n), _c(acc), _c(sf), _c(bs),
                                   _c(LF), _c(ERR), _c(RES), _c(FLAG), int(brk))
        if rc != 0:
            raise RuntimeError(f"sess_search failed (CUDA error {rc})")
        return LF, ERR, RES, FLAG.astype(bool)

    def patch(self, ch, pos, rows):
        pos = np.ascontiguousarray(pos, dtype=np.int64)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        _need(rows, np.int64, "rows", (pos.size, self.nn))
        rc = self._lib.sess_patch(self._h, int(ch), _c(pos), int(pos.size), _c(rows))
        if rc != 0:
            raise RuntimeError(f"sess_patch failed (CUDA error {rc})")

    def rgb(self, idx, sf):
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        n = idx.size
        err = np.zeros(n, dtype=np.float64)
        flag = np.zeros(n, dtype=np.int32)
        rc = self._lib.sess_rgb(self._h, _c(idx), int(n), float(sf), _c(err), _c(flag))
        if rc != 0:
            raise RuntimeError(f"sess_rgb failed (CUDA error {rc})")
        return err, flag.astype(bool)

    def close(self):
        if getattr(self, "_h", None):
            self._lib.sess_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001  (interpreter shutdown)
            pass
