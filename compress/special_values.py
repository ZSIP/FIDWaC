"""Exact restore of NaN / +inf / -inf for the per-band containers (cm=7 .fpred, cm=8 .sz3).

Both containers store one bit per pixel saying "not a regular value" (nodata or non-finite).
That bit alone cannot tell NaN from nodata from inf, so this module adds an optional second
stream with one code per such pixel, in raster order. The stream is written only when some
pixel is non-finite and not plain nodata, so ordinary files do not grow. Files written
without it still decode; every flagged pixel is then restored as NoData.
"""
import zlib

import numpy as np

CODE_NODATA, CODE_NAN, CODE_POSINF, CODE_NEGINF = 0, 1, 2, 3


def encode_special(raw, invalid):
    """zlib-packed code per invalid pixel, or None when every invalid pixel is plain nodata."""
    v = raw[invalid]
    codes = np.zeros(v.shape, dtype=np.uint8)
    codes[np.isnan(v)] = CODE_NAN
    codes[np.isposinf(v)] = CODE_POSINF
    codes[np.isneginf(v)] = CODE_NEGINF
    return zlib.compress(codes.tobytes(), 9) if codes.any() else None


def apply_special(out, invalid, payload, nodata):
    """Write nodata, NaN and +/-inf into `out` (already in its final dtype) at the invalid pixels."""
    if nodata is not None:
        out[invalid] = nodata
    if not payload:
        return
    codes = np.frombuffer(zlib.decompress(payload), dtype=np.uint8)
    rows, cols = np.nonzero(invalid)  # raster order, same as at encode time
    for code, value in ((CODE_NAN, np.nan), (CODE_POSINF, np.inf), (CODE_NEGINF, -np.inf)):
        sel = codes == code
        if sel.any():
            out[rows[sel], cols[sel]] = value
