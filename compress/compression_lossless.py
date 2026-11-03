"""
FIDWAC v2 - Lossless compression path (cm=4)
=============================================
Compresses uint8 bands using zlib deflate with optional Sub filter.
max error = 0 (exact reconstruction).

Entry point: compress_channel_lossless(src, band_idx, config)
"""

import zlib
from typing import Tuple, List

import numpy as np


def compress_channel_lossless(
    src,
    band_idx: int,
) -> Tuple[List, bytes, bool, float, bool, int, Tuple[int, int]]:
    """Compress a uint8 band using deflate (lossless, cm=4).

    Returns
    -------
    (lengths, compressed_data, filter_used, max_error, validity, block_size, padded_shape)

    lengths      : empty list (not used for lossless)
    filter_used  : True if Sub filter was applied (smaller than raw deflate)
    max_error    : always 0.0
    validity     : always True
    block_size   : dummy 1 (not used for lossless)
    """
    data = src.read(band_idx).astype(np.uint8)
    H, W = data.shape

    raw_compressed = zlib.compress(data.tobytes(), level=6)

    # Sub filter: subtract previous pixel horizontally
    filtered = data.copy()
    filtered[:, 1:] = data[:, 1:] - data[:, :-1]
    filtered_compressed = zlib.compress(filtered.tobytes(), level=6)

    if len(filtered_compressed) < len(raw_compressed):
        return [], filtered_compressed, True, 0.0, True, 1, (H, W)
    return [], raw_compressed, False, 0.0, True, 1, (H, W)


__all__ = ["compress_channel_lossless"]
