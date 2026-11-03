"""
FIDWAC v2 - Utility functions: RLE masks, zero interpolation, numpy conversion
"""

from typing import List, Tuple, Any, Optional

import numpy as np

# Integer accuracy mode (cm=5): (centre subtracted before the DCT, min, max) of the decoded values.
INTEGER_RANGES = {
    "uint8": (128.0, 0.0, 255.0),
    "int8": (0.0, -128.0, 127.0),
    "uint16": (32768.0, 0.0, 65535.0),
    "int16": (0.0, -32768.0, 32767.0),
}


def integer_range(src_dtype: Optional[str]) -> Optional[Tuple[float, float, float]]:
    """(centre, min, max) for an integer source dtype, None for float data."""
    return INTEGER_RANGES.get(str(src_dtype)) if src_dtype is not None else None



def encode_mask_rle(mask: np.ndarray) -> List[int]:
    """Encode binary mask as RLE: [start1, length1, start2, length2, ...]."""
    flat = mask.flatten().astype(np.int8)
    # Detect transitions: +1 = start of True run, -1 = end of True run
    changes = np.diff(flat, prepend=np.int8(0), append=np.int8(0))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    rle = np.empty(len(starts) * 2, dtype=np.int64)
    rle[0::2] = starts
    rle[1::2] = lengths
    return rle.tolist()


def decode_mask_rle(rle: List[int], shape: Tuple[int, int]) -> np.ndarray:
    """Decode RLE [start1, length1, ...] into a boolean mask of given shape."""
    flat_mask = np.zeros(shape[0] * shape[1], dtype=bool)

    if len(rle) >= 2:
        rle_arr = np.asarray(rle, dtype=np.int64)
        starts = rle_arr[0::2]
        lengths = rle_arr[1::2]
        indices = np.concatenate(
            [np.arange(s, s + l, dtype=np.int64) for s, l in zip(starts.tolist(), lengths.tolist())]
        )
        if len(indices) > 0:
            flat_mask[indices] = True

    return flat_mask.reshape(shape)


def decode_mask_rle_rows(rle: List[int], width: int, y0: int, height: int) -> np.ndarray:
    """Rows y0..y0+height of the mask that decode_mask_rle would give for an image `width` wide."""
    lo, hi = y0 * width, (y0 + height) * width
    out = np.zeros(height * width, dtype=bool)
    if len(rle) >= 2:
        arr = np.asarray(rle, dtype=np.int64)
        starts, ends = arr[0::2], arr[0::2] + arr[1::2]
        for a, b in zip(np.maximum(starts, lo)[(starts < hi) & (ends > lo)].tolist(),
                        np.minimum(ends, hi)[(starts < hi) & (ends > lo)].tolist()):
            out[a - lo : b - lo] = True
    return out.reshape(height, width)


class RleBuilder:
    """Build the RLE of a mask strip by strip: add() takes the rows of the next strip, in order,
    and joins a run that continues from the previous strip. result() equals encode_mask_rle of the
    whole mask."""

    def __init__(self):
        self._rle: List[int] = []
        self._offset = 0

    def add(self, strip_mask: np.ndarray) -> None:
        part = encode_mask_rle(strip_mask)
        for k in range(0, len(part), 2):
            start, length = part[k] + self._offset, part[k + 1]
            if k == 0 and self._rle and self._rle[-2] + self._rle[-1] == start:
                self._rle[-1] += length
            else:
                self._rle.extend((start, length))
        self._offset += strip_mask.size

    def result(self) -> List[int]:
        return self._rle


def interpolate_zeros(matrix: np.ndarray, nonzero_mask: np.ndarray) -> np.ndarray:
    """Fill zero pixels via nearest-neighbour interpolation (smooths DCT input)."""
    from scipy.interpolate import griddata  # lazy: ~1 s of import time, needed only for NoData/zero blocks

    compression_copy = matrix.copy()

    if not np.any(nonzero_mask):
        return compression_copy

    y_indices, x_indices = np.indices(matrix.shape)
    points = np.column_stack((y_indices[nonzero_mask], x_indices[nonzero_mask]))
    values = matrix[nonzero_mask]

    zero_mask = ~nonzero_mask
    if not np.any(zero_mask) or len(points) == 0:
        return compression_copy

    grid_y = y_indices[zero_mask]
    grid_x = x_indices[zero_mask]
    grid_points = np.column_stack((grid_y, grid_x))

    try:
        interpolated = griddata(points, values, grid_points, method="nearest")
        compression_copy[zero_mask] = interpolated
    except Exception:
        mean_value = np.mean(values) if len(values) > 0 else 0
        compression_copy[zero_mask] = mean_value

    return compression_copy


def numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python (for JSON/msgpack)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    return obj


__all__ = [
    "encode_mask_rle",
    "decode_mask_rle",
    "interpolate_zeros",
    "numpy_to_python",
]
