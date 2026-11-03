"""
FIDWAC v2 - ZigZag + VLQ codec

int → ZigZag → uint → VLQ bytes
Small values (±63) fit in 1 byte; ~91% of DCT coefficients → 3-4× smaller than raw int32.
"""

from typing import List, Union
import numpy as np
import numba as nb



@nb.njit(cache=True)
def _nb_encode_vlq(u: np.ndarray, buf: np.ndarray) -> int:
    """
    Writes uint64 arrays into buf as VLQ bytes.
    Returns the number of bytes written.
    """
    pos = 0
    for v in u:
        while v >= np.uint64(0x80):
            buf[pos] = np.uint8((v & np.uint64(0x7F)) | np.uint64(0x80))
            pos += 1
            v = v >> np.uint64(7)
        buf[pos] = np.uint8(v)
        pos += 1
    return pos


@nb.njit(cache=True)
def _nb_zigzag_encode_vlq(values: np.ndarray, buf: np.ndarray) -> int:
    """
    ZigZag and VLQ encode in a single pass (int64 -> bytes).
    Returns the number of bytes written.
    """
    pos = 0
    for sv in values:
        v = np.uint64(sv << 1) if sv >= 0 else np.uint64((-sv - 1) << 1) | np.uint64(1)
        while v >= np.uint64(0x80):
            buf[pos] = np.uint8((v & np.uint64(0x7F)) | np.uint64(0x80))
            pos += 1
            v = v >> np.uint64(7)
        buf[pos] = np.uint8(v)
        pos += 1
    return pos


@nb.njit(cache=True, nogil=True)
def _nb_decode_vlq(data: np.ndarray, n: int, out: np.ndarray) -> int:
    """
    Decodes VLQ bytes into the uint64 array out.
    Returns the number of decoded values.
    """
    i = 0
    count = 0
    while i < n:
        v = np.uint64(0)
        shift = np.uint64(0)
        while True:
            b = np.uint64(data[i])
            i += 1
            v |= (b & np.uint64(0x7F)) << shift
            shift += np.uint64(7)
            if not b & np.uint64(0x80):
                break
        out[count] = v
        count += 1
    return count


_dummy_u = np.zeros(1, dtype=np.uint64)
_dummy_buf = np.zeros(5, dtype=np.uint8)
_nb_encode_vlq(_dummy_u, _dummy_buf)
_dummy_i64 = np.zeros(1, dtype=np.int64)
_nb_zigzag_encode_vlq(_dummy_i64, _dummy_buf)
_dummy_data = np.zeros(1, dtype=np.uint8)
_dummy_out = np.zeros(1, dtype=np.uint64)
_nb_decode_vlq(_dummy_data, 1, _dummy_out)
del _dummy_u, _dummy_buf, _dummy_i64, _dummy_data, _dummy_out



def encode_vlq(values: List[int]) -> bytes:
    """Encode signed int list → bytes via ZigZag + VLQ."""
    if values is None or (hasattr(values, "__len__") and len(values) == 0):
        return b""
    if isinstance(values, np.ndarray):
        arr = values.astype(np.int64, copy=False)
    else:
        arr = np.asarray(values, dtype=np.int64)
    buf = np.empty(len(arr) * 5, dtype=np.uint8)
    pos = _nb_zigzag_encode_vlq(arr, buf)
    return bytes(buf[:pos])


def decode_vlq_array(data: Union[bytes, bytearray, memoryview]) -> np.ndarray:
    """Decode bytes → signed int64 array via VLQ + ZigZag."""
    if not data:
        return np.empty(0, dtype=np.int64)
    arr = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(len(arr), dtype=np.uint64)  # worst case: 1 value per byte
    count = _nb_decode_vlq(arr, len(arr), out)
    u = out[:count]
    return np.where((u & 1) == 0, (u >> 1).astype(np.int64), -((u >> 1) + 1).astype(np.int64))


def decode_vlq(data: Union[bytes, bytearray, memoryview]) -> List[int]:
    """Decode bytes → signed int list via VLQ + ZigZag."""
    return decode_vlq_array(data).tolist()



@nb.njit(cache=True)
def _nb_median3(a: np.int64, b: np.int64, c: np.int64) -> np.int64:
    """Median of three signed integers."""
    return a + b + c - min(a, b, c) - max(a, b, c)


@nb.njit(cache=True)
def _nb_predict_dc_median(
    left_dc: np.int64,
    has_left: bool,
    top_dc: np.int64,
    has_top: bool,
    topleft_dc: np.int64,
    has_topleft: bool,
) -> np.int64:
    """DC median predictor. Special blocks (L<=0) → neighbour DC = None."""
    if has_left and has_top and has_topleft:
        plane = left_dc + top_dc - topleft_dc
        return _nb_median3(left_dc, top_dc, plane)
    if has_left:
        return left_dc
    if has_top:
        return top_dc
    return np.int64(0)


@nb.njit(cache=True)
def _nb_zz_vlq_one(sv: np.int64, buf: np.ndarray, pos: int) -> int:
    """ZigZag + VLQ encode one signed int64 into buf at pos. Returns new pos."""
    v = np.uint64(sv << 1) if sv >= 0 else np.uint64((-sv - 1) << 1) | np.uint64(1)
    while v >= np.uint64(0x80):
        buf[pos] = np.uint8((v & np.uint64(0x7F)) | np.uint64(0x80))
        pos += 1
        v = v >> np.uint64(7)
    buf[pos] = np.uint8(v)
    pos += 1
    return pos


@nb.njit(cache=True)
def _nb_dc_median_ac_encode(
    coeffs: np.ndarray,
    lengths: np.ndarray,
    blocks_per_row: int,
    dc_buf: np.ndarray,
    ac_buf: np.ndarray,
):
    """
    Encode flat coefficient stream into separate DC/AC byte streams.

    DC: dc_delta = dc - median_predictor(left, top, topleft)
    AC: raw values unchanged

    Returns (dc_pos, ac_pos) — bytes written to each buffer.
    """
    n_blocks = len(lengths)
    dc_values = np.zeros(n_blocks, dtype=np.int64)
    dc_valid = np.zeros(n_blocks, dtype=np.bool_)

    dc_pos = 0
    ac_pos = 0
    coeff_pos = 0

    for i in range(n_blocks):
        L = lengths[i]
        if L > 0:
            dc = coeffs[coeff_pos]
            bx = i % blocks_per_row

            left_i = i - 1
            top_i = i - blocks_per_row
            topleft_i = i - blocks_per_row - 1

            has_left = (bx > 0) and dc_valid[left_i]
            has_top = (top_i >= 0) and dc_valid[top_i]
            has_topleft = (bx > 0 and top_i >= 0) and dc_valid[topleft_i]

            left_dc = dc_values[left_i] if has_left else np.int64(0)
            top_dc = dc_values[top_i] if has_top else np.int64(0)
            topleft_dc = dc_values[topleft_i] if has_topleft else np.int64(0)

            pred = _nb_predict_dc_median(
                left_dc, has_left, top_dc, has_top, topleft_dc, has_topleft
            )
            delta = dc - pred
            dc_pos = _nb_zz_vlq_one(delta, dc_buf, dc_pos)

            for j in range(1, L):
                ac_pos = _nb_zz_vlq_one(coeffs[coeff_pos + j], ac_buf, ac_pos)

            dc_values[i] = dc
            dc_valid[i] = True
            coeff_pos += L

    return dc_pos, ac_pos


@nb.njit(cache=True, nogil=True)
def _nb_dc_median_ac_decode(
    dc_deltas: np.ndarray,
    ac_values: np.ndarray,
    lengths: np.ndarray,
    blocks_per_row: int,
    out_coeffs: np.ndarray,
):
    """
    Decode separate DC-delta / AC streams back into flat coefficient array.

    Reconstructs DC from delta + median_predictor, places AC values unchanged.
    Writes into out_coeffs, returns number of coefficients written.
    """
    n_blocks = len(lengths)
    dc_values = np.zeros(n_blocks, dtype=np.int64)
    dc_valid = np.zeros(n_blocks, dtype=np.bool_)

    dc_idx = 0
    ac_idx = 0
    coeff_pos = 0

    for i in range(n_blocks):
        L = lengths[i]
        if L > 0:
            bx = i % blocks_per_row

            left_i = i - 1
            top_i = i - blocks_per_row
            topleft_i = i - blocks_per_row - 1

            has_left = (bx > 0) and dc_valid[left_i]
            has_top = (top_i >= 0) and dc_valid[top_i]
            has_topleft = (bx > 0 and top_i >= 0) and dc_valid[topleft_i]

            left_dc = dc_values[left_i] if has_left else np.int64(0)
            top_dc = dc_values[top_i] if has_top else np.int64(0)
            topleft_dc = dc_values[topleft_i] if has_topleft else np.int64(0)

            pred = _nb_predict_dc_median(
                left_dc, has_left, top_dc, has_top, topleft_dc, has_topleft
            )
            dc = dc_deltas[dc_idx] + pred
            dc_idx += 1

            out_coeffs[coeff_pos] = dc
            for j in range(1, L):
                out_coeffs[coeff_pos + j] = ac_values[ac_idx]
                ac_idx += 1

            dc_values[i] = dc
            dc_valid[i] = True
            coeff_pos += L

    return coeff_pos


_dummy_len = np.array([1], dtype=np.int64)
_dummy_c = np.array([42], dtype=np.int64)
_dummy_dc_buf = np.zeros(5, dtype=np.uint8)
_dummy_ac_buf = np.zeros(5, dtype=np.uint8)
_nb_dc_median_ac_encode(_dummy_c, _dummy_len, 1, _dummy_dc_buf, _dummy_ac_buf)
_dummy_dc_deltas = np.array([0], dtype=np.int64)
_dummy_ac_vals = np.array([], dtype=np.int64)
_dummy_out = np.zeros(1, dtype=np.int64)
_nb_dc_median_ac_decode(_dummy_dc_deltas, _dummy_ac_vals, _dummy_len, 1, _dummy_out)
del _dummy_len, _dummy_c, _dummy_dc_buf, _dummy_ac_buf
del _dummy_dc_deltas, _dummy_ac_vals, _dummy_out


def dc_median_ac_encode(coeffs: np.ndarray, lengths: list, blocks_per_row: int) -> tuple:
    """
    Encode coefficients with DC median prediction into separate streams.

    Parameters
    ----------
    coeffs : np.ndarray (int64) — flat coefficient stream
    lengths : list[int] — per-block lengths (>0 active, 0 trivial, <0 special)
    blocks_per_row : int

    Returns
    -------
    (dc_bytes, ac_bytes) — two bytes objects
    """
    if len(coeffs) == 0:
        return b"", b""
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    n_active = int(np.sum(lengths_arr > 0))
    total_ac = int(np.sum(np.maximum(lengths_arr - 1, 0)))
    dc_buf = np.empty(n_active * 5, dtype=np.uint8)
    ac_buf = np.empty(max(total_ac * 5, 1), dtype=np.uint8)
    dc_pos, ac_pos = _nb_dc_median_ac_encode(
        coeffs.copy(), lengths_arr, blocks_per_row, dc_buf, ac_buf
    )
    return bytes(dc_buf[:dc_pos]), bytes(ac_buf[:ac_pos])


def dc_median_ac_decode(
    dc_data: bytes, ac_data: bytes, lengths: list, blocks_per_row: int
) -> np.ndarray:
    """
    Decode separate DC/AC streams with DC median prediction.

    Parameters
    ----------
    dc_data : bytes — VLQ-encoded DC deltas
    ac_data : bytes — VLQ-encoded AC values
    lengths : list[int] — per-block lengths
    blocks_per_row : int

    Returns
    -------
    np.ndarray (int64) — flat coefficient stream (DC restored, AC in place)
    """
    dc_deltas = decode_vlq_array(dc_data)
    ac_values = decode_vlq_array(ac_data)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    total_coeffs = int(np.sum(lengths_arr[lengths_arr > 0]))
    out_coeffs = np.empty(max(total_coeffs, 1), dtype=np.int64)
    if total_coeffs == 0:
        return out_coeffs[:0]
    _nb_dc_median_ac_decode(dc_deltas, ac_values, lengths_arr, blocks_per_row, out_coeffs)
    return out_coeffs[:total_coeffs]

