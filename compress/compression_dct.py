"""
FIDWAC v2 - DCT compression path
==================================
Handles float32 elevation and uint8 accuracy-controlled (cm=3/cm=5/cm=6) data:
  - in-memory single-channel  : compress_channel_dct()          → cm=3 or cm=5
  - streaming single-channel  : compress_channel_dct_streaming() → cm=3 or cm=5
  - per-block RGB YCbCr       : compress_channel_rgb()           → cm=6
  - per-channel DCT loop with YCbCr decorrelation: compress_dct_path()

cm=3: float32/int16 elevation — DC Median3 + separate DC/AC streams, max error ≤ε
cm=5: uint8/int8 single-channel — binary search DCT, centering -128, max error ≤ε (grayscale)
cm=6: uint8 RGB accuracy — per-block YCbCr with cascading scaling factors, max error ≤ε (RGB)

All functions return the same 7-tuple:
    (lengths, coeffs_np, masks_data, max_error, validity, block_size, padded_shape)
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
from typing import Optional, Tuple, List
from functools import partial
from copy import deepcopy

import numpy as np
from rasterio.windows import Window
from tqdm import tqdm

from engine.blocks import (
    process_block_batch,
    process_strip_batch,
    process_block_batch_rgb,
    process_rgb_arrays,
    rgb_to_ycbcr,
)
from config import Config
from core.dct import _get_idct_basis
from engine.utils import RleBuilder, interpolate_zeros
from .compression_utils import (
    pad_image,
    extract_blocks,
    auto_select_block_size,
    auto_select_uint8_parameters,
    serialize_compressed_blocks,
    select_scaling_factor,
    scaled_config,
    SF_EXCEPTION_LEVELS,
)

_DctResult = Tuple[np.ndarray, np.ndarray, List, float, bool, int, Tuple[int, int]]

RGB_STRIP_PIXELS = 8 * 1024 * 1024  # pixels per strip of block rows in the cm=6 compression
DCT_STRIP_PIXELS = 16 * 1024 * 1024  # bands above this many pixels are compressed strip by strip

_INTEGER_DTYPES = ("uint8", "int8", "uint16", "int16")


def compress_channel_dct(
    image: np.ndarray,
    block_size: int,
    nodata_value: Optional[float],
    config: Config,
    num_processes: int,
    src_dtype: Optional[str] = None,
    errors_out: Optional[list] = None,
) -> _DctResult:
    """Compress a single channel in-memory.

    For integer dtypes, forces binary search (accuracy mode).
    Retries with smaller block size if validity fails.
    errors_out: when given, receives the float64 array of the error of every block as its only item.
    """
    is_integer = src_dtype in _INTEGER_DTYPES

    if not config.output.quiet and is_integer:
        global_u8_accuracy = float(
            getattr(config.compression, "uint8_accuracy", config.compression.accuracy)
        )
        channel_accuracy = float(config.compression.accuracy)
        if abs(channel_accuracy - global_u8_accuracy) < 1e-9:
            print(f"  uint8 accuracy mode: binary search with accuracy_px={channel_accuracy:g}")
        else:
            print(
                "  uint8 accuracy mode: binary search with "
                f"uint8_accuracy_px={global_u8_accuracy:g}, channel_accuracy_px={channel_accuracy:g}"
            )

    if config.compression.auto_select_block_size:
        if is_integer:
            sf, bs = auto_select_uint8_parameters(image, config, nodata_value)
            config = deepcopy(config)
            config.compression.block_size = bs
            config.compression.uint8_scaling_factor = [sf]
            block_size = bs
        else:
            block_size = auto_select_block_size(image, config, nodata_value, src_dtype)
            config = deepcopy(config)
            config.compression.block_size = block_size

    padded_image, padded_shape = pad_image(image, block_size)
    lengths, coeffs_np, masks_data, max_errors, skipped = _compress_padded_blocks(
        padded_image, block_size, nodata_value, config, is_integer, src_dtype, num_processes
    )
    if not config.output.quiet:
        print(f"  Total blocks: {len(lengths)} (skipped {skipped} trivial)")

    # Use config.compression.accuracy (channel-specific for YCbCr, or global uint8_accuracy)
    max_error_global, validity = _error_summary(max_errors, config.compression.accuracy)

    if not validity and config.compression.auto_select_block_size and block_size > 8:
        return _retry_with_smaller_n(
            image,
            block_size,
            nodata_value,
            config,
            num_processes,
            max_errors,
            src_dtype,
            in_memory=True,
            errors_out=errors_out,
        )
    if errors_out is not None:
        errors_out[:] = [np.asarray(max_errors, dtype=np.float64)]

    return lengths, coeffs_np, masks_data, max_error_global, validity, block_size, padded_shape


def _error_summary(max_errors, accuracy):
    """(largest error, all errors within accuracy) of the per-block errors: a float64 array from the
    array paths, a list from the per-block path."""
    if isinstance(max_errors, np.ndarray):
        largest = float(max_errors.max()) if max_errors.size else 0.0
        return largest, bool((max_errors <= accuracy).all())
    return (max(max_errors) if max_errors else 0.0), all(e <= accuracy for e in max_errors)


def _insert_blocks(lengths, coeffs, blocks):
    """Put per-block results (dict: block index -> compressed block) into the serialised form of an
    array path, where these blocks hold length 0 and no coefficients. Returns (lengths, coeffs,
    masks_data) as serialize_compressed_blocks gives them for all blocks together."""
    idx = np.array(sorted(blocks), dtype=np.int64)
    own_lengths, own_coeffs, own_masks = serialize_compressed_blocks([blocks[int(i)] for i in idx])
    lengths[idx] = own_lengths
    masks_data = [[int(idx[j]), mask] for j, mask in own_masks]
    if own_coeffs.size:
        own = np.zeros(lengths.size, dtype=bool)
        own[idx] = True
        from_own = np.repeat(own, np.maximum(lengths, 0))  # per coefficient of the merged stream
        merged = np.empty(from_own.size, dtype=np.int64)
        merged[from_own] = own_coeffs
        merged[~from_own] = coeffs
        coeffs = merged
    return lengths, coeffs, masks_data


def _compress_padded_blocks(padded_image, block_size, nodata_value, config, is_integer, src_dtype,
                            num_processes):
    """Compress the blocks of a padded image (or strip of block rows): whole arrays on the GPU session /
    numba where they apply, the per-block path otherwise. Returns (lengths, coeffs, masks_data,
    max_errors, skipped_trivial): the blocks as serialize_compressed_blocks gives them, in block
    order, and the error of every block (an array from the array paths, a list otherwise)."""
    fast = _compress_blocks_arrays(
        padded_image, block_size, nodata_value, config, is_integer, src_dtype
    )
    mixed_blocks = []  # float blocks with zeros/NoData inside: the per-block path keeps their masks
    if fast is None:
        f32 = _compress_blocks_f32(padded_image, block_size, nodata_value, config, src_dtype)
        if f32 is not None:
            fast, mixed_blocks = f32[:4], f32[4]
    if fast is not None:  # whole arrays end to end (GPU session / numba), no per-block tuples
        lengths, coeffs, max_errors, skipped = fast
        masks_data = []
        if mixed_blocks:
            mixed = {}
            _run_block_batches(mixed_blocks, mixed, max_errors, num_processes, config)
            lengths, coeffs, masks_data = _insert_blocks(lengths, coeffs, mixed)
    else:
        all_blocks = extract_blocks(padded_image, block_size, nodata_value)
        total_blocks = len(all_blocks)

        compressed_blocks, max_errors = _prefilter_trivial(all_blocks, total_blocks)
        blocks_to_process = _collect_nontrivial(
            all_blocks, compressed_blocks, is_integer, src_dtype
        )
        skipped = total_blocks - len(blocks_to_process)

        _run_block_batches(blocks_to_process, compressed_blocks, max_errors, num_processes, config)
        lengths, coeffs, masks_data = serialize_compressed_blocks(compressed_blocks)
    return lengths, coeffs, masks_data, max_errors, skipped


def compress_channel_dct_strips(
    src,
    band_idx: int,
    block_size: int,
    nodata_value: Optional[float],
    config: Config,
    num_processes: int,
    src_dtype: Optional[str] = None,
    errors_out: Optional[list] = None,
) -> _DctResult:
    """Compress a single channel strip by strip (about DCT_STRIP_PIXELS pixels each) with the same
    array paths as compress_channel_dct, so the whole band never sits in memory. Blocks are
    independent, so with a fixed block size the result equals that of compress_channel_dct. With
    automatic block size the choice comes from a window in the middle of the band (up to 2048 x 2048)."""
    H, W = src.height, src.width
    is_integer = src_dtype in _INTEGER_DTYPES

    if config.compression.auto_select_block_size:
        sample_h, sample_w = min(2048, H), min(2048, W)
        sample = src.read(
            band_idx,
            window=Window((W - sample_w) // 2, (H - sample_h) // 2, sample_w, sample_h),
        ).astype(np.float32)
        config = deepcopy(config)
        if is_integer:
            sf, bs = auto_select_uint8_parameters(sample, config, nodata_value)
            config.compression.uint8_scaling_factor = [sf]
        else:
            bs = auto_select_block_size(sample, config, nodata_value, src_dtype)
        config.compression.block_size = block_size = bs
        del sample

    n = block_size
    padded_H, padded_W, padded_shape = _calc_padded_dims(H, W, n)
    strip_rows = max(1, DCT_STRIP_PIXELS // (padded_W * n))  # block rows per strip
    length_parts: list = []
    coeff_parts: list = []
    error_parts: list = []
    masks_data: list = []
    total_blocks = 0
    skipped_total = 0
    for y0 in range(0, padded_H, strip_rows * n):
        h = min(strip_rows * n, H - y0)
        image = src.read(band_idx, window=Window(0, y0, W, h)).astype(np.float32)
        padded_strip, _ = pad_image(image, n)
        del image
        lengths, coeffs, masks, errors, skipped = _compress_padded_blocks(
            padded_strip, n, nodata_value, config, is_integer, src_dtype, num_processes
        )
        del padded_strip
        length_parts.append(lengths)
        coeff_parts.append(coeffs)
        error_parts.append(errors)
        # mask entries carry the block index within the whole band
        masks_data.extend([total_blocks + i, mask] for i, mask in masks)
        total_blocks += len(lengths)
        skipped_total += skipped
    if not config.output.quiet:
        print(f"  Strip mode: {total_blocks} blocks (skipped {skipped_total} trivial)")

    if all(isinstance(part, np.ndarray) for part in error_parts):
        max_errors = np.concatenate(error_parts)
    else:
        max_errors = [e for part in error_parts for e in part]
    max_error_global, validity = _error_summary(max_errors, config.compression.accuracy)

    if not validity and config.compression.auto_select_block_size and block_size > 8:
        smaller_n = {32: 16, 16: 8}[block_size]
        if not config.output.quiet:
            failed = sum(1 for e in max_errors if e > config.compression.accuracy)
            print(f"  ✗ Validity FAILED ({failed} blocks) - retrying with N={smaller_n}")
        retry_cfg = deepcopy(config)
        retry_cfg.compression.block_size = smaller_n
        retry_cfg.compression.auto_select_block_size = False
        return compress_channel_dct_strips(
            src, band_idx, smaller_n, nodata_value, retry_cfg, num_processes, src_dtype, errors_out
        )
    if errors_out is not None:
        errors_out[:] = [np.asarray(max_errors, dtype=np.float64)]

    lengths = np.concatenate(length_parts)
    coeffs_np = np.concatenate(coeff_parts)
    return lengths, coeffs_np, masks_data, max_error_global, validity, block_size, padded_shape


def compress_channel_dct_streaming(
    src,
    band_idx: int,
    block_size: int,
    nodata_value: Optional[float],
    config: Config,
    num_processes: int,
    src_dtype: Optional[str] = None,
    errors_out: Optional[list] = None,
) -> _DctResult:
    """Compress a single channel via rasterio Window strips (low-memory mode).

    Sends whole strip numpy arrays to workers — eliminates per-block IPC overhead.
    Retries with smaller block size if validity fails.
    """
    H, W = src.height, src.width

    config, block_size = _streaming_detect_and_autoselect(
        src, band_idx, H, W, block_size, nodata_value, config, src_dtype
    )

    is_integer = src_dtype in _INTEGER_DTYPES

    n = block_size

    padded_H, padded_W, padded_shape = _calc_padded_dims(H, W, n)
    total_blocks = (padded_H // n) * (padded_W // n)

    compressed_blocks = [None] * total_blocks
    max_errors = [0.0] * total_blocks

    strip_rows = _calc_strip_rows(W, n)
    blocks_per_row = padded_W // n

    if not config.output.quiet:
        print(f"  Streaming mode: strip={strip_rows} rows, total blocks={total_blocks}")

    effective_processes = _calc_effective_processes(total_blocks, num_processes)

    if effective_processes > 1:
        _run_strips_multiprocess(
            src,
            band_idx,
            H,
            W,
            padded_H,
            padded_W,
            n,
            strip_rows,
            blocks_per_row,
            nodata_value,
            config,
            effective_processes,
            compressed_blocks,
            max_errors,
        )
    else:
        _run_strips_singleprocess(
            src,
            band_idx,
            H,
            W,
            padded_H,
            padded_W,
            n,
            strip_rows,
            blocks_per_row,
            nodata_value,
            config,
            compressed_blocks,
            max_errors,
        )

    trivial_count = sum(1 for b in compressed_blocks if not isinstance(b, np.ndarray))
    if not config.output.quiet:
        print(f"  Total blocks: {total_blocks} (skipped {trivial_count} trivial)")

    max_error_global = max(max_errors) if max_errors else 0.0
    # Use config.compression.accuracy (channel-specific for YCbCr, or global uint8_accuracy)
    validity = all(e <= config.compression.accuracy for e in max_errors)

    if not validity and config.compression.auto_select_block_size and block_size > 8:
        return _retry_with_smaller_n(
            src,
            block_size,
            nodata_value,
            config,
            num_processes,
            max_errors,
            src_dtype,
            in_memory=False,
            band_idx=band_idx,
            errors_out=errors_out,
        )
    if errors_out is not None:
        errors_out[:] = [np.asarray(max_errors, dtype=np.float64)]

    lengths, coeffs_np, masks_data = serialize_compressed_blocks(compressed_blocks)
    return lengths, coeffs_np, masks_data, max_error_global, validity, block_size, padded_shape


def build_ycbcr_images(src, config, num_channels, all_uint8, _data_band_indices):
    """Pre-compute YCbCr images and per-channel accuracy budgets.

    Returns (ycbcr_images, ycbcr_acc, ycbcr_rgb_0).
    All dicts/lists are empty if conditions not met.
    """
    ycbcr_images = {}
    ycbcr_acc = {}
    ycbcr_rgb_0 = []

    rgb_idx_cfg = list(getattr(config.compression, "rgb_channel_indices", []))
    if not (
        len(rgb_idx_cfg) == 3
        and all_uint8
        and getattr(config.compression, "uint8_accuracy_mode", False)
    ):
        return ycbcr_images, ycbcr_acc, ycbcr_rgb_0

    r0, g0, b0 = [int(i) - 1 for i in rgb_idx_cfg]
    if not all(0 <= i < num_channels for i in [r0, g0, b0]):
        return ycbcr_images, ycbcr_acc, ycbcr_rgb_0

    r_arr = src.read(r0 + 1).astype(np.float32)
    g_arr = src.read(g0 + 1).astype(np.float32)
    b_arr = src.read(b0 + 1).astype(np.float32)
    u8a = float(getattr(config.compression, "uint8_accuracy", 1))

    ycbcr_images[r0], ycbcr_images[g0], ycbcr_images[b0] = rgb_to_ycbcr(r_arr, g_arr, b_arr)
    ycbcr_acc[r0] = u8a * config.compression.ycbcr_y_multiplier  # Y
    ycbcr_acc[g0] = u8a * config.compression.ycbcr_cb_multiplier  # Cb
    ycbcr_acc[b0] = u8a * config.compression.ycbcr_cr_multiplier  # Cr
    ycbcr_rgb_0 = [r0, g0, b0]

    del r_arr, g_arr, b_arr

    if not config.output.quiet:
        print(
            f"  YCbCr decorrelation: ch{r0+1}=Y(acc={ycbcr_acc[r0]:.2f}) "
            f"ch{g0+1}=Cb(acc={ycbcr_acc[g0]:.2f}) "
            f"ch{b0+1}=Cr(acc={ycbcr_acc[b0]:.2f}) "
            f"[guarantees ≤{u8a} in RGB]"
        )

    return ycbcr_images, ycbcr_acc, ycbcr_rgb_0


def compress_channel_rgb(
    src,
    r_idx: int,
    g_idx: int,
    b_idx: int,
    block_size: int,
    nodata_value: Optional[float],
    config: Config,
    num_processes: int,
) -> _DctResult:
    """Compress R, G, B channels via per-block YCbCr multiplier prediction (cm=6).

    For each block triplet (Y, Cb, Cr) we:
      1. Predict the starting multiplier index from chroma amplitude
         (skips multipliers that are analytically too large for complex blocks).
      2. Try the predicted multiplier; accept immediately if RGB error ≤ accuracy.
      3. Iterate down the fallback list only when needed (rare for good predictions).

    Returns the standard 7-tuple:
        (lengths, coeffs_dict, masks_data, max_error, validity, block_size, padded_shape)
    """
    H, W = src.height, src.width
    n = block_size
    padded_H, padded_W = H + (n - H % n) % n, W + (n - W % n) % n
    padded_shape = (padded_H, padded_W)
    rows, cols = padded_H // n, padded_W // n
    total_blocks = rows * cols
    strip_rows = max(1, RGB_STRIP_PIXELS // (padded_W * n))  # block rows per strip

    # NoData value inside the uint8 range: per-band masks (R, G, B) over the unpadded image, from
    # which the decoder writes the exact NoData value back. A file without NoData carries the -9999
    # sentinel and gets no masks.
    has_nodata = nodata_value is not None and 0 <= nodata_value <= 255
    nodata_rles = [RleBuilder() for _ in range(3)] if has_nodata else None

    names = ("Y", "Cb", "Cr")
    lens = {ch: [] for ch in names}  # per strip: int64 array of the L of every block
    coefs = {ch: [] for ch in names}  # per strip: int64 array of the stored coefficients
    masks_data: list = []
    sf_cur = config.scaling_factor
    max_error = 0.0
    fallback_n = 0

    n_strips = -(-rows // strip_rows)
    for strip_no, br0 in enumerate(range(0, rows, strip_rows)):
        br1 = min(rows, br0 + strip_rows)
        y0, y1 = br0 * n, min(br1 * n, H)
        window = Window(0, y0, W, y1 - y0)
        bands = [src.read(i + 1, window=window).astype(np.float32) for i in (r_idx, g_idx, b_idx)]
        if has_nodata:
            for builder, band in zip(nodata_rles, bands):
                builder.add(band == nodata_value)
        if not config.output.quiet and n_strips > 1:
            print(f"  RGB strip {strip_no + 1}/{n_strips}: rows {y0}-{y1}")
        result = _compress_rgb_strip(
            bands, (br1 - br0) * n - (y1 - y0), padded_W - W, n, nodata_value, has_nodata, config,
            num_processes, br0 * cols, sf_cur,
        )
        del bands
        strip_lens, strip_coefs, strip_masks, strip_err, strip_fallback = result
        for ch in names:
            lens[ch].append(strip_lens[ch])
            coefs[ch].append(strip_coefs[ch])
        masks_data.extend(strip_masks)
        max_error = max(max_error, strip_err)
        fallback_n += strip_fallback

    base_acc = float(getattr(config.compression, "uint8_accuracy", 5))
    validity = max_error <= base_acc

    if not config.output.quiet:
        print(
            f"  cm=6 per-block YCbCr: max_err={max_error:.2f}, valid={validity},"
            f" fallback_blocks={fallback_n}/{total_blocks}"
            f" ({100*fallback_n/max(1,total_blocks):.1f}%)"
        )

    lengths = {ch: np.concatenate(lens[ch]).tolist() for ch in names}
    coeffs_dict = {
        **{ch: {"coeffs": np.concatenate(coefs[ch]), "multipliers": []} for ch in names},
        "sf": sf_cur if total_blocks else None,
        "fallback": [],
    }
    if has_nodata:
        coeffs_dict["nd"] = [builder.result() for builder in nodata_rles]

    return lengths, coeffs_dict, masks_data, max_error, validity, block_size, padded_shape


def _compress_rgb_strip(
    bands, pad_H, pad_W, n, nodata_value, has_nodata, config, num_processes, first_block, sf_cur
):
    """cm=6 for one strip of block rows: `bands` are the R, G, B rows of the strip, padded below by
    pad_H rows and to the right by pad_W columns (edge values). Blocks are independent, so a strip
    gives the same blocks as the whole image. Returns
    (lengths, coefficients, masks, max_error, fallback_count): lengths and coefficients per channel
    as int64 arrays in block order; mask entries carry the block index within the whole image."""
    r, g, b = bands
    Y, Cb, Cr = rgb_to_ycbcr(r, g, b)

    def _pad(ch: np.ndarray) -> np.ndarray:
        return np.pad(ch, ((0, pad_H), (0, pad_W)), mode="edge")

    Y_p, Cb_p, Cr_p = _pad(Y), _pad(Cb), _pad(Cr)
    r_p, g_p, b_p = _pad(r), _pad(g), _pad(b)
    del Y, Cb, Cr

    padded_H, padded_W = r_p.shape
    rows, cols = padded_H // n, padded_W // n
    total_blocks = rows * cols

    def _blocks(ch: np.ndarray) -> np.ndarray:
        return ch.reshape(rows, n, cols, n).transpose(0, 2, 1, 3).reshape(-1, n, n)

    # pylint: disable=invalid-name
    Y_blk = _blocks(Y_p)
    Cb_blk = _blocks(Cb_p)
    Cr_blk = _blocks(Cr_p)
    r_blk = _blocks(r_p)
    g_blk = _blocks(g_p)
    b_blk = _blocks(b_p)
    del Y_p, Cb_p, Cr_p, r_p, g_p, b_p

    # Trivial blocks are stored as markers, not coefficients: all-zero RGB -> L=0 (decoded as black),
    # all-NoData with NoData != 0 -> L=-1 (restored from the NoData masks). With NoData = 0 both are
    # the same block and get the single L=0 marker.
    zero_blk = ((r_blk == 0) & (g_blk == 0) & (b_blk == 0)).all(axis=(1, 2))
    nd_blk = np.zeros(total_blocks, dtype=bool)
    if has_nodata:
        nd_px = [ch == nodata_value for ch in (r_blk, g_blk, b_blk)]
        hole = nd_px[0] & nd_px[1] & nd_px[2]
        nd_blk = hole.all(axis=(1, 2)) & ~zero_blk
        # Mixed blocks: pixels that are NoData in all three bands (holes) get their neighbours' values
        # before the DCT (fewer coefficients); the decoder writes NoData back from the masks. A value
        # equal to NoData in one band only is ordinary image content and is coded as it is.
        part = hole.any(axis=(1, 2)) & ~nd_blk & ~zero_blk
        for k in np.flatnonzero(part):
            for ch_blk in (r_blk, g_blk, b_blk):
                ch_blk[k] = interpolate_zeros(ch_blk[k], ~hole[k])
        if np.any(part):
            k = np.flatnonzero(part)
            Y_blk[k], Cb_blk[k], Cr_blk[k] = rgb_to_ycbcr(r_blk[k], g_blk[k], b_blk[k])
        del nd_px, hole

    compressed_map: dict = {}
    max_errors_map: dict = {}
    for i in np.flatnonzero(zero_blk | nd_blk):
        L_mark = 0 if zero_blk[i] else -1
        compressed_map[int(i)] = {
            **{ch: {"L": L_mark, "coeffs": np.empty(0, dtype=np.int64), "multiplier": 1.0}
               for ch in ("Y", "Cb", "Cr")},
            "sf": sf_cur,
            "fallback": False,
            "masks": None,
        }
        max_errors_map[int(i)] = 0.0

    # Build list of RGB block tuples: (idx, Y, Cb, Cr, r, g, b, nodata)
    todo = np.flatnonzero(~(zero_blk | nd_blk))
    arrays_result = process_rgb_arrays(
        todo, Y_blk[todo], Cb_blk[todo], Cr_blk[todo], r_blk[todo], g_blk[todo], b_blk[todo], config
    )
    if arrays_result is not None:  # whole arrays end to end: no per-block tuples
        for bidx, metadata, err in arrays_result:
            compressed_map[int(bidx)] = metadata
            max_errors_map[int(bidx)] = err
        all_blocks = []
    else:
        all_blocks = [
            (i, Y_blk[i], Cb_blk[i], Cr_blk[i], r_blk[i], g_blk[i], b_blk[i], nodata_value)
            for i in range(total_blocks)
            if i not in compressed_map
        ]
    del Y_blk, Cb_blk, Cr_blk, r_blk, g_blk, b_blk

    blocks_per_batch = max(256, total_blocks // max(1, num_processes * 4))
    batches = [
        (i // blocks_per_batch, all_blocks[i : i + blocks_per_batch])
        for i in range(0, len(all_blocks), blocks_per_batch)
    ]

    effective = min(num_processes, len(batches))
    # A cm=6 block costs milliseconds (several multipliers x 3 channels), so the pool start-up pays
    # off from about a thousand blocks. Batches are formed the same way for any process count.
    if total_blocks < 1_000:
        effective = 1

    if effective > 1:
        batch_func = partial(process_block_batch_rgb, config=config)
        with multiprocessing.Pool(processes=effective) as pool:
            iter_ = pool.imap(batch_func, batches)
            if not config.output.quiet:
                iter_ = tqdm(iter_, total=len(batches), desc="RGB blocks", ascii=False)
            for batch_results in iter_:
                for idx, metadata, err in batch_results:
                    compressed_map[idx] = metadata
                    max_errors_map[idx] = err
    else:
        for batch_data in tqdm(
            batches, desc="RGB blocks", disable=config.output.quiet, ascii=False
        ):
            for idx, metadata, err in process_block_batch_rgb(batch_data, config):
                compressed_map[idx] = metadata
                max_errors_map[idx] = err
    del all_blocks, batches

    # Keep only the numbers the archive needs; the per-block dicts are dropped with this strip.
    names = ("Y", "Cb", "Cr")
    strip_lens = {ch: np.empty(total_blocks, dtype=np.int64) for ch in names}
    strip_coefs = {ch: [] for ch in names}
    masks: list = []
    fallback_n = 0
    for i in range(total_blocks):
        meta = compressed_map[i]
        for ch in names:
            strip_lens[ch][i] = meta[ch]["L"]
            strip_coefs[ch].append(np.asarray(meta[ch]["coeffs"], dtype=np.int64))
        fallback_n += bool(meta.get("fallback", False))
        if meta.get("masks"):
            masks.append([first_block + i, meta["masks"]])
    strip_coefs = {
        ch: np.concatenate(strip_coefs[ch]) if strip_coefs[ch] else np.empty(0, dtype=np.int64)
        for ch in names
    }
    max_error = max(max_errors_map.values()) if max_errors_map else 0.0
    return strip_lens, strip_coefs, masks, max_error, fallback_n


def _compress_blocks_arrays(padded, n, nodata_value, config, is_integer, src_dtype):
    """cm=5 on whole arrays of n x n (8, 16, 32) uint8 blocks: the same centring, DCT, rounding and zigzag as
    process_block_batch, then one channel search on the GPU session (or numba). With the predictor
    in mode 3 (uint8_use_L_prediction + accept) the predicted L drives accept_predicted on the CPU.
    Applies only when the batched path would apply (lockstep on, no heuristic backend, no NoData,
    no zero masks, uint8, n in 8/16/32); returns (lengths, coeffs, max_errors, trivial_count) or None:
    the length of every block (0 for an all-zero block, as serialize_compressed_blocks stores it),
    the coefficients of all blocks one after another and the error of every block, as arrays."""
    from core.dct import dct2, _get_zigzag_indices
    from engine.blocks import _array_path_enabled, scaled_backscan, search_u8_arrays
    from engine.utils import integer_range

    from engine.blocks import ARRAY_EDGES

    if not (is_integer and src_dtype == "uint8" and n in ARRAY_EDGES):
        return None
    # NoData that no uint8 pixel can equal (None, or the -9999 sentinel of files without NoData):
    # there are no NoData blocks, masks or flips. NoData inside 0..255 keeps the per-block path.
    if nodata_value is not None and 0 <= nodata_value <= 255:
        return None
    if not _array_path_enabled(config):
        return None
    # predictor in mode 3 (uint8_use_L_prediction + accept) runs here; without accept it stays on
    # the per-block path
    predict = bool(config.model.uint8_use_L_prediction)
    if predict and (n != 8 or not config.model.accept_prediction_if_within_accuracy):
        return None
    if (config.model.backend.lower() in ("heuristic",) or config.model.advanced_heuristic
            or config.compression.uint8_zero_mask):
        return None
    H, W = padded.shape
    blocks = padded.reshape(H // n, n, W // n, n).transpose(0, 2, 1, 3).reshape(-1, n, n)
    N = blocks.shape[0]
    allzero = (blocks == 0).all(axis=(1, 2))
    lengths = np.zeros(N, dtype=np.int64)  # 0: ALL_ZEROS, as _collect_nontrivial
    max_errors = np.zeros(N)
    todo = np.flatnonzero(~allzero)
    if todo.size == 0:
        return lengths, np.empty(0, dtype=np.int64), max_errors, int(allzero.sum())
    center = integer_range(src_dtype)[0]
    A = blocks[todo] - center  # float32 - python float stays float32, as the per-block path
    dct = np.round(dct2(A, config.compression.dct_type), config.compression.decimal_places)
    m = todo.size
    ZZ = np.ascontiguousarray(dct.reshape(m, -1)[:, _get_zigzag_indices(n)].astype(np.float64))
    OR = np.ascontiguousarray(A.astype(np.float64).reshape(m, n * n))
    LP = scales = None
    if predict:
        from engine.blocks import predict_L_arrays

        LP = predict_L_arrays(ZZ, config, config.compression.accuracy)
        scales = tuple(float(x) for x in config.model.uint8_L_prediction_scales)
    LF, ERR, RES = search_u8_arrays(
        ZZ, OR, np.full(m, float(config.compression.accuracy)),
        np.full(m, float(config.scaling_factor)),
        np.full(m, scaled_backscan(config.model.minimize_backscan, n), dtype=np.int64),
        LP=LP, scales=scales, brk=int(config.model.backscan_break_after),
    )
    lengths[todo] = LF
    max_errors[todo] = ERR
    coeffs = RES[np.arange(n * n)[None, :] < LF[:, None]].astype(np.int64, copy=False)
    return lengths, coeffs, max_errors, int(allzero.sum())


def _f32_array_enabled(config) -> bool:
    """Array search for float32 blocks: used for every backend except the heuristic predictors."""
    return config.model.backend.lower() not in ("heuristic",) and not config.model.advanced_heuristic


def _compress_blocks_f32(padded, n, nodata_value, config, src_dtype):
    """cm=3 (float32) on whole arrays. Clean blocks (finite, no zeros, no NoData inside) go through
    engine.f32_arrays.search_f32 with the same results as refine_dct_array; all-NoData blocks get the
    length -1 and all-zero blocks the length 0, as serialize_compressed_blocks stores the sentinels of
    _collect_nontrivial. Blocks that mix zeros or NoData with values are returned as (idx, block,
    nodata) tuples for the per-block path, which writes their masks; here they hold length 0.
    Returns (lengths, coeffs, max_errors, skipped, mixed_blocks) or None: the length and the error of
    every block and the coefficients of all blocks one after another, as arrays."""
    from core.dct import dct2, _get_zigzag_indices
    from engine.f32_arrays import search_f32

    # N = 32 stays on the per-block pool: its reconstructions (tensordot, single-threaded BLAS) make the
    # array search slower than the 32 workers, 25 s against 7 s on a 4096 x 4096 crop
    if src_dtype != "float32" or n > 16 or not _f32_array_enabled(config):
        return None
    H, W = padded.shape
    blocks = padded.reshape(H // n, n, W // n, n).transpose(0, 2, 1, 3).reshape(-1, n, n)
    N = blocks.shape[0]
    nd_eq = (blocks == nodata_value) if nodata_value is not None else np.zeros(blocks.shape, bool)
    allnd = nd_eq.all(axis=(1, 2))
    allzero = (blocks == 0).all(axis=(1, 2)) & ~allnd
    zero_any = (blocks == 0).any(axis=(1, 2))
    clean = np.isfinite(blocks).all(axis=(1, 2)) & ~zero_any & ~nd_eq.any(axis=(1, 2))
    lengths = np.zeros(N, dtype=np.int64)  # 0: ALL_ZEROS
    lengths[allnd] = -1  # ALL_NODATA
    coeffs = np.empty(0, dtype=np.int64)
    max_errors = np.zeros(N)
    mixed = np.flatnonzero(~(allnd | allzero | clean))
    mixed_blocks = [(int(i), blocks[i], nodata_value) for i in mixed]
    todo = np.flatnonzero(clean)
    if todo.size:
        zz_idx = _get_zigzag_indices(n)
        A = np.ascontiguousarray(blocks[todo], dtype=np.float32)
        dct = np.round(dct2(A, config.compression.dct_type), config.compression.decimal_places)
        zz = np.ascontiguousarray(dct.reshape(todo.size, -1)[:, zz_idx], dtype=np.float32)
        acc = np.full(todo.size, float(config.compression.accuracy))
        LF, ERR, coeffs = search_f32(
            zz, A, acc, config.scaling_factor, config.compression.dct_type,
            backscan=int(config.model.minimize_backscan),
            brk=int(config.model.backscan_break_after),
        )
        lengths[todo] = LF
        max_errors[todo] = ERR
    skipped = int(allnd.sum() + allzero.sum())
    return lengths, coeffs, max_errors, skipped, mixed_blocks


def _prefilter_trivial(_all_blocks, total_blocks):
    compressed_blocks = [None] * total_blocks
    max_errors = [0.0] * total_blocks
    return compressed_blocks, max_errors


def _trivial_flags(all_blocks):
    """Vectorised 'all zero' / 'all NoData' flags of the blocks (same tests as the per-block loop).
    None when the blocks cannot be stacked (mixed shapes or several NoData values)."""
    if not all_blocks:
        return None
    nds = {nd for _, _, nd in all_blocks}
    if len(nds) > 1:
        return None
    try:
        A = np.stack([np.asarray(b) for _, b, _ in all_blocks])
    except ValueError:
        return None
    ax = tuple(range(1, A.ndim))
    nd = next(iter(nds))
    all_nodata = (A == nd).all(axis=ax) if nd is not None else np.zeros(A.shape[0], dtype=bool)
    return (A == 0).all(axis=ax), all_nodata


def _collect_nontrivial(all_blocks, compressed_blocks, is_integer, src_dtype):
    blocks_to_process = []
    flags = _trivial_flags(all_blocks)
    if flags is not None:
        all_zero, all_nodata = flags
        for k, (idx, block, nodata) in enumerate(all_blocks):
            if nodata is not None and all_nodata[k]:
                compressed_blocks[idx] = 1  # ALL_NODATA
            elif all_zero[k]:
                compressed_blocks[idx] = 0  # ALL_ZEROS
            else:
                blocks_to_process.append((idx, block, nodata))
    else:
        for idx, block, nodata in all_blocks:
            if nodata is not None and np.all(block == nodata):
                compressed_blocks[idx] = 1  # ALL_NODATA
            elif np.all(block == 0):
                compressed_blocks[idx] = 0  # ALL_ZEROS
            else:
                blocks_to_process.append((idx, block, nodata))

    if is_integer and blocks_to_process:
        blocks_to_process = [(idx, blk, nd, src_dtype) for idx, blk, nd in blocks_to_process]

    return blocks_to_process


def _run_block_batches(blocks_to_process, compressed_blocks, max_errors, num_processes, config):
    if not blocks_to_process:
        return

    # The cost of a block grows with its pixels, so the smallest batch and the serial threshold
    # hold the same number of pixels for every block size: 2048 and 4000 blocks of 8 x 8,
    # 512 and 1000 of 16 x 16, 128 and 250 of 32 x 32. Only with forked workers, which share the IDCT
    # basis built below; a spawned worker costs an import and builds its own basis, so there the
    # counts stay those of 8 x 8.
    forked = multiprocessing.get_start_method() == "fork"
    pixel_scale = max(1, int(np.shape(blocks_to_process[0][1])[0]) ** 2 // 64) if forked else 1
    blocks_per_batch = max(max(1, 2048 // pixel_scale), len(blocks_to_process) // num_processes)
    batches = [
        (i // blocks_per_batch, blocks_to_process[i : i + blocks_per_batch])
        for i in range(0, len(blocks_to_process), blocks_per_batch)
    ]

    effective = min(num_processes, len(batches))

    # For small files the Pool start-up + IPC pickle overhead exceeds compute gains; above the
    # threshold the batches run in parallel. Every block is searched on its own, so the result
    # does not depend on the batch size or the process count.
    _SERIAL_THRESHOLD = 4_000 // pixel_scale
    if len(blocks_to_process) < _SERIAL_THRESHOLD:
        effective = 1

    if effective > 1:
        # Built here, the IDCT basis is inherited by the forked workers and they all read the same
        # pages. A copy per worker (8 MB for 32 x 32) pushes the block search out of the CPU cache.
        _get_idct_basis(int(np.shape(blocks_to_process[0][1])[0]), config.compression.dct_type)
        batch_func = partial(process_block_batch, config=config)
        with multiprocessing.Pool(processes=effective) as pool:
            if not config.output.quiet:
                results = list(
                    tqdm(
                        pool.imap(batch_func, batches),
                        total=len(batches),
                        desc="Compression",
                        ascii=False,
                    )
                )
            else:
                results = pool.map(batch_func, batches)
        for batch_results in results:
            for idx, compressed, error in batch_results:
                compressed_blocks[idx] = compressed
                max_errors[idx] = error
    else:
        for batch_data in tqdm(
            batches, desc="Compression", disable=config.output.quiet, ascii=False
        ):
            for idx, compressed, error in process_block_batch(batch_data, config):
                compressed_blocks[idx] = compressed
                max_errors[idx] = error


def _retry_with_smaller_n(
    image_or_src,
    block_size,
    nodata_value,
    config,
    num_processes,
    max_errors,
    src_dtype,
    in_memory,
    band_idx=None,
    errors_out=None,
):
    smaller_n = {32: 16, 16: 8}[block_size]
    if not config.output.quiet:
        failed = sum(1 for e in max_errors if e > config.compression.accuracy)
        print(f"  ✗ Validity FAILED ({failed} blocks) - retrying with N={smaller_n}")

    retry_cfg = deepcopy(config)
    retry_cfg.compression.block_size = smaller_n
    retry_cfg.compression.auto_select_block_size = False

    if in_memory:
        return compress_channel_dct(
            image_or_src, smaller_n, nodata_value, retry_cfg, num_processes, src_dtype=src_dtype,
            errors_out=errors_out,
        )
    return compress_channel_dct_streaming(
        image_or_src,
        band_idx,
        smaller_n,
        nodata_value,
        retry_cfg,
        num_processes,
        src_dtype=src_dtype,
        errors_out=errors_out,
    )


def _streaming_detect_and_autoselect(
    src, band_idx, H, W, block_size, nodata_value, config, src_dtype
):
    sample_h = min(1024, H)
    sample_w = min(1024, W)
    row_off = max(0, (H - sample_h) // 2)
    col_off = max(0, (W - sample_w) // 2)
    sample = src.read(band_idx, window=Window(col_off, row_off, sample_w, sample_h)).astype(
        np.float32
    )

    is_integer = src_dtype in _INTEGER_DTYPES

    if config.compression.auto_select_block_size:
        block_size = auto_select_block_size(sample, config, nodata_value, src_dtype)
        config = deepcopy(config)
        config.compression.block_size = block_size

    return config, block_size


def _calc_padded_dims(H, W, n):
    padded_H = H + (n - H % n) % n
    padded_W = W + (n - W % n) % n
    return padded_H, padded_W, (padded_H, padded_W)


def _calc_strip_rows(W, n):
    strip_rows = max(n, (64 * 1024 * 1024) // (W * 4))
    return ((strip_rows + n - 1) // n) * n


def compress_channel_dct_adaptive(
    src, band_idx, block_size, nodata_value, config, num_processes, src_dtype, use_streaming
):
    """cm=3 (float) with the scaling factor chosen from the accuracy (select_scaling_factor) and
    per-block exceptions: every block whose error exceeds the accuracy with that factor is compressed
    again with sf * 10, sf * 100, sf * 1000 until it holds it, and replaces its own entry in the
    streams (lengths, coefficients, masks). Blocks that are all zero or all NoData are markers without
    coefficients and never need it; blocks that mix NoData/zeros with data carry coefficients and
    go through the same check.
    Returns the 7-tuple of compress_channel_dct plus (sf, exceptions): exceptions is
    [gaps, levels] - block index = running sum of the gaps, scaling factor of the block
    sf * 10**level - or None when every block holds the accuracy with sf."""
    H, W = src.height, src.width
    image = None if use_streaming else src.read(band_idx).astype(np.float32)
    if use_streaming:
        sh, sw = min(2048, H), min(2048, W)
        sample = src.read(band_idx, window=Window((W - sw) // 2, (H - sh) // 2, sw, sh)).astype(np.float32)
    else:
        sample = image
    if config.compression.auto_select_block_size:
        block_size = auto_select_block_size(sample, config, nodata_value, src_dtype)
    sf = select_scaling_factor(sample, block_size, nodata_value, config)
    del sample
    cfg = scaled_config(config, sf)
    cfg.compression.auto_select_block_size = False  # block failures are handled by the exceptions
    cfg.compression.block_size = block_size

    errors_out: list = []
    if use_streaming:
        result = compress_channel_dct_streaming(
            src, band_idx, block_size, nodata_value, cfg, num_processes, src_dtype, errors_out=errors_out
        )
    else:
        result = compress_channel_dct(
            image, block_size, nodata_value, cfg, num_processes, src_dtype, errors_out=errors_out
        )
    lengths, coeffs, masks_data, _, _, n, padded_shape = result
    errors = errors_out[0]
    lengths = np.asarray(lengths, dtype=np.int64)
    coeffs = np.asarray(coeffs, dtype=np.int64)
    masks_data = list(masks_data)
    acc = float(config.compression.accuracy)
    padded_W = padded_shape[1]
    bpr = padded_W // n
    padded = None if image is None else pad_image(image, n)[0]
    del image

    def block_row(r):
        if padded is not None:
            return padded[r * n : (r + 1) * n]
        h = min(n, H - r * n)
        rows = src.read(band_idx, window=Window(0, r * n, W, h)).astype(np.float32)
        return np.pad(rows, ((0, n - h), (0, padded_W - W)), mode="edge")

    level = np.zeros(lengths.size, dtype=np.int64)
    fail = np.flatnonzero(errors > acc)
    blocks = _gather_blocks(block_row, fail, n, bpr)
    for k in range(1, SF_EXCEPTION_LEVELS + 1):
        if fail.size == 0:
            break
        # the failing blocks side by side form an image of one block row; blocks are independent
        row = np.ascontiguousarray(blocks.transpose(1, 0, 2).reshape(n, fail.size * n))
        err_k: list = []
        new_l, new_c, new_m, *_ = compress_channel_dct(
            row, n, nodata_value, scaled_config(cfg, sf * 10**k, quiet=True), num_processes,
            src_dtype, errors_out=err_k,
        )
        lengths, coeffs, masks_data = _replace_blocks(
            lengths, coeffs, masks_data, fail, new_l, new_c, new_m
        )
        errors[fail] = err_k[0]
        level[fail] = k
        keep = err_k[0] > acc
        fail, blocks = fail[keep], blocks[keep]

    max_error = float(errors.max()) if errors.size else 0.0
    validity = bool((errors <= acc).all())
    exc = np.flatnonzero(level)
    if not config.output.quiet:
        if exc.size:
            per_level = ", ".join(f"x{10**k}: {int((level == k).sum())}" for k in range(1, SF_EXCEPTION_LEVELS + 1))
            print(
                f"  Adaptive sf={sf}: {exc.size}/{lengths.size} blocks ({100 * exc.size / lengths.size:.2f}%)"
                f" with a finer step ({per_level}) -> archive format v4"
            )
        else:
            print(f"  Adaptive sf={sf}: every block holds the accuracy, no per-block exceptions")
        if not validity:
            print(f"  ✗ {int((errors > acc).sum())} blocks miss the accuracy even with sf={sf * 10**SF_EXCEPTION_LEVELS}")
    exceptions = [np.diff(exc, prepend=0).tolist(), level[exc].tolist()] if exc.size else None
    return lengths, coeffs, masks_data, max_error, validity, n, padded_shape, sf, exceptions


def _gather_blocks(block_row, idx, n, bpr):
    """Blocks idx (sorted) of a padded band as an (len(idx), n, n) float32 array; block_row(r) gives
    the padded block row r (n rows, bpr * n columns)."""
    out = np.empty((idx.size, n, n), dtype=np.float32)
    rows = idx // bpr
    for r in np.unique(rows):
        sel = np.flatnonzero(rows == r)
        strip = block_row(int(r)).reshape(n, bpr, n)
        out[sel] = strip[:, idx[sel] % bpr, :].transpose(1, 0, 2)
    return out


def _replace_blocks(lengths, coeffs, masks_data, idx, new_lengths, new_coeffs, new_masks):
    """Streams of a channel with the blocks idx (sorted) replaced by their new compressed form:
    new_lengths / new_coeffs / new_masks as compress_channel_dct gives them for these blocks in
    the order of idx (mask entries carry the position within idx)."""
    keep = np.ones(lengths.size, dtype=bool)
    keep[idx] = False
    kept = coeffs[np.repeat(keep, np.maximum(lengths, 0))]
    lengths = lengths.copy()
    lengths[idx] = np.asarray(new_lengths, dtype=np.int64)
    from_new = np.repeat(~keep, np.maximum(lengths, 0))
    merged = np.empty(from_new.size, dtype=np.int64)
    merged[from_new] = np.asarray(new_coeffs, dtype=np.int64)
    merged[~from_new] = kept
    replaced = set(idx.tolist())
    masks = [m for m in masks_data if int(m[0]) not in replaced]
    masks += [[int(idx[int(j)]), mask] for j, mask in new_masks]
    masks.sort(key=lambda m: int(m[0]))
    return lengths, merged, masks


def _calc_effective_processes(total_blocks, num_processes):
    min_per_worker = max(256, total_blocks // (num_processes * 4))
    return min(num_processes, max(1, total_blocks // min_per_worker))


def _read_and_pad_strip(src, band_idx, y_start, H, W, strip_rows, n, pad_W):
    y_end = min(y_start + strip_rows, H)
    read_h = y_end - y_start
    if read_h <= 0:
        return None, 0

    strip_data = src.read(band_idx, window=Window(0, y_start, W, read_h)).astype(np.float32)

    if pad_W > 0:
        strip_data = np.pad(strip_data, ((0, 0), (0, pad_W)), mode="edge")
    strip_pad_h = (n - read_h % n) % n if read_h % n != 0 else 0
    if strip_pad_h > 0:
        strip_data = np.pad(strip_data, ((0, strip_pad_h), (0, 0)), mode="edge")

    return strip_data, read_h


def _collect_strip_results(ar_or_results, compressed_blocks, max_errors):
    for idx, compressed, error in ar_or_results:
        compressed_blocks[idx] = compressed
        max_errors[idx] = error


def _run_strips_multiprocess(
    src,
    band_idx,
    H,
    W,
    padded_H,
    padded_W,
    n,
    strip_rows,
    blocks_per_row,
    nodata_value,
    config,
    effective_processes,
    compressed_blocks,
    max_errors,
):
    pad_W = padded_W - W
    total_blocks = len(compressed_blocks)
    strip_func = partial(process_strip_batch, config=config)

    pbar = (
        None
        if config.output.quiet
        else tqdm(total=total_blocks, desc="Compression", unit="blk", ascii=False)
    )
    pending = []

    _get_idct_basis(n, config.compression.dct_type)  # shared by the forked workers, as in _run_block_batches
    with multiprocessing.Pool(processes=effective_processes) as pool:
        for y_start in range(0, padded_H, strip_rows):
            strip_data, _ = _read_and_pad_strip(src, band_idx, y_start, H, W, strip_rows, n, pad_W)
            if strip_data is None:
                break

            ar = pool.apply_async(
                strip_func, ((strip_data, y_start, blocks_per_row, nodata_value),)
            )
            pending.append(ar)

            if pbar:
                sH, sW = strip_data.shape
                pbar.update((sH // n) * (sW // n))

            del strip_data

            if len(pending) > effective_processes * 2:
                _flush_pending(pending, compressed_blocks, max_errors, limit=effective_processes)

        for ar in pending:
            _collect_strip_results(ar.get(), compressed_blocks, max_errors)

    if pbar:
        pbar.close()


def _run_strips_singleprocess(
    src,
    band_idx,
    H,
    W,
    padded_H,
    padded_W,
    n,
    strip_rows,
    blocks_per_row,
    nodata_value,
    config,
    compressed_blocks,
    max_errors,
):
    pad_W = padded_W - W
    total_blocks = len(compressed_blocks)
    pbar = (
        None
        if config.output.quiet
        else tqdm(total=total_blocks, desc="Compression", unit="blk", ascii=False)
    )

    for y_start in range(0, padded_H, strip_rows):
        strip_data, _ = _read_and_pad_strip(src, band_idx, y_start, H, W, strip_rows, n, pad_W)
        if strip_data is None:
            break

        results = process_strip_batch((strip_data, y_start, blocks_per_row, nodata_value), config)
        _collect_strip_results(results, compressed_blocks, max_errors)

        if pbar:
            sH, sW = strip_data.shape
            pbar.update((sH // n) * (sW // n))

        del strip_data

    if pbar:
        pbar.close()


def _flush_pending(pending, compressed_blocks, max_errors, limit):
    still_pending = []
    for ar in pending:
        if ar.ready():
            _collect_strip_results(ar.get(), compressed_blocks, max_errors)
        else:
            still_pending.append(ar)
            if len(still_pending) >= limit:
                _collect_strip_results(still_pending.pop(0).get(), compressed_blocks, max_errors)
    pending[:] = still_pending


__all__ = [
    "compress_channel_dct",
    "compress_channel_dct_streaming",
    "build_ycbcr_images",
    "compress_channel_rgb",
]
