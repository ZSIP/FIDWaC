"""
FIDWAC v2 - Shared compression utilities
=========================================
Common helpers used across compression paths:
  - data characteristics detection
  - block padding / extraction
  - block size auto-selection
  - compressed-blocks serialization
  - archive packing helpers
"""

import os
import re
import shutil
import subprocess
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import msgpack
import py7zr

from config import Config
from engine.blocks import process_block
from engine.utils import numpy_to_python

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RASTER_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".asc", ".img", ".gif"}

_STREAMING_THRESHOLD = 256 * 1024 * 1024  # bytes — ~256 MB for float32


# ---------------------------------------------------------------------------
# Padding / block extraction
# ---------------------------------------------------------------------------


def pad_image(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad image to multiple of block_size."""
    H, W = image.shape
    pad_H = (block_size - H % block_size) % block_size
    pad_W = (block_size - W % block_size) % block_size

    padded = np.pad(image, ((0, pad_H), (0, pad_W)), mode="edge") if (pad_H or pad_W) else image
    return padded, (H + pad_H, W + pad_W)


def extract_blocks(
    image: np.ndarray, block_size: int, nodata_value: Optional[float]
) -> List[Tuple[int, np.ndarray, Optional[float]]]:
    """Extract all (index, block, nodata) tuples from image (vectorized)."""
    H, W = image.shape
    n = block_size
    blocks_3d = image.reshape(H // n, n, W // n, n).transpose(0, 2, 1, 3).reshape(-1, n, n)
    return [(i, blocks_3d[i], nodata_value) for i in range(blocks_3d.shape[0])]


# ---------------------------------------------------------------------------
# Auto block-size selection
# ---------------------------------------------------------------------------


def _sample_results_per_block(sample_blocks, test_n, config):
    """Sample results of the per-block path: (compressed_list, errors) as process_block returns them."""
    compressed_list = []
    errors = []
    for block_data in sample_blocks:
        try:
            _, compressed, error = process_block(block_data, config)
            compressed_list.append(compressed)
            errors.append(error)
        except Exception:
            compressed_list.append([0] * (test_n * test_n))
            errors.append(0.0)
    return compressed_list, errors


def _sample_results_f32(grid, idx, test_n, config, nodata_value, dtype=np.float32):
    """Sample results of float32 blocks on arrays, the same values process_block gives. grid (m, n, n)
    holds the blocks, idx the sampled ones. All-zero blocks give 0, all-NoData blocks 1, clean blocks
    the search of engine.f32_arrays (identical to refine_dct_array), and the blocks that mix zeros or
    NoData with values go through process_block. process_block transforms a single block with
    dct2 on 2-D input (rows first), so the sample DCT uses the same axis order."""
    from scipy.fft import dct
    from core.dct import _get_zigzag_indices
    from engine.f32_arrays import search_f32

    m = len(idx)
    A = np.ascontiguousarray(grid[idx], dtype=dtype)
    nd_eq = (A == nodata_value) if nodata_value is not None else np.zeros(A.shape, dtype=bool)
    allzero = (A == 0).all(axis=(1, 2))
    allnd = nd_eq.all(axis=(1, 2)) & ~allzero
    clean = np.isfinite(A).all(axis=(1, 2)) & ~(A == 0).any(axis=(1, 2)) & ~nd_eq.any(axis=(1, 2))
    compressed = [None] * m
    errors = [0.0] * m
    for i in np.flatnonzero(allzero):
        compressed[int(i)] = 0
    for i in np.flatnonzero(allnd):
        compressed[int(i)] = 1
    todo = np.flatnonzero(clean)
    if todo.size:
        zz_idx = _get_zigzag_indices(test_n)
        C = np.ascontiguousarray(A[todo])
        t = config.compression.dct_type
        dct_c = dct(dct(C, axis=1, norm="ortho", type=t), axis=2, norm="ortho", type=t)
        dct_c = np.round(dct_c, config.compression.decimal_places)
        zz = np.ascontiguousarray(dct_c.reshape(todo.size, -1)[:, zz_idx], dtype=np.float32)
        LF, ERR, RES = search_f32(
            zz, C, np.full(todo.size, float(config.compression.accuracy)), config.scaling_factor,
            t, backscan=int(config.model.minimize_backscan),
            brk=int(config.model.backscan_break_after),
        )
        for j, i in enumerate(todo):
            compressed[int(i)] = RES[j, : int(LF[j])].tolist()
            errors[int(i)] = float(ERR[j])
    for i in np.flatnonzero(~(allzero | allnd | clean)):
        _, comp, err = process_block((int(idx[i]), grid[idx[i]], nodata_value), config)
        compressed[int(i)] = comp
        errors[int(i)] = err
    return compressed, errors


def auto_select_block_size(
    image: np.ndarray,
    config: Config,
    nodata_value: Optional[float] = None,
    src_dtype: Optional[str] = None,
) -> int:
    """Auto-select optimal block size (8, 16, 32) based on image std and sample compression.
    float32 data samples on arrays (same results as the per-block path, no per-block refinement)."""
    if not config.output.quiet:
        print("  Auto-selecting block size...")

    valid_data = image[image != nodata_value] if nodata_value is not None else image.flatten()
    image_std = np.std(valid_data) if len(valid_data) > 0 else 0

    std_high = config.compression.auto_select_std_threshold_high
    std_medium = config.compression.auto_select_std_threshold_medium

    test_sizes = [8, 16, 32]
    if image_std > std_high:
        test_sizes = [8]
        if not config.output.quiet:
            print(f"    Image std={image_std:.2f} (>{std_high}) - using only N=8")
    elif image_std > std_medium:
        test_sizes = [8, 16]
        if not config.output.quiet:
            print(f"    Image std={image_std:.2f} (>{std_medium}) - excluding N=32")

    allowed = config.compression.allowed_block_sizes
    test_sizes = [n for n in test_sizes if n in allowed] or [min(allowed)]
    if not config.output.quiet and len(test_sizes) < 3:
        print(f"    Allowed block sizes: {allowed} → testing {test_sizes}")

    results = {}
    for test_n in test_sizes:
        padded, _ = pad_image(image, test_n)
        max_sample = config.compression.auto_select_sample_size
        if src_dtype == "float32":
            # sample straight from the block array: the same evenly spaced blocks as blocks[::step]
            H, W = padded.shape
            grid = padded.reshape(H // test_n, test_n, W // test_n, test_n).transpose(0, 2, 1, 3)
            grid = grid.reshape(-1, test_n, test_n)
            total = grid.shape[0]
            step = max(1, total // min(max_sample, total))
            idx = np.arange(0, total, step)[:max_sample]
            compressed_list, errors = _sample_results_f32(grid, idx, test_n, config, nodata_value)
            sample_size = len(idx)
        else:
            blocks = extract_blocks(padded, test_n, nodata_value)
            total = len(blocks)
            step = max(1, total // min(max_sample, total))
            sample_blocks = blocks[::step][:max_sample]
            compressed_list, errors = _sample_results_per_block(sample_blocks, test_n, config)
            sample_size = len(sample_blocks)
        total_error = 0.0
        for error in errors:
            total_error += error

        sample_msgpack_size = len(msgpack.packb(compressed_list, use_bin_type=True))
        estimated_size = (sample_msgpack_size / sample_size) * total
        avg_error = total_error / sample_size

        results[test_n] = {"estimated_size": estimated_size, "avg_error": avg_error}
        if not config.output.quiet:
            print(
                f"    N={test_n}: blocks={total}, est_size={estimated_size/1024:.1f}KB, avg_error={avg_error:.6f}"
            )

    best_n = min(results, key=lambda n: results[n]["estimated_size"])
    if not config.output.quiet:
        print(f"  Selected block size: {best_n}")
    return best_n


def auto_select_uint8_parameters(
    image: np.ndarray,
    config: Config,
    nodata_value: Optional[float] = None,
) -> Tuple[int, int]:
    """Auto-select optimal (scaling_factor, block_size) for uint8 compression.

    Prefers fast lookup grid if available, falls back to sample-based testing.

    Returns:
        (scaling_factor, block_size) with minimum estimated_size
    """
    if not config.output.quiet:
        print("  Auto-selecting uint8 parameters (sf, bs)...")

    # --- Try lookup grid first (fast O(1) lookup) ---
    try:
        from pathlib import Path
        from predictor.predictor_uint8 import load_uint8_lookup_grid, predict_uint8_parameters

        grid_path = Path("models/lookup_uint8_grid.npz")
        if grid_path.exists():
            grid_data = load_uint8_lookup_grid(str(grid_path))
            if grid_data is not None:
                # Extract features from full image
                valid_data = (
                    image[image != nodata_value] if nodata_value is not None else image.flatten()
                )

                if len(valid_data) > 0:
                    from core.dct import dct2, to_zigzag

                    # Sample a few blocks to compute average ac_abs_mean and zero_ratio
                    ref_block_size = getattr(config.compression, "block_size", 16)
                    padded, _ = pad_image(
                        image, ref_block_size
                    )  # Use configured reference block size
                    blocks = extract_blocks(padded, ref_block_size, nodata_value)

                    sample_size_cfg = getattr(config.compression, "auto_select_sample_size", 100)
                    max_sample = min(
                        sample_size_cfg, len(blocks)
                    )  # Sample configured blocks for feature estimation
                    step = max(1, len(blocks) // max_sample)
                    sample_blocks = blocks[::step][:max_sample]

                    acm_values = []
                    zr_values = []

                    for block_data in sample_blocks:
                        try:
                            dct = dct2(block_data)
                            zigzag = to_zigzag(dct)

                            # Calculate ac_abs_mean and zero_ratio
                            ac_zigzag = zigzag[1:]  # Skip DC
                            ac_abs_mean = np.mean(np.abs(ac_zigzag))
                            zero_ratio = np.sum(ac_zigzag == 0) / len(ac_zigzag)

                            acm_values.append(ac_abs_mean)
                            zr_values.append(zero_ratio)
                        except Exception:
                            pass

                    if acm_values and zr_values:
                        # Use median of sampled features for prediction
                        avg_acm = np.median(acm_values)
                        avg_zr = np.median(zr_values)

                        sf, bs = predict_uint8_parameters(avg_acm, avg_zr, grid_data)

                        if not config.output.quiet:
                            print(
                                f"  Using lookup grid: acm={avg_acm:.4f}, zr={avg_zr:.4f} → sf={sf}, bs={bs}"
                            )

                        return sf, bs
    except Exception as e:
        if not config.output.quiet:
            print(f"  Lookup grid unavailable ({e}), falling back to sample-based selection...")

    # --- Fallback: sample-based testing ---

    valid_data = image[image != nodata_value] if nodata_value is not None else image.flatten()
    image_std = np.std(valid_data) if len(valid_data) > 0 else 0

    sf_cfg = getattr(config.compression, "uint8_scaling_factor", 1)
    sf_candidates = list(sf_cfg) if isinstance(sf_cfg, (list, tuple)) else [sf_cfg]

    allowed_bs = list(
        getattr(config.compression, "allowed_block_sizes", [config.compression.block_size])
    )
    current_bs = int(getattr(config.compression, "block_size", 8))
    if current_bs in allowed_bs:
        bs_candidates = [current_bs]
    else:
        bs_candidates = [int(allowed_bs[0])] if allowed_bs else [current_bs]

    test_configs = [(int(sf), int(bs)) for sf in sf_candidates for bs in bs_candidates]
    if not config.output.quiet:
        print(f"    Image std={image_std:.2f} - testing uint8 configs: {test_configs}")

    results = {}
    for test_sf, test_bs in test_configs:
        # Prepare image in uint8 (already is) and sample blocks
        padded, _ = pad_image(image, test_bs)
        max_sample = config.compression.auto_select_sample_size
        H, W = padded.shape
        grid = padded.reshape(H // test_bs, test_bs, W // test_bs, test_bs).transpose(0, 2, 1, 3)
        grid = grid.reshape(-1, test_bs, test_bs)
        total = grid.shape[0]
        step = max(1, total // min(max_sample, total))
        idx = np.arange(0, total, step)[:max_sample]

        # Evaluate this (scaling factor, block size) pair; the original config values are restored below.
        orig_sf = config.compression.uint8_scaling_factor
        orig_override = getattr(config, "_scaling_factor_override", None)
        orig_decimal = config.compression.decimal_places
        config.compression.uint8_scaling_factor = [test_sf]
        config._scaling_factor_override = test_sf
        if test_sf > 1:
            config.compression.decimal_places = int(np.ceil(np.log10(test_sf)))
        else:
            config.compression.decimal_places = 0

        # uint8 blocks are sampled as process_block does them (no src dtype): float64 values and DCT
        compressed_list, errors = _sample_results_f32(
            grid, idx, test_bs, config, nodata_value, np.float64)
        total_error = 0.0
        for error in errors:
            total_error += error
        max_error = max(errors) if errors else 0.0

        # Restore original sf
        config.compression.uint8_scaling_factor = orig_sf
        config.compression.decimal_places = orig_decimal
        if orig_override is None:
            if hasattr(config, "_scaling_factor_override"):
                delattr(config, "_scaling_factor_override")
        else:
            config._scaling_factor_override = orig_override

        sample_size = len(idx)
        sample_msgpack_size = len(msgpack.packb(compressed_list, use_bin_type=True))
        estimated_size = (sample_msgpack_size / sample_size) * total
        avg_error = total_error / sample_size
        valid = max_error <= float(
            getattr(config.compression, "uint8_accuracy", config.compression.accuracy)
        )

        results[(test_sf, test_bs)] = {
            "estimated_size": estimated_size,
            "avg_error": avg_error,
            "max_error": max_error,
            "valid": valid,
        }
        if not config.output.quiet:
            valid_txt = "valid" if valid else "invalid"
            print(
                f"    sf={test_sf}, N={test_bs}: blocks={len(blocks)}, "
                f"est_size={estimated_size/1024:.1f}KB, avg_error={avg_error:.6f}, "
                f"max_error={max_error:.6f}, {valid_txt}"
            )

    # Select the smallest valid (sf, bs) combination; if none are valid, fall back to smallest estimate.
    valid_results = {k: v for k, v in results.items() if v["valid"]}
    candidates = valid_results if valid_results else results
    best_config = min(candidates, key=lambda k: candidates[k]["estimated_size"])
    best_sf, best_bs = best_config

    if not config.output.quiet:
        print(f"  Selected uint8 parameters: sf={best_sf}, block_size={best_bs}")

    return best_sf, best_bs


# ---------------------------------------------------------------------------
# Compressed-blocks serialization (shared by DCT paths)
# ---------------------------------------------------------------------------


def serialize_compressed_blocks(
    compressed_blocks: List,
) -> Tuple[List[int], np.ndarray, List]:
    """Convert compressed_blocks list → (lengths, coeffs_np, masks_data).

    Sentinel values:
        0  → ALL_ZEROS  → length 0
        1  → ALL_NODATA → length -1
        np.ndarray / list of ints → normal block
        list of lists/arrays → mask block (length -2)
    """
    lengths = []
    coeff_arrays = []
    masks_data = []

    for i, b in enumerate(compressed_blocks):
        if isinstance(b, np.ndarray):
            lengths.append(len(b))
            coeff_arrays.append(b.astype(np.int64, copy=False))
        elif isinstance(b, list):
            if b and isinstance(b[0], (list, np.ndarray)):
                lengths.append(-2)
                masks_data.append([i, b])
            else:
                lengths.append(len(b))
                coeff_arrays.append(np.asarray(b, dtype=np.int64))
        elif b == 0:
            lengths.append(0)
        elif b == 1:
            lengths.append(-1)
        else:
            lengths.append(0)

    coeffs_np = np.concatenate(coeff_arrays) if coeff_arrays else np.empty(0, dtype=np.int64)
    return lengths, coeffs_np, masks_data


def serialize_compressed_blocks_rgb(
    compressed_blocks: List,
) -> Tuple[List[int], Dict[str, Any], List]:
    """Convert RGB compressed blocks with per-block YCbCr multipliers
    → (lengths, coeffs_dict, masks_data).

    For per-block YCbCr mode (cm=6), each block has metadata:
        {
            'Y': {'L': L_y, 'coeffs': coeffs_y, 'multiplier': mult_y},
            'Cb': {'L': L_cb, 'coeffs': coeffs_cb, 'multiplier': mult_cb},
            'Cr': {'L': L_cr, 'coeffs': coeffs_cr, 'multiplier': mult_cr},
            'sf': global_sf,
            'masks': masks (if any)
        }

    Returns:
        lengths: dict with per-channel L values, e.g.
                 {'Y': [...], 'Cb': [...], 'Cr': [...]}
        coeffs_dict: dict with flattened coefficients and per-block
                     multipliers
        masks_data: list of mask data (if any)
    """
    lengths: Dict[str, List[int]] = {"Y": [], "Cb": [], "Cr": []}
    coeffs_dict: Dict[str, Any] = {
        "Y": {"coeffs": [], "multipliers": []},
        "Cb": {"coeffs": [], "multipliers": []},
        "Cr": {"coeffs": [], "multipliers": []},
        "sf": None,
        "fallback": [],
    }
    masks_data: List[Any] = []

    for idx, metadata in compressed_blocks:
        # Store coefficients, multipliers, and lengths for each channel
        for ch in ["Y", "Cb", "Cr"]:
            L_ch = metadata[ch]["L"]
            lengths[ch].append(L_ch)

            coeffs = metadata[ch]["coeffs"]
            # Ensure numpy arrays are converted to plain Python lists
            if isinstance(coeffs, np.ndarray):
                coeffs = coeffs.tolist()
            coeffs_dict[ch]["coeffs"].extend(coeffs)
            coeffs_dict[ch]["multipliers"].append(float(metadata[ch]["multiplier"]))

        # Store global scaling factor (same for all blocks)
        if coeffs_dict["sf"] is None:
            coeffs_dict["sf"] = metadata["sf"]

        # Store fallback flag
        coeffs_dict["fallback"].append(metadata.get("fallback", False))

        # Store masks if present
        if metadata.get("masks"):
            masks_data.append([idx, metadata["masks"]])

    return lengths, coeffs_dict, masks_data


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------


def _find_7z_cmd() -> Optional[str]:
    """Return first available 7z executable name, or None."""
    return next((c for c in ["7zz", "7zip", "7z"] if shutil.which(c)), None)


def pack_to_7z(
    output_path: str,
    *source_files: str,
    store: bool = False,
    method: str = "LZMA2",
    lzma2_block_size: str = "",
) -> None:
    """Pack source_files into a .7z archive.

    store=True  → no compression (-mx=0), used for JPEG strips
    store=False → max compression (-mx=9) with given method
    lzma2_block_size: LZMA2 block size ("1m", "2m", ...; "" = 7-Zip default). 7-Zip compresses the
    blocks in parallel, so smaller blocks are much faster with nearly the same size.
    """
    if os.path.exists(output_path):
        os.remove(output_path)

    cpu_count = os.cpu_count() or 4
    cmd = _find_7z_cmd()

    cwd = None
    if source_files:
        first_file = source_files[0]
        cwd = os.path.dirname(os.path.abspath(first_file))
        source_files_rel = []
        for f in source_files:
            abs_f = os.path.abspath(f)
            if os.path.dirname(abs_f) == cwd:
                source_files_rel.append(os.path.basename(abs_f))
            else:
                source_files_rel.append(abs_f)
    else:
        source_files_rel = list(source_files)

    abs_output_path = os.path.abspath(output_path)

    if cmd:
        if store:
            args = [cmd, "a", "-mx=0", f"-mmt={cpu_count}", abs_output_path, *source_files_rel]
        else:
            if method not in ("LZMA2", "PPMD", "BZIP2", "DEFLATE"):
                method = "LZMA2"
            m0 = method
            block = (lzma2_block_size or "").strip().lower()
            if method == "LZMA2" and block:
                if not re.fullmatch(r"\d+[kmg]?", block):
                    raise ValueError(f"lzma2_block_size: expected e.g. 1m, 2m, 4m, got {lzma2_block_size!r}")
                m0 = f"LZMA2:c={block}"
            args = [
                cmd,
                "a",
                "-mx=9",
                f"-m0={m0}",
                f"-mmt={cpu_count}",
                abs_output_path,
                *source_files_rel,
            ]
        subprocess.run(args, cwd=cwd, capture_output=True, check=True)
    else:
        with py7zr.SevenZipFile(abs_output_path, "w") as archive:
            for src in source_files:
                archive.write(src, os.path.basename(src))


def write_msgpack(path: str, data: Any) -> None:
    """Serialize data to msgpack file."""
    with open(path, "wb") as f:
        f.write(msgpack.packb(numpy_to_python(data), use_bin_type=True))


def remove_if_exists(path: str) -> None:
    """Remove file silently if it exists."""
    try:
        os.remove(path)
    except OSError:
        pass


__all__ = [
    "RASTER_EXTENSIONS",
    "_STREAMING_THRESHOLD",
    "pad_image",
    "extract_blocks",
    "auto_select_block_size",
    "serialize_compressed_blocks",
    "pack_to_7z",
    "write_msgpack",
    "remove_if_exists",
]
