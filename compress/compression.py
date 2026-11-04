"""
FIDWAC v2 - Main compression logic (orchestrator)
==================================================
This module is the single public entry point for compression.
Actual encoding logic lives in sub-modules:

  compression_utils.py   — shared helpers (padding, serialization, archive I/O)
  compression_jpeg.py    — JPEG turbojpeg path (uint8 RGB, cm=1)
  compression_lossless.py— lossless mode (cm=4): PNG for 3-band uint8, GeoTIFF otherwise
  compression_dct.py     — DCT path (float32 elevation + uint8 accuracy, cm=3/5/6)

Compression mode codes (cm):
  cm=0  Interleaved VLQ stream without DC prediction (kept for backward compatibility)
  cm=1  TurboJPEG (SIMD) — quality-based JPEG strips via libjpeg-turbo for uint8 RGB.
        No formal max-error guarantee; lossiness depends on JPEG quality parameter.
        Archive format: rgb.bin + meta.msgpack (mode='jpeg_strips'), not the standard
        msgpack 'cm' field.
  cm=3  DCT with DC Median3 Predictor + separate DC/AC streams — for float32/int16
        elevation data (e.g. DEM, LiDAR height). Binary search guarantees max error ≤ε.
  cm=4  Lossless (--lossless, any data type): the raster unchanged in PNG + world file (3-band
        uint8) or GeoTIFF DEFLATE + predictor, packed into .7z. max error = 0. Older archives
        (msgpack, PNG Sub filter + zlib per band) are still decoded.
  cm=5  Standard DCT accuracy mode (uint8/int8 single-channel) — binary search with
        centering (-128), adaptive scaling factor. Guarantees max error ≤ε in grayscale
        (0–255). No color-space transform. Used for LiDAR intensity, Hillshade, etc.
  cm=6  Per-block YCbCr with cascading multipliers (uint8 RGB accuracy mode) —
        hybrid per-block DCT in YCbCr space with per-block scaling-factor cascade.
        Validates final RGB error ≤ε. Used for multi-channel RGB with accuracy mode.
"""

import os
import time
from typing import Optional, List
from copy import deepcopy

import numpy as np
import rasterio
import msgpack

from engine.utils import numpy_to_python
from config import Config
from compress.output_crs import crs_label, resolve_output_crs

from .compression_utils import (
    RASTER_EXTENSIONS,
    _STREAMING_THRESHOLD,
    pack_to_7z,
    remove_if_exists,
)
from .compression_jpeg import compress_jpeg_path
from .compression_lossless import compress_lossless
from . import compression_dct
from .compression_dct import (
    compress_channel_dct,
    compress_channel_dct_streaming,
    compress_channel_dct_strips,
)



def find_raster_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all supported raster files in a directory (recursively by default)."""
    files = []
    if recursive:
        for root, _dirs, fnames in os.walk(directory):
            for fname in fnames:
                if os.path.splitext(fname)[1].lower() in RASTER_EXTENSIONS:
                    files.append(os.path.join(root, fname))
    else:
        import glob as _glob

        for ext in RASTER_EXTENSIONS:
            files.extend(_glob.glob(os.path.join(directory, f"*{ext}")))
            files.extend(_glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    return sorted(set(files))



def _select_threading_layer(config: Config) -> None:
    """Use numba's fork-safe workqueue layer for the heuristic backend, then warm up the kernels.

    The default GNU OpenMP layer aborts forked pool workers that run a parallel kernel after the
    parent has run one ("fork() called from a process already using GNU OpenMP"), and the pool then
    waits forever. The advanced heuristic does exactly that. The layer can only be chosen before the
    first parallel launch, so the warm-up that follows fixes it; the binary backend keeps numba's default.
    """
    if str(config.model.backend).lower() == "heuristic":
        import numba

        numba.config.THREADING_LAYER = "workqueue"
    from predictor.predictor import warm_up_parallel_kernels

    warm_up_parallel_kernels()


def compress_image(
    file_path: str,
    config: Optional[Config] = None,
    num_processes: Optional[int] = None,
    output_dir: Optional[str] = None,
    skip_archive: bool = False,
) -> str:
    """Compress raster image to .7z archive.

    Routing:
      - uint8 RGB (3 bands, quality mode)  → JPEG turbojpeg strips (cm=1)
      - uint8 RGB (3+ bands, accuracy mode,
        ycbcr_per_block enabled)           → per-block YCbCr DCT (cm=6)
      - any data, --lossless               → PNG (3-band uint8) / GeoTIFF in .7z (cm=4)
      - uint8 any bands, accuracy mode     → DCT binary search (cm=5)
      - float32/64 elevation               → DCT binary search (cm=3)
      - no cm field (older files)          → interleaved VLQ (cm=0)

    For large rasters (>256 MB/channel) uses streaming via rasterio Window.
    """
    if config is None:
        from config import load_config

        config = load_config()
    if num_processes is None:
        num_processes = config.num_processes_int
    if output_dir is None:
        output_dir = config.results_dir

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    _select_threading_layer(config)

    if not config.output.quiet:
        print(f"Loading: {file_path}")

    src, meta = _open_raster(file_path, config)
    config._source_has_nodata = meta["has_nodata"]
    H, W = meta["H"], meta["W"]

    if config.compression.lossless:
        result = compress_lossless(
            src, file_path, meta["raster_crs"], meta["transform"], config, output_dir, start_time
        )
        src.close()
        return result

    u8_accuracy_mode = getattr(config.compression, "uint8_accuracy_mode", False)
    if (
        meta["all_uint8"]
        and meta["num_channels"] == 3
        and not u8_accuracy_mode
        and not config.compression.lossless
    ):
        result = compress_jpeg_path(
            src,
            file_path,
            H,
            W,
            meta["data_band_indices"],
            meta["has_nodata"],
            meta["nodata_value"],
            meta["file_crs"],
            meta["transform"],
            config,
            output_dir,
            start_time,
        )
        src.close()
        return result

    # --no-uint8-accuracy-mode selects JPEG (cm=1), which exists only for 3-band uint8 RGB. Integer
    # rasters of any other shape have no quality-mode path, so they are refused instead of guessed.
    if not u8_accuracy_mode and any(np.dtype(d).kind in "iu" for d in meta["band_dtypes"]):
        src.close()
        raise ValueError(
            f"--no-uint8-accuracy-mode selects JPEG (cm=1), which is only for 3-band uint8 RGB; "
            f"this raster has {meta['num_channels']} band(s) of {', '.join(sorted(set(meta['band_dtypes'])))}. "
            "Use --uint8-accuracy (DCT with an error bound) or --lossless."
        )

    result = _compress_per_channel(
        src, file_path, meta, config, num_processes, output_dir, start_time, skip_archive
    )
    src.close()
    return result



def _open_raster(file_path: str, config: Config):
    """Open raster and collect metadata dict."""
    from rasterio.enums import ColorInterp

    src = rasterio.open(file_path)
    H, W = src.height, src.width
    all_ci = src.colorinterp
    data_band_indices = [i + 1 for i, ci in enumerate(all_ci) if ci != ColorInterp.alpha]
    alpha_band_indices = [i + 1 for i, ci in enumerate(all_ci) if ci == ColorInterp.alpha]

    if alpha_band_indices and not config.output.quiet:
        print(f"  Skipping alpha channel(s): band(s) {alpha_band_indices}")

    num_channels = len(data_band_indices)
    band_dtypes = tuple(src.dtypes[i - 1] for i in data_band_indices)
    all_uint8 = all(dt == "uint8" for dt in band_dtypes)

    nodata_value = src.nodata
    has_nodata = nodata_value is not None
    if nodata_value is None:
        nodata_value = -9999.0  # sentinel — won't match any real pixel

    file_crs = src.crs
    transform = src.transform
    raster_crs = resolve_output_crs(file_crs, config.compression.crs)

    if not config.output.quiet and file_crs is None and not transform.is_identity:
        out_crs = "none" if raster_crs is None else raster_crs.to_string()
        print(f"  WARNING: no CRS in file — transform from world file, output CRS: {out_crs}")

    channel_bytes = H * W * 4
    use_streaming = channel_bytes > _STREAMING_THRESHOLD

    if not config.output.quiet:
        alpha_info = f" (+ {len(alpha_band_indices)} alpha)" if alpha_band_indices else ""
        print(
            f"  Size: {W}x{H}, Channels: {num_channels}{alpha_info}, "
            f"NoData: {nodata_value}, Mode: {'streaming' if use_streaming else 'in-memory'}"
        )

    return src, {
        "H": H,
        "W": W,
        "data_band_indices": data_band_indices,
        "num_channels": num_channels,
        "band_dtypes": band_dtypes,
        "all_uint8": all_uint8,
        "nodata_value": nodata_value,
        "has_nodata": has_nodata,
        "file_crs": file_crs,
        "raster_crs": raster_crs,
        "transform": transform,
        "use_streaming": use_streaming,
    }



def _compress_per_channel_rgb(
    src, file_path, meta, config, num_processes, output_dir, start_time, skip_archive
):
    """Hybrid compression: RGB channels with per-block YCbCr (cm=6),
    remaining channels with standard (cm=5)."""
    from .compression_dct import compress_channel_rgb

    H, W = meta["H"], meta["W"]
    data_band_indices = meta["data_band_indices"]
    num_channels = meta["num_channels"]
    band_dtypes = meta["band_dtypes"]
    all_uint8 = meta["all_uint8"]
    nodata_value = meta["nodata_value"]
    use_streaming = meta["use_streaming"]

    block_size = config.compression.block_size
    u8_accuracy_mode = getattr(config.compression, "uint8_accuracy_mode", False)
    rgb_indices = getattr(config.compression, "rgb_channel_indices", [])

    rgb_indices_0based = [int(i) - 1 for i in rgb_indices]

    rgb_channels = [i for i in rgb_indices_0based if i < num_channels]
    non_rgb_channels = [i for i in range(num_channels) if i not in rgb_channels]

    if not config.output.quiet:
        print("  Using hybrid compression mode")
        print(f"  RGB channels (per-block YCbCr): {[i+1 for i in rgb_channels]}")
        print(f"  Other channels (standard): {[i+1 for i in non_rgb_channels]}")

    from .compression_dct import build_ycbcr_images

    if len(rgb_channels) == 3:
        # The YCbCr images only exist for the three RGB channels, which compress_channel_rgb
        # (cm=6) handles itself; the loop over the other channels never looks them up.
        ycbcr_images, ycbcr_acc = {}, {}
    else:
        ycbcr_images, ycbcr_acc, _ = build_ycbcr_images(
            src, config, num_channels, all_uint8, data_band_indices
        )

    channels_data = [None] * num_channels
    max_error_global = 0.0
    all_valid = True

    if len(rgb_channels) == 3:
        r_idx, g_idx, b_idx = rgb_channels
        if not config.output.quiet:
            print("  Compressing RGB channels with per-block YCbCr...")

        # Prepare custom config with adaptive scaling-factor fallback.
        # cm=6 validates final RGB error, so sf=1 can be too coarse for strict accuracy=2.
        u8_sf_cfg = getattr(config.compression, "uint8_scaling_factor", 1)
        if isinstance(u8_sf_cfg, list):
            sfs_to_try = list(u8_sf_cfg)
        else:
            sfs_to_try = [u8_sf_cfg]

        last_result = None
        u8_cfg = None
        for sf_idx, try_sf in enumerate(sfs_to_try):
            u8_cfg = deepcopy(config)
            u8_cfg._scaling_factor_override = try_sf
            if try_sf > 1:
                u8_cfg.compression.decimal_places = int(np.ceil(np.log10(try_sf)))
            else:
                u8_cfg.compression.decimal_places = 0

            if not config.output.quiet:
                print(f"  Trying cm=6 RGB scaling factor: sf={try_sf}")

            last_result = compress_channel_rgb(
                src, r_idx, g_idx, b_idx, block_size, nodata_value, u8_cfg, num_processes
            )
            lengths, coeffs_dict, masks_data, max_err, validity, block_size, padded_shape = (
                last_result
            )
            if validity:
                break
            if not config.output.quiet and sf_idx < len(sfs_to_try) - 1:
                print(
                    f"  ⚠ cm=6 sf={try_sf} did not meet RGB accuracy "
                    f"(max_err={max_err:.1f}), retrying with sf={sfs_to_try[sf_idx + 1]}"
                )

        if last_result is None or u8_cfg is None:
            raise RuntimeError("cm=6 compression failed: no scaling factor candidates")

        # Build channel data for RGB (single "channel" containing all 3)
        ch_data = {
            "l": lengths,
            "cm": 6,  # per-block YCbCr
            "m": masks_data,
            "bs": block_size,
            "sf": coeffs_dict["sf"],
        }

        ch_data = _encode_dct_channel_rgb(ch_data, coeffs_dict, block_size, padded_shape, u8_cfg)
        if coeffs_dict.get("nd") is not None:
            ch_data["nd"] = coeffs_dict["nd"]  # per-band NoData RLE masks (R, G, B)
            flips = _rgb_nodata_flips(
                ch_data, block_size, padded_shape, src, (r_idx, g_idx, b_idx), nodata_value
            )
            if flips is not None:
                ch_data["nu"] = flips

        # Store in first RGB slot; other RGB slots get empty marker
        channels_data[rgb_channels[0]] = ch_data
        for i in rgb_channels[1:]:
            channels_data[i] = {"cm": 6}  # marker: part of RGB group

        if max_err > max_error_global:
            max_error_global = max_err
        if not validity:
            all_valid = False

    for ch_idx in non_rgb_channels:
        band_idx = data_band_indices[ch_idx]
        if not config.output.quiet:
            print(f"  Channel {ch_idx + 1}/{num_channels} (standard):")

        ch_dtype = band_dtypes[ch_idx]
        is_8bit_ch = ch_dtype in ("uint8", "uint16", "int8", "int16")

        ch_data, coeffs_np, block_size_ch, padded_shape_ch, max_err, validity = (
            _compress_one_channel(
                src,
                band_idx,
                ch_dtype,
                is_8bit_ch,
                block_size,
                nodata_value,
                config,
                num_processes,
                use_streaming,
                u8_accuracy_mode,
                ycbcr_images,
                ycbcr_acc,
            )
        )

        if coeffs_np is not None:
            ch_data = _encode_dct_channel(ch_data, coeffs_np, block_size_ch, padded_shape_ch)

        channels_data[ch_idx] = ch_data

        if max_err > max_error_global:
            max_error_global = max_err
        if not validity:
            all_valid = False

    dcv_compress = {
        "v": 3,
        "ch": num_channels,
        "n": block_size,
        "s": [H, W],
        "p": [H, W],
        "f": config.scaling_factor,
        "d": nodata_value,
        "channels": channels_data,
    }

    if len(rgb_channels) == 3:
        dcv_compress["ycbcr_rgb"] = rgb_channels

    if not config.output.quiet:
        print(f"  Hybrid compression: max_error={max_error_global:.2f}, validity={all_valid}")

    return _finalize_dct_archive(
        file_path,
        meta,
        config,
        output_dir,
        start_time,
        skip_archive,
        channels_data,
        [block_size] * len(channels_data),
        rgb_channels if len(rgb_channels) == 3 else None,
        max_error_global,
        all_valid,
        all_uint8,
        u8_accuracy_mode,
    )



def _compress_per_channel(
    src, file_path, meta, config, num_processes, output_dir, start_time, skip_archive
):
    data_band_indices = meta["data_band_indices"]
    num_channels = meta["num_channels"]
    band_dtypes = meta["band_dtypes"]
    all_uint8 = meta["all_uint8"]
    nodata_value = meta["nodata_value"]
    use_streaming = meta["use_streaming"]

    block_size = config.compression.block_size
    u8_accuracy_mode = getattr(config.compression, "uint8_accuracy_mode", False)
    ycbcr_per_block_mode = getattr(config.compression, "ycbcr_per_block_mode", False)

    # Check if we should use per-block RGB path (cm=6)
    # Conditions: uint8 accuracy mode + 3+ channels + RGB indices defined + per-block mode enabled
    rgb_indices = getattr(config.compression, "rgb_channel_indices", [])
    if (
        u8_accuracy_mode
        and not config.compression.lossless
        and len(rgb_indices) == 3
        and ycbcr_per_block_mode
        and all_uint8
        and num_channels >= 3
    ):
        return _compress_per_channel_rgb(
            src, file_path, meta, config, num_processes, output_dir, start_time, skip_archive
        )

    max_error_global = 0.0
    all_valid = True
    channel_block_sizes = []
    ch_data_list = []

    for ch_local, band_idx in enumerate(data_band_indices):
        if not config.output.quiet:
            print(f"  Channel {ch_local + 1}/{num_channels} (standard):")

        ch_dtype = band_dtypes[ch_local]
        is_8bit_ch = ch_dtype in ("uint8", "uint16", "int8", "int16")

        ch_data, coeffs_np, block_size_ch, padded_shape_ch, max_err, validity = (
            _compress_one_channel(
                src,
                band_idx,
                ch_dtype,
                is_8bit_ch,
                block_size,
                nodata_value,
                config,
                num_processes,
                use_streaming,
                u8_accuracy_mode,
                None,
                None,
            )
        )

        if coeffs_np is not None:
            ch_data = _encode_dct_channel(ch_data, coeffs_np, block_size_ch, padded_shape_ch)

        if max_err > max_error_global:
            max_error_global = max_err
        if not validity:
            all_valid = False

        channel_block_sizes.append(block_size_ch)
        ch_data_list.append(ch_data)

    return _finalize_dct_archive(
        file_path,
        meta,
        config,
        output_dir,
        start_time,
        skip_archive,
        ch_data_list,
        channel_block_sizes,
        None,
        max_error_global,
        all_valid,
        all_uint8,
        u8_accuracy_mode,
    )


def _compress_one_channel(
    src,
    band_idx,
    ch_dtype,
    is_8bit_ch,
    block_size,
    nodata_value,
    config,
    num_processes,
    use_streaming,
    u8_accuracy_mode,
    ycbcr_images,
    ycbcr_acc,
):
    """Dispatch to the right compression path for one band.

    Returns (ch_data_partial, coeffs_np_or_None, block_size, max_err, validity).
    ch_data_partial lacks 'c' key when coeffs_np is not None (added later by caller).
    """
    ch_idx_0 = band_idx - 1
    coeffs_np = None

    if is_8bit_ch and u8_accuracy_mode:
        lengths, coeffs_np, masks_data, max_err, validity, block_size, padded_shape, ch_sf = (
            _compress_uint8_accuracy(
                src,
                band_idx,
                ch_dtype,
                ch_idx_0,
                block_size,
                nodata_value,
                config,
                num_processes,
                ycbcr_images,
                ycbcr_acc,
            )
        )
        ch_data = {"l": lengths, "cm": 5, "m": masks_data, "bs": block_size}
        if ch_dtype != "uint8":
            ch_data["dt"] = ch_dtype  # integer range for the decoder (uint8 when absent)
        if ch_sf is not None:
            ch_data["sf"] = ch_sf

    elif use_streaming:
        base_dec = config.compression.decimal_places
        dec = base_dec
        for i, dec in enumerate([base_dec, base_dec + 1, base_dec + 2, base_dec + 3]):
            ch_cfg = deepcopy(config)
            ch_cfg.compression.decimal_places = dec
            lengths, coeffs_np, masks_data, max_err, validity, block_size, padded_shape = (
                compress_channel_dct_streaming(
                    src, band_idx, block_size, nodata_value, ch_cfg, num_processes, src_dtype=ch_dtype
                )
            )
            if validity:
                break
            if not config.output.quiet:
                print(f"  dec={dec} does not meet the accuracy (max_err={max_err:.4g}), retrying with dec={dec + 1}")
        ch_data = {"l": lengths, "cm": 3, "m": masks_data, "bs": block_size}
        if dec != base_dec:
            ch_data["sf"] = 10 ** dec

    else:
        image_ch = src.read(band_idx).astype(np.float32)
        base_dec = config.compression.decimal_places
        dec = base_dec
        for i, dec in enumerate([base_dec, base_dec + 1, base_dec + 2, base_dec + 3]):
            ch_cfg = deepcopy(config)
            ch_cfg.compression.decimal_places = dec
            lengths, coeffs_np, masks_data, max_err, validity, block_size, padded_shape = (
                compress_channel_dct(
                    image_ch, block_size, nodata_value, ch_cfg, num_processes, src_dtype=ch_dtype
                )
            )
            if validity:
                break
            if not config.output.quiet:
                print(f"  dec={dec} does not meet the accuracy (max_err={max_err:.4g}), retrying with dec={dec + 1}")
        del image_ch
        ch_data = {"l": lengths, "cm": 3, "m": masks_data, "bs": block_size}
        if dec != base_dec:
            ch_data["sf"] = 10 ** dec

    return ch_data, coeffs_np, block_size, padded_shape, max_err, validity


def _compress_uint8_accuracy(
    src,
    band_idx,
    ch_dtype,
    ch_idx_0,
    block_size,
    nodata_value,
    config,
    num_processes,
    ycbcr_images,
    ycbcr_acc,
):
    """uint8 accuracy-controlled path (cm=5) with adaptive scaling factor fallback."""
    u8_accuracy = float(getattr(config.compression, "uint8_accuracy", 1))
    u8_sf_cfg = getattr(config.compression, "uint8_scaling_factor", 1)

    # A band above DCT_STRIP_PIXELS is compressed strip by strip, so it never sits in memory whole.
    # (The globally decorrelated YCbCr images are already in memory.)
    by_strips = False
    if ycbcr_images is not None and ch_idx_0 in ycbcr_images:
        image_ch = ycbcr_images[ch_idx_0]
        u8_accuracy = ycbcr_acc[ch_idx_0]
    else:
        by_strips = src.height * src.width > compression_dct.DCT_STRIP_PIXELS
        image_ch = None if by_strips else src.read(band_idx).astype(np.float32)

    if isinstance(u8_sf_cfg, list):
        sfs_to_try = list(u8_sf_cfg)
    else:
        sfs_to_try = []
        for exp in range(0, 4):  # try 1, 10, 100, 1000
            try_sf = 10**exp
            if try_sf >= u8_sf_cfg:
                sfs_to_try.append(try_sf)
        if not sfs_to_try or sfs_to_try[0] != u8_sf_cfg:
            sfs_to_try.insert(0, u8_sf_cfg)

    ch_sf = None
    for i, try_sf in enumerate(sfs_to_try):
        u8_cfg = deepcopy(config)
        u8_cfg.compression.accuracy = u8_accuracy

        u8_cfg._scaling_factor_override = try_sf
        if try_sf > 1:
            u8_cfg.compression.decimal_places = int(np.ceil(np.log10(try_sf)))
        else:
            u8_cfg.compression.decimal_places = 0

        if by_strips:
            lengths, coeffs_np, masks_data, max_err, validity, block_size, padded_shape = (
                compress_channel_dct_strips(
                    src, band_idx, block_size, nodata_value, u8_cfg, num_processes, src_dtype=ch_dtype
                )
            )
        else:
            lengths, coeffs_np, masks_data, max_err, validity, block_size, padded_shape = (
                compress_channel_dct(
                    image_ch, block_size, nodata_value, u8_cfg, num_processes, src_dtype=ch_dtype
                )
            )
        del u8_cfg

        ch_sf = try_sf
        if validity:
            break

        if not config.output.quiet and i < len(sfs_to_try) - 1:
            next_sf = sfs_to_try[i + 1]
            print(
                f"  ⚠ sf={try_sf} did not meet accuracy (max_err={max_err:.1f}), "
                f"retrying with sf={next_sf}"
            )

    del image_ch
    return lengths, coeffs_np, masks_data, max_err, validity, block_size, padded_shape, ch_sf


def _rgb_nodata_flips(ch_data, block_size, padded_shape, src, rgb_idx, nodata_value):
    """cm=6 with NoData inside 1..254: per-band RLE of valid pixels that decode to NoData and lie
    below it, so the decoder steps them down instead of up (see _apply_rgb_nodata). The archive
    is decoded with the decoder's own function, strip by strip, so the masks match its output exactly."""
    if not (0 < nodata_value < 255):
        return None
    from rasterio.windows import Window

    from engine.utils import RleBuilder, decode_mask_rle_rows
    from .decompression import _iter_rgb_strips

    H, W = src.height, src.width
    builders = [RleBuilder() for _ in range(3)]
    any_flip = False
    for y0, *planes in _iter_rgb_strips(ch_data, block_size, tuple(padded_shape), (H, W)):
        h = planes[0].shape[0]
        for plane, band_idx, rle, builder in zip(planes, rgb_idx, ch_data["nd"], builders):
            orig = src.read(band_idx + 1, window=Window(0, y0, W, h))
            flip = (
                (plane == nodata_value)
                & ~decode_mask_rle_rows(rle, W, y0, h)
                & (orig < nodata_value)
            )
            any_flip |= bool(np.any(flip))
            builder.add(flip)
    return [builder.result() for builder in builders] if any_flip else None


def _encode_dct_channel_rgb(
    ch_data: dict, coeffs_dict: dict, block_size: int, padded_shape: tuple, config=None
) -> dict:
    """Encode DCT coefficients for per-block RGB mode (cm=6).

    coeffs_dict contains:
        {
            'Y': {'coeffs': [...], 'multipliers': [...]},
            'Cb': {'coeffs': [...], 'multipliers': [...]},
            'Cr': {'coeffs': [...], 'multipliers': [...]},
            'sf': global_sf,
            'fallback': [...]
        }
    """
    from core.codec import dc_median_ac_encode

    if config is None:
        from config import load_config

        config = load_config()

    lengths = ch_data["l"]  # dict: {'Y': [...], 'Cb': [...], 'Cr': [...]}
    padded_W = padded_shape[1] if padded_shape and len(padded_shape) >= 2 else 0
    blocks_per_row = max(1, padded_W // block_size) if block_size > 0 and padded_W > 0 else 1

    encoded_channels = {}
    for ch in ["Y", "Cb", "Cr"]:
        # coeffs_dict[ch]['coeffs'] is already flattened from serialize_compressed_blocks_rgb
        coeffs = np.asarray(coeffs_dict[ch]["coeffs"], dtype=np.int64)
        ch_lengths = lengths[ch]

        if len(coeffs) > 0:
            dc_bytes, ac_bytes = dc_median_ac_encode(coeffs, ch_lengths, blocks_per_row)
            encoded_channels[ch] = [dc_bytes, ac_bytes]
        else:
            encoded_channels[ch] = b""

    # Store encoded data as flat list. Per-block multipliers and fallback flags are not stored:
    # the decoder does not use them (the coefficients are already quantized).
    # [Y_dc, Y_ac, Cb_dc, Cb_ac, Cr_dc, Cr_ac]
    ch_data["c"] = [
        encoded_channels["Y"][0] if isinstance(encoded_channels["Y"], list) else b"",
        encoded_channels["Y"][1] if isinstance(encoded_channels["Y"], list) else b"",
        encoded_channels["Cb"][0] if isinstance(encoded_channels["Cb"], list) else b"",
        encoded_channels["Cb"][1] if isinstance(encoded_channels["Cb"], list) else b"",
        encoded_channels["Cr"][0] if isinstance(encoded_channels["Cr"], list) else b"",
        encoded_channels["Cr"][1] if isinstance(encoded_channels["Cr"], list) else b"",
    ]

    # Flatten lengths: [Y_lengths, Cb_lengths, Cr_lengths]
    ch_data["l"] = [lengths["Y"], lengths["Cb"], lengths["Cr"]]
    return ch_data


def _encode_dct_channel(
    ch_data: dict, coeffs_np: np.ndarray, block_size: int, padded_shape: tuple
) -> dict:
    """Encode DCT coefficients using DC-median predictor + separate AC stream."""
    from core.codec import dc_median_ac_encode

    lengths = ch_data["l"]

    padded_W = padded_shape[1] if padded_shape and len(padded_shape) >= 2 else 0
    blocks_per_row = max(1, padded_W // block_size) if block_size > 0 and padded_W > 0 else 1

    if len(coeffs_np) > 0:
        dc_bytes, ac_bytes = dc_median_ac_encode(coeffs_np, lengths, blocks_per_row)
        ch_data["c"] = [dc_bytes, ac_bytes]
    else:
        ch_data["c"] = b""

    del coeffs_np
    return ch_data



def _finalize_dct_archive(
    file_path,
    meta,
    config,
    output_dir,
    start_time,
    skip_archive,
    ch_data_list,
    channel_block_sizes,
    ycbcr_rgb_0,
    max_error_global,
    all_valid,
    all_uint8,
    u8_accuracy_mode,
):
    H, W = meta["H"], meta["W"]
    num_channels = meta["num_channels"]
    raster_crs = meta["raster_crs"]
    transform = meta["transform"]
    nodata_value = meta["nodata_value"]
    has_nodata = meta["has_nodata"]

    block_size = channel_block_sizes[0] if channel_block_sizes else config.compression.block_size

    dcv_compress = {
        "v": 3,
        "ch": num_channels,
        "n": block_size,
        "s": [H, W],
        "p": [H, W],  # padded_shape placeholder; real value per channel
        "t": list(transform)[:6],
        "r": None if raster_crs is None else raster_crs.to_string(),
        "d": nodata_value if has_nodata else None,
        "dn": has_nodata,
        "f": config.scaling_factor,
    }
    if ycbcr_rgb_0:
        dcv_compress["ycbcr_rgb"] = ycbcr_rgb_0

    dcv_compress = _assemble_channels(dcv_compress, ch_data_list, num_channels)

    dcv_compress = numpy_to_python(dcv_compress)

    validity_suffix = _validity_suffix(config, all_valid)
    output_path = _build_dct_output_path(
        file_path,
        config,
        output_dir,
        channel_block_sizes,
        meta["band_dtypes"],
        u8_accuracy_mode,
        validity_suffix=validity_suffix,
        crs=raster_crs,
    )

    if not config.output.quiet:
        print(f"  Saving: {output_path}")

    packed_data = msgpack.packb(dcv_compress, use_bin_type=True)
    del dcv_compress

    if skip_archive:
        base = os.path.splitext(os.path.basename(output_path))[0]
        temp_msgpack = os.path.join(output_dir, f"{base}.msgpack")
        with open(temp_msgpack, "wb") as f:
            f.write(packed_data)
        if not config.output.quiet:
            print("  Skipped archiving (skip_archive=True)")
        return temp_msgpack

    base = os.path.splitext(os.path.basename(output_path))[0]
    shm_dir = "/dev/shm"
    if os.path.isdir(shm_dir) and os.access(shm_dir, os.W_OK):
        temp_msgpack = os.path.join(shm_dir, f"{base}_{id(output_path)}.msgpack")
    else:
        temp_msgpack = os.path.join(output_dir, f"{base}.msgpack")

    with open(temp_msgpack, "wb") as f:
        f.write(packed_data)
    del packed_data

    method = config.output.compression_method.upper()
    pack_to_7z(output_path, temp_msgpack, store=False, method=method,
               lzma2_block_size=config.output.lzma2_block_size)

    if config.output.delete_temp_files:
        remove_if_exists(temp_msgpack)

    if not config.output.quiet:
        _print_dct_stats(
            file_path,
            output_path,
            config,
            all_uint8,
            u8_accuracy_mode,
            max_error_global,
            all_valid,
            start_time,
        )

    return output_path


def _assemble_channels(dcv_compress, ch_data_list, num_channels):
    if num_channels == 1:
        ch_packed = ch_data_list[0]
        dcv_compress["l"] = ch_packed.get("l", [])
        dcv_compress["c"] = ch_packed.get("c", b"")
        dcv_compress["m"] = ch_packed.get("m", [])
        if "cm" in ch_packed:
            dcv_compress["cm"] = ch_packed["cm"]
        if "sf" in ch_packed:
            dcv_compress["sf"] = ch_packed["sf"]
        if "dt" in ch_packed:
            dcv_compress["dt"] = ch_packed["dt"]
    else:
        dcv_compress["channels"] = ch_data_list
    return dcv_compress


def _validity_suffix(config, all_valid):
    return "VT" if all_valid else "VF"


def _build_dct_output_path(
    file_path,
    config,
    output_dir,
    channel_block_sizes,
    band_dtypes,
    u8_accuracy_mode,
    validity_suffix="VT",
    crs=None,
):
    """Archive name. The accuracy tag shows the bound the bands are checked against: U8acc for
    integer bands (cm=5/cm=6), Acc for float bands (cm=3), both for a raster mixing the two."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    crs_clean = crs_label(crs)

    if config.compression.auto_select_block_size:
        mode_suffix = "AUTO"
    else:
        mode_suffix = "N"

    unique_bs = (
        sorted(set(channel_block_sizes)) if channel_block_sizes else [config.compression.block_size]
    )
    bs_label = str(unique_bs[0]) if len(unique_bs) == 1 else "-".join(str(n) for n in unique_bs)

    is_int = [np.dtype(d).kind in "iu" for d in band_dtypes]
    tags = []
    if not all(is_int):
        tags.append(f"Acc{config.compression.accuracy}")
    if any(is_int) and u8_accuracy_mode:
        u8sf = getattr(config.compression, "uint8_scaling_factor", 10)
        u8sf_str = "-".join(str(x) for x in u8sf) if isinstance(u8sf, list) else str(u8sf)
        tags.append(f"U8acc{config.compression.uint8_accuracy}_sf{u8sf_str}")
    acc_str = "_".join(tags) or f"Acc{config.compression.accuracy}"

    output_name = (
        f"{base_name}_{mode_suffix}{bs_label}"
        f"_{acc_str}"
        f"_tdct{config.compression.dct_type}"
        f"_dec{config.compression.decimal_places}"
        f"_CRS{crs_clean}"
        f"_{validity_suffix}"
    )
    return os.path.join(output_dir, f"{output_name}.7z")


def _print_dct_stats(
    file_path,
    output_path,
    config,
    all_uint8,
    u8_accuracy_mode,
    max_error_global,
    all_valid,
    start_time,
):
    elapsed = time.time() - start_time
    original_size = os.path.getsize(file_path)
    compressed_size = os.path.getsize(output_path)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    print(f"  Original: {original_size / 1024:.1f} KB")
    print(f"  Compressed: {compressed_size / 1024:.1f} KB")
    print(f"  Ratio: {ratio:.2f}x")
    if u8_accuracy_mode and all_uint8:
        u8_acc = config.compression.uint8_accuracy
        u8_sf = getattr(config.compression, "uint8_scaling_factor", 10)
        if isinstance(u8_sf, list):
            u8_sf_str = "-".join(str(x) for x in u8_sf)
        else:
            u8_sf_str = str(u8_sf)
        print(f"  Accuracy: ±{u8_acc} (uint8 mode, sf={u8_sf_str})")
        print(f"  Max error: {max_error_global:.1f}")
    else:
        print(f"  Max error: {max_error_global:.6f} (target: {config.compression.accuracy})")
    validity_str = "✓ TRUE" if all_valid else "✗ FALSE"
    print(f"  Validity: {validity_str}")
    print(f"  Time: {elapsed:.2f}s")


__all__ = [
    "compress_image",
    "find_raster_files",
]
