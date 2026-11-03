"""
FIDWAC v2 - JPEG compression path (cm=1)
==========================================
Handles uint8 RGB (exactly 3 bands) via turbojpeg SIMD:
  - strips → JPEG bytes
  - rgb.bin + meta.msgpack → .7z (store, -mx=0)

cm=1: TurboJPEG (SIMD) — quality-based JPEG strips via libjpeg-turbo for uint8 RGB.
      No formal max-error guarantee; lossiness depends on JPEG quality parameter.
      Archive format: rgb.bin + meta.msgpack (mode='jpeg_strips'), not the standard
      msgpack 'cm' field.

Entry point: compress_jpeg_path(src, ...)

REQUIRES: libjpeg-turbo 3.0 or later (for high-performance SIMD encoding of large files)
"""

import os
import time
from typing import List

import numpy as np
from rasterio.windows import Window
from tqdm import tqdm

from config import Config
from compress.output_crs import crs_label, resolve_output_crs
from .compression_utils import pack_to_7z, write_msgpack, remove_if_exists

def compress_jpeg_path(
    src,
    file_path: str,
    H: int,
    W: int,
    data_band_indices: List[int],
    has_nodata: bool,
    nodata_value: float,
    file_crs,
    transform,
    config: Config,
    output_dir: str,
    start_time: float,
) -> str:
    """Compress RGB uint8 raster using turbojpeg strips.

    Returns path to the output .7z archive.
    """
    from turbojpeg import TJPF_RGB, TJSAMP_420
    from compress.turbojpeg_loader import open_turbojpeg

    quality = config.compression.rgb_quality
    pad_W = (8 - W % 8) % 8
    padded_W = W + pad_W

    strip_rows = max(8, (256 * 1024 * 1024) // (padded_W * 3))
    strip_rows = (strip_rows // 8) * 8
    total_strips = (H + strip_rows - 1) // strip_rows

    if not config.output.quiet:
        print("  RGB channels: JPEG strips via turbojpeg")
        print(f"  quality={quality}, strip={strip_rows} rows, total={total_strips}")

    tj = open_turbojpeg()
    jpeg_strips, strip_sizes, strip_offsets, nodata_masks = _encode_strips(
        src,
        H,
        W,
        pad_W,
        strip_rows,
        data_band_indices,
        has_nodata,
        nodata_value,
        quality,
        tj,
        TJPF_RGB,
        TJSAMP_420,
        config.output.quiet,
    )

    nodata_rle = _build_nodata_rle(has_nodata, nodata_masks, strip_offsets, H, W)
    del nodata_masks

    out_crs = resolve_output_crs(file_crs, config.compression.crs)
    output_path = _build_jpeg_output_path(file_path, config, output_dir, quality, out_crs)

    _write_jpeg_archive(
        jpeg_strips,
        strip_sizes,
        strip_offsets,
        nodata_rle,
        H,
        W,
        padded_W,
        nodata_value,
        has_nodata,
        out_crs,
        transform,
        config,
        output_dir,
        output_path,
        len(data_band_indices),
    )
    del jpeg_strips

    if not config.output.quiet:
        _print_jpeg_stats(file_path, output_path, quality, start_time)

    return output_path



def _encode_strips(
    src,
    H,
    W,
    pad_W,
    strip_rows,
    data_band_indices,
    has_nodata,
    nodata_value,
    quality,
    tj,
    TJPF_RGB,
    TJSAMP_420,
    quiet,
):
    jpeg_strips = []
    strip_sizes = []
    strip_offsets = []
    nodata_masks = []

    pbar = (
        None
        if quiet
        else tqdm(total=(H + strip_rows - 1) // strip_rows, desc="JPEG", unit="strip", ascii=False)
    )

    for y_start in range(0, H, strip_rows):
        read_h = min(strip_rows, H - y_start)
        if read_h <= 0:
            break

        rgb_bands = [src.read(bi, window=Window(0, y_start, W, read_h)) for bi in data_band_indices]
        rgb = np.stack(rgb_bands, axis=-1)
        if pad_W > 0:
            rgb = np.pad(rgb, ((0, 0), (0, pad_W), (0, 0)), mode="edge")

        if has_nodata and 0 <= nodata_value <= 255:
            mask = np.any(rgb[:, :W, :] == int(nodata_value), axis=2)
            nodata_masks.append(mask)
            rgb[mask] = 0
        else:
            nodata_masks.append(None)

        jpg = tj.encode(
            rgb,
            quality=quality,
            pixel_format=TJPF_RGB,
            jpeg_subsample=TJSAMP_420,
        )
        jpeg_strips.append(jpg)
        strip_sizes.append(len(jpg))
        strip_offsets.append(y_start)

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    return jpeg_strips, strip_sizes, strip_offsets, nodata_masks


def _build_nodata_rle(has_nodata, nodata_masks, strip_offsets, H, W):
    if not has_nodata or not any(m is not None for m in nodata_masks):
        return None

    from engine.utils import encode_mask_rle

    full_mask = np.zeros((H, W), dtype=bool)
    for i, m in enumerate(nodata_masks):
        if m is not None:
            y0 = strip_offsets[i]
            h = m.shape[0]
            full_mask[y0 : y0 + h, :] = m[:h, :W]
    return encode_mask_rle(full_mask) if full_mask.any() else None


def _build_jpeg_output_path(file_path, config, output_dir, quality, out_crs):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    crs_clean = crs_label(out_crs)
    mode_suffix = "AUTO" if config.compression.auto_select_block_size else "N"
    output_name = f"{base_name}_{mode_suffix}8_Q{quality}_CRS{crs_clean}"
    return os.path.join(output_dir, f"{output_name}.7z")


def _write_jpeg_archive(
    jpeg_strips,
    strip_sizes,
    strip_offsets,
    nodata_rle,
    H,
    W,
    padded_W,
    nodata_value,
    src_has_nodata,
    out_crs,
    transform,
    config,
    output_dir,
    output_path,
    num_bands,
):
    unique_id = id(compress_jpeg_path)

    rgb_bin_path = os.path.join(output_dir, f"_fidwac_rgb_{unique_id}.bin")
    with open(rgb_bin_path, "wb") as f:
        for s in jpeg_strips:
            f.write(s)

    meta = {
        "mode": "jpeg_strips",
        "strip_sizes": strip_sizes,
        "strip_offsets": strip_offsets,
        "nodata_rle": nodata_rle,
        "quality": config.compression.rgb_quality,
        "original_shape": [H, W],
        "padded_shape": [H, padded_W],
        "crs": None if out_crs is None else out_crs.to_string(),
        "transform": list(transform)[:6],
        "nodata": nodata_value if src_has_nodata else None,
        "has_nodata": src_has_nodata,
        "num_bands": num_bands,
    }
    meta_path = os.path.join(output_dir, f"_fidwac_meta_{unique_id}.msgpack")
    write_msgpack(meta_path, meta)

    pack_to_7z(output_path, rgb_bin_path, meta_path, store=True)

    if config.output.delete_temp_files:
        remove_if_exists(rgb_bin_path)
        remove_if_exists(meta_path)


def _print_jpeg_stats(file_path, output_path, quality, start_time):
    elapsed = time.time() - start_time
    original_size = os.path.getsize(file_path)
    compressed_size = os.path.getsize(output_path)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    print(f"  Original: {original_size / 1024:.1f} KB")
    print(f"  Compressed: {compressed_size / 1024:.1f} KB")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Quality: JPEG Q{quality}")
    print(f"  Time: {elapsed:.2f}s")


__all__ = ["compress_jpeg_path"]
