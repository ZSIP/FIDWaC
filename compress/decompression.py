"""
FIDWAC v2 - Decompression logic (uniform grid, adaptive blocks)
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

import numpy as np
import rasterio
from rasterio.transform import Affine
import msgpack
import py7zr
from tqdm import tqdm

from scipy.fft import idct

from engine.blocks import reconstruct_block, ycbcr_to_rgb
from engine.utils import decode_mask_rle, decode_mask_rle_rows, integer_range
from config import Config, load_config
from compress.output_crs import crs_from_header
from core.codec import decode_vlq, dc_median_ac_decode
from core.dct import _get_zigzag_indices

STRIP_PIXELS = 16 * 1024 * 1024  # pixels per strip of the strip-wise decoders
STREAMING_CHANNEL_BYTES = 256 * 1024 * 1024  # channels larger than this are decoded strip by strip into the GeoTIFF

# Batched reconstruction of the cm=6 Y/Cb/Cr planes: one IDCT call per batch of blocks (falls back
# to the block-by-block reconstruct_block when a plane cannot be batched).


def load_compressed_data(file_path: str) -> Dict[str, Any]:
    """Extract and load compressed data from a .7z archive."""
    import tempfile
    import subprocess

    temp_dir = tempfile.mkdtemp()

    try:
        if shutil.which("7zz"):
            cmd = "7zz"
        elif shutil.which("7zip"):
            cmd = "7zip"
        elif shutil.which("7z"):
            cmd = "7z"
        else:
            cmd = None

        if cmd:
            subprocess.run(
                [cmd, "x", "-y", f"-o{temp_dir}", file_path],
                capture_output=True,
                check=True,
            )
            names = os.listdir(temp_dir)
        else:
            with py7zr.SevenZipFile(file_path, "r") as archive:
                names = archive.getnames()
                if not names:
                    raise ValueError(f"Archive {file_path} is empty")
                archive.extractall(path=temp_dir)

        if not names:
            raise ValueError(f"Archive {file_path} is empty")

        # Detect format:
        # A) jpeg_strips: rgb.bin + meta.msgpack
        # B) single GeoTIFF (JPEG GeoTIFF, legacy)
        # C) single msgpack (DCT path)
        # D) single .fpred payload (cm=7 near-lossless predictive mode)
        # E) single .sz3 payload (cm=8 external SZ3 mode)
        has_rgb_bin = any(n.endswith(".bin") and "rgb" in n.lower() for n in names)
        has_meta_msgpack = any(n.endswith(".msgpack") and "meta" in n.lower() for n in names)
        tif_files = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
        msgpack_files = [n for n in names if n.lower().endswith(".msgpack")]
        fpred_files = [n for n in names if n.lower().endswith(".fpred")]
        sz3_files = [n for n in names if n.lower().endswith(".sz3")]

        if fpred_files:
            fpred_path = os.path.join(temp_dir, fpred_files[0])
            with open(fpred_path, "rb") as f:
                fpred_bytes = f.read()
            return {"mode": "fpred", "fpred_bytes": fpred_bytes}

        if sz3_files:
            sz3_path = os.path.join(temp_dir, sz3_files[0])
            with open(sz3_path, "rb") as f:
                sz3_bytes = f.read()
            return {"mode": "sz3", "sz3_bytes": sz3_bytes}

        if has_rgb_bin and has_meta_msgpack:
            # Format A: jpeg_strips
            rgb_bin_data = None
            meta_data = None
            for name in names:
                p = os.path.join(temp_dir, name)
                if not os.path.exists(p):
                    continue
                if name.endswith(".bin") and "rgb" in name.lower():
                    with open(p, "rb") as f:
                        rgb_bin_data = f.read()
                elif name.endswith(".msgpack") and "meta" in name.lower():
                    with open(p, "rb") as f:
                        meta_data = f.read()
            if rgb_bin_data is None or meta_data is None:
                raise ValueError(
                    f"JPEG strips format: missing rgb.bin or meta.msgpack in {file_path}"
                )
            result = msgpack.unpackb(meta_data, raw=False, strict_map_key=False)
            result["_rgb_bin"] = rgb_bin_data
            return result

        if tif_files and not msgpack_files:
            # Format B: single GeoTIFF
            return os.path.join(temp_dir, tif_files[0])

        # Format C: single msgpack (DCT)
        msgpack_data = None
        for name in names:
            p = os.path.join(temp_dir, name)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    msgpack_data = f.read()
                break
        if msgpack_data is None:
            raise ValueError(f"msgpack data not found in {file_path}")
        return msgpack.unpackb(msgpack_data, raw=False, strict_map_key=False)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _reconstruct_plane_batched(coeffs, lengths, block_size, scaling_factor, padded_H, padded_W):
    """Reconstruct a Y, Cb or Cr plane batch-wise: scipy idct over batches of blocks instead of a
    reconstruct_block loop. The axis order and operations match idct2 in 2D, so the result is identical.
    length == 0 gives the neutral 128 (planes are stored offset by 128, so a block with no
    coefficients is neutral, not black) and length == -1 gives -9999, as in the loop.
    Returns None when the data is inconsistent; the caller then uses the block-by-block loop."""
    n = block_size
    lengths = np.asarray(lengths, dtype=np.int64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    bpr = padded_W // n
    pos = np.nonzero(lengths > 0)[0]
    Ls = lengths[pos]
    if len(lengths) != (padded_H // n) * bpr or coeffs.size < int(Ls.sum()):
        return None
    return _reconstruct_plane_strip(coeffs, lengths, n, scaling_factor, padded_W, 0, padded_H // n)


def _reconstruct_plane_strip(coeffs, lengths, n, scaling_factor, padded_W, row0, row1, coeff_start=0):
    """Block rows row0..row1 of a Y, Cb or Cr plane (see _reconstruct_plane_batched). `coeffs` and
    `lengths` are numpy arrays of the whole plane; coeff_start is the position in `coeffs` of the
    first block of the strip. Returns (row1 - row0) * n rows of padded_W columns."""
    bpr = padded_W // n
    ln = lengths[row0 * bpr : row1 * bpr]
    pos = np.nonzero(ln > 0)[0]
    Ls = ln[pos]
    out = np.full((len(ln), n, n), 128.0, dtype=np.float64)
    out[ln == -1] = -9999.0
    strip_coeffs = coeffs[coeff_start : coeff_start + int(Ls.sum())]
    for p, rec in _idct_block_batches(strip_coeffs, pos, Ls, n, scaling_factor):
        out[p] = np.clip(rec + 128.0, 0.0, 255.0)
    return out.reshape(row1 - row0, bpr, n, n).transpose(0, 2, 1, 3).reshape((row1 - row0) * n, padded_W)


def _idct_block_batches(coeffs, pos, Ls, n, scaling_factor):
    """Yield (block indices, float64 IDCT) for the blocks `pos` that store `Ls` coefficients each,
    in batches of about 4M coefficients. `coeffs` is the flat array in block order. The axis order
    is the one of idct2 on a single 2D block, so every block equals the one of reconstruct_block."""
    idx = _get_zigzag_indices(n)[: n * n]
    starts = np.concatenate(([0], np.cumsum(Ls)[:-1])).astype(np.int64)
    chunk = max(256, (1 << 22) // (n * n))
    for a in range(0, len(pos), chunk):
        p, Lc = pos[a : a + chunk], Ls[a : a + chunk]
        s0, tot = int(starts[a]), int(Lc.sum())
        # the k-th stored coefficient of block j goes to zigzag position k of row j of mat
        cols = np.arange(tot) - np.repeat(np.cumsum(Lc) - Lc, Lc)
        dest = np.repeat(np.arange(len(p)) * (n * n), Lc) + idx[cols]
        mat = np.zeros((len(p), n * n), dtype=np.float64)
        mat.reshape(-1)[dest] = coeffs[s0 : s0 + tot] / scaling_factor
        yield p, idct(
            idct(mat.reshape(-1, n, n), axis=1, norm="ortho", type=2), axis=2, norm="ortho", type=2
        )


def _channel_batchable(coeffs, lengths, n, padded_H, padded_W):
    """True when the flat coefficient array and block lengths are consistent enough to decode in batches."""
    pos = lengths > 0
    return (
        len(lengths) == (padded_H // n) * (padded_W // n)
        and coeffs.size >= int(lengths[pos].sum())
        and (not pos.any() or int(lengths.max()) <= n * n)
    )


def _reconstruct_channel_batched(
    coeffs, lengths, masks_dict, cm, int_dtype, result_dtype, n, scaling_factor,
    padded_W, nodata_value, nodata_guard, row0, row1, coeff_start=0,
):
    """cm=3 / cm=5 block rows row0..row1 of a channel as the block loop of _decompress_single_channel
    builds them, with one IDCT call per batch of blocks. Blocks with stored coefficients go through
    the batch; empty, NoData and masked (length -2) blocks are filled as in the loop.
    `coeffs` and `lengths` are numpy arrays of the whole channel (see _channel_batchable) and
    coeff_start is the position in `coeffs` of the first block of the strip.
    Returns the strip of (row1 - row0) * n rows and padded_W columns."""
    bpr = padded_W // n
    first, last = row0 * bpr, row1 * bpr
    ln = lengths[first:last]
    pos = np.nonzero(ln > 0)[0]
    Ls = ln[pos]
    uint8_mode = cm == 5
    if uint8_mode:
        centre, lo, hi = integer_range(int_dtype or "uint8")
    out = np.zeros((len(ln), n, n), dtype=result_dtype)
    nodata_blocks = ln == -1
    if nodata_blocks.any():
        out[nodata_blocks] = nodata_value if nodata_value is not None else 0
    strip_coeffs = coeffs[coeff_start : coeff_start + int(Ls.sum())]
    for p, rec in _idct_block_batches(strip_coeffs, pos, Ls, n, scaling_factor):
        if uint8_mode:
            rec = np.clip(np.round(rec + centre), lo, hi).astype(np.float32)
            if nodata_guard is not None and lo <= nodata_guard <= hi:
                # a valid pixel must not decode to the NoData value (see reconstruct_block)
                rec[rec == nodata_guard] = nodata_guard + (-1.0 if nodata_guard >= hi else 1.0)
        out[p] = rec
    for k in np.nonzero(ln == -2)[0]:
        idx = first + int(k)
        if idx in masks_dict:
            out[k] = reconstruct_block(
                masks_dict[idx],
                n,
                scaling_factor,
                nodata_value,
                uint8_mode=uint8_mode,
                src_dtype=int_dtype,
                nodata_guard=nodata_guard,
            )
    return out.reshape(row1 - row0, bpr, n, n).transpose(0, 2, 1, 3).reshape((row1 - row0) * n, padded_W)


def _rgb_nodata_rles(ch_data, nodata_value):
    """cm=6 NoData of the three bands as RLE lists, (nd, flip), or None without NoData.
    Header "nd": per-band RLE over the unpadded image; "nu": per-band RLE of the pixels that step
    towards their original value (only for NoData inside 1..254), None for a band without any."""
    nd_rles = ch_data.get("nd")
    if nodata_value is None or not nd_rles:
        return None
    return nd_rles, ch_data.get("nu") or [None, None, None]


def _apply_rgb_nodata(planes, nodata_rles, nodata_value, y0=0):
    """cm=6: write the exact NoData value back on the pixels that held it in each band. A valid pixel
    that decodes to the NoData value is moved one step into the data range, so it is not read as
    NoData: down for NoData 255, up otherwise; pixels in the "nu" masks step down, towards their
    original value. `planes` are the rows y0.. of the unpadded image (the whole image for y0 = 0)."""
    nd_rles, flip_rles = nodata_rles
    step = -1.0 if nodata_value >= 255 else 1.0
    for plane, rle, flip_rle in zip(planes, nd_rles, flip_rles):
        height, width = plane.shape
        mask = decode_mask_rle_rows(rle, width, y0, height)
        hit = (plane == nodata_value) & ~mask
        plane[hit] = nodata_value + step
        if flip_rle:
            plane[hit & decode_mask_rle_rows(flip_rle, width, y0, height)] = nodata_value - step
        plane[mask] = nodata_value


def _restore_rgb_nodata(R, G, B, ch_data, nodata_value):
    """cm=6: put the NoData value back on the whole decoded image (see _apply_rgb_nodata)."""
    nodata_rles = _rgb_nodata_rles(ch_data, nodata_value)
    if nodata_rles is not None:
        _apply_rgb_nodata((R, G, B), nodata_rles, nodata_value)


def _ycbcr_to_rgb_float32(Y, Cb, Cr):
    """ycbcr_to_rgb in float32 planes, computed in row bands on a few threads (element-wise, so the
    values equal those of one call)."""
    R, G, B = (np.empty(Y.shape, dtype=np.float32) for _ in range(3))
    workers = max(1, min(4, os.cpu_count() or 1)) if Y.size >= (1 << 20) else 1
    bounds = np.linspace(0, Y.shape[0], workers + 1).astype(int)

    def band(i):
        a, b = bounds[i], bounds[i + 1]
        R[a:b], G[a:b], B[a:b] = ycbcr_to_rgb(Y[a:b], Cb[a:b], Cr[a:b])

    if workers == 1:
        band(0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(band, range(workers)))
    return R, G, B


def _rgb_channel_maps(ch_data):
    """cm=6 entry -> (lengths per plane, encoded streams per plane, scaling factor)."""
    raw_l = ch_data.get("l", [])
    raw_c = ch_data.get("c", [])
    scaling_factor = ch_data.get("sf", 1)

    # Flat format: l = [Y_lengths, Cb_lengths, Cr_lengths]
    # Flat format: c = [Y_dc, Y_ac, Cb_dc, Cb_ac, Cr_dc, Cr_ac, mult_y, mult_cb, mult_cr, fallback]
    if isinstance(raw_l, list) and len(raw_l) == 3 and not isinstance(raw_l[0], dict):
        lengths_map = {"Y": raw_l[0], "Cb": raw_l[1], "Cr": raw_l[2]}
    else:
        # legacy dict format
        lengths_map = raw_l

    if isinstance(raw_c, list) and len(raw_c) in (6, 10):
        encoded_map = {
            "Y": [raw_c[0], raw_c[1]],
            "Cb": [raw_c[2], raw_c[3]],
            "Cr": [raw_c[4], raw_c[5]],
        }
    else:
        # legacy dict format
        encoded_map = {ch: raw_c.get(ch, b"") for ch in ["Y", "Cb", "Cr"]}
    return lengths_map, encoded_map, scaling_factor


def _rgb_marker_blocks(lengths_map, block_size, padded_H, blocks_per_row):
    """Boolean array over the blocks that decode to black RGB: NoData markers (L == -1, restored later
    from the "nd" masks; the -9999 fill would not survive the YCbCr conversion) and the all-zero RGB
    marker (L == 0 in all three channels). A block with L == 0 in only some channels is not a marker:
    those planes are neutral (128). None when there is no such block or the lengths are inconsistent."""
    ly = np.asarray(lengths_map["Y"], dtype=np.int64)
    lcb = np.asarray(lengths_map["Cb"], dtype=np.int64)
    lcr = np.asarray(lengths_map["Cr"], dtype=np.int64)
    if ly.size == (padded_H // block_size) * blocks_per_row and np.any(ly <= 0):
        return (ly < 0) | ((ly == 0) & (lcb == 0) & (lcr == 0))
    return None


def _rgb_geometry(ch_data, block_size, original_shape):
    """(block size, padded shape) of a cm=6 entry. The header field "p" only repeats the image shape,
    so the padded shape is derived from the block size of the entry."""
    n = ch_data.get("bs", block_size)
    H, W = original_shape
    return n, (-(-H // n) * n, -(-W // n) * n)


def _rgb_streamable(ch_data, block_size, padded_shape):
    """True when a cm=6 entry can be decoded in strips: DC/AC streams and consistent block counts."""
    lengths_map, encoded_map, _ = _rgb_channel_maps(ch_data)
    padded_H, padded_W = padded_shape
    count = (padded_H // block_size) * (padded_W // block_size)
    return all(
        isinstance(encoded_map[ch], list)
        and len(encoded_map[ch]) == 2
        and len(lengths_map[ch]) == count
        for ch in ("Y", "Cb", "Cr")
    )


def _iter_rgb_strips(ch_data, block_size, padded_shape, original_shape):
    """Decode a cm=6 entry strip by strip (about STRIP_PIXELS pixels each) and yield
    (y0, R, G, B): the rows y0.. of the unpadded image as float32 planes, black blocks included, NoData
    not yet restored. The planes of the whole image never sit in memory. Same values as
    _decode_dct_channel_rgb."""
    H, W = original_shape
    padded_H, padded_W = padded_shape
    n = block_size
    bpr = padded_W // n
    names = ("Y", "Cb", "Cr")
    lengths_map, encoded_map, scaling_factor = _rgb_channel_maps(ch_data)
    lens = {ch: np.asarray(lengths_map[ch], dtype=np.int64) for ch in names}

    def _coeffs(ch):
        return dc_median_ac_decode(encoded_map[ch][0], encoded_map[ch][1], lens[ch], bpr)

    with ThreadPoolExecutor(max_workers=3) as pool:
        coeffs = dict(zip(names, pool.map(_coeffs, names)))
        starts = {ch: np.concatenate(([0], np.cumsum(np.maximum(lens[ch], 0)))) for ch in names}
        marker = _rgb_marker_blocks(lengths_map, n, padded_H, bpr)

        block_rows = -(-H // n)
        strip_rows = max(1, STRIP_PIXELS // (padded_W * n))
        for r0 in range(0, block_rows, strip_rows):
            r1 = min(block_rows, r0 + strip_rows)
            planes = list(
                pool.map(
                    lambda ch: _reconstruct_plane_strip(
                        coeffs[ch], lens[ch], n, scaling_factor, padded_W, r0, r1,
                        int(starts[ch][r0 * bpr]),
                    ),
                    names,
                )
            )
            R, G, B = _ycbcr_to_rgb_float32(*planes)
            if marker is not None:
                trivial = marker[r0 * bpr : r1 * bpr].reshape(r1 - r0, bpr)
                trivial = np.repeat(np.repeat(trivial, n, axis=0), n, axis=1)
                R[trivial] = 0.0
                G[trivial] = 0.0
                B[trivial] = 0.0
            h = min((r1 - r0) * n, H - r0 * n)
            yield r0 * n, R[:h, :W], G[:h, :W], B[:h, :W]


def _decode_rgb_streaming(
    ch_data, block_size, padded_shape, original_shape, dst, first_band, nodata_value, out_dtype, quiet
):
    """cm=6 image written to bands first_band.. of the GeoTIFF strip by strip (see _iter_rgb_strips)."""
    from rasterio.windows import Window

    H, W = original_shape
    nodata_rles = _rgb_nodata_rles(ch_data, nodata_value)
    n_strips = -(-(-(-H // block_size)) // max(1, STRIP_PIXELS // (padded_shape[1] * block_size)))
    for y0, R, G, B in tqdm(
        _iter_rgb_strips(ch_data, block_size, padded_shape, original_shape),
        total=n_strips, desc="RGB", disable=quiet,
    ):
        strip = [R, G, B]
        if nodata_rles is not None:
            _apply_rgb_nodata(strip, nodata_rles, nodata_value, y0)
        for k, plane in enumerate(strip):
            data = np.round(plane) if np.issubdtype(out_dtype, np.integer) else plane
            dst.write(data.astype(out_dtype), first_band + k, window=Window(0, y0, W, R.shape[0]))


def _decode_dct_channel_rgb(
    ch_data: Dict[str, Any], block_size: int, padded_shape: tuple
) -> Dict[str, np.ndarray]:
    """Decode DCT coefficients for per-block RGB mode (cm=6).

    Returns dict with reconstructed R, G, B channels.
    """
    lengths_map, encoded_map, scaling_factor = _rgb_channel_maps(ch_data)
    padded_H, padded_W = padded_shape
    blocks_per_row = padded_W // block_size

    # Decode the three channels side by side: the numba decoders release the GIL, and so do the
    # numpy / scipy calls of the plane reconstruction.
    def _decode_one(ch):
        encoded = encoded_map[ch]
        if isinstance(encoded, list) and len(encoded) == 2:
            coeffs = dc_median_ac_decode(encoded[0], encoded[1], lengths_map[ch], blocks_per_row)
            plane = _reconstruct_plane_batched(
                coeffs, lengths_map[ch], block_size, scaling_factor, padded_H, padded_W
            )
            return coeffs, plane
        return np.array([], dtype=np.int64), None

    with ThreadPoolExecutor(max_workers=3) as pool:
        decoded = dict(zip(("Y", "Cb", "Cr"), pool.map(_decode_one, ("Y", "Cb", "Cr"))))

    channels = {}
    for ch in ["Y", "Cb", "Cr"]:
        lengths = lengths_map[ch]
        encoded = encoded_map[ch]
        coeffs, plane = decoded[ch]

        result = np.zeros((padded_H, padded_W), dtype=np.float64)
        coeff_pos = 0

        if plane is not None:
            channels[ch] = plane.astype(result.dtype, copy=False)
            continue

        for idx, length in enumerate(lengths):
            row = idx // blocks_per_row
            col = idx % blocks_per_row
            y_start = row * block_size
            x_start = col * block_size

            if length == 0:
                # No coefficients: the plane is stored offset by 128, so this block is neutral
                # (128), not black. The integer refinement can zero out a whole channel.
                block = np.full((block_size, block_size), 128.0, dtype=np.float32)
            elif length == -1:
                block = np.full(
                    (block_size, block_size), -9999, dtype=np.float32
                )  # length == -1 marks a NoData block
            else:
                block_coeffs = coeffs[coeff_pos : coeff_pos + length]
                coeff_pos += length

                # Reconstruct YCbCr without uint8_mode (coefficients already compressed with appropriate accuracy)
                block = reconstruct_block(
                    block_coeffs.tolist(),
                    block_size,
                    scaling_factor,
                    -9999,
                    uint8_mode=False,
                )
                # Clip to [0,255]: IDCT can overshoot; matches compression error check
                block = np.clip(block + 128.0, 0.0, 255.0)

            result[y_start : y_start + block_size, x_start : x_start + block_size] = block

        channels[ch] = result

    Y = channels["Y"]
    Cb = channels["Cb"]
    Cr = channels["Cr"]

    R, G, B = _ycbcr_to_rgb_float32(Y, Cb, Cr)

    marker = _rgb_marker_blocks(lengths_map, block_size, padded_H, blocks_per_row)
    if marker is not None:
        trivial = marker.reshape(padded_H // block_size, blocks_per_row)
        trivial = np.repeat(np.repeat(trivial, block_size, axis=0), block_size, axis=1)
        R[trivial] = 0.0
        G[trivial] = 0.0
        B[trivial] = 0.0

    return {"R": R, "G": G, "B": B}


def _decompress_single_channel(
    ch_data: Dict[str, Any],
    block_size: int,
    original_shape: tuple,
    nodata_value: float,
    scaling_factor: int,
    version: int,
    nodata_guard: Optional[float] = None,
) -> np.ndarray:
    """Decompress a single channel from flat format data. Returns 2D array.
    nodata_guard: source NoData value (None without NoData) - see reconstruct_block."""
    lengths = ch_data.get("l", [])
    cm = ch_data.get("cm", 0)
    raw_c = ch_data.get("c", [])
    scaling_factor = ch_data.get("sf", scaling_factor)
    block_size = ch_data.get("bs", block_size)

    H, W = original_shape
    # Compute padded_shape from per-channel block_size (header value may differ)
    padded_H = ((H + block_size - 1) // block_size) * block_size
    padded_W = ((W + block_size - 1) // block_size) * block_size
    blocks_per_row = padded_W // block_size

    if cm == 4 and isinstance(raw_c, (bytes, bytearray)):
        import zlib

        filter_used = ch_data.get("f")
        decompressed = zlib.decompress(raw_c)
        filtered = np.frombuffer(decompressed, dtype=np.uint8).reshape(H, W)
        if filter_used is None:
            # Single-band cm=4 archives did not store the filter flag. The encoder kept the Sub filter
            # when it compressed better, i.e. for smooth images, so take the smoother reading.
            def roughness(a):
                return np.abs(np.diff(a.astype(np.int16), axis=1)).mean() if W > 1 else 0.0

            filter_used = roughness(np.cumsum(filtered, axis=1, dtype=np.uint8)) < roughness(filtered)
        if filter_used:
            # Inverse Sub filter: original[i] = filtered[i] + original[i-1]
            result = np.cumsum(filtered, axis=1, dtype=np.uint8)
        else:
            result = filtered
        return result.astype(np.float32)
    if cm == 5:
        # uint8 accuracy mode: DC-AC format, coefficients in centered space
        if isinstance(raw_c, list) and len(raw_c) == 2:
            dc_data, ac_data = raw_c
            coeffs_arr = dc_median_ac_decode(dc_data, ac_data, lengths, blocks_per_row)
            coeffs = coeffs_arr.tolist()
        else:
            coeffs = []
    elif cm == 3 and isinstance(raw_c, list) and len(raw_c) == 2:
        dc_data, ac_data = raw_c
        coeffs_arr = dc_median_ac_decode(dc_data, ac_data, lengths, blocks_per_row)
        coeffs = coeffs_arr.tolist()
    elif version >= 2 and isinstance(raw_c, (bytes, bytearray)):
        coeffs = decode_vlq(raw_c)
    else:
        coeffs = list(raw_c) if raw_c else []

    masks_data = ch_data.get("m", [])
    masks_dict = {item[0]: item[1] for item in masks_data}

    int_dtype = ch_data.get("dt", "uint8") if cm == 5 else None
    result_dtype = np.dtype(int_dtype) if cm == 5 else (np.uint8 if cm == 4 else np.float32)
    lengths_np = np.asarray(lengths, dtype=np.int64)
    coeffs_np = np.asarray(coeffs, dtype=np.float64)
    if _channel_batchable(coeffs_np, lengths_np, block_size, padded_H, padded_W):
        plane = _reconstruct_channel_batched(
            coeffs_np, lengths_np, masks_dict, cm, int_dtype, result_dtype, block_size,
            scaling_factor, padded_W, nodata_value, nodata_guard, 0, padded_H // block_size,
        )
        return plane[:H, :W]
    result = np.zeros((padded_H, padded_W), dtype=result_dtype)

    coeff_pos = 0
    for idx, length in enumerate(lengths):

        row = idx // blocks_per_row
        col = idx % blocks_per_row
        y = row * block_size
        x = col * block_size

        if length == 0:
            result[y : y + block_size, x : x + block_size] = 0
        elif length == -1:
            result[y : y + block_size, x : x + block_size] = (
                nodata_value if nodata_value is not None else 0
            )
        elif length == -2:
            if idx in masks_dict:
                block = reconstruct_block(
                    masks_dict[idx],
                    block_size,
                    scaling_factor,
                    nodata_value,
                    uint8_mode=(cm == 5),
                    src_dtype=int_dtype,
                    nodata_guard=nodata_guard,
                )
                result[y : y + block_size, x : x + block_size] = block
        else:
            block_coeffs = coeffs[coeff_pos : coeff_pos + length]
            coeff_pos += length
            block = reconstruct_block(
                block_coeffs,
                block_size,
                scaling_factor,
                nodata_value,
                uint8_mode=(cm == 5),
                src_dtype=int_dtype,
                nodata_guard=nodata_guard,
            )
            result[y : y + block_size, x : x + block_size] = block

    return result[:H, :W]


def _decompress_single_channel_streaming(
    ch_data: Dict[str, Any],
    block_size: int,
    original_shape: tuple,
    nodata_value: float,
    scaling_factor: int,
    version: int,
    config: Config,
    dst,
    band_idx: int,
    nodata_guard: Optional[float] = None,
) -> None:
    """Decompress a single channel and write blocks directly to GeoTIFF via window.

    Avoids allocating the full padded image in RAM. Writes each reconstructed
    block directly to the output raster using rasterio's write(window=...).
    """
    from rasterio.windows import Window

    lengths = ch_data.get("l", [])
    cm = ch_data.get("cm", 0)
    raw_c = ch_data.get("c", [])
    scaling_factor = ch_data.get("sf", scaling_factor)
    block_size = ch_data.get("bs", block_size)

    H, W = original_shape
    # Compute padded_shape from per-channel block_size (header value may differ)
    padded_H = ((H + block_size - 1) // block_size) * block_size
    padded_W = ((W + block_size - 1) // block_size) * block_size
    blocks_per_row = padded_W // block_size

    if cm == 5:
        # uint8 accuracy mode: DC-AC format, coefficients in centered space
        if isinstance(raw_c, list) and len(raw_c) == 2:
            dc_data, ac_data = raw_c
            coeffs_arr = dc_median_ac_decode(dc_data, ac_data, lengths, blocks_per_row)
            coeffs = coeffs_arr
        else:
            coeffs = []
    elif cm == 3 and isinstance(raw_c, list) and len(raw_c) == 2:
        dc_data, ac_data = raw_c
        coeffs_arr = dc_median_ac_decode(dc_data, ac_data, lengths, blocks_per_row)
        coeffs = coeffs_arr
    elif version >= 2 and isinstance(raw_c, (bytes, bytearray)):
        coeffs = decode_vlq(raw_c)
    else:
        coeffs = list(raw_c) if raw_c else []

    masks_data = ch_data.get("m", [])
    masks_dict = {item[0]: item[1] for item in masks_data}

    int_dtype = ch_data.get("dt", "uint8") if cm == 5 else None
    block_dtype = np.dtype(int_dtype) if cm == 5 else (np.uint8 if cm == 4 else np.float32)

    if cm != 4:
        lengths_np = np.asarray(lengths, dtype=np.int64)
        coeffs_np = np.asarray(coeffs, dtype=np.float64)
        if _channel_batchable(coeffs_np, lengths_np, block_size, padded_H, padded_W):
            # Whole strips of block rows at once (about 64 MB each); same values as the block loop below.
            starts = np.concatenate(([0], np.cumsum(np.maximum(lengths_np, 0))))
            blocks_per_col = -(-H // block_size)
            strip_rows = max(1, STRIP_PIXELS // (padded_W * block_size))
            for r0 in tqdm(
                range(0, blocks_per_col, strip_rows), desc=f"Ch{band_idx}", disable=config.output.quiet
            ):
                r1 = min(blocks_per_col, r0 + strip_rows)
                strip = _reconstruct_channel_batched(
                    coeffs_np, lengths_np, masks_dict, cm, int_dtype, block_dtype, block_size,
                    scaling_factor, padded_W, nodata_value, nodata_guard, r0, r1,
                    int(starts[r0 * blocks_per_row]),
                )
                h = min((r1 - r0) * block_size, H - r0 * block_size)
                window = Window(0, r0 * block_size, W, h)
                dst.write(strip[:h, :W].astype(np.float32), band_idx, window=window)
            return

    coeff_pos = 0
    for idx in tqdm(range(len(lengths)), desc=f"Ch{band_idx}", disable=config.output.quiet):
        length = lengths[idx]

        brow = idx // blocks_per_row
        bcol = idx % blocks_per_row
        y = brow * block_size
        x = bcol * block_size

        if y >= H or x >= W:
            coeff_pos += max(0, length)
            continue

        if length == 0:
            block = np.zeros((block_size, block_size), dtype=block_dtype)
        elif length == -1:
            block = np.full(
                (block_size, block_size),
                nodata_value if nodata_value is not None else 0,
                dtype=block_dtype,
            )
        elif length == -2:
            if idx in masks_dict:
                block = reconstruct_block(
                    masks_dict[idx],
                    block_size,
                    scaling_factor,
                    nodata_value,
                    uint8_mode=(cm == 5),
                    src_dtype=int_dtype,
                    nodata_guard=nodata_guard,
                )
            else:
                block = np.zeros((block_size, block_size), dtype=block_dtype)
        else:
            block_coeffs = coeffs[coeff_pos : coeff_pos + length]
            if isinstance(block_coeffs, np.ndarray):
                block_coeffs = block_coeffs.tolist()
            coeff_pos += length
            block = reconstruct_block(
                block_coeffs,
                block_size,
                scaling_factor,
                nodata_value,
                uint8_mode=(cm == 5),
                src_dtype=int_dtype,
                nodata_guard=nodata_guard,
            )

        write_h = min(block_size, H - y)
        write_w = min(block_size, W - x)
        if write_h <= 0 or write_w <= 0:
            continue

        window = Window(x, y, write_w, write_h)
        dst.write(block[:write_h, :write_w].astype(np.float32), band_idx, window=window)


def decompress_image(
    dcv_compress: Dict[str, Any],
    config: Optional[Config] = None,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> str:
    """Decompress image from in-memory compression data dict. Returns output path."""
    if config is None:
        config = load_config()

    if output_dir is None:
        output_dir = config.results_dir

    os.makedirs(output_dir, exist_ok=True)

    if "n" in dcv_compress:
        # flat format
        block_size = dcv_compress.get("n", config.compression.block_size)
        version = dcv_compress.get("v", 1)
        num_channels = dcv_compress.get("ch", 1)
        original_shape = dcv_compress.get("s", [0, 0])
        padded_shape = dcv_compress.get("p", original_shape)
        transform_list = dcv_compress.get("t", [1, 0, 0, 0, 1, 0])
        crs_str = dcv_compress.get("r", config.compression.crs)
        nodata_value = dcv_compress.get("d", -9999)
        # Separate: internal nodata for block reconstruction vs output nodata
        # If source had no nodata (d=None, dn=False), use -9999 internally but write None to GeoTIFF
        internal_nodata = nodata_value if nodata_value is not None else -9999.0
        output_nodata = nodata_value  # None if source had no nodata
        scaling_factor = dcv_compress.get("sf", dcv_compress.get("f", config.scaling_factor))
        is_flat_format = True
    else:
        # legacy format
        block_size = dcv_compress.get("block_size", config.compression.block_size)
        version = 1
        original_shape = dcv_compress.get("original_shape", [0, 0])
        padded_shape = dcv_compress.get("padded_shape", original_shape)
        transform_list = dcv_compress.get("transform", [1, 0, 0, 0, 1, 0])
        crs_str = dcv_compress.get("crs", config.compression.crs)
        nodata_value = dcv_compress.get("nodata", -9999)
        scaling_factor = dcv_compress.get("scaling_factor", config.scaling_factor)
        num_channels = 1
        is_flat_format = False
        internal_nodata = nodata_value
        output_nodata = nodata_value

    H, W = original_shape
    padded_H, padded_W = padded_shape

    # Decide: streaming vs in-memory (>256 MB per channel → streaming)
    channel_bytes = H * W * 4
    use_streaming = channel_bytes > STREAMING_CHANNEL_BYTES
    if use_streaming and is_flat_format:
        # cm=6 (per-block RGB) is decoded in strips too, unless its streams are not in the DC/AC form
        use_streaming = all(
            _rgb_streamable(ch, *_rgb_geometry(ch, block_size, (H, W)))
            for ch in dcv_compress.get("channels", [])
            if ch.get("cm") == 6 and "c" in ch
        )

    if not config.output.quiet:
        mode_str = "streaming" if use_streaming else "in-memory"
        print(
            f"  Decompressing: {W}x{H}, channels={num_channels}, block_size={block_size}, Mode: {mode_str}"
        )

    if len(transform_list) >= 6:
        transform = Affine(
            transform_list[0],
            transform_list[1],
            transform_list[2],
            transform_list[3],
            transform_list[4],
            transform_list[5],
        )
    else:
        transform = Affine.identity()

    # CRS chosen at compression time (input CRS, config CRS or none); None = GeoTIFF without CRS
    crs = crs_from_header(crs_str)

    if output_filename is None:
        output_filename = "decompressed.tif"

    output_path = os.path.join(output_dir, output_filename)

    tiff_compress = config.output.tiff_compression.upper()
    if tiff_compress not in ("DEFLATE", "LZW", "LZMA", "ZSTD", "NONE"):
        tiff_compress = "DEFLATE"

    is_uint8_mode = False
    if is_flat_format:
        if num_channels == 1:
            cm = dcv_compress.get("cm", 0)
            is_uint8_mode = cm == 5 or cm == 4
        else:
            channel_list = dcv_compress.get("channels", [])
            if channel_list:
                is_uint8_mode = all(ch.get("cm", 0) in (5, 4, 6) for ch in channel_list)

    out_dtype = np.uint8 if is_uint8_mode else np.float32
    if is_uint8_mode and is_flat_format:
        # cm=5 channels may hold uint16 / int8 / int16 (header "dt"); use the common integer type
        if num_channels == 1:
            int_dts = [dcv_compress.get("dt", "uint8")]
        else:
            ch_list = dcv_compress.get("channels", [])
            int_dts = [ch.get("dt", "uint8") for ch in ch_list if ch.get("cm") == 5]
            if any(ch.get("cm") in (4, 6) for ch in ch_list):
                int_dts.append("uint8")
        if int_dts:
            out_dtype = np.result_type(*[np.dtype(d) for d in int_dts])
    out_nodata = output_nodata
    if out_nodata is not None and np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        if not (info.min <= out_nodata <= info.max):
            out_nodata = None  # e.g. -9999 declared on uint8/uint16 data: no pixel can carry it

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=num_channels,
        dtype=out_dtype,
        crs=crs,
        transform=transform,
        nodata=out_nodata,
        compress=tiff_compress if tiff_compress != "NONE" else None,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        if use_streaming and is_flat_format:
            if num_channels == 1:
                ch_data = {
                    "l": dcv_compress.get("l", []),
                    "c": dcv_compress.get("c", []),
                    "m": dcv_compress.get("m", []),
                    "cm": dcv_compress.get("cm", 0),
                    "dt": dcv_compress.get("dt", "uint8"),
                }
                _decompress_single_channel_streaming(
                    ch_data,
                    block_size,
                    (H, W),
                    internal_nodata,
                    scaling_factor,
                    version,
                    config,
                    dst,
                    1,
                    nodata_guard=output_nodata,
                )
            else:
                channel_list = dcv_compress.get("channels", [])
                band = 1  # a cm=6 entry fills three bands, its marker entries none
                rgb_decoded = False
                for ch_data in channel_list:
                    if ch_data.get("cm") == 6:
                        if not rgb_decoded and "c" in ch_data:
                            _decode_rgb_streaming(
                                ch_data,
                                *_rgb_geometry(ch_data, block_size, (H, W)),
                                (H, W),
                                dst,
                                band,
                                output_nodata,
                                out_dtype,
                                config.output.quiet,
                            )
                            band += 3
                            rgb_decoded = True
                        continue
                    _decompress_single_channel_streaming(
                        ch_data,
                        block_size,
                        (H, W),
                        internal_nodata,
                        scaling_factor,
                        version,
                        config,
                        dst,
                        band,
                        nodata_guard=output_nodata,
                    )
                    band += 1
        else:
            channels = []

            if is_flat_format and num_channels == 3:
                channel_list = dcv_compress.get("channels", [])
                if channel_list and channel_list[0].get("cm") == 6:
                    ch_data = channel_list[0]
                    rgb_channels = _decode_dct_channel_rgb(
                        ch_data, *_rgb_geometry(ch_data, block_size, (H, W))
                    )
                    rgb_planes = [rgb_channels[c][:H, :W] for c in ("R", "G", "B")]
                    _restore_rgb_nodata(*rgb_planes, ch_data, output_nodata)
                    channels.extend(rgb_planes)
                else:
                    for ch_idx, ch_data in enumerate(channel_list):
                        ch_result = _decompress_single_channel(
                            ch_data,
                            block_size,
                            (H, W),
                            internal_nodata,
                            scaling_factor,
                            version,
                            nodata_guard=output_nodata,
                        )
                        channels.append(ch_result)
            elif is_flat_format and num_channels == 1:
                ch_data = {
                    "l": dcv_compress.get("l", []),
                    "c": dcv_compress.get("c", []),
                    "m": dcv_compress.get("m", []),
                    "cm": dcv_compress.get("cm", 0),
                    "dt": dcv_compress.get("dt", "uint8"),
                }
                ch_result = _decompress_single_channel(
                    ch_data,
                    block_size,
                    (H, W),
                    internal_nodata,
                    scaling_factor,
                    version,
                    nodata_guard=output_nodata,
                )
                channels.append(ch_result)
            elif is_flat_format and num_channels > 1:
                channel_list = dcv_compress.get("channels", [])
                rgb_decoded = False
                for ch_idx, ch_data in enumerate(channel_list):
                    if ch_data.get("cm") == 6:
                        if not rgb_decoded and "c" in ch_data:
                            if not config.output.quiet:
                                print(f"  Channel {ch_idx + 1}/{num_channels} (RGB per-block):")
                            rgb_channels = _decode_dct_channel_rgb(
                                ch_data, *_rgb_geometry(ch_data, block_size, (H, W))
                            )
                            rgb_planes = [rgb_channels[c][:H, :W] for c in ("R", "G", "B")]
                            _restore_rgb_nodata(*rgb_planes, ch_data, output_nodata)
                            channels.extend(rgb_planes)
                            rgb_decoded = True
                        # Skip marker cm=6 entries (they have no "c" key)
                        continue
                    if not config.output.quiet:
                        print(f"  Channel {ch_idx + 1}/{num_channels}:")
                    ch_result = _decompress_single_channel(
                        ch_data,
                        block_size,
                        (H, W),
                        internal_nodata,
                        scaling_factor,
                        version,
                        nodata_guard=output_nodata,
                    )
                    channels.append(ch_result)
            else:
                # legacy format (single channel only)
                compressed_blocks = dcv_compress.get("compressed_blocks", [])
                result = np.zeros((padded_H, padded_W), dtype=np.float32)
                blocks_per_row = padded_W // block_size

                for idx, compressed_data in enumerate(
                    tqdm(
                        compressed_blocks,
                        desc="Dekompresja",
                        disable=config.output.quiet,
                    )
                ):
                    if compressed_data is None:
                        continue

                    row = idx // blocks_per_row
                    col = idx % blocks_per_row
                    y = row * block_size
                    x = col * block_size

                    try:
                        block = reconstruct_block(
                            compressed_data, block_size, scaling_factor, internal_nodata
                        )
                        result[y : y + block_size, x : x + block_size] = block
                    except Exception as e:
                        if not config.output.quiet:
                            print(f"  Warning: Block {idx} reconstruction failed: {e}")

                channels.append(result[:H, :W])

            # Inverse YCbCr if channels were decorrelated during compression
            # Skip for cm=6 (per-block RGB) - channels are already R,G,B
            ycbcr_rgb = dcv_compress.get("ycbcr_rgb", [])
            has_cm6 = any(ch.get("cm") == 6 for ch in dcv_compress.get("channels", []))
            if len(ycbcr_rgb) == 3 and not has_cm6:
                r0, g0, b0 = ycbcr_rgb
                Y = channels[r0].astype(np.float32)
                Cb = channels[g0].astype(np.float32)
                Cr = channels[b0].astype(np.float32)
                R, G, B = ycbcr_to_rgb(Y, Cb, Cr)
                channels[r0] = R.astype(np.float32)
                channels[g0] = G.astype(np.float32)
                channels[b0] = B.astype(np.float32)

            for ch_idx, ch_result in enumerate(channels):
                if np.issubdtype(out_dtype, np.integer):
                    write_data = np.round(ch_result).astype(out_dtype)
                else:
                    write_data = ch_result.astype(out_dtype)
                dst.write(write_data, ch_idx + 1)

    if not config.output.quiet:
        print(f"  Saved: {output_path}")

    return output_path


def _find_payload_member(archive_path: str) -> Optional[str]:
    """Name of the cm=7/cm=8 container (.fpred/.sz3) stored in a .7z archive, or None."""
    with py7zr.SevenZipFile(archive_path, "r") as archive:
        for name in archive.getnames():
            if name.lower().endswith((".fpred", ".sz3")):
                return name
    return None


def _decode_archived_payload(archive_path: str, member: str, output_dir: str) -> str:
    """Extract the container from a .7z archive and decode it to <archive stem>.tif."""
    import tempfile

    base_name = os.path.splitext(os.path.basename(archive_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.tif")
    with tempfile.TemporaryDirectory() as tmp_dir:
        with py7zr.SevenZipFile(archive_path, "r") as archive:
            archive.extract(path=tmp_dir, targets=[member])
        payload = os.path.join(tmp_dir, member)
        if member.lower().endswith(".fpred"):
            from compress.compression_pred import decompress_pred

            return decompress_pred(payload, output_path)
        from compress.compression_sz3 import decompress_sz3

        return decompress_sz3(payload, output_path)


def decompress_file(
    file_path: str, config: Optional[Config] = None, output_dir: Optional[str] = None
) -> str:
    """Decompress .7z archive to GeoTIFF. Returns output path."""
    if config is None:
        config = load_config()

    if output_dir is None:
        output_dir = config.results_dir

    os.makedirs(output_dir, exist_ok=True)

    if file_path.lower().endswith(".fpred"):
        from compress.compression_pred import decompress_pred

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.tif")
        return decompress_pred(file_path, output_path)

    if file_path.lower().endswith(".sz3"):
        from compress.compression_sz3 import decompress_sz3

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.tif")
        return decompress_sz3(file_path, output_path)

    if file_path.lower().endswith(".7z"):
        member = _find_payload_member(file_path)
        if member is not None:
            return _decode_archived_payload(file_path, member, output_dir)
        from compress.compression_lossless import decompress_lossless, is_lossless_archive

        if is_lossless_archive(file_path):
            return decompress_lossless(file_path, output_dir, config)

    if not config.output.quiet:
        print(f"Loading: {file_path}")

    dcv_compress = load_compressed_data(file_path)

    # .fpred / .sz3 wrapped in a 7z archive
    if isinstance(dcv_compress, dict) and dcv_compress.get("mode") in {"fpred", "sz3"}:
        import tempfile

        mode = dcv_compress.get("mode")
        if mode == "fpred":
            from compress.compression_pred import decompress_pred

            fpred_bytes = dcv_compress.get("fpred_bytes")
            if fpred_bytes is None:
                raise ValueError("Corrupt .fpred archive payload: missing compressed data")

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.tif")
            with tempfile.NamedTemporaryFile(suffix=".fpred", delete=False) as tmp:
                tmp.write(fpred_bytes)
                fpred_path = tmp.name
            try:
                return decompress_pred(fpred_path, output_path)
            finally:
                if os.path.exists(fpred_path):
                    os.remove(fpred_path)

        from compress.compression_sz3 import decompress_sz3
        sz3_bytes = dcv_compress.get("sz3_bytes")
        if sz3_bytes is None:
            raise ValueError("Corrupt .sz3 archive payload: missing compressed data")

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.tif")
        with tempfile.NamedTemporaryFile(suffix=".sz3", delete=False) as tmp:
            tmp.write(sz3_bytes)
            sz3_path = tmp.name
        try:
            return decompress_sz3(sz3_path, output_path)
        finally:
            if os.path.exists(sz3_path):
                os.remove(sz3_path)

    # Format B: single GeoTIFF (JPEG GeoTIFF, legacy)
    if isinstance(dcv_compress, str) and os.path.exists(dcv_compress):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.tif")
        shutil.move(dcv_compress, output_path)
        if not config.output.quiet:
            print(f"  Saved: {output_path}")
        return output_path

    # Format A: jpeg_strips (rgb.bin + meta.msgpack)
    if isinstance(dcv_compress, dict) and dcv_compress.get("mode") == "jpeg_strips":
        from turbojpeg import TJPF_RGB
        from compress.turbojpeg_loader import open_turbojpeg
        from engine.utils import decode_mask_rle

        meta = dcv_compress
        rgb_bin = meta["_rgb_bin"]
        strip_sizes = meta["strip_sizes"]
        strip_offsets = meta["strip_offsets"]
        H, W = meta["original_shape"]
        num_bands = meta["num_bands"]
        nodata_rle = meta.get("nodata_rle")
        src_nodata = meta.get("nodata")
        crs_str = meta.get("crs")
        t = meta["transform"]
        crs = crs_from_header(crs_str)

        nodata_mask = None
        if nodata_rle:
            nodata_mask = decode_mask_rle(nodata_rle, (H, W))

        strips = []
        pos = 0
        for sz in strip_sizes:
            strips.append(rgb_bin[pos : pos + sz])
            pos += sz
        del rgb_bin

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.tif")

        tiff_compress = config.output.tiff_compression.upper()
        if tiff_compress not in ("DEFLATE", "LZW", "LZMA", "ZSTD", "NONE"):
            tiff_compress = "DEFLATE"

        tj = open_turbojpeg()

        if not config.output.quiet:
            print(f"  Allocating {H*W*num_bands/1024/1024:.0f} MB RAM...")
        full_img = np.empty((H, W, num_bands), dtype=np.uint8)

        pbar = (
            None
            if config.output.quiet
            else tqdm(total=len(strips), desc="Decompress", unit="strip", ascii=False)
        )

        for i, jpg_bytes in enumerate(strips):
            y_start = strip_offsets[i]
            rgb = tj.decode(jpg_bytes, pixel_format=TJPF_RGB)  # (strip_h, padded_W, 3) RGB
            strip_h = min(rgb.shape[0], H - y_start)
            if strip_h <= 0:
                continue
            full_img[y_start : y_start + strip_h, :, :] = rgb[:strip_h, :W, :]
            del rgb
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        del strips

        if nodata_mask is not None:
            fill = int(src_nodata) if src_nodata is not None and 0 <= src_nodata <= 255 else 0
            full_img[nodata_mask] = fill

        if not config.output.quiet:
            print("  Writing GeoTIFF...")

        out_nodata = src_nodata if src_nodata is not None and 0 <= src_nodata <= 255 else None
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=num_bands,
            dtype=np.uint8,
            crs=crs,
            transform=Affine(*t[:6]),
            nodata=out_nodata,
            compress=tiff_compress if tiff_compress != "NONE" else None,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(np.moveaxis(full_img, -1, 0))
        del full_img

        if not config.output.quiet:
            print(f"  Saved: {output_path}")
        return output_path

    # Format C: DCT msgpack path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}.tif"
    return decompress_image(dcv_compress, config, output_dir, output_filename)


__all__ = [
    "load_compressed_data",
    "decompress_image",
    "decompress_file",
]
