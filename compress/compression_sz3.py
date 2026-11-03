"""
cm=8: external SZ3 backend for rasters. Every band is encoded separately.

Each band is compressed with the SZ3 absolute error bound. The payload layout matches the cm=7
format: a JSON header, the compressed data, the NoData mask and an optional section for NaN and
inf codes. CRS, affine transform, NoData value and dtype are stored in the header, so the shared
decoder restores the raster.
"""

import json
import os
import struct
import zlib

import numpy as np
import rasterio

from compress.float_storage import codec_bound
from compress.multiband import pack_bands, split_bands
from compress.output_crs import resolve_output_crs
from compress.special_values import apply_special, encode_special
try:
    from pysz import sz, szConfig, szErrorBoundMode
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "SZ3 mode (cm=8) needs the pysz binding. Run install/setup_linux.sh "
        "(see the SZ3 section of install/README.md)."
    ) from exc

CM = 8
MAGIC = b"SZ3D"


def _pack_mask(valid: np.ndarray) -> bytes:
    return zlib.compress(np.packbits(valid.astype(np.uint8)).tobytes(), 9)


def _unpack_mask(mask_bytes: bytes, h: int, w: int) -> np.ndarray:
    buf = np.frombuffer(zlib.decompress(mask_bytes), dtype=np.uint8)
    bits = np.unpackbits(buf)
    return bits[: h * w].reshape((h, w)).astype(bool)


def _cast_to_dtype(data: np.ndarray, dtype_like) -> np.ndarray:
    """Convert a float64 array to the raster dtype. Integer types are rounded and clipped to their range."""
    out_dtype = np.dtype(dtype_like)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        return np.clip(np.rint(data), info.min, info.max).astype(out_dtype)
    return data.astype(out_dtype)


def _encode_band(raw, eps, meta):
    """Encode one band. Returns the payload, the maximum error over valid pixels and the SZ3 compression ratio."""
    nodata = meta["nodata"]
    dtype_name = meta["dtype"]
    valid = np.isfinite(raw)
    if nodata is not None:
        valid &= raw != nodata

    work = raw.astype(np.float64, copy=True)
    work[~valid] = 0.0

    bound = codec_bound(float(eps), raw[valid], dtype_name)
    cfg = szConfig()
    cfg.errorBoundMode = szErrorBoundMode.ABS
    cfg.absErrorBound = bound
    compressed, ratio = sz.compress(work, cfg)

    recon, _ = sz.decompress(compressed, np.float64, work.shape)
    recon_cast = _cast_to_dtype(recon, dtype_name).astype(np.float64)
    if nodata is not None:
        recon_cast[~valid] = nodata
    max_err = 0.0 if not valid.any() else float(np.max(np.abs(recon_cast[valid] - raw[valid])))

    mask_bytes = _pack_mask(valid)
    special_bytes = encode_special(raw, ~valid)
    header = {
        "cm": CM,
        "H": raw.shape[0],
        "W": raw.shape[1],
        "eps": float(eps),
        "bound": bound,
        "nodata": nodata,
        "dtype": dtype_name,
        "crs_wkt": meta["crs_wkt"],
        "transform": meta["transform"],
        "ratio": float(ratio),
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    comp_bytes = np.asarray(compressed, dtype=np.uint8).tobytes()
    payload = b"".join([
        MAGIC + struct.pack("<I", len(hb)) + hb,
        struct.pack("<Q", len(comp_bytes)) + comp_bytes,
        struct.pack("<Q", len(mask_bytes)) + mask_bytes,
        # Optional trailing section with the NaN and inf codes.
        struct.pack("<Q", len(special_bytes)) + special_bytes if special_bytes is not None else b"",
    ])
    return payload, max_err, ratio


def compress_sz3(src_path, out_path, eps, quiet=False, crs="", eps_int=None):
    """Encode every band with the SZ3 absolute error bound into out_path.

    Integer bands use eps_int when given (the uint8 accuracy bound in DN), float bands use eps.
    Returns (max_err, valid): max_err is the largest measured error over all bands; valid is
    True when every band's error stays within its own bound.
    """
    with rasterio.open(src_path) as src:
        out_crs = resolve_output_crs(src.crs, crs)
        payloads, max_err, valid = [], 0.0, True
        for band in range(1, src.count + 1):
            band_dtype = src.dtypes[band - 1]
            band_eps = (
                eps_int
                if (eps_int is not None and np.dtype(band_dtype).kind in "iu")
                else eps
            )
            meta = {
                "nodata": src.nodata,
                "dtype": band_dtype,
                "crs_wkt": None if out_crs is None else out_crs.to_wkt(),
                "transform": list(src.transform)[:6],
            }
            payload, err, ratio = _encode_band(src.read(band).astype(np.float64), band_eps, meta)
            payloads.append(payload)
            max_err = max(max_err, err)
            valid = valid and err <= band_eps
            if not quiet:
                print(
                    f"  cm=8 band {band}/{src.count} eps={band_eps:g} ratio={ratio:.2f} "
                    f"max_err={err:.5f} payload={len(payload)/1e6:.2f} MB"
                )
    with open(out_path, "wb") as f:
        f.write(pack_bands(payloads))
    return max_err, valid


def compress_sz3_archive(src_path, out_path, eps, quiet=False, crs="", eps_int=None):
    """Write a cm=8 payload and optionally wrap it in a .7z archive.

    Returns a tuple (result_path, max_error, valid), where result_path is the final
    compressed artifact (.sz3 or .7z), max_error is the measured error over
    valid pixels and valid is True when every band meets its bound.
    """
    out_path = os.fspath(out_path)
    lower = out_path.lower()
    if lower.endswith(".7z"):
        import tempfile

        from compress.compression_utils import pack_to_7z

        stem = os.path.splitext(os.path.basename(out_path))[0]
        base_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        with tempfile.TemporaryDirectory(dir=base_dir) as tmp_dir:
            payload = os.path.join(tmp_dir, f"{stem}.sz3")
            max_err, valid = compress_sz3(src_path, payload, eps, quiet=quiet, crs=crs, eps_int=eps_int)
            pack_to_7z(out_path, payload, store=False, method="LZMA2")
        return out_path, max_err, valid

    if lower.endswith(".sz3"):
        max_err, valid = compress_sz3(src_path, out_path, eps, quiet=quiet, crs=crs, eps_int=eps_int)
        return out_path, max_err, valid

    out_file = os.path.splitext(out_path)[0] + ".sz3"
    max_err, valid = compress_sz3(src_path, out_file, eps, quiet=quiet, crs=crs, eps_int=eps_int)
    return out_file, max_err, valid


def _parse_band(payload):
    if payload[:4] != MAGIC:
        raise ValueError("not an .sz3 payload")
    (header_len,) = struct.unpack_from("<I", payload, 4)
    header = json.loads(payload[8 : 8 + header_len].decode())
    pos = 8 + header_len
    (payload_len,) = struct.unpack_from("<Q", payload, pos)
    comp = payload[pos + 8 : pos + 8 + payload_len]
    pos += 8 + payload_len
    (mask_len,) = struct.unpack_from("<Q", payload, pos)
    mask_bytes = payload[pos + 8 : pos + 8 + mask_len]
    pos += 8 + mask_len
    special_bytes = None
    if pos < len(payload):
        (special_len,) = struct.unpack_from("<Q", payload, pos)
        special_bytes = payload[pos + 8 : pos + 8 + special_len]
    return header, comp, mask_bytes, special_bytes


def _decode_band(parsed):
    header, comp, mask_bytes, special_bytes = parsed
    h = int(header["H"])
    w = int(header["W"])
    valid = _unpack_mask(mask_bytes, h, w)
    data_arr, _ = sz.decompress(np.frombuffer(comp, dtype=np.uint8), np.float64, (h, w))
    out = _cast_to_dtype(data_arr.astype(np.float64), np.dtype(header["dtype"]))
    apply_special(out, ~valid, special_bytes, header.get("nodata"))
    return out


def decompress_sz3(in_path, out_tif):
    from rasterio.transform import Affine

    with open(os.fspath(in_path), "rb") as f:
        bands = [_parse_band(payload) for payload in split_bands(f.read())]
    header = bands[0][0]
    profile = {
        "driver": "GTiff",
        "height": int(header["H"]),
        "width": int(header["W"]),
        "count": len(bands),
        "dtype": np.dtype(header["dtype"]).name,
        "nodata": header.get("nodata"),
        "transform": Affine(*header["transform"]),
    }
    if header.get("crs_wkt"):
        profile["crs"] = header["crs_wkt"]

    os.makedirs(os.path.dirname(os.path.abspath(out_tif)) or ".", exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        for band, parsed in enumerate(bands, start=1):
            dst.write(_decode_band(parsed), band)
    return out_tif
