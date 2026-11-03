"""
cm=7: near-lossless predictive coding for float and integer rasters.

Encoding steps
--------------
1. Fill NoData pixels with the nearest valid pixel. The NoData mask is stored separately.
2. Quantize the pixel values uniformly with step Q = 2 * bound * SAFETY, where bound is the codec
   bound derived from the requested accuracy (compress.float_storage). The maximum pixel error is
   Q / 2 and stays within the bound. This is the only lossy step; the error is measured after decoding.
3. Predict the integer quantization indices with MED (LOCO-I). Indices and residuals are integers,
   so the prediction chain is reversible and adds no further error.
4. Map the residuals to unsigned integers (zigzag) and pack them with the smaller of two codecs
   (see _encode_residuals): LZMA2 over a byte stream with an escape list for large values, or bzip2
   over 16- or 32-bit words. The NoData mask is packed with packbits and compressed with zlib.

File format (.fpred)
--------------------
Each band is one payload: magic b'FPRD' | uint32 header_len | JSON header | uint64 len_res
| res_payload | uint64 len_mask | mask_zlib [| uint64 len_special | special_codes]. The bands of a
multi-band raster are stored one after another by compress/multiband.py.

The header field "res_codec" selects the residual codec: "lzma-be" or "bz2-zz". Files without this
field were written with zlib over int16/int32 residuals and are still decoded.
"""
import bz2
import json
import lzma
import os
import struct
import zlib

import numpy as np
import rasterio
from numba import njit
from scipy import ndimage

from compress.float_storage import codec_bound
from compress.multiband import pack_bands, split_bands
from compress.output_crs import resolve_output_crs
from compress.special_values import apply_special, encode_special

CM = 7
MAGIC = b"FPRD"
SAFETY = 0.999
_ESC = 255  # Zigzag values >= _ESC go to the escape list of the "lzma-be" codec.


def _med_pred_vec(q):
    """Vectorized MED prediction of integer quantization indices (encoder side).

    Neighbours outside the raster are taken as 0, as in _med_decode.
    """
    a = np.zeros_like(q)
    b = np.zeros_like(q)
    c = np.zeros_like(q)
    a[:, 1:] = q[:, :-1]
    b[1:, :] = q[:-1, :]
    c[1:, 1:] = q[:-1, :-1]
    mx = np.maximum(a, b)
    mn = np.minimum(a, b)
    return np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))


@njit(cache=True)
def _med_decode(res):
    """Sequential MED reconstruction of indices from residuals (decoder side)."""
    H, W = res.shape
    q = np.zeros((H, W), np.int32)
    for i in range(H):
        for j in range(W):
            a = q[i, j - 1] if j > 0 else 0
            b = q[i - 1, j] if i > 0 else 0
            c = q[i - 1, j - 1] if (i > 0 and j > 0) else 0
            mx = max(a, b)
            mn = min(a, b)
            if c >= mx:
                p = mn
            elif c <= mn:
                p = mx
            else:
                p = a + b - c
            q[i, j] = res[i, j] + p
    return q


def _cast_to_dtype(data: np.ndarray, dtype_Like) -> np.ndarray:
    """Convert a float64 array to the raster dtype. Integer types are rounded and clipped to their range."""
    out_dtype = np.dtype(dtype_Like)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        return np.clip(np.rint(data), info.min, info.max).astype(out_dtype)
    return data.astype(out_dtype)


def _zigzag(res):
    r = res.astype(np.int64)
    return ((r << 1) ^ (r >> 63)).astype(np.uint32)


def _unzigzag(z):
    z = z.astype(np.int64)
    return ((z >> 1) ^ -(z & 1)).astype(np.int32)


def _lzma_filters():
    # lc, lp and pb = 0: the input is a byte stream, so literal context, literal position and alignment bits give no gain.
    return [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME,
             "dict_size": 1 << 24, "lc": 0, "lp": 0, "pb": 0}]


def _encode_residuals(res):
    """Pack MED residuals with the smaller of two codecs. Returns (payload, header_fields)."""
    z = _zigzag(res).ravel()
    esc = z >= _ESC
    be = np.minimum(z, _ESC).astype(np.uint8).tobytes() + z[esc].astype("<u4").tobytes()
    best = (lzma.compress(be, format=lzma.FORMAT_RAW, filters=_lzma_filters()),
            {"res_codec": "lzma-be", "n_esc": int(esc.sum())})
    zz_dtype = "<u2" if z.max(initial=0) < 65536 else "<u4"
    alt = bz2.compress(z.astype(zz_dtype).tobytes(), 9)
    if len(alt) < len(best[0]):
        best = (alt, {"res_codec": "bz2-zz", "zz_dtype": zz_dtype})
    return best


def _decode_residuals(payload, h):
    H, W = h["H"], h["W"]
    codec = h.get("res_codec")
    if codec is None:  # written before the res_codec field existed: zlib over int16/int32 residuals
        return np.frombuffer(zlib.decompress(payload), dtype=h["res_dtype"]).reshape(H, W).astype(np.int32)
    if codec == "lzma-be":
        raw = lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=_lzma_filters())
        small = np.frombuffer(raw, dtype=np.uint8, count=H * W)
        z = small.astype(np.uint32)
        z[small == _ESC] = np.frombuffer(raw, dtype="<u4", offset=H * W, count=h["n_esc"])
    elif codec == "bz2-zz":
        z = np.frombuffer(bz2.decompress(payload), dtype=h["zz_dtype"])
    else:
        raise ValueError(f"unknown residual codec: {codec!r}")
    return _unzigzag(z).reshape(H, W)


def _encode_band(raw, eps, meta):
    """Encode one band. Returns the single-band payload and its maximum error over valid pixels."""
    nodata = meta["nodata"]
    dtype_name = meta["dtype"]
    H0, W0 = raw.shape
    valid = np.isfinite(raw)
    if nodata is not None:
        valid &= raw != nodata

    if valid.any() and not valid.all():
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = raw[idx[0], idx[1]]
    else:
        filled = np.where(valid, raw, 0.0)

    bound = codec_bound(float(eps), raw[valid], dtype_name)
    Q = 2.0 * bound * SAFETY
    q = np.round(filled / Q).astype(np.int32)
    res = q - _med_pred_vec(q)

    res_z, codec_fields = _encode_residuals(res)
    mask_z = zlib.compress(np.packbits(~valid).tobytes(), 9)
    special_z = encode_special(raw, ~valid)

    header = {
        "cm": CM, "H": H0, "W": W0, "eps": eps, "Q": Q,
        "nodata": nodata, "dtype": dtype_name,
        "crs_wkt": meta["crs_wkt"], "transform": meta["transform"], **codec_fields,
    }
    hb = json.dumps(header).encode()
    payload = b"".join([
        MAGIC + struct.pack("<I", len(hb)) + hb,
        struct.pack("<Q", len(res_z)) + res_z,
        struct.pack("<Q", len(mask_z)) + mask_z,
        # Optional trailing section with the NaN and inf codes.
        struct.pack("<Q", len(special_z)) + special_z if special_z is not None else b"",
    ])
    recon = _cast_to_dtype(q.astype(np.float64) * Q, dtype_name).astype(np.float64)
    max_err = float(np.abs(recon - raw)[valid].max()) if valid.any() else 0.0
    return payload, max_err


def compress_pred(src_path, out_path, eps, quiet=False, crs="", eps_int=None):
    """Encode every band separately into out_path.

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
            payload, err = _encode_band(src.read(band).astype(np.float64), band_eps, meta)
            payloads.append(payload)
            max_err = max(max_err, err)
            valid = valid and err <= band_eps
            if not quiet:
                print(f"  cm=7 band {band}/{src.count} eps={band_eps:g} max_err={err:.5f} payload={len(payload)/1e6:.2f} MB")
    with open(out_path, "wb") as f:
        f.write(pack_bands(payloads))
    return max_err, valid


def compress_pred_archive(src_path, out_path, eps, quiet=False, crs="", eps_int=None):
    """Write a cm=7 payload and optionally wrap it in a .7z archive.

    Returns a tuple (result_path, max_error, valid), where result_path is the final
    compressed artifact (.fpred or .7z), max_error is the measured error over
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
            payload = os.path.join(tmp_dir, f"{stem}.fpred")
            max_err, valid = compress_pred(src_path, payload, eps, quiet=quiet, crs=crs, eps_int=eps_int)
            pack_to_7z(out_path, payload, store=False, method="LZMA2")
        return out_path, max_err, valid

    if lower.endswith(".fpred"):
        max_err, valid = compress_pred(src_path, out_path, eps, quiet=quiet, crs=crs, eps_int=eps_int)
        return out_path, max_err, valid

    pred_path = os.path.splitext(out_path)[0] + ".fpred"
    max_err, valid = compress_pred(src_path, pred_path, eps, quiet=quiet, crs=crs, eps_int=eps_int)
    return pred_path, max_err, valid


def _parse_band(payload):
    """Header and sections of one single-band payload (the optional special section is last)."""
    if payload[:4] != MAGIC:
        raise ValueError("not an .fpred payload")
    (hl,) = struct.unpack_from("<I", payload, 4)
    header = json.loads(payload[8 : 8 + hl].decode())
    pos = 8 + hl
    (lr,) = struct.unpack_from("<Q", payload, pos)
    res_z = payload[pos + 8 : pos + 8 + lr]
    pos += 8 + lr
    (lm,) = struct.unpack_from("<Q", payload, pos)
    mask_z = payload[pos + 8 : pos + 8 + lm]
    pos += 8 + lm
    special_z = None
    if pos < len(payload):
        (ls,) = struct.unpack_from("<Q", payload, pos)
        special_z = payload[pos + 8 : pos + 8 + ls]
    return header, res_z, mask_z, special_z


def _decode_band(parsed):
    h, res_z, mask_z, special_z = parsed
    H, W = h["H"], h["W"]
    q = _med_decode(_decode_residuals(res_z, h))
    data = q.astype(np.float64) * h["Q"]
    mask = np.unpackbits(np.frombuffer(zlib.decompress(mask_z), np.uint8))[: H * W].reshape(H, W).astype(bool)
    out = _cast_to_dtype(data, h["dtype"])
    apply_special(out, mask, special_z, h["nodata"])
    return out


def decompress_pred(in_path, out_tif):
    from rasterio.transform import Affine

    with open(os.fspath(in_path), "rb") as f:
        bands = [_parse_band(payload) for payload in split_bands(f.read())]
    h = bands[0][0]
    profile = {
        "driver": "GTiff",
        "height": h["H"],
        "width": h["W"],
        "count": len(bands),
        "dtype": h["dtype"],
        "nodata": h["nodata"],
        "transform": Affine(*h["transform"]),
    }
    if h["crs_wkt"]:
        profile["crs"] = h["crs_wkt"]
    os.makedirs(os.path.dirname(os.path.abspath(out_tif)) or ".", exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        for band, parsed in enumerate(bands, start=1):
            dst.write(_decode_band(parsed), band)
    return out_tif
