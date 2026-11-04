"""
FIDWAC v2 - lossless mode (--lossless, cm=4)
=============================================
The raster is stored without any change of values, in a standard container packed into .7z:

  - uint8 with exactly 3 bands (RGB): PNG + world file (.pgw)
  - every other raster (float, int8/int16/uint16, uint8 with 1 or 4+ bands): GeoTIFF with
    DEFLATE and a predictor (2 = horizontal differencing for integers, 3 = floating point)

lossless_meta.msgpack keeps CRS, transform, NoData and band types, so decompression always writes
a GeoTIFF with the full georeferencing. The mode only brings the data into the FIDWAC archive;
the size reduction is that of PNG / DEFLATE, i.e. little or none when the input file is already
compressed. The archive is tagged LF only after the written file was read back and compared with
the source value by value.

Archives written before this mode (cm=4 msgpack: PNG Sub filter + zlib per band) are still decoded
by compress/decompression.py.
"""

import os
import shutil
import tempfile
import time

import msgpack
import numpy as np
import py7zr
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from compress.compression_utils import pack_to_7z, write_msgpack
from compress.output_crs import crs_from_header, crs_label

CM = 4
META_NAME = "lossless_meta.msgpack"
_STRIP_ROWS = 1024


def _use_png(src) -> bool:
    return src.count == 3 and all(d == "uint8" for d in src.dtypes)


def _strips(ds):
    for y in range(0, ds.height, _STRIP_ROWS):
        yield Window(0, y, ds.width, min(_STRIP_ROWS, ds.height - y))


def _world_file(path, t: Affine) -> None:
    """ESRI world file: pixel size, rotation, and the centre of the upper-left pixel."""
    lines = [t.a, t.d, t.b, t.e, t.c + t.a / 2 + t.b / 2, t.f + t.d / 2 + t.e / 2]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(repr(float(v)) for v in lines) + "\n")


def _write_png(src, path) -> None:
    # The PNG driver writes through a copy, so the whole raster is read at once.
    # CRS and transform are kept in the world file and the meta file, not in a GDAL .aux.xml.
    data = src.read()
    with rasterio.open(path, "w", driver="PNG", height=src.height, width=src.width,
                       count=src.count, dtype="uint8", ZLEVEL=9) as dst:
        dst.write(data)


def _write_geotiff(src, path, crs, transform, dtype) -> None:
    nodata = src.nodata
    if nodata is not None and np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if not info.min <= nodata <= info.max:
            nodata = None
    profile = {
        "driver": "GTiff", "height": src.height, "width": src.width, "count": src.count,
        "dtype": dtype, "crs": crs, "transform": transform, "nodata": nodata,
        "compress": "DEFLATE", "zlevel": 9, "predictor": 3 if dtype.kind == "f" else 2,
        "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        for win in _strips(src):
            dst.write(src.read(window=win).astype(dtype, copy=False), window=win)


def _identical(src, path) -> bool:
    """True when the written file holds exactly the source values (NaN compared as equal)."""
    with rasterio.open(path) as chk:
        if (chk.count, chk.height, chk.width) != (src.count, src.height, src.width):
            return False
        for win in _strips(src):
            a = src.read(window=win)
            b = chk.read(window=win)
            if a.dtype.kind == "f" or b.dtype.kind == "f":
                if not np.array_equal(a.astype(np.float64), b.astype(np.float64), equal_nan=True):
                    return False
            elif not np.array_equal(a, b):
                return False
    return True


def compress_lossless(src, file_path, out_crs, transform, config, output_dir, start_time) -> str:
    """Write the lossless archive for an open raster. Returns the path of the .7z archive."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    png = _use_png(src)
    dtype = np.result_type(*[np.dtype(d) for d in src.dtypes])
    quiet = config.output.quiet
    if not quiet:
        kind = "PNG + world file" if png else f"GeoTIFF DEFLATE, predictor {3 if dtype.kind == 'f' else 2}"
        print(f"  Lossless: {src.count} band(s) {', '.join(sorted(set(src.dtypes)))} -> {kind}")

    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        payload = os.path.join(tmp, f"{stem}.{'png' if png else 'tif'}")
        files = [payload]
        if png:
            _write_png(src, payload)
            pgw = os.path.join(tmp, f"{stem}.pgw")
            _world_file(pgw, transform)
            files.append(pgw)
        else:
            _write_geotiff(src, payload, out_crs, transform, dtype)

        exact = _identical(src, payload)
        meta = {
            "mode": "lossless",
            "cm": CM,
            "container": "png" if png else "tif",
            "payload": os.path.basename(payload),
            "shape": [src.height, src.width],
            "count": src.count,
            "dtypes": list(src.dtypes),
            "crs": None if out_crs is None else out_crs.to_wkt(),
            "transform": list(transform)[:6],
            "nodata": src.nodata,
        }
        meta_path = os.path.join(tmp, META_NAME)
        write_msgpack(meta_path, meta)
        files.append(meta_path)

        tag = "LF" if exact else "VF"
        output_path = os.path.join(output_dir, f"{stem}_Lossless_CRS{crs_label(out_crs)}_{tag}.7z")
        pack_to_7z(output_path, *files, method=config.output.compression_method.upper(),
                   lzma2_block_size=config.output.lzma2_block_size)

    if not quiet:
        size_in = os.path.getsize(file_path)
        size_out = os.path.getsize(output_path)
        print(f"  Input file: {size_in / 1024:.1f} KB, archive: {size_out / 1024:.1f} KB "
              f"({size_in / max(size_out, 1):.2f}x), identical: {exact}, time: {time.time() - start_time:.2f}s")
        if not exact:
            print("  WARNING: the written file differs from the source; archive tagged VF")
    return output_path


def is_lossless_archive(archive_path: str) -> bool:
    with py7zr.SevenZipFile(archive_path, "r") as archive:
        return META_NAME in archive.getnames()


def decompress_lossless(archive_path: str, output_dir: str, config) -> str:
    """Decode a lossless archive to <archive stem>.tif (GeoTIFF with CRS, transform and NoData)."""
    base = os.path.splitext(os.path.basename(archive_path))[0]
    output_path = os.path.join(output_dir, f"{base}.tif")
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        with py7zr.SevenZipFile(archive_path, "r") as archive:
            archive.extractall(path=tmp)
        with open(os.path.join(tmp, META_NAME), "rb") as f:
            meta = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        payload = os.path.join(tmp, meta["payload"])

        if meta["container"] == "tif":
            shutil.move(payload, output_path)
        else:
            tiff_compress = config.output.tiff_compression.upper()
            if tiff_compress not in ("DEFLATE", "LZW", "LZMA", "ZSTD", "NONE"):
                tiff_compress = "DEFLATE"
            nodata = meta.get("nodata")
            if nodata is not None and not 0 <= nodata <= 255:
                nodata = None
            with rasterio.open(payload) as png:
                profile = {
                    "driver": "GTiff", "height": png.height, "width": png.width,
                    "count": png.count, "dtype": "uint8", "crs": crs_from_header(meta.get("crs")),
                    "transform": Affine(*meta["transform"]), "nodata": nodata,
                    "compress": None if tiff_compress == "NONE" else tiff_compress,
                    "tiled": True, "blockxsize": 256, "blockysize": 256,
                }
                with rasterio.open(output_path, "w", **profile) as dst:
                    for win in _strips(png):
                        dst.write(png.read(window=win), window=win)

    if not config.output.quiet:
        print(f"  Lossless archive ({meta['container']}) -> {output_path}")
    return output_path


__all__ = ["compress_lossless", "is_lossless_archive", "decompress_lossless", "META_NAME"]
