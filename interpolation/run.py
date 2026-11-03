"""Pipeline for one point file: read, geoid correction, IDW grid, smoothing, exports and optional compression."""

import os
import subprocess
import sys
from typing import Callable, Dict, Iterable, Optional, Tuple

from config import InterpolationConfig
from interpolation import export, idw, points

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPRESS_PY = os.path.join(PROJECT_ROOT, "compress.py")
EXTRA_OUTPUTS = ("csv", "surfer", "shp", "png")


def run_interpolation(
    points_path: str,
    out_dir: str,
    icfg: InterpolationConfig,
    outputs: Iterable[str] = (),
    compress: Optional[Tuple[str, float]] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, str]:
    """Interpolate one point file into out_dir.

    outputs: extra files besides the GeoTIFF, a subset of EXTRA_OUTPUTS.
    compress: None, or (method, accuracy): compress the GeoTIFF with compress.py.
    Returns the written paths: geotiff, the extras, and compressed/validity when compressed.
    """
    outputs = set(outputs)
    unknown = outputs - set(EXTRA_OUTPUTS)
    if unknown:
        raise ValueError(f"Unknown outputs: {', '.join(sorted(unknown))}")
    stem = os.path.splitext(os.path.basename(points_path))[0]
    base = f"{stem}-N-{icfg.neighbours}-R-{icfg.resolution}-dist-{icfg.max_distance}"
    os.makedirs(out_dir, exist_ok=True)

    las_classes = icfg.filter_classes if icfg.filter_las else None
    xyz = points.read_points(points_path, z_field=icfg.z_field_name, las_classes=las_classes)
    if xyz.shape[0] == 0:
        raise ValueError(f"No valid points in {points_path}")
    log(f"Points: {xyz.shape[0]}")
    if icfg.geoid_correction:
        xyz, corrected = points.apply_geoid_correction(
            xyz, icfg.geoid_correction_file, icfg.geoid_crs, icfg.source_crs, icfg.geoid_max_distance, icfg.leafsize
        )
        log(f"Geoid correction applied to {corrected} points")

    grid = idw.make_grid(xyz[:, :2], icfg.resolution)
    log(f"Grid: {grid.nx} x {grid.ny} cells, resolution {icfg.resolution}")
    r_min = icfg.r_min if icfg.r_min_function else None
    raw = idw.interpolate_grid(
        xyz[:, :2], xyz[:, 2], grid, icfg.neighbours, icfg.max_distance, icfg.weights, icfg.leafsize, r_min
    )
    values = idw.smooth_gaussian(raw, icfg.smooth_level) if icfg.smooth_result else raw

    result: Dict[str, str] = {}
    geotiff = os.path.join(out_dir, base + "_geotif.tif")
    export.write_geotiff(geotiff, values, grid, icfg.source_crs)
    result["geotiff"] = geotiff
    if "csv" in outputs:
        result["csv"] = os.path.join(out_dir, base + "_result.csv")
        export.write_csv(result["csv"], raw, grid)
    if "surfer" in outputs:
        result["surfer"] = os.path.join(out_dir, base + "_grid.grd")
        export.write_surfer(result["surfer"], values, grid)
    if "shp" in outputs:
        result["shp"] = os.path.join(out_dir, base + "_points.shp")
        export.write_points_shapefile(result["shp"], xyz, icfg.source_crs)
    if "png" in outputs:
        result["png"] = os.path.join(out_dir, base + "_preview.png")
        export.write_preview(result["png"], values, stem)
    for key in ("csv", "surfer", "shp", "png"):
        if key in result:
            log(f"Wrote {key}: {result[key]}")
    log(f"Wrote GeoTIFF: {geotiff}")

    if compress is not None:
        method, accuracy = compress
        archive, validity = _compress_geotiff(geotiff, out_dir, method, accuracy)
        result["compressed"] = archive
        if validity:
            result["validity"] = validity
    return result


def _compress_geotiff(geotiff: str, out_dir: str, method: str, accuracy: float) -> Tuple[str, Optional[str]]:
    """Run compress.py on the GeoTIFF; returns (archive path, VT/VF tag)."""
    cmd = [
        sys.executable, COMPRESS_PY, "-i", os.path.abspath(geotiff), "-o", os.path.abspath(out_dir),
        "--method", method, "--accuracy", str(accuracy), "--quiet",
    ]
    proc = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    output, validity = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("Output file: "):
            output = line[len("Output file: "):]
        elif line.startswith("VALIDITY:"):
            validity = line[len("VALIDITY:"):]
    if proc.returncode != 0 or not output:
        errors = [line for line in proc.stdout.splitlines() if line.startswith("Error")]
        message = errors[-1] if errors else f"compression failed (exit code {proc.returncode})"
        raise RuntimeError(message.removeprefix("Error:").strip())
    return output, validity
