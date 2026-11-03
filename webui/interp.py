"""Web GUI support for IDW interpolation: form defaults, validation, interpolate.py arguments, input collection."""

import dataclasses
import os
from typing import Any, Dict, List, Optional

from rasterio.crs import CRS

from config import InterpolationConfig, load_config, save_config
from interpolation.points import BUNDLED_GEOID, POINT_EXTENSIONS, resolve_geoid_path
from webui.settings import CONFIG_PATH, PROJECT_ROOT

INTERP_CLI = os.path.join(PROJECT_ROOT, "interpolate.py")
GEOID_EXTENSIONS = (".7z", ".csv", ".txt")
PIPELINES = ("interp_only", "interp_compress")
METHODS = ("dct", "pred", "sz3")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "results", "interpolation")
BUNDLED_GEOID_PATH = resolve_geoid_path(BUNDLED_GEOID)


def browse_extensions(mode: str, default):
    """File types shown by the folder browser in a file mode."""
    if mode == "points":
        return POINT_EXTENSIONS
    if mode == "geoid":
        return GEOID_EXTENSIONS
    return default


def form_defaults(icfg: InterpolationConfig) -> Dict[str, Any]:
    """Initial values of the Interpolation tab, taken from config.json."""
    return {
        "ip_input_type": "file",
        "ip_input_path": "",
        "ip_output_dir": DEFAULT_OUTPUT,
        "ip_pipeline": "interp_only",
        "ip_neighbours": icfg.neighbours,
        "ip_resolution": icfg.resolution,
        "ip_max_distance": icfg.max_distance,
        "ip_weights": icfg.weights,
        "ip_leafsize": icfg.leafsize,
        "ip_smooth": icfg.smooth_result,
        "ip_smooth_level": icfg.smooth_level,
        "ip_r_min_function": icfg.r_min_function,
        "ip_r_min": icfg.r_min,
        "ip_las_filter": icfg.filter_las,
        "ip_las_classes": ",".join(str(c) for c in icfg.filter_classes),
        "ip_z_field": icfg.z_field_name,
        "ip_geoid": icfg.geoid_correction,
        "ip_geoid_file": os.path.join(PROJECT_ROOT, icfg.geoid_correction_file),
        "ip_geoid_crs": icfg.geoid_crs,
        "ip_source_crs": icfg.source_crs,
        "ip_geoid_max_distance": icfg.geoid_max_distance,
        "ip_csv": icfg.csv_result,
        "ip_surfer": icfg.surfer_grd,
        "ip_shp": icfg.save_points_shp,
        "ip_png": icfg.preview_png,
    }


def _number(data: Dict[str, Any], key: str, lo: float, hi: float, integer: bool = False):
    label = key.removeprefix("ip_").replace("_", " ")
    try:
        value = int(data.get(key)) if integer else float(data.get(key))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a {'whole number' if integer else 'number'}.") from None
    if not lo <= value <= hi:
        raise ValueError(f"{label} must be between {lo} and {hi}.")
    return value


def _absolute(value: Any) -> str:
    text = str(value or "").strip()
    return os.path.abspath(os.path.expanduser(text)) if text else ""


def geoid_info(path: str) -> Dict[str, Any]:
    """State of a geoid model path for the Interpolation tab, resolved the same way as a job."""
    full = _absolute(path)
    if not full:
        return {"path": "", "exists": False, "supported": False, "size": None}
    exists = os.path.isfile(full)
    return {
        "path": full,
        "exists": exists,
        "supported": full.lower().endswith(GEOID_EXTENSIONS),
        "size": os.path.getsize(full) if exists else None,
    }


def _crs(value: str, label: str) -> str:
    try:
        CRS.from_user_input(value)
    except Exception:
        raise ValueError(f"{label}: unknown CRS '{value}' (for example epsg:2180).") from None
    return value


def _classes(text: Any) -> List[int]:
    try:
        classes = [int(part) for part in str(text or "").split(",") if part.strip()]
    except ValueError:
        raise ValueError("LAS classes must be numbers separated by commas, for example 2,6.") from None
    if not classes or any(not 0 <= c <= 255 for c in classes):
        raise ValueError("LAS classes must be numbers from 0 to 255, for example 2,6.")
    return classes


def _classes_or_default(text: Any) -> List[int]:
    """Classes stored when the LAS filter is off: the typed list when valid, else the ground class 2."""
    try:
        return _classes(text)
    except ValueError:
        return [2]


def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the Interpolation tab (and the method and accuracy of the Compression tab)."""
    input_type = str(data.get("ip_input_type", ""))
    if input_type not in ("file", "directory"):
        raise ValueError("Input type must be a point file or a folder.")
    input_path = _absolute(data.get("ip_input_path"))
    if not input_path:
        raise ValueError("Select an input point file or folder.")
    if input_type == "file" and not os.path.isfile(input_path):
        raise ValueError(f"Input file not found: {input_path}")
    if input_type == "file" and not input_path.lower().endswith(POINT_EXTENSIONS):
        raise ValueError("Input file must be a point file: TXT, CSV, LAS, LAZ or SHP.")
    if input_type == "directory" and not os.path.isdir(input_path):
        raise ValueError(f"Input folder not found: {input_path}")
    output_dir = _absolute(data.get("ip_output_dir"))
    if not output_dir:
        raise ValueError("Select an output folder.")
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError(f"Output path is not a folder: {output_dir}")
    pipeline = str(data.get("ip_pipeline", ""))
    if pipeline not in PIPELINES:
        raise ValueError("Choose the pipeline: interpolation only, or interpolation and compression.")

    las_filter = bool(data.get("ip_las_filter"))
    geoid = bool(data.get("ip_geoid"))
    geoid_file = _absolute(data.get("ip_geoid_file"))
    if geoid and not os.path.isfile(geoid_file):
        raise ValueError(f"Geoid file not found: {geoid_file}")
    method, accuracy = None, None
    if pipeline == "interp_compress":
        method = str(data.get("method", ""))
        if method not in METHODS:
            raise ValueError("Compression method must be dct, pred or sz3.")
        accuracy = _number(data, "accuracy", 0.001, 1.0)

    return {
        "kind": "interpolate",
        "input_type": input_type,
        "input_path": input_path,
        "output_dir": output_dir,
        "pipeline": pipeline,
        "neighbours": _number(data, "ip_neighbours", 1, 500, integer=True),
        "resolution": _number(data, "ip_resolution", 1e-6, 1e6),
        "max_distance": _number(data, "ip_max_distance", 1e-6, 1e9),
        "weights": _number(data, "ip_weights", 0.1, 10.0),
        "leafsize": _number(data, "ip_leafsize", 1, 10000, integer=True),
        "smooth": bool(data.get("ip_smooth")),
        "smooth_level": _number(data, "ip_smooth_level", 0.1, 20.0),
        "r_min_function": bool(data.get("ip_r_min_function")),
        "r_min": _number(data, "ip_r_min", 0.0, 1e9),
        "las_filter": las_filter,
        "las_classes": _classes(data.get("ip_las_classes")) if las_filter else _classes_or_default(data.get("ip_las_classes")),
        "z_field": str(data.get("ip_z_field") or "z").strip() or "z",
        "geoid": geoid,
        "geoid_file": geoid_file,
        "geoid_crs": _crs(str(data.get("ip_geoid_crs") or "epsg:4326").strip(), "Geoid CRS"),
        "source_crs": _crs(str(data.get("ip_source_crs") or "epsg:2180").strip(), "Source CRS"),
        "geoid_max_distance": _number(data, "ip_geoid_max_distance", 1.0, 1e6),
        "csv": bool(data.get("ip_csv")),
        "surfer": bool(data.get("ip_surfer")),
        "shp": bool(data.get("ip_shp")),
        "png": bool(data.get("ip_png")),
        "method": method,
        "accuracy": accuracy,
        "processes": max(1, min(int(data.get("processes") or 1), os.cpu_count() or 1)),
        "files_parallel": max(1, min(int(data.get("files_parallel") or 1), os.cpu_count() or 1)),
    }


def interp_args(form: Dict[str, Any], points_path: str, out_dir: str) -> List[str]:
    """interpolate.py arguments for one point file (values were validated by normalize)."""
    def onoff(flag, on, off):
        return on if flag else off

    args = [
        "-i", points_path, "-o", out_dir,
        "-N", str(form["neighbours"]),
        "--resolution", str(form["resolution"]),
        "--max-distance", str(form["max_distance"]),
        "--weights", str(form["weights"]),
        "--leafsize", str(form["leafsize"]),
        onoff(form["smooth"], "--smooth", "--no-smooth"),
        "--smooth-level", str(form["smooth_level"]),
        onoff(form["r_min_function"], "--r-min-function", "--no-r-min-function"),
        "--r-min", str(form["r_min"]),
        "--z-field", form["z_field"],
        onoff(form["geoid"], "--geoid", "--no-geoid"),
        "--source-crs", form["source_crs"],
        "--geoid-crs", form["geoid_crs"],
        "--geoid-max-distance", str(form["geoid_max_distance"]),
        onoff(form["csv"], "--csv", "--no-csv"),
        onoff(form["surfer"], "--surfer", "--no-surfer"),
        onoff(form["shp"], "--shp", "--no-shp"),
        onoff(form["png"], "--png", "--no-png"),
        "--quiet",
    ]
    if form["las_filter"]:
        args += ["--las-classes", ",".join(str(c) for c in form["las_classes"])]
    if form["geoid"]:
        args += ["--geoid-file", form["geoid_file"]]
    if form["pipeline"] == "interp_compress":
        args += ["--compress", "--method", form["method"], "--accuracy", str(form["accuracy"])]
    return args


def _store_path(path: str) -> str:
    """Paths inside the project are stored relative to it, so config.json stays portable."""
    prefix = PROJECT_ROOT + os.sep
    return os.path.relpath(path, PROJECT_ROOT) if path.startswith(prefix) else path


def save_settings(form: Dict[str, Any]) -> None:
    """Store the Interpolation tab in config.json (the CLI reads these values as defaults)."""
    cfg = load_config(CONFIG_PATH)
    cfg.interpolation = dataclasses.replace(
        cfg.interpolation,
        neighbours=form["neighbours"],
        resolution=form["resolution"],
        max_distance=form["max_distance"],
        weights=form["weights"],
        leafsize=form["leafsize"],
        smooth_result=form["smooth"],
        smooth_level=form["smooth_level"],
        r_min_function=form["r_min_function"],
        r_min=form["r_min"],
        filter_las=form["las_filter"],
        filter_classes=form["las_classes"],
        z_field_name=form["z_field"],
        geoid_correction=form["geoid"],
        geoid_correction_file=_store_path(form["geoid_file"]),
        geoid_crs=form["geoid_crs"],
        source_crs=form["source_crs"],
        geoid_max_distance=form["geoid_max_distance"],
        csv_result=form["csv"],
        surfer_grd=form["surfer"],
        save_points_shp=form["shp"],
        preview_png=form["png"],
    )
    save_config(cfg, CONFIG_PATH)


def collect_points(input_type: str, input_path: str, output_dir: str) -> List[str]:
    """Point files to interpolate: the file itself, or every point file below a folder.

    Results of earlier runs (names containing -dist-) inside the input folder are skipped.
    """
    if input_type == "file":
        return [os.path.abspath(input_path)]
    root = os.path.abspath(input_path)
    out = os.path.abspath(output_dir)
    skip_prefix: Optional[str] = out + os.sep if out != root and out.startswith(root + os.sep) else None
    found = []
    for folder, _dirs, names in os.walk(root):
        for name in names:
            if not name.lower().endswith(POINT_EXTENSIONS) or "-dist-" in name:
                continue
            full = os.path.join(folder, name)
            if skip_prefix is None or not full.startswith(skip_prefix):
                found.append(full)
    return sorted(found)
