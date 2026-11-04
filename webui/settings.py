"""Web GUI settings: form defaults from config.json, validation, presets and the compress.py arguments they map to."""

import json
import os
import re
import sys
from typing import Any, Dict, List

from compress.compression import RASTER_EXTENSIONS
from config import Config, load_config, save_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
COMPRESS_PY = os.path.join(PROJECT_ROOT, "compress.py")

# Recension test suite (analysis/tests): the "Tests" tab runs run_tests.py as a job.
TESTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "tests")
TESTS_CONFIG = os.path.join(TESTS_DIR, "config.json")
RUN_TESTS_PY = os.path.join(TESTS_DIR, "run_tests.py")

ARCHIVE_EXTENSIONS = (".7z", ".fpred", ".sz3")
BROWSE_EXTENSIONS = tuple(sorted(set(RASTER_EXTENSIONS) | set(ARCHIVE_EXTENSIONS)))

METHODS = ("dct", "pred", "sz3")
METHOD_HELP = {
    "dct": "Block DCT with per-block error control (cm=3/5/6). Archive: .7z.",
    "pred": "Near-lossless MED prediction (cm=7). Archive: .7z with a .fpred payload.",
    "sz3": "External SZ3 compressor with an absolute error bound (cm=8). Archive: .7z with a .sz3 payload.",
}
BLOCK_SIZES = (8, 16, 32)
BACKENDS = ("binary", "heuristic")
UINT8_MODES = ("quality", "accuracy", "lossless")
ARCHIVE_METHODS = ("LZMA2", "PPMD", "BZIP2", "DEFLATE")
TIFF_COMPRESSIONS = ("DEFLATE", "LZW", "LZMA", "ZSTD", "NONE")
ACCURACY_BUTTONS = (0.01, 0.05, 0.1, 0.5, 1.0)

# Preset form values; the preset buttons of the web GUI apply one of these entries.
PRESETS: Dict[str, Dict[str, Any]] = {
    "Quick": {
        "accuracy": 0.5,
        "block_size": 8,
        "auto_select": False,
        "rgb_quality": 70,
        "uint8_mode": "quality",
        "files_parallel": 2,
        "description": "Fast compression, moderate quality. Best for large batches.",
    },
    "Balanced": {
        "accuracy": 0.1,
        "block_size": 16,
        "auto_select": True,
        "rgb_quality": 85,
        "uint8_mode": "accuracy",
        "files_parallel": 1,
        "description": "Balanced speed and quality. Auto-selects the block size.",
    },
    "High Quality": {
        "accuracy": 0.01,
        "block_size": 32,
        "auto_select": False,
        "rgb_quality": 95,
        "uint8_mode": "accuracy",
        "files_parallel": 1,
        "description": "Maximum quality, slower compression.",
    },
    "Lossless": {
        "accuracy": 0.001,
        "block_size": 32,
        "auto_select": False,
        "rgb_quality": 100,
        "uint8_mode": "lossless",
        "files_parallel": 1,
        "description": "Lossless: data stored unchanged (PNG / GeoTIFF) in the FIDWAC archive; little or no size reduction.",
    },
}


def form_defaults(cfg: Config) -> Dict[str, Any]:
    """Initial form values taken from config.json."""
    comp, model, out, perf = cfg.compression, cfg.model, cfg.output, cfg.performance
    if comp.lossless:
        uint8_mode = "lossless"
    elif comp.uint8_accuracy_mode:
        uint8_mode = "accuracy"
    else:
        uint8_mode = "quality"
    rgb = list(comp.rgb_channel_indices or [1, 2, 3])
    cpu = os.cpu_count() or 1
    return {
        "input_type": "file",
        "input_path": "",
        "output_dir": os.path.normpath(os.path.join(PROJECT_ROOT, cfg.directories.results)),
        "method": comp.method or "dct",
        "accuracy": float(comp.accuracy),
        "uint8_accuracy": int(comp.uint8_accuracy),
        "auto_select": bool(comp.auto_select_block_size),
        "block_size": int(comp.block_size),
        "allowed_block_sizes": [n for n in BLOCK_SIZES if n in (comp.allowed_block_sizes or [])],
        "backend": model.backend,
        "uint8_mode": uint8_mode,
        "rgb_quality": int(comp.rgb_quality),
        "rgb_channels": rgb if len(rgb) == 3 else [None, None, None],
        "ycbcr_per_block": bool(comp.ycbcr_per_block_mode),
        "ycbcr_y": float(comp.ycbcr_y_multiplier),
        "ycbcr_cb": float(comp.ycbcr_cb_multiplier),
        "ycbcr_cr": float(comp.ycbcr_cr_multiplier),
        "advanced_heuristic": bool(model.advanced_heuristic),
        "accept_prediction": bool(model.accept_prediction_if_within_accuracy),
        "minimize_backscan": int(model.minimize_backscan),
        "backscan_break_after": int(model.backscan_break_after),
        "decimal_places": int(comp.decimal_places),
        "processes": min(max(1, cfg.num_processes_int), cpu),
        "gpu": cfg.performance.gpu,
        "files_parallel": 1,
        "compression_method": out.compression_method,
        "lzma2_block_size": out.lzma2_block_size,
        "tiff_compression": out.tiff_compression,
        "crs": "" if comp.crs in ("", "none", None) else comp.crs,
        "no_crs": comp.crs == "none",
        "delete_temp": bool(out.delete_temp_files),
        "verify_full_idct": bool(perf.verify_with_full_idct),
        "l2_precheck": bool(perf.l2_precheck_enabled),
        "incremental_backscan": bool(perf.incremental_backscan),
    }


def _abs_path(value: Any) -> str:
    text = str(value or "").strip()
    return os.path.abspath(os.path.expanduser(text)) if text else ""


def _float(data: Dict[str, Any], key: str, lo: float, hi: float) -> float:
    label = key.replace("_", " ")
    try:
        value = float(data.get(key))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not lo <= value <= hi:
        raise ValueError(f"{label} must be between {lo} and {hi}.")
    return value


def _int(data: Dict[str, Any], key: str, lo: int, hi: int) -> int:
    label = key.replace("_", " ")
    try:
        value = int(data.get(key))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.") from None
    if not lo <= value <= hi:
        raise ValueError(f"{label} must be between {lo} and {hi}.")
    return value


GPU_MODES = ("auto", "on", "off")


def _choice(data: Dict[str, Any], key: str, choices: tuple) -> str:
    value = str(data.get(key, ""))
    if value not in choices:
        raise ValueError(f"{key.replace('_', ' ')}: unknown value {value!r}.")
    return value


def _lzma2_block_size(raw: Any) -> str:
    """LZMA2 block size like '1m', '2m', '4m'; '' = 7-Zip default (same rule as pack_to_7z)."""
    value = str(raw or "").strip().lower()
    if value and not re.fullmatch(r"\d+[kmg]?", value):
        raise ValueError(f"lzma2 block size: expected e.g. 1m, 2m, 4m, got {value!r}.")
    return value


def _rgb_channels(raw: Any) -> List[int]:
    """Three 1-based band numbers, or [] when all three are empty (no RGB handling)."""
    values = [None if v in (None, "") else v for v in (raw or [])]
    if all(v is None for v in values):
        return []
    if len(values) != 3 or any(v is None for v in values):
        raise ValueError("Fill in all three RGB channel indices, or leave them all empty.")
    try:
        bands = [int(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError("RGB channel indices must be whole numbers.") from None
    if min(bands) < 1:
        raise ValueError("RGB channel indices start at 1.")
    return bands


def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the posted form and return it with proper types. Raises ValueError with a message for the user."""
    input_type = _choice(data, "input_type", ("file", "directory"))
    input_path = _abs_path(data.get("input_path"))
    output_dir = _abs_path(data.get("output_dir"))
    if not input_path:
        raise ValueError("Select an input file or folder.")
    if input_type == "file" and not os.path.isfile(input_path):
        raise ValueError(f"Input file not found: {input_path}")
    if input_type == "directory" and not os.path.isdir(input_path):
        raise ValueError(f"Input folder not found: {input_path}")
    if not output_dir:
        raise ValueError("Select an output folder.")
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError(f"Output path is not a folder: {output_dir}")

    block_size = _int(data, "block_size", min(BLOCK_SIZES), max(BLOCK_SIZES))
    if block_size not in BLOCK_SIZES:
        raise ValueError("block size must be 8, 16 or 32.")
    backend = _choice(data, "backend", BACKENDS)
    cpu = os.cpu_count() or 1
    return {
        "input_type": input_type,
        "input_path": input_path,
        "output_dir": output_dir,
        "method": _choice(data, "method", METHODS),
        "accuracy": _float(data, "accuracy", 0.001, 1.0),
        "uint8_accuracy": _int(data, "uint8_accuracy", 2, 255),
        "auto_select": bool(data.get("auto_select")),
        "block_size": block_size,
        "allowed_block_sizes": sorted({int(n) for n in (data.get("allowed_block_sizes") or [])} & set(BLOCK_SIZES)),
        "backend": backend,
        "uint8_mode": _choice(data, "uint8_mode", UINT8_MODES),
        "rgb_quality": _int(data, "rgb_quality", 1, 100),
        "rgb_channels": _rgb_channels(data.get("rgb_channels")),
        "ycbcr_per_block": bool(data.get("ycbcr_per_block")),
        "ycbcr_y": _float(data, "ycbcr_y", 0.0, 10.0),
        "ycbcr_cb": _float(data, "ycbcr_cb", 0.0, 10.0),
        "ycbcr_cr": _float(data, "ycbcr_cr", 0.0, 10.0),
        # The advanced heuristic applies only to the heuristic backend.
        "advanced_heuristic": bool(data.get("advanced_heuristic")) and backend == "heuristic",
        "accept_prediction": bool(data.get("accept_prediction")),
        "minimize_backscan": _int(data, "minimize_backscan", 0, 1000),
        "backscan_break_after": _int(data, "backscan_break_after", 0, 100),
        "decimal_places": _int(data, "decimal_places", 0, 6),
        "processes": _int(data, "processes", 1, cpu),
        "gpu": _choice(data, "gpu", GPU_MODES),
        "files_parallel": _int(data, "files_parallel", 1, cpu),
        "compression_method": _choice(data, "compression_method", ARCHIVE_METHODS),
        "lzma2_block_size": _lzma2_block_size(data.get("lzma2_block_size")),
        "tiff_compression": _choice(data, "tiff_compression", TIFF_COMPRESSIONS),
        "crs": str(data.get("crs") or "").strip(),
        "no_crs": bool(data.get("no_crs")),
        "delete_temp": bool(data.get("delete_temp")),
        "verify_full_idct": bool(data.get("verify_full_idct")),
        "l2_precheck": bool(data.get("l2_precheck")),
        "incremental_backscan": bool(data.get("incremental_backscan")),
    }


def compress_args(form: Dict[str, Any], input_path: str, output_dir: str, processes: int, uint8_input: bool) -> List[str]:
    """Build the compress.py arguments for one raster from a validated form."""
    accuracy = form["accuracy"]
    if form["method"] in ("pred", "sz3") and uint8_input:
        accuracy = form["uint8_accuracy"]
    args = ["-i", input_path, "-o", output_dir, "--method", form["method"], "--accuracy", str(accuracy)]
    args += ["--auto"] if form["auto_select"] else ["--block-size", str(form["block_size"])]
    args += [
        "--backend", form["backend"],
        "--minimize-backscan", str(form["minimize_backscan"]),
        "--backscan-break-after", str(form["backscan_break_after"]),
        "--advanced-heuristic" if form["advanced_heuristic"] else "--no-advanced-heuristic",
        "--accept-prediction" if form["accept_prediction"] else "--no-accept-prediction",
        "--decimal-places", str(form["decimal_places"]),
        "--processes", str(processes),
        "--gpu", form["gpu"],
        "--quiet",
    ]
    mode = form["uint8_mode"]
    if mode == "accuracy":
        args += ["--uint8-accuracy-mode", "--uint8-accuracy", str(form["uint8_accuracy"])]
    else:
        args += ["--no-uint8-accuracy-mode"]
    args += ["--rgb-quality", str(form["rgb_quality"])]
    if mode == "lossless":
        args += ["--lossless"]
    if form["rgb_channels"]:
        args += ["--rgb-channel-indices", ",".join(str(v) for v in form["rgb_channels"])]
        if form["ycbcr_per_block"]:
            args += ["--ycbcr-per-block"]
    args += [
        "--ycbcr-y-multiplier", str(form["ycbcr_y"]),
        "--ycbcr-cb-multiplier", str(form["ycbcr_cb"]),
        "--ycbcr-cr-multiplier", str(form["ycbcr_cr"]),
    ]
    if form["allowed_block_sizes"]:
        args += ["--allowed-block-sizes", ",".join(str(n) for n in form["allowed_block_sizes"])]
    args += [
        "--crs", "none" if form["no_crs"] else form["crs"],
        "--compression-method", form["compression_method"],
        "--lzma2-block-size", form["lzma2_block_size"],
        "--verify-with-full-idct" if form["verify_full_idct"] else "--no-verify-with-full-idct",
        "--l2-precheck" if form["l2_precheck"] else "--no-l2-precheck",
        "--incremental-backscan" if form["incremental_backscan"] else "--no-incremental-backscan",
    ]
    return args


def decompress_args(archive_path: str, output_dir: str) -> List[str]:
    """compress.py arguments that decode one archive (.7z, .fpred or .sz3) to GeoTIFF."""
    return ["-i", archive_path, "-o", output_dir, "--quiet"]


def fidwac_python() -> str:
    """Python for test runs: the installer venv when present, else this interpreter."""
    p = os.path.expanduser("~/.fidwac/venv/bin/python3")
    return p if os.path.isfile(p) else sys.executable


def load_tests_config() -> Dict[str, Any]:
    """Raw analysis/tests/config.json ({} when unreadable)."""
    try:
        with open(TESTS_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def tests_results_dir(tcfg: Dict[str, Any]) -> str:
    """Resolved results_dir of the tests config — same rules as run_tests.resolve_paths."""
    v = str((tcfg.get("paths") or {}).get("results_dir") or "").strip()
    if not v:
        return os.path.join(TESTS_DIR, "results")
    v = os.path.expanduser(v)
    return v if os.path.isabs(v) else os.path.normpath(os.path.join(TESTS_DIR, v))


def tests_defaults(tcfg: Dict[str, Any]) -> Dict[str, Any]:
    """Initial Tests-tab values taken from analysis/tests/config.json."""
    run = tcfg.get("run") or {}
    return {
        "t_variants": list((tcfg.get("variants") or {}).keys()),
        "t_max_files": str(run.get("max_files_per_dataset") or "all"),
        "t_workers": int(run.get("workers") or 1),
        "t_resume": bool(run.get("resume", True)),
        "t_dry_run": False,
    }


TEST_KINDS = ("float", "uint8")
TEST_METRICS = ("grad", "ssim", "dist", "de", "dem")
TEST_FILE_EXTENSIONS = (".tif", ".tiff", ".png")  # build_plan globs only these


def _band_list(raw: Any, label: str) -> List[int]:
    """Comma-separated 1-based band numbers; empty input -> []."""
    text = raw if isinstance(raw, str) else ",".join(str(v) for v in (raw or []))
    if not text.strip():
        return []
    try:
        bands = [int(v) for v in text.replace(";", ",").split(",") if str(v).strip()]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be comma-separated whole numbers.") from None
    if min(bands) < 1:
        raise ValueError(f"{label}: band numbers start at 1.")
    return bands


def _thresholds(raw: Any) -> List[float]:
    vals = raw if isinstance(raw, list) else str(raw or "").split(",")
    try:
        out = [float(v) for v in vals if str(v).strip() != ""]
    except (TypeError, ValueError):
        raise ValueError("Thresholds must be comma-separated numbers.") from None
    if not out or any(v <= 0 for v in out):
        raise ValueError("Thresholds must be a non-empty list of positive numbers.")
    return out


def _dataset_dir(raw: Any) -> str:
    path = os.path.abspath(os.path.expanduser(str(raw or "").strip()))
    if not str(raw or "").strip():
        raise ValueError("Every dataset needs a folder path.")
    if not os.path.isdir(path):
        raise ValueError(f"Dataset folder not found: {path}")
    return path


def tests_normalize(data: Dict[str, Any], tcfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the Tests-tab form. Datasets come as a free-form list built in the GUI;
    the runner gets them through a generated runtime config (see build_tests_runtime_cfg)."""
    raw_sets = data.get("datasets") or []
    datasets, seen = [], set()
    for i, d in enumerate(raw_sets, 1):
        name = str(d.get("name") or "").strip() or f"dataset_{i}"
        if not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError(f"Dataset name {name!r}: only letters, digits, '_' and '-'.")
        if name in seen:
            raise ValueError(f"Duplicate dataset name: {name!r}.")
        seen.add(name)
        kind = str(d.get("kind") or "")
        if kind not in TEST_KINDS:
            raise ValueError(f"Dataset {name}: kind must be 'float' or 'uint8'.")
        tests = [t for t in (d.get("tests") or []) if t in TEST_METRICS]
        if "de" in tests and not _band_list(d.get("rgb_bands"), "rgb_bands"):
            raise ValueError(f"Dataset {name}: test 'de' needs rgb_bands (e.g. 1,2,3).")
        datasets.append({
            "name": name,
            "dir": _dataset_dir(d.get("dir")),
            "kind": kind,
            "thresholds": _thresholds(d.get("thresholds")),
            "tests": tests or list(TEST_METRICS[:3]),
            "rgb_bands": _band_list(d.get("rgb_bands"), "rgb_bands"),
            "dem_bands": _band_list(d.get("dem_bands"), "dem_bands"),
            "fidwac_extra": [str(a) for a in (d.get("fidwac_extra") or [])],
        })
    if not datasets:
        raise ValueError("Add at least one dataset.")
    variants = [v for v in (data.get("t_variants") or []) if v in (tcfg.get("variants") or {})]
    if not variants:
        raise ValueError("Select at least one variant.")
    mf = str(data.get("t_max_files") or "").strip() or "all"
    if mf != "all":
        try:
            if int(mf) < 1:
                raise ValueError
        except ValueError:
            raise ValueError("Files per dataset must be a whole number or 'all'.") from None
    raw_workers = str(data.get("t_workers") or "").strip() or "1"
    form = {
        "kind": "test",
        "datasets": datasets,
        "variants": variants,
        "max_files": mf,
        "workers": _int({"t_workers": raw_workers}, "t_workers", 1, os.cpu_count() or 1),
        "resume": bool(data.get("t_resume")),
        "dry_run": bool(data.get("t_dry_run")),
    }
    form["config_path"] = build_tests_runtime_cfg(tcfg, datasets)
    return form


def build_tests_runtime_cfg(tcfg: Dict[str, Any], datasets: List[Dict[str, Any]]) -> str:
    """Merge GUI-defined datasets into the base tests config and write it next to the
    results — the file documents exactly what was run. Dataset dirs are absolute, so
    data_root becomes '/' (os.path.join keeps the absolute part)."""
    cfg = json.loads(json.dumps(tcfg))
    # The runtime config lives in results/, so every relative path from the base
    # config must become absolute now (run_tests resolves them against the cfg dir).
    paths = cfg.setdefault("paths", {})
    for key in ("fidwac_dir", "fidwac_python", "data_root", "work_dir",
                "results_dir", "sz3_bin", "7z_dir"):
        v = os.path.expanduser(str(paths.get(key) or "").strip())
        if v and not os.path.isabs(v):
            paths[key] = os.path.normpath(os.path.join(TESTS_DIR, v))
        elif v:
            paths[key] = v
    paths["data_root"] = os.sep  # dataset dirs are absolute already
    cfg.setdefault("run", {})["order"] = [d["name"] for d in datasets]
    cfg["datasets"] = {
        d["name"]: {"dir": d["dir"], "kind": d["kind"], "thresholds": d["thresholds"],
                    "tests": d["tests"], "rgb_bands": d["rgb_bands"],
                    "dem_bands": d["dem_bands"], "fidwac_extra": d["fidwac_extra"]}
        for d in datasets
    }
    path = os.path.join(tests_results_dir(tcfg), "run_gui_config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return path


def save_tests_gui_datasets(datasets: List[Dict[str, Any]]) -> None:
    """Remember the dataset rows between GUI sessions (key 'gui_datasets')."""
    tcfg = load_tests_config()
    tcfg["gui_datasets"] = datasets
    try:
        with open(TESTS_CONFIG, "w", encoding="utf-8") as f:
            json.dump(tcfg, f, indent=1)
    except OSError:
        pass


def tests_datasets(tcfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dataset rows for the Tests tab: saved gui_datasets, else config datasets with
    dirs resolved against data_root (only those that exist on disk)."""
    saved = tcfg.get("gui_datasets")
    if saved:
        return saved
    root = str((tcfg.get("paths") or {}).get("data_root") or "")
    root = os.path.expanduser(root)
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(TESTS_DIR, root)) if root else ""
    out = []
    for name, ds in (tcfg.get("datasets") or {}).items():
        d = os.path.join(root, str(ds.get("dir") or ""))
        if not os.path.isdir(d):
            continue
        out.append({"name": name, "dir": d, "kind": ds.get("kind", "float"),
                    "thresholds": ds.get("thresholds") or [],
                    "tests": ds.get("tests") or ["grad", "ssim", "dist"],
                    "rgb_bands": ds.get("rgb_bands") or [],
                    "dem_bands": ds.get("dem_bands") or [],
                    "fidwac_extra": ds.get("fidwac_extra") or []})
    return out


def scan_dataset_dir(path: str) -> Dict[str, Any]:
    """Inspect a dataset folder and propose its test parameters.

    Reads up to 3 first .tif/.png files (build_plan's scope), detects dtype, band
    count and nodata, then applies the same defaults the recension config used:
    uint8 -> thresholds 2/5/10/20, RGB on bands 1-3; float -> 0.01..1.0, DEM tests.
    """
    path = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not os.path.isdir(path):
        raise ValueError(f"Folder not found: {path}")
    files = sorted(f for f in os.listdir(path)
                   if os.path.splitext(f)[1].lower() in TEST_FILE_EXTENSIONS)
    if not files:
        raise ValueError(f"No .tif/.png files in {path} (only the top level is scanned).")
    import rasterio
    dtypes, counts, nodatas, crs_list = set(), set(), set(), set()
    for name in files[:3]:
        try:
            with rasterio.open(os.path.join(path, name)) as src:
                dtypes.update(src.dtypes)
                counts.add(src.count)
                nodatas.add(src.nodata)
                crs_list.add(str(src.crs) if src.crs else "")
        except Exception as exc:
            raise ValueError(f"Cannot read {name}: {exc}") from None
    all_uint8 = dtypes == {"uint8"}
    kind = "uint8" if all_uint8 else "float"
    bands = max(counts) if counts else 1
    thresholds = [2, 5, 10, 20] if kind == "uint8" else [0.01, 0.05, 0.1, 0.5, 1.0]
    tests, rgb_bands, dem_bands, extra = ["grad", "ssim", "dist"], [], [], []
    if bands >= 3:
        tests.append("de")
        rgb_bands = [1, 2, 3]
        if kind == "uint8":
            extra = ["--rgb-channel-indices", "1,2,3", "--ycbcr-per-block"]
    if kind == "float":
        dem_bands = [1] if bands == 1 else ([bands - 1, bands] if bands >= 5 else [])
        if dem_bands:
            tests.append("dem")
    notes = []
    if len(counts) > 1:
        notes.append(f"band count varies between files: {sorted(counts)}")
    if not all_uint8 and not any(str(d).startswith("float") for d in dtypes):
        notes.append("unusual dtype — forced kind 'float'")
    return {
        "name": os.path.basename(path) or "dataset",
        "dir": path,
        "kind": kind,
        "thresholds": thresholds,
        "tests": tests,
        "rgb_bands": rgb_bands,
        "dem_bands": dem_bands,
        "fidwac_extra": extra,
        "info": {
            "files": len(files), "bands": sorted(counts), "dtypes": sorted(dtypes),
            "nodata": sorted(n for n in nodatas if n is not None),
            "crs": sorted(crs_list), "notes": notes,
        },
    }


def tests_args(form: Dict[str, Any]) -> List[str]:
    """run_tests.py arguments for a validated Tests-tab form."""
    args = ["--config", form["config_path"], "--max-files", str(form["max_files"]),
            "--workers", str(form["workers"])]
    for name in form["variants"]:
        args += ["--variant", name]
    if not form["resume"]:
        args.append("--no-resume")
    if form["dry_run"]:
        args.append("--dry-run")
    return args


def save_settings(form: Dict[str, Any]) -> None:
    """Store the form in config.json. The command-line tools read these values as their defaults."""
    cfg = load_config(CONFIG_PATH)
    comp, model, out, perf = cfg.compression, cfg.model, cfg.output, cfg.performance
    if form["input_type"] == "directory":
        cfg.directories.source = form["input_path"]
    else:
        cfg.directories.source = os.path.dirname(form["input_path"])
    cfg.directories.results = form["output_dir"]
    comp.method = form["method"]
    comp.accuracy = form["accuracy"]
    comp.uint8_accuracy = form["uint8_accuracy"]
    comp.auto_select_block_size = form["auto_select"]
    if not form["auto_select"]:
        comp.block_size = form["block_size"]
    comp.allowed_block_sizes = form["allowed_block_sizes"]
    comp.rgb_quality = form["rgb_quality"]
    comp.lossless = form["uint8_mode"] == "lossless"
    comp.uint8_accuracy_mode = form["uint8_mode"] == "accuracy"
    comp.rgb_channel_indices = form["rgb_channels"]
    comp.ycbcr_per_block_mode = form["ycbcr_per_block"]
    comp.ycbcr_y_multiplier = form["ycbcr_y"]
    comp.ycbcr_cb_multiplier = form["ycbcr_cb"]
    comp.ycbcr_cr_multiplier = form["ycbcr_cr"]
    comp.decimal_places = form["decimal_places"]
    comp.crs = "none" if form["no_crs"] else form["crs"]
    model.backend = form["backend"]
    model.advanced_heuristic = form["advanced_heuristic"]
    model.accept_prediction_if_within_accuracy = form["accept_prediction"]
    model.minimize_backscan = form["minimize_backscan"]
    model.backscan_break_after = form["backscan_break_after"]
    out.compression_method = form["compression_method"]
    out.lzma2_block_size = form["lzma2_block_size"]
    out.tiff_compression = form["tiff_compression"]
    out.delete_temp_files = form["delete_temp"]
    perf.verify_with_full_idct = form["verify_full_idct"]
    perf.l2_precheck_enabled = form["l2_precheck"]
    perf.incremental_backscan = form["incremental_backscan"]
    perf.gpu = form["gpu"]
    save_config(cfg, CONFIG_PATH)
