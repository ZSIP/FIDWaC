"""Flask app: the page, a folder browser, and the job start/status/stop API."""

import atexit
import csv
import glob
import json
import os
import subprocess

import rasterio
from flask import Flask, abort, jsonify, render_template, request

from config import load_config
from webui import settings
from webui.jobs import Job, JobManager, collect_inputs
from webui import interp


def _pick_sample_raster(path: str) -> str | None:
    """Return the first raster file under a given input path, or the file itself."""
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in settings.RASTER_EXTENSIONS:
            return path
        return None
    if not os.path.isdir(path):
        return None
    for root, _dirs, files in os.walk(path):
        for name in sorted(files, key=str.lower):
            full = os.path.join(root, name)
            if os.path.splitext(name)[1].lower() in settings.RASTER_EXTENSIONS:
                return full
    return None


def detect_input_kind(path: str) -> str:
    """Return 'uint8', 'float', 'other', or 'unknown' for the selected input."""
    sample = _pick_sample_raster(path)
    if not sample or not os.path.exists(sample):
        return "unknown"
    try:
        with rasterio.open(sample) as src:
            dtypes = tuple(src.dtypes)
    except Exception:
        return "unknown"
    if not dtypes:
        return "unknown"
    if all(dtype == "uint8" for dtype in dtypes):
        return "uint8"
    if any(str(dtype).lower().startswith(("float", "double")) for dtype in dtypes):
        return "float"
    if any(str(dtype).lower().startswith(("int", "uint")) for dtype in dtypes):
        return "other"
    return "other"


def create_app() -> Flask:
    app = Flask(__name__)
    jobs = JobManager()
    atexit.register(jobs.shutdown)

    @app.context_processor
    def asset_version() -> dict:
        """Changes when a static file changes, so the browser never runs an old script or stylesheet."""
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        stamps = [os.path.getmtime(os.path.join(static_dir, name))
                  for name in os.listdir(static_dir) if os.path.isfile(os.path.join(static_dir, name))]
        return {"asset_v": int(max(stamps, default=0))}

    @app.get("/")
    def index():
        cfg = load_config(settings.CONFIG_PATH)
        tests_cfg = settings.load_tests_config()
        initial = {
            "defaults": settings.form_defaults(cfg),
            "interpDefaults": interp.form_defaults(cfg.interpolation),
            "testsDefaults": settings.tests_defaults(tests_cfg),
            "testsDatasets": settings.tests_datasets(tests_cfg),
            "bundledGeoid": interp.BUNDLED_GEOID_PATH,
            "presets": settings.PRESETS,
            "methodHelp": settings.METHOD_HELP,
            "accuracyButtons": list(settings.ACCURACY_BUTTONS),
            "cpuCount": os.cpu_count() or 1,
        }
        tests = {
            "variants": list((tests_cfg.get("variants") or {}).keys()),
            "metrics": list(settings.TEST_METRICS),
            "available": os.path.isfile(settings.RUN_TESTS_PY),
        }
        return render_template("index.html", initial=initial, tests=tests)

    @app.get("/api/browse")
    def browse():
        mode = request.args.get("mode", "dir")
        path = os.path.abspath(os.path.expanduser(request.args.get("path") or "~"))
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if not os.path.isdir(path):
            return jsonify(error=f"Not a folder: {path}"), 400
        try:
            names = sorted(os.listdir(path), key=str.lower)
        except OSError as exc:
            return jsonify(error=str(exc)), 403
        entries = []
        for name in names:
            if name.startswith("."):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                entries.append({"name": name, "path": full, "kind": "dir"})
            elif mode != "dir" and name.lower().endswith(interp.browse_extensions(mode, settings.BROWSE_EXTENSIONS)):
                entries.append({"name": name, "path": full, "kind": "file"})
        parent = os.path.dirname(path)
        return jsonify(path=path, parent=parent if parent != path else None, entries=entries)

    @app.get("/api/inspect-input")
    def inspect_input():
        path = request.args.get("path", "")
        if not path:
            return jsonify(kind="unknown")
        return jsonify(kind=detect_input_kind(path), sample=_pick_sample_raster(path))

    @app.get("/api/geoid-info")
    def geoid_info():
        return jsonify(interp.geoid_info(request.args.get("path", "")))

    def start_interpolation(data):
        form = interp.normalize(data)
        inputs = interp.collect_points(form["input_type"], form["input_path"], form["output_dir"])
        if not inputs:
            raise ValueError("No point files (TXT, CSV, LAS, LAZ, SHP) found in the input.")
        os.makedirs(form["output_dir"], exist_ok=True)
        interp.save_settings(form)
        root = form["input_path"] if form["input_type"] == "directory" else None
        return jobs.start(Job(inputs, root, form["output_dir"], form))

    @app.post("/api/run")
    def run():
        if jobs.busy():
            return jsonify(error="A job is already running. Wait for it to finish or press Stop."), 400
        try:
            data = request.get_json(silent=True) or {}
            if data.get("kind") == "interpolate":
                return jsonify(start_interpolation(data).status()), 202
            if data.get("kind") == "test":
                tests_cfg = settings.load_tests_config()
                form = settings.tests_normalize(data, tests_cfg)
                out_dir = settings.tests_results_dir(tests_cfg)
                os.makedirs(out_dir, exist_ok=True)
                settings.save_tests_gui_datasets(form["datasets"])
                job = jobs.start(Job([settings.RUN_TESTS_PY], None, out_dir, form))
                return jsonify(job.status()), 202
            form = settings.normalize(data)
            inputs = collect_inputs(form["input_type"], form["input_path"], form["output_dir"])
            if not inputs:
                raise ValueError("No raster or archive files found in the input.")
            os.makedirs(form["output_dir"], exist_ok=True)
            settings.save_settings(form)
            root = form["input_path"] if form["input_type"] == "directory" else None
            job = jobs.start(Job(inputs, root, form["output_dir"], form))
        except (ValueError, RuntimeError, OSError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(job.status()), 202

    @app.get("/api/status")
    def status():
        job = jobs.job
        if job is None:
            return jsonify(state="idle", done=0, total=0, current=[], results=[], errors=[], log=[], log_offset=0)
        offset = request.args.get("offset", default=0, type=int)
        if request.args.get("job") != job.id:
            offset = 0  # The page follows another job: return the complete log of the current job.
        return jsonify(job.status(offset))

    @app.post("/api/stop")
    def stop():
        if jobs.job is not None:
            jobs.job.request_stop()
        return jsonify(ok=True)

    @app.get("/api/tests-info")
    def tests_info():
        """Resolved test-suite paths plus environment problems shown in the Tests tab."""
        tcfg = settings.load_tests_config()
        try:
            r = subprocess.run(
                [settings.fidwac_python(), settings.RUN_TESTS_PY,
                 "--config", settings.TESTS_CONFIG, "--print-config"],
                capture_output=True, text=True, timeout=60)
            info = json.loads(r.stdout) if r.returncode == 0 else {}
        except (OSError, subprocess.TimeoutExpired, ValueError):
            info = {}
        p = info.get("paths") or {}
        problems = []
        if not p.get("data_root") or not os.path.isdir(p["data_root"]):
            problems.append(f"data_root not found: {p.get('data_root') or '(empty)'}")
        if not p.get("fidwac_dir") or not os.path.isfile(os.path.join(p.get("fidwac_dir", ""), "compress.py")):
            problems.append(f"compress.py not found in fidwac_dir: {p.get('fidwac_dir') or '(empty)'}")
        if not p.get("sz3_bin"):
            problems.append("sz3 binary not found — the 'sz3' variants are unavailable "
                            "(run install/setup_linux.sh or build SZ3).")
        info["problems"] = problems
        info["results_dir"] = p.get("results_dir") or settings.tests_results_dir(tcfg)
        return jsonify(info)

    @app.get("/api/tests-scan")
    def tests_scan():
        """Probe a dataset folder and suggest its test parameters (kind, thresholds,
        rgb/de and dem tests, fidwac extra flags). Used by the Scan button in the
        Tests tab."""
        try:
            return jsonify(settings.scan_dataset_dir(request.args.get("path", "")))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/tests/results")
    def tests_results():
        """Simple viewer for the results_*.csv and features.csv files of the test suite."""
        tcfg = settings.load_tests_config()
        res_dir = settings.tests_results_dir(tcfg)
        names = sorted(os.path.basename(f)
                       for f in glob.glob(os.path.join(res_dir, "*.csv")))
        sel = request.args.get("f", names[0] if names else "")
        if sel and sel not in names:
            abort(404)
        header, rows = [], []
        if sel:
            with open(os.path.join(res_dir, sel), newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))[-300:]
                header = list(rows[0].keys()) if rows else []
        return render_template("tests_results.html", files=names, sel=sel,
                               header=header, rows=rows, res_dir=res_dir)

    return app
