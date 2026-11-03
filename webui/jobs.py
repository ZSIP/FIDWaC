"""Compression and decompression jobs. Every raster or archive runs compress.py as a subprocess."""

import glob
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import rasterio

from compress.compression import find_raster_files
from webui.settings import (
    ARCHIVE_EXTENSIONS,
    COMPRESS_PY,
    PROJECT_ROOT,
    RUN_TESTS_PY,
    TESTS_DIR,
    compress_args,
    decompress_args,
    fidwac_python,
    load_tests_config,
    tests_args,
    tests_results_dir,
)
from webui import interp

OUTPUT_PREFIX = "Output file: "
VALIDITY_PREFIX = "VALIDITY:"

# run_tests.py with FIDWAC_NO_TQDM=1 prints "postep <done>/<total> ..." per combination.
PROGRESS_RE = re.compile(r"postep\s+(\d+)\s*/\s*(\d+)")


def is_archive(path: str) -> bool:
    return path.lower().endswith(ARCHIVE_EXTENSIONS)


def is_uint8_raster(path: str) -> bool:
    """True when the file is a raster and every band is uint8."""
    try:
        with rasterio.open(path) as src:
            return all(dtype == "uint8" for dtype in src.dtypes)
    except Exception:  # GDAL cannot open the file; the CLI reports the actual error
        return False


def collect_inputs(input_type: str, input_path: str, output_dir: str) -> List[str]:
    """The file itself, or every raster and archive below a folder.

    Results written into a subfolder of the input folder are skipped, so a second run does not decode them.
    """
    if input_type == "file":
        return [os.path.abspath(input_path)]
    root = os.path.abspath(input_path)
    out = os.path.abspath(output_dir)
    found = set(find_raster_files(root, recursive=True))
    for folder, _dirs, names in os.walk(root):
        found.update(os.path.join(folder, name) for name in names if is_archive(name))
    skip_prefix = out + os.sep if out != root and out.startswith(root + os.sep) else None
    files = [os.path.abspath(p) for p in found]
    return sorted(p for p in files if skip_prefix is None or not p.startswith(skip_prefix))


def _output_dir(job: "Job", path: str) -> str:
    """Output folder for one input; folder directories keep their relative structure."""
    if job.input_root is None:
        return job.output_dir
    rel = os.path.relpath(os.path.dirname(path), job.input_root)
    return job.output_dir if rel == "." else os.path.join(job.output_dir, rel)


def _terminate(proc: subprocess.Popen) -> None:
    """Stop the CLI together with its worker processes (it runs in its own process group)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()


def _run_cli(job: "Job", args: List[str], script: str = COMPRESS_PY,
             cwd: str = PROJECT_ROOT, env: Optional[Dict[str, str]] = None,
             python: Optional[str] = None) -> Tuple[int, Optional[str], str, str]:
    """Run a CLI script, stream its output into the job log.

    Returns (exit code, output path, validity tag, last error line).
    """
    proc = subprocess.Popen(
        [python or sys.executable, script, *args],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    with job._lock:
        job._procs.add(proc)
        stopping = job._stop_requested
    if stopping:
        _terminate(proc)
    output, validity, last_error = None, "", ""
    try:
        for line in proc.stdout:
            text = line.rstrip()
            if not text:
                continue
            if text.startswith(OUTPUT_PREFIX):
                output = text[len(OUTPUT_PREFIX):]
            elif text.startswith(VALIDITY_PREFIX):
                validity = text[len(VALIDITY_PREFIX):]
            elif text.startswith("Error"):
                last_error = text
            m = PROGRESS_RE.search(text)
            if m:
                job.set_progress(int(m.group(1)), int(m.group(2)))
            job.log(text)
    finally:
        proc.stdout.close()
        with job._lock:
            job._procs.discard(proc)
    return proc.wait(), output, validity, last_error


def _result(path: str, output: str, validity: str, archive: bool) -> Dict[str, Any]:
    size_in = os.path.getsize(path)
    size_out = os.path.getsize(output) if os.path.exists(output) else 0
    return {
        "input": path,
        "output": output,
        "validity": validity,
        "size_in": size_in,
        "size_out": size_out,
        "ratio": None if archive or not size_out else round(size_in / size_out, 2),
    }


def _collect_test_results(job: "Job") -> None:
    """After a finished test run, list the produced CSVs in the Results table."""
    res_dir = tests_results_dir(load_tests_config())
    csvs = sorted(glob.glob(os.path.join(res_dir, "results_*.csv")))
    features = os.path.join(res_dir, "features.csv")
    if os.path.isfile(features):
        csvs.append(features)
    for p in csvs:
        job.record_result({
            "input": "test suite",
            "output": p,
            "validity": "",
            "size_in": 0,
            "size_out": os.path.getsize(p),
            "ratio": None,
        })


def _process_test(job: "Job", path: str) -> None:
    """Run analysis/tests/run_tests.py as a single managed subprocess."""
    if job.stop_requested:
        return
    job.begin_file(path)
    try:
        job.log("Starting the test suite (log lines show the progress).")
        env = {**os.environ, "FIDWAC_NO_TQDM": "1", "PYTHONUNBUFFERED": "1"}
        code, _output, _validity, last_error = _run_cli(
            job, tests_args(job.form), RUN_TESTS_PY,
            cwd=TESTS_DIR, env=env, python=fidwac_python())
        if job.stop_requested:
            return
        if code != 0:
            reason = f": {last_error}" if last_error else ""
            job.record_error(f"test run failed (exit code {code}){reason}")
            return
        job.set_progress(job.total, job.total)
        _collect_test_results(job)
    except Exception as exc:
        job.record_error(f"test run: {exc}")
    finally:
        job.end_file(path)


def _process_one(job: "Job", index: int, path: str, processes: int) -> None:
    if job.stop_requested:
        return
    if job.form.get("kind") == "test":
        _process_test(job, path)
        return
    name = os.path.basename(path)
    job.begin_file(path)
    try:
        job.log(f"[{index}/{job.total}] {path}")
        out_dir = _output_dir(job, path)
        os.makedirs(out_dir, exist_ok=True)
        archive = is_archive(path)
        interpolating = job.form.get("kind") == "interpolate"
        if interpolating:
            args, script = interp.interp_args(job.form, path, out_dir), interp.INTERP_CLI
        elif archive:
            args, script = decompress_args(path, out_dir), COMPRESS_PY
        else:
            args, script = compress_args(job.form, path, out_dir, processes, is_uint8_raster(path)), COMPRESS_PY
        code, output, validity, last_error = _run_cli(job, args, script)
        if job.stop_requested:
            return
        if code != 0 or not output:
            reason = last_error or (f"exit code {code}" if code else "no output reported")
            job.record_error(f"{name}: {reason}")
            return
        job.record_result(_result(path, output, validity, archive or interpolating))
    except Exception as exc:  # one failing file must not stop the others
        job.record_error(f"{name}: {exc}")
    finally:
        job.end_file(path)


def _run(job: "Job") -> None:
    """Worker thread: processes the files, `files_parallel` at a time, then marks the job finished."""
    try:
        workers = max(1, min(job.form.get("files_parallel", 1), job.total))
        processes = max(1, job.form.get("processes", 1) // workers)
        if job.form.get("kind") == "test":
            ds_names = ", ".join(d["name"] if isinstance(d, dict) else str(d)
                                 for d in job.form["datasets"])
            job.log(f"Test suite: {len(job.form['datasets'])} dataset(s) [{ds_names}], "
                    f"{len(job.form['variants'])} variant(s), "
                    f"{job.form['workers']} workers, max {job.form['max_files']} file(s) per dataset.")
        else:
            job.log(f"Found {job.total} files: {workers} at a time, {processes} processes each.")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_process_one, job, index, path, processes)
                for index, path in enumerate(job.inputs, 1)
            ]
            for future in as_completed(futures):
                future.result()
    except Exception as exc:
        job.record_error(str(exc))
    finally:
        job.finish()


class Job:
    """Progress, results, errors and the log of one run. The page reads it with an offset."""

    def __init__(self, inputs: List[str], input_root: Optional[str], output_dir: str, form: Dict[str, Any]):
        self.id = uuid.uuid4().hex[:8]
        self.inputs = inputs
        self.input_root = input_root
        self.output_dir = output_dir
        self.form = form
        self.state = "running"
        self.done = 0
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.started = time.time()
        self.finished: Optional[float] = None
        self._running: set = set()
        self._procs: set = set()
        self._log: List[str] = []
        self._stop_requested = False
        self._total_override: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self._total_override or len(self.inputs)

    def set_progress(self, done: int, total: int) -> None:
        """Progress reported by the running CLI itself (test suite runs)."""
        with self._lock:
            self._total_override = total
            self.done = done

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def log(self, line: str) -> None:
        with self._lock:
            self._log.append(line)

    def begin_file(self, path: str) -> None:
        with self._lock:
            self._running.add(path)

    def end_file(self, path: str) -> None:
        with self._lock:
            self._running.discard(path)
            if self._total_override is None:
                self.done += 1

    def record_result(self, result: Dict[str, Any]) -> None:
        with self._lock:
            self.results.append(result)
        self.log(f"Output: {result['output']}")

    def record_error(self, message: str) -> None:
        with self._lock:
            self.errors.append(message)
        self.log(f"Error: {message}")

    def request_stop(self) -> None:
        with self._lock:
            if self.state != "running":
                return
            self._stop_requested = True
            procs = list(self._procs)
        for proc in procs:
            _terminate(proc)

    def finish(self) -> None:
        with self._lock:
            self.finished = time.time()
            if self._stop_requested:
                self.state = "stopped"
                self._log.append("Stopped.")
            elif self.errors and not self.results:
                self.state = "error"
                self._log.append("Finished with errors.")
            else:
                self.state = "done"
                self._log.append("Done, with some errors (see above)." if self.errors else "Done.")

    def status(self, offset: int = 0) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "state": self.state,
                "done": self.done,
                "total": self.total,
                "current": [os.path.basename(p) for p in sorted(self._running)],
                "results": list(self.results),
                "errors": list(self.errors),
                "log": self._log[offset:],
                "log_offset": len(self._log),
            }


class JobManager:
    """Runs one job at a time; the last job stays available for the status API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[Job] = None

    @property
    def job(self) -> Optional[Job]:
        return self._job

    def busy(self) -> bool:
        return self._job is not None and self._job.state == "running"

    def start(self, job: Job) -> Job:
        with self._lock:
            if self.busy():
                raise RuntimeError("A job is already running. Wait for it to finish or press Stop.")
            self._job = job
        threading.Thread(target=_run, args=(job,), daemon=True).start()
        return job

    def shutdown(self) -> None:
        if self._job is not None:
            self._job.request_stop()
