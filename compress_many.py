#!/usr/bin/env python3
"""Compress many files with one initialisation per worker process (batch driver).

Starting `compress.py` once per file pays the imports (scipy, numba cache, CUDA runtime) and the
GPU context for every file. Here every worker imports and builds the configuration once and then
compresses its files one after another; files are handed out dynamically, so a few workers keep
all cores busy and the GPU is shared.

usage:
  compress_many.py --list files.txt -o OUTDIR [--workers K] [--results results.csv] \
      <the usual compress.py options, without -i/-o>
e.g.
  python3 compress_many.py --list files.txt -o out \
      --uint8-accuracy 5 --block-size 8 --rgb-channel-indices 1,2,3 --ycbcr-per-block --quiet
"""

import argparse
import csv
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATE = {}


def _load_cli():
    """compress.py as a module (the name `compress` is the package, so load it by path)."""
    spec = importlib.util.spec_from_file_location("fidwac_compress_cli", os.path.join(_HERE, "compress.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init_worker(rest_args, outdir, slot_counter=None, gpu_slots=0):
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    cli = _load_cli()
    args = cli._build_parser().parse_args(rest_args + ["-i", "placeholder", "-o", outdir])  # noqa: SLF001
    config = cli.load_config(args.config)
    cli._apply_args_to_config(args, config)  # noqa: SLF001
    if slot_counter is not None and config.performance.gpu != "off":
        # only as many workers as the GPU slots get a CUDA context; the others run on the CPU
        with slot_counter.get_lock():
            index = slot_counter.value
            slot_counter.value += 1
        if index >= gpu_slots:
            cli.set_gpu_mode("off")
            config.performance.gpu = "off"
    _STATE.update(cli=cli, config=config, outdir=outdir, procs=1)


def _one_file(path):
    cli = _STATE["cli"]
    t0 = time.perf_counter()
    try:
        out = cli.compress_image(path, deepcopy(_STATE["config"]), _STATE["procs"], _STATE["outdir"])
        return path, out, cli._extract_validity_tag(out), time.perf_counter() - t0, ""  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001  (one bad file must not stop the batch)
        return path, "", "ERR", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


# Memory model per worker: a fixed part for the imports and the process, plus a multiple of the largest
# raster held in memory.
_WORKER_BASE_MIB = 500
_WORKER_RASTER_FACTOR = 6


def _available_mib():
    """MemAvailable from /proc/meminfo in MiB; None when it cannot be read (no cap then)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        return None
    return None


def _largest_raster_bytes(files):
    """Largest decoded raster (width * height * bands * itemsize) read from the file headers."""
    try:
        import rasterio
    except ImportError:
        return 0
    best = 0
    for path in files:
        try:
            with rasterio.open(path) as src:
                size = src.width * src.height * src.count * np.dtype(src.dtypes[0]).itemsize
        except Exception:  # noqa: BLE001  (not a raster or unreadable: no estimate for it)
            continue
        best = max(best, size)
    return best


def memory_workers(files, max_workers):
    """Number of workers that fit into the available memory (80% of MemAvailable)."""
    avail = _available_mib()
    biggest = _largest_raster_bytes(files)
    if avail is None or biggest == 0:
        return max_workers
    per_worker = _WORKER_BASE_MIB + _WORKER_RASTER_FACTOR * (biggest >> 20)
    fit = int(0.8 * avail) // per_worker
    return max(1, min(max_workers, fit))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="text file: one input path per line")
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: performance.parallel.num_processes of the config, auto = CPU count)")
    ap.add_argument("--results", default=None, help="CSV with path, output, validity, seconds, error")
    ns, rest = ap.parse_known_args()
    with open(ns.list, encoding="utf-8") as fh:
        files = [ln.strip() for ln in fh if ln.strip()]
    os.makedirs(ns.output_dir, exist_ok=True)

    cli = _load_cli()
    cfg_args = cli._build_parser().parse_args(rest + ["-i", "placeholder", "-o", ns.output_dir])  # noqa: SLF001
    config = cli.load_config(cfg_args.config)
    cli._apply_args_to_config(cfg_args, config)  # noqa: SLF001
    if ns.workers is None:
        ns.workers = config.num_processes_int
    ns.workers = max(1, min(ns.workers, len(files)))
    fit = memory_workers(files, ns.workers)
    if fit < ns.workers:
        print(f"  note: {fit} of {ns.workers} workers fit into the available memory")
        ns.workers = fit

    # GPU workers: as many as this machine lets hold a CUDA context at the same time (measured and cached)
    gpu_slots = 0
    if config.performance.gpu != "off":
        from engine.gpu_u8.slots import gpu_slots as measure_slots

        gpu_slots = measure_slots(ns.workers)
        if gpu_slots == 0 and config.performance.gpu == "on":
            raise SystemExit("performance.gpu = on, but this machine gives no GPU context")
        if gpu_slots == 0:
            print("  note: GPU not found, all workers use the CPU (performance.gpu = auto)")
        elif gpu_slots < ns.workers:
            print(f"  note: {gpu_slots} of {ns.workers} workers use the GPU (measured CUDA context limit)")


    t0 = time.perf_counter()
    rows = []
    if ns.workers <= 1:
        _init_worker(rest, ns.output_dir, None, gpu_slots)
        for f in files:
            rows.append(_one_file(f))
    else:
        ctx = get_context("spawn")
        counter = ctx.Value("i", 0)
        with ProcessPoolExecutor(max_workers=ns.workers, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(rest, ns.output_dir, counter, gpu_slots)) as ex:
            rows = list(ex.map(_one_file, files, chunksize=1))
    wall = time.perf_counter() - t0

    bad = [r for r in rows if r[2] == "ERR"]
    print(f"files={len(files)} workers={ns.workers} wall={wall:.1f}s "
          f"per_file={wall / max(1, len(files)):.2f}s errors={len(bad)}")
    for r in bad:
        print("ERROR", r[0], r[4])
    if ns.results:
        with open(ns.results, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["input", "output", "validity", "seconds", "error"])
            w.writerows(rows)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
