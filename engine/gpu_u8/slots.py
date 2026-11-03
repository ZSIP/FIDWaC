"""How many worker processes can hold a CUDA context at the same time on this machine.

Each process that uses the GPU creates its own CUDA context. On some systems (WSL2 in particular) the
driver refuses new contexts after a certain number of processes, although the card has memory left.
The limit is measured here, by creating contexts in short-lived child processes, and cached per GPU.
Only that many workers of a batch run use the GPU; the others run on the CPU.
"""

import ctypes
import json
import os
import subprocess
from multiprocessing import get_context

_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "fidwac", "gpu_slots.json")
_HOLD_SECONDS = 2.0


def _gpu_key():
    """Name and memory of the first GPU (nvidia-smi); None when there is no NVIDIA driver."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    return line or None


def _probe_child(hold):
    """Create a CUDA context, allocate a little, keep both for `hold` seconds (runs in a child process)."""
    import time

    from engine.gpu_u8.phases import _cudart_candidates

    rt = None
    for cand in _cudart_candidates():
        try:
            rt = ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    if rt is None:
        return False
    ptr = ctypes.c_void_p()
    ok = rt.cudaFree(0) == 0 and rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(64 << 20)) == 0
    time.sleep(hold)
    return ok


def _all_ok(n):
    with get_context("spawn").Pool(n) as pool:
        return all(pool.map(_probe_child, [_HOLD_SECONDS] * n, chunksize=1))


def _read_cache():
    try:
        with open(_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write_cache(data):
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def gpu_slots(max_workers):
    """Largest number (<= max_workers) of processes that get a CUDA context at the same time. 0 when there
    is no GPU. A measured limit below max_workers is cached per GPU; otherwise the test runs each time."""
    key = _gpu_key()
    if key is None:
        return 0
    cache = _read_cache()
    if key in cache and int(cache[key]) < max_workers:
        return int(cache[key])
    if _all_ok(max_workers):
        return max_workers
    lo, hi = 1, max_workers - 1  # lo is known to work only after the check below
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _all_ok(mid):
            lo = mid
        else:
            hi = mid - 1
    if not _all_ok(lo):
        lo = 0
    cache[key] = lo
    _write_cache(cache)
    return lo
