"""GPU backend for the batched uint8 and float32 search.

The libraries are built by build.sh; their loaders are in phases.py (phase library), session.py
(device-resident session) and rgb.py (RGB check). A missing library, runtime or device makes the
caller fall back to the CPU (numba prange path) with a warning.
"""

import warnings


def warn_once(msg: str) -> None:
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
