"""Accuracy budget for the codec when the reconstruction is stored in the raster's dtype.

SZ3 and the MED path reconstruct in float64 and keep the error within the bound there. Storing
the result as float32 rounds each value to the nearest float32, which adds up to half an ulp at
the data's magnitude. Without a margin a float32 result can exceed the requested accuracy by that
rounding alone. Integer outputs are compared exactly after rounding and need no margin.
"""

import numpy as np


def storage_margin(values: np.ndarray, dtype, eps: float) -> float:
    """One ulp of `dtype` at the largest reconstructed magnitude (max |value| + eps); 0 for integers."""
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer) or values.size == 0:
        return 0.0
    magnitude = dt.type(float(np.max(np.abs(values))) + eps)
    return float(np.spacing(magnitude))


def codec_bound(eps: float, values: np.ndarray, dtype) -> float:
    """Bound to hand to the codec so that the stored raster still meets `eps`."""
    bound = eps - storage_margin(values, dtype, eps)
    if bound <= 0:
        raise ValueError(
            f"accuracy {eps} is below the {np.dtype(dtype).name} resolution at this data magnitude; "
            "use a larger accuracy or a float64 raster"
        )
    return bound
