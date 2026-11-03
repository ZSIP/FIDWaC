"""FIDWAC v2 interpolation: IDW gridding of scattered points (ported from ZSIP/FIDWaC, MIT License)."""

from interpolation.run import EXTRA_OUTPUTS, run_interpolation
from interpolation.points import POINT_EXTENSIONS

__all__ = ["EXTRA_OUTPUTS", "POINT_EXTENSIONS", "run_interpolation"]
