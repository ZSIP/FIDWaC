"""IDW gridding ported from ZSIP/FIDWaC (MIT License, Copyright (c) 2025 ZSIP, https://github.com/ZSIP/FIDWaC).

The algorithm follows ZSIP/FIDWaC interpolation.py: k nearest points within max_distance (KD-tree), a
half-minimum rule for zero distances, an optional r_min reduction and Gaussian smoothing. Two behaviours
differ from the original:

- Each cell uses only the neighbours that exist. The original Dask path returns values close to 0 for
  cells with fewer than N neighbours inside max_distance.
- The grid covers the full extent of the points. Queries run in blocks to bound memory use.

Gaussian smoothing is unchanged: NaN values spread to neighbouring cells.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage, spatial

QUERY_BLOCK = 500_000  # Query points per KD-tree call, which bounds the memory of one query.
ZERO_DISTANCE_EPS = 1e-10  # used only when every neighbour of a point sits exactly on its centre
MAX_CELLS = 400_000_000  # Upper limit on grid cells; larger grids are rejected to avoid exhausting memory.


@dataclass(frozen=True)
class Grid:
    """Regular grid. Row 0 is the southern row; cell (row j, column i) starts at x0 + i*res and y0 + j*res."""

    x0: float
    y0: float
    resolution: float
    nx: int
    ny: int

    def centres(self):
        """Cell-centre coordinates along x and y."""
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.resolution
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.resolution
        return xs, ys


def make_grid(xy: np.ndarray, resolution: float) -> Grid:
    """Grid aligned to multiples of resolution that covers every point."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    x0 = np.floor(xy[:, 0].min() / resolution) * resolution
    y0 = np.floor(xy[:, 1].min() / resolution) * resolution
    nx = int(np.floor((xy[:, 0].max() - x0) / resolution)) + 1
    ny = int(np.floor((xy[:, 1].max() - y0) / resolution)) + 1
    if nx * ny > MAX_CELLS:
        raise ValueError(f"grid {nx} x {ny} is too large; use a coarser resolution")
    return Grid(float(x0), float(y0), float(resolution), nx, ny)


def _reduce_close_distances(distance: np.ndarray, r_min: float) -> np.ndarray:
    """Replace distances below r_min with half of the smallest distance above r_min (ZSIP/FIDWaC rule)."""
    above = np.where(distance > r_min, distance, np.inf)
    row_min = above.min(axis=1)
    replace = (distance < r_min) & np.isfinite(row_min)[:, None]
    out = distance.copy()
    out[replace] = np.broadcast_to((row_min / 2.0)[:, None], distance.shape)[replace]
    return out


def _zero_distance_rule(distance: np.ndarray) -> np.ndarray:
    """Replace zero distances with half of the smallest non-zero distance of the same point (ZSIP/FIDWaC rule)."""
    zero = distance == 0
    if not zero.any():
        return distance
    positive = np.where(zero | ~np.isfinite(distance), np.inf, distance)
    row_min = positive.min(axis=1)
    replacement = np.where(np.isfinite(row_min), row_min / 2.0, ZERO_DISTANCE_EPS)
    return np.where(zero, replacement[:, None], distance)


def _idw_rows(distance: np.ndarray, index: np.ndarray, values: np.ndarray, weights: float,
              r_min: Optional[float]) -> np.ndarray:
    """IDW for query rows. Missing neighbours come from cKDTree as inf distance and index len(values)."""
    missing = ~np.isfinite(distance)
    d = np.where(missing, np.nan, distance)
    v = np.where(missing, np.nan, values[np.where(missing, 0, index)])
    if r_min is not None:
        d = _reduce_close_distances(d, r_min)
    d = _zero_distance_rule(d)
    w = 1.0 / d ** weights
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.nansum(w * v, axis=1)
        den = np.nansum(w, axis=1)
        return np.where(den > 0, num / den, np.nan)


def idw_points(xy: np.ndarray, z: np.ndarray, centres: np.ndarray, neighbours: int, max_distance: float,
               weights: float, leafsize: int = 10, r_min: Optional[float] = None) -> np.ndarray:
    """IDW value for every query point (NaN where no point lies within max_distance). The weight of a
    neighbour at distance d is 1 / d**weights, so `weights` is the power p of the distance."""
    if len(xy) == 0:
        raise ValueError("no points to interpolate")
    if not weights > 0:
        raise ValueError(f"weights (the power p of the weights 1 / d^p) must be positive, got {weights}")
    k = int(neighbours)
    tree = spatial.cKDTree(xy, leafsize=leafsize, balanced_tree=True, compact_nodes=True)
    out = np.empty(len(centres), dtype=float)
    for start in range(0, len(centres), QUERY_BLOCK):
        block = centres[start:start + QUERY_BLOCK]
        distance, index = tree.query(block, k=k, workers=-1, distance_upper_bound=max_distance, eps=0, p=2)
        distance = np.asarray(distance, dtype=float).reshape(len(block), k)
        index = np.asarray(index).reshape(len(block), k)
        out[start:start + len(block)] = _idw_rows(distance, index, z, weights, r_min)
    return out


def interpolate_grid(xy: np.ndarray, z: np.ndarray, grid: Grid, neighbours: int, max_distance: float,
                     weights: float, leafsize: int = 10, r_min: Optional[float] = None) -> np.ndarray:
    """Unsmoothed IDW values with shape (ny, nx). Row 0 is the southern row, as in ZSIP/FIDWaC."""
    xs, ys = grid.centres()
    xx, yy = np.meshgrid(xs, ys)
    flat = idw_points(xy, z, np.column_stack([xx.ravel(), yy.ravel()]), neighbours, max_distance, weights,
                      leafsize, r_min)
    return flat.reshape(grid.ny, grid.nx)


def smooth_gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing as in ZSIP/FIDWaC. NaN values spread to neighbouring cells."""
    return ndimage.gaussian_filter(values, sigma=sigma)
