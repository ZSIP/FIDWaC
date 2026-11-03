"""
NPY format support for AdvancedHeuristicPredictor.

2D quantile grid format (preferred): O(log B) lookup where B=64,
grid ~8KB vs 347MB NPY+PKL. No cKDTree at runtime.

Uses (ac_abs_mean, zero_ratio) — top-2 features from analysis:
  ac_abs_mean: Spearman #1 (0.911), MI #1 (1.099)
  zero_ratio:  RF #1 (87.9%), Spearman #2 (0.903)
2D grid 64×64 = 4096 cells.
"""

import os
import numpy as np
import pickle


def load_multi_lookup_npy(predictor, npy_file: str):
    """Load lookup table — prefers quantile grid, fallback to cKDTree."""

    # --- Format 1: 2D quantile grid (preferred) ---
    grid_file = npy_file.replace("_optimized.npy", "_grid.npz")
    if os.path.exists(grid_file):
        data = np.load(grid_file)
        predictor._grid = data["grid"]
        predictor._grid_edges_acm = data["edges_acm"]
        predictor._grid_edges_zr = data["edges_zr"]
        predictor._use_grid = True
        predictor._use_kdtree = False
        predictor._top_features = ["ac_abs_mean", "zero_ratio"]
        predictor._use_npy_format = True
        return

    # --- Format 2: Full NPY + cKDTree (old format, fallback) ---
    predictor._use_grid = False
    predictor._multi_lookup_array = np.load(npy_file)

    tree_file = npy_file.replace("_optimized.npy", "_tree.pkl")
    try:
        with open(tree_file, "rb") as f:
            predictor._kdtree = pickle.load(f)
        predictor._use_kdtree = True
    except FileNotFoundError:
        predictor._use_kdtree = False

    predictor._top_features = ["mean_val", "std_dev", "ac_std"]
    predictor._use_npy_format = True


def predict_batch_from_npy(predictor, query_points: np.ndarray) -> np.ndarray:
    """
    Batch lookup for multiple blocks at once.
    query_points: (n_blocks, 2) — [ac_abs_mean, zero_ratio]
    Returns: (n_blocks,) int32, 0 = no data in this cell
    """
    if getattr(predictor, "_use_grid", False):
        return _batch_grid_lookup(
            predictor._grid,
            predictor._grid_edges_acm,
            predictor._grid_edges_zr,
            query_points,
        )
    # Fallback: cKDTree (K=1), columns [ac_abs_mean, zero_ratio]
    indices = predictor._kdtree.query(query_points[:, :2], k=1)[1]
    return predictor._multi_lookup_array[indices, 4].astype(np.int32)


def predict_from_multi_lookup_npy(predictor, features: np.ndarray) -> tuple:
    """
    Single-block lookup.
    features[0]=ac_abs_mean, features[1]=zero_ratio, features[2]=ac_std, features[3]=std_dev
    """
    if not getattr(predictor, "_use_grid", False) and not hasattr(predictor, "_multi_lookup_array"):
        return None, None

    try:
        ac_abs_mean = float(features[0])
        zero_ratio = float(features[1])

        if getattr(predictor, "_use_grid", False):
            pred = _single_grid_lookup(
                predictor._grid,
                predictor._grid_edges_acm,
                predictor._grid_edges_zr,
                ac_abs_mean,
                zero_ratio,
            )
        else:
            q = np.array([ac_abs_mean, zero_ratio], dtype=np.float32)
            _, idx = predictor._kdtree.query(q, k=1)
            pred = int(predictor._multi_lookup_array[idx, 4])

        return pred if pred > 0 else None, []

    except Exception:
        return None, []


# ---------------------------------------------------------------------------
# Internal quantile grid functions
# ---------------------------------------------------------------------------


def _single_grid_lookup(grid, e_acm, e_zr, ac_abs_mean: float, zero_ratio: float) -> int:
    """O(log B) lookup — searchsorted on quantile thresholds.
    e_acm is in log1p(ac_abs_mean) space; e_zr without transformation."""
    i = min(int(np.searchsorted(e_acm[1:-1], np.log1p(ac_abs_mean))), grid.shape[0] - 1)
    j = min(int(np.searchsorted(e_zr[1:-1], zero_ratio)), grid.shape[1] - 1)
    return int(grid[i, j])


def _batch_grid_lookup(grid, e_acm, e_zr, query_points: np.ndarray) -> np.ndarray:
    """
    Vectorized batch lookup O(n · log B).
    query_points: (n, 2) — [ac_abs_mean, zero_ratio]
    e_acm is in log1p(ac_abs_mean) space — apply log1p to query before searchsorted.
    e_zr without transformation.
    """
    i = np.clip(np.searchsorted(e_acm[1:-1], np.log1p(query_points[:, 0])), 0, grid.shape[0] - 1)
    j = np.clip(np.searchsorted(e_zr[1:-1], query_points[:, 1]), 0, grid.shape[1] - 1)
    return grid[i, j].astype(np.int32)
