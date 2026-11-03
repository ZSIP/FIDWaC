#!/usr/bin/env python3
"""
Build per-accuracy YCbCr lookup grids from collected feature .npy files.
Loads ALL parts into RAM at once (requires ~28 GB RAM per sf), then builds
all accuracy grids from the in-memory data without re-loading.

Output: one NPZ per (block_size, scaling_factor, accuracy) saved directly to models_dir:
  lookup_uint8_ycbcr_L_N8_sf10_acc3_grid.npz
  keys: grid_L_Y, grid_L_Cb, grid_L_Cr, edges_acm, edges_zr,
        coverage_Y, coverage_Cb, coverage_Cr

Usage:
  python3 build_ycbcr_lookup.py \
      --features /abs/path/results/ycbcr_features \
      --models   /abs/path/models \
      --block-sizes 8 \
      --scaling-factors 1,10 \
      --accuracies 2,3,5,10,20,30 \
      --percentile 90
"""

import argparse
import glob
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "channel_id",
    "std_dev",
    "mean_val",
    "dc_value",
    "ac_mean",
    "ac_std",
    "ac_abs_mean",
    "ac_abs_max",
    "zero_ratio",
    "small_vals_ratio",
    "medium_vals_ratio",
    "large_vals_ratio",
    "energy_ratio",
    "entropy",
    "zero_run_count",
    "zero_run_mean",
    "zero_run_max",
]
N_FEATURES = len(FEATURE_NAMES)
IDX_CHANNEL = FEATURE_NAMES.index("channel_id")
IDX_ACM = FEATURE_NAMES.index("ac_abs_mean")
IDX_ZR = FEATURE_NAMES.index("zero_ratio")

CHANNEL_IDS = {0: "Y", 1: "Cb", 2: "Cr"}
ACCURACIES = [2, 3, 5, 10, 20, 30]
N_BINS_ACM = 39  # → edges_acm.shape=(40,), grid.shape=(39,10)
N_BINS_ZR = 10
MAX_L = 64  # N=8 → max coefficients = 8*8


def _acc_col(accuracy: int) -> int:
    return N_FEATURES + ACCURACIES.index(accuracy)


def _load_all(features_dir: Path, block_size: int, sf: int) -> np.ndarray | None:
    """Load ALL feature .npy parts into a single contiguous array."""
    pattern = str(features_dir / f"uint8_ycbcr_features_N{block_size}_sf{sf}_part*.npy")
    parts = sorted(glob.glob(pattern))
    if not parts:
        print(f"  No feature files found: {pattern}", flush=True)
        return None

    print(f"  Loading {len(parts)} parts for N={block_size} sf={sf} into RAM ...", flush=True)
    chunks = []
    needed_cols = N_FEATURES + len(ACCURACIES)
    for i, p in enumerate(parts):
        arr = np.load(p)
        if arr.ndim == 2 and arr.shape[1] >= needed_cols:
            chunks.append(arr)
        if (i + 1) % 1000 == 0:
            print(
                f"    {i+1}/{len(parts)} parts loaded ({sum(c.shape[0] for c in chunks):,} rows so far)",
                flush=True,
            )

    if not chunks:
        return None

    data = np.concatenate(chunks, axis=0)
    print(f"  Total rows in RAM: {len(data):,}  ({data.nbytes / 1e9:.1f} GB)", flush=True)
    return data


def _build_edges_acm(data: np.ndarray) -> np.ndarray:
    """Quantile-based edges for ac_abs_mean.
    N_BINS_ACM=39 bins → 40 edges (shape=(40,)) — matches existing uint8 grid format.
    """
    acm = data[:, IDX_ACM].astype(np.float32)
    pcts = np.linspace(0, 100, N_BINS_ACM + 1)
    edges = np.percentile(acm, pcts).astype(np.float64)
    edges[0] = 0.0
    edges[-1] = edges[-1] * 1.001
    return edges


def _fill_empty_cells(grid: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Pure-numpy neighbour fill for empty cells, then monotone constraint on acm axis."""
    grid = grid.copy()
    empty = coverage == 0

    # Iterative 3×3 neighbour fill — no scipy needed
    for _ in range(max(N_BINS_ACM, N_BINS_ZR)):
        if not empty.any():
            break
        for i in range(N_BINS_ACM):
            for j in range(N_BINS_ZR):
                if not empty[i, j]:
                    continue
                nbrs = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N_BINS_ACM and 0 <= nj < N_BINS_ZR and not empty[ni, nj]:
                            nbrs.append(grid[ni, nj])
                if nbrs:
                    grid[i, j] = float(np.mean(nbrs))
                    empty[i, j] = False

    # Global fallback for any remaining empty cells
    filled_vals = grid[~empty]
    fallback = float(np.median(filled_vals)) if len(filled_vals) else 32.0
    grid[empty] = fallback

    # Monotone constraint along ac_abs_mean axis:
    # higher ac_abs_mean → at least as many coefficients as lower ac_abs_mean (same zero_ratio)
    for j in range(N_BINS_ZR):
        for i in range(1, N_BINS_ACM):
            if grid[i, j] < grid[i - 1, j]:
                grid[i, j] = grid[i - 1, j]

    return grid


def _build_grid(
    sub_acm: np.ndarray,
    sub_zr: np.ndarray,
    sub_L: np.ndarray,
    edges_acm: np.ndarray,
    percentile: float,
):
    """Build (N_BINS_ACM, N_BINS_ZR) lookup grid and coverage.
    All inputs are pre-filtered for a single channel.
    Returns (grid, coverage) both shape (N_BINS_ACM, N_BINS_ZR).
    """
    if len(sub_acm) == 0:
        default = np.full((N_BINS_ACM, N_BINS_ZR), 32.0, dtype=np.float32)
        return default, np.zeros((N_BINS_ACM, N_BINS_ZR), dtype=np.int32)

    L = np.clip(sub_L.astype(np.int32), 0, MAX_L)

    # Bin indices — same formula as predict_uint8_L_batch for exact consistency
    i_bins = np.clip(
        np.searchsorted(edges_acm[1:-1].astype(np.float32), sub_acm),
        0,
        N_BINS_ACM - 1,
    ).astype(np.int32)
    j_bins = np.clip(
        (sub_zr * N_BINS_ZR).astype(np.int32),
        0,
        N_BINS_ZR - 1,
    )

    # Histogram (N_BINS_ACM, N_BINS_ZR, MAX_L+1)
    hist = np.zeros((N_BINS_ACM, N_BINS_ZR, MAX_L + 1), dtype=np.int32)
    np.add.at(hist, (i_bins, j_bins, L), 1)

    coverage = hist.sum(axis=2).astype(np.int32)
    total = coverage.astype(np.float64)
    target = total * (percentile / 100.0)
    cumsum = np.cumsum(hist, axis=2).astype(np.float64)

    grid = np.zeros((N_BINS_ACM, N_BINS_ZR), dtype=np.float32)
    for i in range(N_BINS_ACM):
        for j in range(N_BINS_ZR):
            if coverage[i, j] == 0:
                continue
            idx = int(np.searchsorted(cumsum[i, j], target[i, j]))
            grid[i, j] = float(min(idx, MAX_L))

    grid = _fill_empty_cells(grid, coverage)
    return np.maximum(1.0, grid).astype(np.float32), coverage


def build_all(
    features_dir: Path,
    models_dir: Path,
    block_sizes,
    scaling_factors,
    accuracies,
    percentile: float,
):
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {models_dir.resolve()}", flush=True)

    for bs in block_sizes:
        for sf in scaling_factors:
            # --- Load all data once into RAM ---
            data = _load_all(features_dir, bs, sf)
            if data is None:
                print(f"  No data for N={bs} sf={sf}, skipping.", flush=True)
                continue

            # Pre-split by channel into contiguous views to avoid repeated masking
            ch_ids = data[:, IDX_CHANNEL].astype(np.int32)
            mask_Y = ch_ids == 0
            mask_Cb = ch_ids == 1
            mask_Cr = ch_ids == 2
            acm_Y, zr_Y = data[mask_Y, IDX_ACM].astype(np.float32), data[mask_Y, IDX_ZR].astype(
                np.float32
            )
            acm_Cb, zr_Cb = data[mask_Cb, IDX_ACM].astype(np.float32), data[mask_Cb, IDX_ZR].astype(
                np.float32
            )
            acm_Cr, zr_Cr = data[mask_Cr, IDX_ACM].astype(np.float32), data[mask_Cr, IDX_ZR].astype(
                np.float32
            )
            print(
                f"  Channel split: Y={mask_Y.sum():,}  Cb={mask_Cb.sum():,}  Cr={mask_Cr.sum():,}",
                flush=True,
            )

            # Build shared edges from the full dataset (all channels)
            edges_acm = _build_edges_acm(data)
            edges_zr = np.linspace(0.0, 1.0, N_BINS_ZR + 1, dtype=np.float64)

            for acc in accuracies:
                if acc not in ACCURACIES:
                    print(f"  accuracy={acc} not in training set, skip.", flush=True)
                    continue

                col_l = _acc_col(acc)
                L_Y = data[mask_Y, col_l].astype(np.float32)
                L_Cb = data[mask_Cb, col_l].astype(np.float32)
                L_Cr = data[mask_Cr, col_l].astype(np.float32)

                print(f"  Building grid N={bs} sf={sf} acc={acc} ...", flush=True)
                grid_Y, cov_Y = _build_grid(acm_Y, zr_Y, L_Y, edges_acm, percentile)
                grid_Cb, cov_Cb = _build_grid(acm_Cb, zr_Cb, L_Cb, edges_acm, percentile)
                grid_Cr, cov_Cr = _build_grid(acm_Cr, zr_Cr, L_Cr, edges_acm, percentile)

                out = models_dir / f"lookup_uint8_ycbcr_L_N{bs}_sf{sf}_acc{acc}_grid.npz"
                np.savez_compressed(
                    str(out),
                    grid_L_Y=grid_Y,
                    grid_L_Cb=grid_Cb,
                    grid_L_Cr=grid_Cr,
                    edges_acm=edges_acm.astype(np.float64),
                    edges_zr=edges_zr.astype(np.float64),
                    coverage_Y=cov_Y,
                    coverage_Cb=cov_Cb,
                    coverage_Cr=cov_Cr,
                )
                print(f"  Saved: {out.resolve()}", flush=True)

            del data, acm_Y, zr_Y, acm_Cb, zr_Cb, acm_Cr, zr_Cr
            print(f"  Done sf={sf}.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Absolute path to ycbcr_features dir")
    parser.add_argument("--models", required=True, help="Absolute path to models output dir")
    parser.add_argument("--block-sizes", default="8")
    parser.add_argument("--scaling-factors", default="1,10")
    parser.add_argument("--accuracies", default="2,3,5,10,20,30")
    parser.add_argument("--percentile", type=float, default=90.0)
    args = parser.parse_args()

    build_all(
        features_dir=Path(args.features).resolve(),
        models_dir=Path(args.models).resolve(),
        block_sizes=[int(x) for x in args.block_sizes.split(",")],
        scaling_factors=[int(x) for x in args.scaling_factors.split(",")],
        accuracies=[int(x) for x in args.accuracies.split(",")],
        percentile=args.percentile,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
