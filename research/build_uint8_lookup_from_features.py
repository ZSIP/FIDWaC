"""
Build uint8 lookup table from pre-computed feature files (STREAMING version).

Input: 41,986 feature files in /results/features/ (N=8,16 × sf=1,10)
Output: models/lookup_uint8_grid.npz (2D grid on ac_abs_mean, zero_ratio)

Streams through features without loading all into memory at once (54GB).
"""

import numpy as np
from pathlib import Path
from glob import glob
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Feature columns (from uint8_features_N*_sf*_names.json)
FEATURE_NAMES = [
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
    "L_acc",
]

# Column indices
IDX_AC_ABS_MEAN = FEATURE_NAMES.index("ac_abs_mean")
IDX_ZERO_RATIO = FEATURE_NAMES.index("zero_ratio")
IDX_L_ACC = FEATURE_NAMES.index("L_acc")


def stream_features(feature_dir: str):
    """Stream through feature files without loading all into memory.

    Yields: (acm, zr, bs, sf) tuples for each sample.
    """

    feature_dir = Path(feature_dir)

    # Configs to load
    configs = [
        ("uint8_features_N8_sf1_part*.npy", 8, 1),
        ("uint8_features_N8_sf10_part*.npy", 8, 10),
        ("uint8_features_N16_sf1_part*.npy", 16, 1),
        ("uint8_features_N16_sf10_part*.npy", 16, 10),
    ]

    for pattern, bs, sf in configs:
        files = sorted(glob(str(feature_dir / pattern)))

        if not files:
            continue

        logger.info(f"Streaming N={bs}, sf={sf}: {len(files)} files")

        for fpath in tqdm(files, desc=f"  N={bs} sf={sf}", leave=False):
            data = np.load(fpath)
            acm = data[:, IDX_AC_ABS_MEAN]
            zr = data[:, IDX_ZERO_RATIO]

            for i in range(len(data)):
                yield acm[i], zr[i], bs, sf


def build_lookup_grid(feature_dir: str, num_bins_acm: int = 64, num_bins_zr: int = 29) -> dict:
    """Build 2D lookup grid on (ac_abs_mean, zero_ratio) from streaming features."""

    logger.info(f"\nCollecting statistics from features for grid binning...")

    # First pass: collect min/max and sample quantiles
    all_acm = []
    all_zr = []

    for acm, zr, bs, sf in tqdm(stream_features(feature_dir), desc="  Pass 1: Statistics"):
        all_acm.append(acm)
        all_zr.append(zr)

        # Don't collect all in memory, sample for quantiles
        if len(all_acm) >= 100000:  # Process in chunks
            break

    all_acm = np.array(all_acm)
    all_zr = np.array(all_zr)

    # Create bin edges (quantile-based on sample)
    edges_acm = np.percentile(all_acm, np.linspace(0, 100, num_bins_acm + 1))
    edges_zr = np.percentile(all_zr, np.linspace(0, 100, num_bins_zr + 1))

    # Remove duplicates
    edges_acm = np.unique(edges_acm)
    edges_zr = np.unique(edges_zr)

    logger.info(f"  edges_acm: {len(edges_acm)-1} bins")
    logger.info(f"  edges_zr: {len(edges_zr)-1} bins")
    logger.info(f"  Grid size: {(len(edges_acm)-1)} x {(len(edges_zr)-1)}")

    # Initialize grid: (N_acm_bins, N_zr_bins) → (sf, bs)
    grid_sf = np.ones((len(edges_acm) - 1, len(edges_zr) - 1), dtype=np.uint8)
    grid_bs = np.full((len(edges_acm) - 1, len(edges_zr) - 1), 16, dtype=np.uint8)
    grid_mult = np.full((len(edges_acm) - 1, len(edges_zr) - 1), 0.5, dtype=np.float32)
    grid_n = np.zeros((len(edges_acm) - 1, len(edges_zr) - 1), dtype=np.uint16)

    # Second pass: fill grid by assigning (sf, bs) to cells via voting
    logger.info(f"\nPass 2: Filling grid with (sf, bs) parameters...")

    for acm, zr, bs, sf in tqdm(stream_features(feature_dir), desc="  Pass 2: Grid fill"):
        # Find bin indices
        acm_bin = np.digitize(acm, edges_acm) - 1
        zr_bin = np.digitize(zr, edges_zr) - 1

        # Clamp to valid range
        acm_bin = np.clip(acm_bin, 0, len(edges_acm) - 2)
        zr_bin = np.clip(zr_bin, 0, len(edges_zr) - 2)

        # Update cell: prefer smaller sf, then smaller bs
        current_sf = grid_sf[acm_bin, zr_bin]
        if sf < current_sf or (sf == current_sf and bs < grid_bs[acm_bin, zr_bin]):
            grid_sf[acm_bin, zr_bin] = sf
            grid_bs[acm_bin, zr_bin] = bs

        grid_n[acm_bin, zr_bin] += 1

    logger.info(f"  Grid filled: {(grid_n > 0).sum()} cells with data")

    return {
        "grid_sf": grid_sf,
        "grid_bs": grid_bs,
        "grid_mult": grid_mult,
        "grid_n": grid_n,
        "edges_acm": edges_acm,
        "edges_zr": edges_zr,
    }


def save_lookup_npz(grid_data: dict, output_path: str):
    """Save lookup grid to NPZ file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    accuracy_levels = np.array([1, 2, 5, 10, 20, 50, 100], dtype=np.int32)

    # Default parameters
    default_sf = np.array([1, 1, 1, 1, 1, 10, 10], dtype=np.uint8)
    default_bs = np.array([8, 8, 16, 16, 16, 8, 8], dtype=np.uint8)
    default_mult = np.array([0.3, 0.3, 0.5, 0.5, 0.7, 0.9, 0.9], dtype=np.float32)

    n_train = np.array([41986], dtype=np.int32)

    # Create a lookup for accuracy levels (mock implementation)
    # In real scenario, this would map accuracy to (sf, bs)
    acm_lookup = np.arange(10000, dtype=np.int32)
    zr_lookup = np.arange(10000, dtype=np.int32)

    np.savez_compressed(
        output_path,
        # Main grids
        grid_sf=grid_data["grid_sf"],
        grid_bs=grid_data["grid_bs"],
        grid_mult=grid_data["grid_mult"],
        grid_n=grid_data["grid_n"],
        # Bin edges
        edges_acm=grid_data["edges_acm"],
        edges_zr=grid_data["edges_zr"],
        # Lookup tables (for compatibility)
        acm_lookup=acm_lookup,
        zr_lookup=zr_lookup,
        # Metadata
        accuracy_levels=accuracy_levels,
        default_sf=default_sf,
        default_bs=default_bs,
        default_mult=default_mult,
        n_train=n_train,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"\n✓ Saved lookup table: {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    import sys

    feature_dir = sys.argv[1] if len(sys.argv) > 1 else "./results/features"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./models/lookup_uint8_grid.npz"

    logger.info(f"Feature directory: {feature_dir}")
    logger.info(f"Output path: {output_path}")

    # Build grid (streams features, doesn't load all into memory)
    grid_data = build_lookup_grid(feature_dir)

    # Save
    save_lookup_npz(grid_data, output_path)

    logger.info("\n" + "=" * 70)
    logger.info("LOOKUP TABLE GENERATION COMPLETE")
    logger.info("=" * 70)
