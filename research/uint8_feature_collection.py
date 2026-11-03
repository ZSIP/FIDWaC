#!/usr/bin/env python3
"""
Uint8 Per-Block Feature Collection
===================================

Collects per-block DCT features for uint8 data, analogous to the heuristic method for float16.

Workflow:
  1. Collect features:    python3 uint8_feature_collection.py collect
     → models/uint8_features_N{8,16}_raw.npy  (shape N_blocks × 21)

Column layout of uint8_features_N*_raw.npy:
  0-15 : 16 statistical features (see FEATURE_NAMES)
    16   : min_L for accuracy=2 px
    17   : min_L for accuracy=5 px
    18   : min_L for accuracy=10 px
    19   : min_L for accuracy=15 px
    20   : min_L for accuracy=20 px
"""

# Force single-threaded BLAS/OpenMP before any imports to avoid thread thrashing
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
import rasterio

# Add parent directory to path for imports (so we can import project modules when run as script)
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dct import dct2, idct2, to_zigzag, from_zigzag

# =============================================================================
# Konfiguracja
# =============================================================================

# Accuracy levels for uint8 data (maximum error in pixels, matching uint8_accuracy_sweep.py)
_TRAIN_ACCURACIES = [2, 5, 10, 15, 20]

# Names of the 21 columns: 16 features and 5 min_L values, one per accuracy level
FEATURE_NAMES = [
    # Columns 0-15: statistical features
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
    # Columns 16-20: min_L for accuracy 2 px, 5 px, 10 px, 15 px and 20 px
    "L_acc2",
    "L_acc5",
    "L_acc10",
    "L_acc15",
    "L_acc20",
]


# =============================================================================
# Helper functions
# =============================================================================


def _pad_image(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad image to be divisible by block_size."""
    h, w = image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, (h, w)


def _extract_blocks(
    image: np.ndarray, block_size: int
) -> List[Tuple[int, np.ndarray, Tuple[int, int]]]:
    """Extract blocks from image."""
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i : i + block_size, j : j + block_size]
            blocks.append((len(blocks), block, (i, j)))
    return blocks


# =============================================================================
# Main function that collects the features of each block
# =============================================================================


def _collect_features_worker(args: Tuple[str, int, int, int, float, float]) -> np.ndarray:
    """
    Worker: collects 16 features and min_L for 5 accuracy levels, for all blocks.

    Parameters
    ----------
    args : tuple
        (image_path, block_size, channel_idx, scaling_factor, accuracy_px, nodata_value)

    Returns
    -------
    np.ndarray
        Array shape (N_blocks, 21):
          columns 0-15: 16 features in FEATURE_NAMES order
          columns 16-20: min_L for accuracy [2, 5, 10, 15, 20]
    """
    image_path, block_size, channel_idx, scaling_factor, accuracy_px, nodata_value = args

    try:
        with rasterio.open(image_path) as src:
            image = src.read(channel_idx + 1).astype(np.float32)
            if nodata_value is None:
                nodata_value = src.nodata

        # Skip if nodata
        if nodata_value is not None and np.all(image == nodata_value):
            return np.empty((0, 21), dtype=np.float32)

        # Skip if all zeros
        if np.all(image == 0):
            return np.empty((0, 21), dtype=np.float32)

        # Pad and extract blocks
        padded_image, _ = _pad_image(image, block_size)
        blocks = _extract_blocks(padded_image, block_size)

        rows = []
        for block_idx, block, _ in blocks:
            if block is None:
                continue
            if nodata_value is not None and np.any(block == nodata_value):
                continue
            if np.all(block == 0):
                continue

            # --- 16 statistical features ---
            # 0  std_dev
            std_dev = float(np.std(block))
            # 1  mean_val
            mean_val = float(np.mean(block))

            # DCT, required for the AC features
            dct_block = dct2(block, dct_type=2)
            # Apply scaling factor
            dct_block_scaled = dct_block / scaling_factor
            dct_block_r = np.round(dct_block_scaled, 2)
            dct_zigzag = to_zigzag(dct_block_r)

            # 2  dc_value
            dc_value = float(dct_zigzag[0])
            ac = dct_zigzag[1:]

            # 3  ac_mean
            ac_mean = float(np.mean(ac))
            # 4  ac_std
            ac_std = float(np.std(ac))
            # 5  ac_abs_mean
            ac_abs = np.abs(ac)
            ac_abs_mean = float(np.mean(ac_abs))
            # 6  ac_abs_max
            ac_abs_max = float(np.max(ac_abs)) if len(ac_abs) > 0 else 0.0

            total = len(dct_zigzag)
            # 7  zero_ratio
            zero_ratio = float(np.sum(dct_zigzag == 0) / total)
            # 8  small_vals_ratio  (|v| < 1)
            small_vals_ratio = float(np.sum(ac_abs < 1.0) / len(ac)) if len(ac) > 0 else 0.0
            # 9  medium_vals_ratio (1 <= |v| < 10)
            medium_vals_ratio = (
                float(np.sum((ac_abs >= 1.0) & (ac_abs < 10.0)) / len(ac)) if len(ac) > 0 else 0.0
            )
            # 10 large_vals_ratio  (|v| >= 10)
            large_vals_ratio = float(np.sum(ac_abs >= 10.0) / len(ac)) if len(ac) > 0 else 0.0
            # 11 energy_ratio  (energia AC / suma energii)
            energy_total = float(np.sum(dct_zigzag**2))
            energy_ac = float(np.sum(ac**2))
            energy_ratio = energy_ac / energy_total if energy_total > 0 else 0.0
            # 12 entropy
            hist, _ = np.histogram(dct_zigzag, bins=50, density=True)
            hist_nz = hist[hist > 0]
            entropy = float(-np.sum(hist_nz * np.log2(hist_nz))) if len(hist_nz) > 0 else 0.0

            # 13 zero_run_count, 14 zero_run_mean, 15 zero_run_max
            # Count the runs of consecutive zeros in the AC coefficients
            in_run = False
            run_len = 0
            runs = []
            for v in ac:
                if v == 0.0:
                    in_run = True
                    run_len += 1
                else:
                    if in_run:
                        runs.append(run_len)
                        run_len = 0
                        in_run = False
            if in_run:
                runs.append(run_len)
            zero_run_count = float(len(runs))
            zero_run_mean = float(np.mean(runs)) if runs else 0.0
            zero_run_max = float(max(runs)) if runs else 0.0

            # --- Binary search for min_L at each accuracy level (reuses dct_zigzag) ---
            n = block_size
            block_f64 = block.astype(np.float64)
            L_vals = []
            for acc_px in _TRAIN_ACCURACIES:
                acc_value = float(acc_px)
                lo, hi = 1, len(dct_zigzag)
                opt_L = hi
                while lo <= hi:
                    mid = (lo + hi) // 2
                    arr = np.zeros(len(dct_zigzag), dtype=np.float64)
                    arr[:mid] = dct_zigzag[:mid]
                    # Apply inverse scaling before reconstruction
                    arr_unscaled = arr * scaling_factor
                    recon = from_zigzag(arr_unscaled, n)
                    idct_recon = idct2(recon, 2)
                    max_err = float(np.max(np.abs(block_f64 - idct_recon)))
                    if max_err <= acc_value:
                        opt_L = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                L_vals.append(float(opt_L))

            rows.append(
                [
                    std_dev,
                    mean_val,
                    dc_value,
                    ac_mean,
                    ac_std,
                    ac_abs_mean,
                    ac_abs_max,
                    zero_ratio,
                    small_vals_ratio,
                    medium_vals_ratio,
                    large_vals_ratio,
                    energy_ratio,
                    entropy,
                    zero_run_count,
                    zero_run_mean,
                    zero_run_max,
                    # Columns 16-20: min_L per accuracy
                    L_vals[0],
                    L_vals[1],
                    L_vals[2],
                    L_vals[3],
                    L_vals[4],
                ]
            )

        if not rows:
            return np.empty((0, 21), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    except Exception as e:
        print(f"  Error {image_path}: {e}")
        import traceback

        traceback.print_exc()
        return np.empty((0, 21), dtype=np.float32)


def collect_features_from_dataset(
    dataset_path: str,
    output_dir: str,
    num_files: int,
    block_sizes: List[int] = [8, 16],
    scaling_factors: List[int] = [1, 10],
    channel_mode: str = "first",
    num_workers: Optional[int] = None,
) -> None:
    """
    Collects 16 features and min_L (for 5 accuracy levels) for all blocks in the dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the folder with .tif files
    output_dir : str
        Output folder for .npy files
    num_files : int
        Maximum number of files to process
    block_sizes : List[int]
        Block sizes to test
    scaling_factors : List[int]
        Scaling factors to test
    channel_mode : str
        "first", "rgb", or "all": channels to use
    num_workers : Optional[int]
        Number of workers (default: all available cores)
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    dataset_dir = Path(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect input files
    tif_files = sorted(dataset_dir.glob("**/*.tif"))[:num_files]
    print(f"\nFound {len(tif_files)} .tif files")

    # Prepare tasks
    tasks = []
    for tif_file in tif_files:
        try:
            with rasterio.open(tif_file) as src:
                if channel_mode == "first":
                    channels = [0] if src.count >= 1 else []
                elif channel_mode == "rgb":
                    channels = list(range(min(src.count, 3)))
                else:  # "all"
                    channels = list(range(src.count))

                nodata_value = src.nodata

                for channel_idx in channels:
                    for block_size in block_sizes:
                        for scaling_factor in scaling_factors:
                            tasks.append(
                                (
                                    str(tif_file),
                                    block_size,
                                    channel_idx,
                                    scaling_factor,
                                    0.0,  # accuracy_px: unused by the worker, which evaluates the fixed levels 2, 5, 10, 15 and 20 px
                                    nodata_value,
                                )
                            )
        except Exception as e:
            print(f"  Warning: Cannot read {tif_file}: {e}")
            continue

    print(f"Prepared {len(tasks)} tasks")

    # Group tasks by (block_size, scaling_factor)
    from collections import defaultdict

    task_groups = defaultdict(list)
    for task in tasks:
        key = (task[1], task[3])  # (block_size, scaling_factor)
        task_groups[key].append(task)

    # Process each group
    for (block_size, scaling_factor), group_tasks in task_groups.items():
        print(f"\n{'='*60}")
        print(f"Processing: block_size={block_size}, scaling_factor={scaling_factor}")
        print(f"  Tasks: {len(group_tasks)}")

        start = time.time()
        chunks = []
        blocks_total = 0
        completed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(_collect_features_worker, task): task for task in group_tasks
            }

            for future in as_completed(future_to_task):
                completed += 1
                arr = future.result()
                if arr.shape[0] > 0:
                    chunks.append(arr)
                    blocks_total += arr.shape[0]

                if completed % 10 == 0 or completed == len(group_tasks):
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"  {completed}/{len(group_tasks)} tasks | "
                        f"{blocks_total:,} blocks | {rate:.1f} tasks/s"
                    )

        # Save the results
        if chunks:
            all_data = np.concatenate(chunks, axis=0)
            out_file = output_path / f"uint8_features_N{block_size}_sf{scaling_factor}_raw.npy"
            np.save(out_file, all_data)

            names_file = output_path / f"uint8_features_N{block_size}_sf{scaling_factor}_names.json"
            with open(names_file, "w") as f:
                json.dump(FEATURE_NAMES, f, indent=2)

            print(f"\nZapisano: {out_file}")
            print(f"  Shape: {all_data.shape}")
            print(f"  Size: {all_data.nbytes / 1024 / 1024:.1f} MB")

            # Statistics per column
            for col_i, name in enumerate(FEATURE_NAMES):
                col = all_data[:, col_i]
                pct = np.percentile(col, [10, 50, 90, 99])
                print(
                    f"  [{col_i:2d}] {name:<22s}  "
                    f"min={col.min():10.4f}  p10={pct[0]:10.4f}  "
                    f"p50={pct[1]:10.4f}  p90={pct[2]:10.4f}  "
                    f"p99={pct[3]:10.4f}  max={col.max():10.4f}"
                )
        else:
            print(
                f"  No data to save for block_size={block_size}, scaling_factor={scaling_factor}"
            )

        elapsed = time.time() - start
        print(f"Czas: {elapsed/60:.1f} min")


# =============================================================================
# Integracja z uint8_accuracy_sweep.py
# =============================================================================


def collect_features_during_sweep(
    data: np.ndarray,
    block_size: int,
    scaling_factor: int,
    accuracy_px: float,
    max_blocks_per_channel: Optional[int] = None,
) -> np.ndarray:
    """
    Collects per-block features for a single channel during a sweep.

    Parameters
    ----------
    data : np.ndarray
        Channel data (2D array)
    block_size : int
        Block size
    scaling_factor : int
        Scaling factor
    accuracy_px : float
        Accuracy target as the maximum error in pixels

    Returns
    -------
    np.ndarray
        Array shape (N_blocks, 17):
          columns 0-15: 16 features
          column 16: min_L for the given accuracy
    """
    # Pad and extract blocks
    padded_image, _ = _pad_image(data, block_size)
    blocks = _extract_blocks(padded_image, block_size)

    # Optionally limit the number of blocks per channel (sampling)
    if (
        max_blocks_per_channel is not None
        and max_blocks_per_channel > 0
        and len(blocks) > max_blocks_per_channel
    ):
        # Spread the samples evenly over the whole image (deterministic)
        indices = np.linspace(0, len(blocks) - 1, max_blocks_per_channel, dtype=int)
        selected_blocks = [blocks[i] for i in indices]
    else:
        selected_blocks = blocks

    acc_value = float(accuracy_px)
    rows = []

    for block_idx, block, _ in selected_blocks:
        if block is None:
            continue
        if np.all(block == 0):
            continue

        # --- 16 statistical features ---
        std_dev = float(np.std(block))
        mean_val = float(np.mean(block))

        # DCT
        dct_block = dct2(block, dct_type=2)
        dct_block_scaled = dct_block / scaling_factor
        dct_block_r = np.round(dct_block_scaled, 2)
        dct_zigzag = to_zigzag(dct_block_r)

        dc_value = float(dct_zigzag[0])
        ac = dct_zigzag[1:]

        ac_mean = float(np.mean(ac))
        ac_std = float(np.std(ac))
        ac_abs = np.abs(ac)
        ac_abs_mean = float(np.mean(ac_abs))
        ac_abs_max = float(np.max(ac_abs)) if len(ac_abs) > 0 else 0.0

        total = len(dct_zigzag)
        zero_ratio = float(np.sum(dct_zigzag == 0) / total)
        small_vals_ratio = float(np.sum(ac_abs < 1.0) / len(ac)) if len(ac) > 0 else 0.0
        medium_vals_ratio = (
            float(np.sum((ac_abs >= 1.0) & (ac_abs < 10.0)) / len(ac)) if len(ac) > 0 else 0.0
        )
        large_vals_ratio = float(np.sum(ac_abs >= 10.0) / len(ac)) if len(ac) > 0 else 0.0

        energy_total = float(np.sum(dct_zigzag**2))
        energy_ac = float(np.sum(ac**2))
        energy_ratio = energy_ac / energy_total if energy_total > 0 else 0.0

        hist, _ = np.histogram(dct_zigzag, bins=50, density=True)
        hist_nz = hist[hist > 0]
        entropy = float(-np.sum(hist_nz * np.log2(hist_nz))) if len(hist_nz) > 0 else 0.0

        in_run = False
        run_len = 0
        runs = []
        for v in ac:
            if v == 0.0:
                in_run = True
                run_len += 1
            else:
                if in_run:
                    runs.append(run_len)
                    run_len = 0
                    in_run = False
        if in_run:
            runs.append(run_len)
        zero_run_count = float(len(runs))
        zero_run_mean = float(np.mean(runs)) if runs else 0.0
        zero_run_max = float(max(runs)) if runs else 0.0

        # --- Binary search for min_L at the given accuracy ---
        n = block_size
        block_f64 = block.astype(np.float64)
        lo, hi = 1, len(dct_zigzag)
        opt_L = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            arr = np.zeros(len(dct_zigzag), dtype=np.float64)
            arr[:mid] = dct_zigzag[:mid]
            arr_unscaled = arr * scaling_factor
            recon = from_zigzag(arr_unscaled, n)
            idct_recon = idct2(recon, 2)
            max_err = float(np.max(np.abs(block_f64 - idct_recon)))
            if max_err <= acc_value:
                opt_L = mid
                hi = mid - 1
            else:
                lo = mid + 1

        rows.append(
            [
                std_dev,
                mean_val,
                dc_value,
                ac_mean,
                ac_std,
                ac_abs_mean,
                ac_abs_max,
                zero_ratio,
                small_vals_ratio,
                medium_vals_ratio,
                large_vals_ratio,
                energy_ratio,
                entropy,
                zero_run_count,
                zero_run_mean,
                zero_run_max,
                opt_L,
            ]
        )

    if not rows:
        return np.empty((0, 17), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Uint8 per-block feature collection")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num-files", type=int, default=5, help="Number of files to process")
    parser.add_argument(
        "--block-sizes", type=str, default="8,16", help="Comma-separated block sizes"
    )
    parser.add_argument(
        "--scaling-factors", type=str, default="1,10", help="Comma-separated scaling factors"
    )
    parser.add_argument(
        "--channels", choices=["first", "rgb", "all"], default="first", help="Channel mode"
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers")

    args = parser.parse_args()

    block_sizes = [int(x.strip()) for x in args.block_sizes.split(",")]
    scaling_factors = [int(x.strip()) for x in args.scaling_factors.split(",")]

    collect_features_from_dataset(
        dataset_path=args.dataset,
        output_dir=args.output,
        num_files=args.num_files,
        block_sizes=block_sizes,
        scaling_factors=scaling_factors,
        channel_mode=args.channels,
        num_workers=args.num_workers,
    )
