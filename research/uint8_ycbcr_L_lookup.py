#!/usr/bin/env python3
"""
Collect YCbCr-specific uint8 raw feature shards for RGB cm=6 training.

This script is separate from the single-channel uint8 trainers. It uses only
the first three bands of each multiband raster as RGB, converts RGB blocks to
YCbCr, and collects per-channel Y/Cb/Cr DCT features with RGB-validated L
targets. It intentionally saves raw .npy shards first. A separate builder can
convert these shards to the final compact NPZ model later, just like
research/build_uint8_lookup_from_features.py does for the old uint8 pipeline.

Typical workflow from the repository root:

python3 research/uint8_ycbcr_L_lookup.py collect \
  --dataset /home/infostrateg/temp/source/multi_uint_6ch \
  --output results/ycbcr_features \
  --block-sizes 8 \
  --scaling-factors 1,10 \
  --accuracies 2,3,5,10,20,30 \
  --fallback-multipliers 1.0,0.9,0.8,0.7,0.6,0.5,0.3,0.2 \
  --num-workers 8

Outputs:
  results/ycbcr_features/uint8_ycbcr_features_N8_sf1_part00000.npy
    results/ycbcr_features/uint8_ycbcr_features_N8_sf10_part00000.npy
    results/ycbcr_features/uint8_ycbcr_feature_names.json
    results/ycbcr_features/uint8_ycbcr_collection_config.json
    results/ycbcr_features/uint8_ycbcr_completed_files.jsonl
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dct import dct2, from_zigzag, idct2, to_zigzag

FEATURE_NAMES = [
    "channel_id",  # 0=Y, 1=Cb, 2=Cr
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

CHANNEL_IDS = {"Y": 0, "Cb": 1, "Cr": 2}
CHANNEL_NAMES = {value: key for key, value in CHANNEL_IDS.items()}

N_BINS_ACM = 40
N_BINS_ZR = 10
SAMPLE_N = 200_000
MIN_CELL = 5
SHARD_ROWS = 250_000
TEMP_OUTPUT_ROOTS = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def is_under_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_persistent_output(output: Path) -> None:
    for temp_root in TEMP_OUTPUT_ROOTS:
        if is_under_path(output, temp_root):
            raise SystemExit(
                f"Output directory {output} is temporary. Use a persistent directory, "
                "for example results/ycbcr_features, so training can resume after interruption."
            )


def decimal_places_for_sf(scaling_factor: int) -> int:
    return int(math.ceil(math.log10(scaling_factor))) if scaling_factor > 1 else 0


def pad_image(image: np.ndarray, block_size: int) -> np.ndarray:
    height, width = image.shape
    pad_h = (block_size - height % block_size) % block_size
    pad_w = (block_size - width % block_size) % block_size
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")


def iter_block_slices(height: int, width: int, block_size: int) -> Iterable[tuple[slice, slice]]:
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield slice(row, row + block_size), slice(col, col + block_size)


def rgb_to_ycbcr(
    red: np.ndarray, green: np.ndarray, blue: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.clip(0.299 * red + 0.587 * green + 0.114 * blue, 0, 255)
    cb = np.clip(-0.169 * red - 0.331 * green + 0.500 * blue + 128.0, 0, 255)
    cr = np.clip(0.500 * red - 0.419 * green - 0.081 * blue + 128.0, 0, 255)
    return y, cb, cr


def compute_features(spatial_uint8: np.ndarray, zigzag: np.ndarray, channel_id: int) -> list[float]:
    ac = zigzag[1:]
    ac_abs = np.abs(ac)
    energy_total = float(np.sum(zigzag * zigzag))
    energy_ac = float(np.sum(ac * ac))

    hist, _ = np.histogram(zigzag, bins=50, density=True)
    hist_nz = hist[hist > 0]
    entropy = float(-np.sum(hist_nz * np.log2(hist_nz))) if len(hist_nz) else 0.0

    runs = []
    run_len = 0
    for value in ac:
        if value == 0.0:
            run_len += 1
        elif run_len:
            runs.append(run_len)
            run_len = 0
    if run_len:
        runs.append(run_len)

    return [
        float(channel_id),
        float(np.std(spatial_uint8)),
        float(np.mean(spatial_uint8)),
        float(zigzag[0]),
        float(np.mean(ac)),
        float(np.std(ac)),
        float(np.mean(ac_abs)),
        float(np.max(ac_abs)) if len(ac_abs) else 0.0,
        float(np.sum(ac == 0)) / max(1, len(ac)),
        float(np.sum(ac_abs < 1.0)) / max(1, len(ac)),
        float(np.sum((ac_abs >= 1.0) & (ac_abs < 10.0))) / max(1, len(ac)),
        float(np.sum(ac_abs >= 10.0)) / max(1, len(ac)),
        energy_ac / energy_total if energy_total > 0 else 0.0,
        entropy,
        float(len(runs)),
        float(np.mean(runs)) if runs else 0.0,
        float(max(runs)) if runs else 0.0,
    ]


def min_l_for_channel_accuracy(
    zigzag: np.ndarray, centered_block: np.ndarray, accuracy: float, scaling_factor: int
) -> int:
    total_len = len(zigzag)
    block_size = centered_block.shape[0]
    lo, hi = 1, total_len
    best = total_len
    orig_u8 = centered_block + 128.0

    while lo <= hi:
        mid = (lo + hi) // 2
        coeffs = np.zeros(total_len, dtype=np.float64)
        quantized = np.round(zigzag[:mid] * scaling_factor).astype(np.float64) / scaling_factor
        coeffs[:mid] = quantized
        reconstructed = idct2(from_zigzag(coeffs, block_size), 2)
        recon_u8 = np.clip(np.round(reconstructed + 128.0), 0, 255)
        diff = orig_u8 - recon_u8
        err = max(float(diff.max()), float(-diff.min()))
        if err <= accuracy:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return int(best)


def reconstruct_channel(
    zigzag: np.ndarray, length: int, block_size: int, scaling_factor: int
) -> np.ndarray:
    coeffs = np.zeros(len(zigzag), dtype=np.float64)
    quantized = np.round(zigzag[:length] * scaling_factor).astype(np.float64) / scaling_factor
    coeffs[:length] = quantized
    reconstructed = idct2(from_zigzag(coeffs, block_size), 2)
    return np.clip(reconstructed + 128.0, 0.0, 255.0)


def rgb_max_error(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
) -> float:
    red_rec = np.clip(np.round(y + 1.403 * (cr - 128.0)), 0, 255)
    green_rec = np.clip(np.round(y - 0.344 * (cb - 128.0) - 0.714 * (cr - 128.0)), 0, 255)
    blue_rec = np.clip(np.round(y + 1.773 * (cb - 128.0)), 0, 255)
    return float(
        max(
            np.max(np.abs(red - red_rec)),
            np.max(np.abs(green - green_rec)),
            np.max(np.abs(blue - blue_rec)),
        )
    )


def target_l_triplet_for_rgb(
    zigzags: dict[str, np.ndarray],
    centered_blocks: dict[str, np.ndarray],
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    accuracy: float,
    scaling_factor: int,
    fallback_multipliers: list[float],
) -> dict[str, int]:
    block_size = centered_blocks["Y"].shape[0]
    best_lengths = {channel: len(zigzag) for channel, zigzag in zigzags.items()}
    best_error = float("inf")

    for multiplier in fallback_multipliers:
        channel_accuracy = multiplier * accuracy
        lengths = {
            channel: min_l_for_channel_accuracy(
                zigzag, centered_blocks[channel], channel_accuracy, scaling_factor
            )
            for channel, zigzag in zigzags.items()
        }
        y_rec = reconstruct_channel(zigzags["Y"], lengths["Y"], block_size, scaling_factor)
        cb_rec = reconstruct_channel(zigzags["Cb"], lengths["Cb"], block_size, scaling_factor)
        cr_rec = reconstruct_channel(zigzags["Cr"], lengths["Cr"], block_size, scaling_factor)
        error = rgb_max_error(red, green, blue, y_rec, cb_rec, cr_rec)
        if error < best_error:
            best_error = error
            best_lengths = lengths
        if error <= accuracy:
            return lengths

    return best_lengths


def collect_file(
    worker_args: tuple[str, list[int], list[int], list[float], list[float], int],
) -> dict[tuple[int, int], np.ndarray]:
    (
        file_path,
        block_sizes,
        scaling_factors,
        accuracies,
        fallback_multipliers,
        max_blocks_per_file,
    ) = worker_args
    rows_by_cfg: dict[tuple[int, int], list[list[float]]] = {
        (block_size, scaling_factor): []
        for block_size in block_sizes
        for scaling_factor in scaling_factors
    }

    with rasterio.open(file_path) as src:
        if src.count < 3:
            return empty_result(rows_by_cfg, len(accuracies))
        red = src.read(1).astype(np.float64)
        green = src.read(2).astype(np.float64)
        blue = src.read(3).astype(np.float64)
        nodata = src.nodata

    for block_size in block_sizes:
        red_p = pad_image(red, block_size)
        green_p = pad_image(green, block_size)
        blue_p = pad_image(blue, block_size)
        height, width = red_p.shape
        slices = list(iter_block_slices(height, width, block_size))

        if max_blocks_per_file > 0 and len(slices) > max_blocks_per_file:
            selected = set(
                np.linspace(0, len(slices) - 1, max_blocks_per_file, dtype=np.int64).tolist()
            )
        else:
            selected = None

        for block_index, (row_slice, col_slice) in enumerate(slices):
            if selected is not None and block_index not in selected:
                continue

            r_block = red_p[row_slice, col_slice]
            g_block = green_p[row_slice, col_slice]
            b_block = blue_p[row_slice, col_slice]
            if np.all(r_block == 0) and np.all(g_block == 0) and np.all(b_block == 0):
                continue
            if nodata is not None and (
                np.any(r_block == nodata) or np.any(g_block == nodata) or np.any(b_block == nodata)
            ):
                continue

            y_block, cb_block, cr_block = rgb_to_ycbcr(r_block, g_block, b_block)
            channels = (
                (CHANNEL_IDS["Y"], y_block, y_block - 128.0),
                (CHANNEL_IDS["Cb"], cb_block, cb_block - 128.0),
                (CHANNEL_IDS["Cr"], cr_block, cr_block - 128.0),
            )

            for scaling_factor in scaling_factors:
                decimal_places = decimal_places_for_sf(scaling_factor)
                zigzags: dict[str, np.ndarray] = {}
                centered_blocks: dict[str, np.ndarray] = {}
                rows: dict[str, list[float]] = {}
                for channel_name, (channel_id, spatial_u8, centered) in zip(
                    ("Y", "Cb", "Cr"), channels
                ):
                    dct_block = np.round(dct2(centered, dct_type=2), decimal_places)
                    zigzag = to_zigzag(dct_block)
                    zigzags[channel_name] = zigzag
                    centered_blocks[channel_name] = centered
                    rows[channel_name] = compute_features(spatial_u8, zigzag, channel_id)

                targets_by_accuracy = [
                    target_l_triplet_for_rgb(
                        zigzags,
                        centered_blocks,
                        r_block,
                        g_block,
                        b_block,
                        accuracy,
                        scaling_factor,
                        fallback_multipliers,
                    )
                    for accuracy in accuracies
                ]

                for channel_name in ("Y", "Cb", "Cr"):
                    row = rows[channel_name]
                    row.extend(targets[channel_name] for targets in targets_by_accuracy)
                    rows_by_cfg[(block_size, scaling_factor)].append(row)

    return {
        key: (
            np.asarray(rows, dtype=np.float32)
            if rows
            else np.empty((0, len(FEATURE_NAMES) + len(accuracies)), dtype=np.float32)
        )
        for key, rows in rows_by_cfg.items()
    }


def profile_one_file(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).expanduser()
    block_sizes = parse_int_list(args.block_sizes)
    scaling_factors = parse_int_list(args.scaling_factors)
    accuracies = parse_float_list(args.accuracies)
    fallback_multipliers = parse_float_list(args.fallback_multipliers)
    block_size = block_sizes[0]
    files = sorted(dataset.glob("**/*.tif"))
    if not files:
        raise SystemExit(f"No .tif files found in {dataset}")
    file_path = str(files[0])
    max_blocks = args.max_blocks_per_file if args.max_blocks_per_file > 0 else 200

    print(f"Profiling one real file: {file_path}")
    start = time.perf_counter()
    with rasterio.open(file_path) as src:
        print(
            f"  raster={src.width}x{src.height} bands={src.count} dtype={src.dtypes[:3]} nodata={src.nodata}"
        )
        red = src.read(1).astype(np.float64)
        green = src.read(2).astype(np.float64)
        blue = src.read(3).astype(np.float64)
        nodata = src.nodata
    read_sec = time.perf_counter() - start

    start = time.perf_counter()
    red_p = pad_image(red, block_size)
    green_p = pad_image(green, block_size)
    blue_p = pad_image(blue, block_size)
    slices = list(iter_block_slices(*red_p.shape, block_size))
    pad_sec = time.perf_counter() - start
    selected = slices[: min(max_blocks, len(slices))]

    timings = {"ycbcr": 0.0, "dct_features": 0.0, "targets": 0.0}
    rows = 0
    skipped = 0
    start_all = time.perf_counter()
    for row_slice, col_slice in tqdm(
        selected, desc="Profile blocks", unit="blk", dynamic_ncols=True
    ):
        r_block = red_p[row_slice, col_slice]
        g_block = green_p[row_slice, col_slice]
        b_block = blue_p[row_slice, col_slice]
        if np.all(r_block == 0) and np.all(g_block == 0) and np.all(b_block == 0):
            skipped += 1
            continue
        if nodata is not None and (
            np.any(r_block == nodata) or np.any(g_block == nodata) or np.any(b_block == nodata)
        ):
            skipped += 1
            continue

        t0 = time.perf_counter()
        y_block, cb_block, cr_block = rgb_to_ycbcr(r_block, g_block, b_block)
        timings["ycbcr"] += time.perf_counter() - t0
        channels = (
            ("Y", CHANNEL_IDS["Y"], y_block, y_block - 128.0),
            ("Cb", CHANNEL_IDS["Cb"], cb_block, cb_block - 128.0),
            ("Cr", CHANNEL_IDS["Cr"], cr_block, cr_block - 128.0),
        )

        for scaling_factor in scaling_factors:
            decimal_places = decimal_places_for_sf(scaling_factor)
            zigzags: dict[str, np.ndarray] = {}
            centered_blocks: dict[str, np.ndarray] = {}
            t0 = time.perf_counter()
            for channel_name, channel_id, spatial_u8, centered in channels:
                dct_block = np.round(dct2(centered, dct_type=2), decimal_places)
                zigzag = to_zigzag(dct_block)
                zigzags[channel_name] = zigzag
                centered_blocks[channel_name] = centered
                compute_features(spatial_u8, zigzag, channel_id)
            timings["dct_features"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            for accuracy in accuracies:
                target_l_triplet_for_rgb(
                    zigzags,
                    centered_blocks,
                    r_block,
                    g_block,
                    b_block,
                    accuracy,
                    scaling_factor,
                    fallback_multipliers,
                )
            timings["targets"] += time.perf_counter() - t0
            rows += 3

    total_sec = time.perf_counter() - start_all
    processed = len(selected) - skipped
    print("\nProfile summary")
    print(f"  read: {read_sec:.3f}s")
    print(f"  pad_and_slices: {pad_sec:.3f}s")
    print(f"  total_blocks_in_file: {len(slices)}")
    print(f"  profiled_blocks: {len(selected)} processed={processed} skipped={skipped}")
    print(f"  scaling_factors: {scaling_factors} accuracies: {accuracies}")
    print(f"  output_rows_if_collected: {rows}")
    print(f"  measured_processing: {total_sec:.3f}s ({(total_sec / max(1, processed)):.4f}s/block)")
    for name, seconds in timings.items():
        pct = 100.0 * seconds / total_sec if total_sec > 0 else 0.0
        print(f"  {name}: {seconds:.3f}s ({pct:.1f}%)")
    estimated = (total_sec / max(1, processed)) * len(slices)
    print(
        f"  estimated_one_file_processing: {estimated:.1f}s ({estimated / 60.0:.1f} min) on one process"
    )


def empty_result(
    keys: dict[tuple[int, int], list[list[float]]], accuracy_count: int
) -> dict[tuple[int, int], np.ndarray]:
    return {
        key: np.empty((0, len(FEATURE_NAMES) + accuracy_count), dtype=np.float32) for key in keys
    }


def flush_shards(
    output_dir: Path,
    buffers: dict[tuple[int, int], list[np.ndarray]],
    counters: dict[tuple[int, int], int],
) -> None:
    for key, arrays in list(buffers.items()):
        if not arrays:
            continue
        block_size, scaling_factor = key
        data = np.vstack(arrays)
        part = counters.get(key, 0)
        path = (
            output_dir / f"uint8_ycbcr_features_N{block_size}_sf{scaling_factor}_part{part:05d}.npy"
        )
        np.save(path, data)
        counters[key] = part + 1
        buffers[key] = []
        print(f"  saved {path} shape={data.shape}")


def init_part_counters(
    output_dir: Path, block_sizes: list[int], scaling_factors: list[int]
) -> dict[tuple[int, int], int]:
    counters: dict[tuple[int, int], int] = {}
    for block_size in block_sizes:
        for scaling_factor in scaling_factors:
            pattern = f"uint8_ycbcr_features_N{block_size}_sf{scaling_factor}_part*.npy"
            max_part = -1
            for path in output_dir.glob(pattern):
                stem = path.stem
                try:
                    part = int(stem.rsplit("part", 1)[1])
                except (IndexError, ValueError):
                    continue
                max_part = max(max_part, part)
            counters[(block_size, scaling_factor)] = max_part + 1
    return counters


def load_completed_files(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    completed: set[str] = set()
    with open(progress_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_path = item.get("file")
            if file_path:
                completed.add(file_path)
    return completed


def append_completed_file(progress_path: Path, file_path: str) -> None:
    with open(progress_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"file": file_path, "time": time.time()}) + "\n")


def collect_features(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).expanduser()
    output = Path(args.output).expanduser()
    validate_persistent_output(output)
    output.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, int(args.num_workers))

    block_sizes = parse_int_list(args.block_sizes)
    scaling_factors = parse_int_list(args.scaling_factors)
    accuracies = parse_float_list(args.accuracies)
    fallback_multipliers = parse_float_list(args.fallback_multipliers)
    files = sorted(dataset.glob("**/*.tif"))
    if args.num_files > 0:
        files = files[: args.num_files]
    if not files:
        raise SystemExit(f"No .tif files found in {dataset}")

    progress_path = output / "uint8_ycbcr_completed_files.jsonl"
    if args.resume:
        completed_files = load_completed_files(progress_path)
        if completed_files:
            before = len(files)
            files = [path for path in files if str(path) not in completed_files]
            print(f"Resume enabled: skipped {before - len(files)} completed file(s)")
    if not files:
        print("No files left to process")
        return

    names = FEATURE_NAMES + [
        f"L_acc{int(acc) if float(acc).is_integer() else str(acc).replace('.', 'p')}"
        for acc in accuracies
    ]
    with open(output / "uint8_ycbcr_feature_names.json", "w", encoding="utf-8") as handle:
        json.dump(names, handle, indent=2)
    with open(output / "uint8_ycbcr_collection_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": str(dataset),
                "num_files": args.num_files,
                "block_sizes": block_sizes,
                "scaling_factors": scaling_factors,
                "accuracies": accuracies,
                "fallback_multipliers": fallback_multipliers,
                "max_blocks_per_file": args.max_blocks_per_file,
                "num_workers": worker_count,
                "flush_every_file": bool(args.flush_every_file),
                "resume": bool(args.resume),
                "feature_names": names,
                "format": "uint8_ycbcr_raw_features_v1",
                "rgb_bands": [1, 2, 3],
            },
            handle,
            indent=2,
        )

    print(f"Collecting YCbCr features from {len(files)} file(s)")
    print(f"  dataset={dataset}")
    print(f"  output={output}")
    print(f"  first three bands are treated as RGB")
    print(f"  block_sizes={block_sizes} scaling_factors={scaling_factors} accuracies={accuracies}")
    print(f"  fallback_multipliers={fallback_multipliers}")
    print(f"  worker_processes={worker_count}")
    print(f"  checkpoint_after_each_file={args.flush_every_file} resume={args.resume}")

    buffers: dict[tuple[int, int], list[np.ndarray]] = {}
    counters = init_part_counters(output, block_sizes, scaling_factors)
    buffer_rows: dict[tuple[int, int], int] = {}
    start = time.time()

    worker_args = [
        (
            str(path),
            block_sizes,
            scaling_factors,
            accuracies,
            fallback_multipliers,
            args.max_blocks_per_file,
        )
        for path in files
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(collect_file, item): item[0] for item in worker_args}
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="YCbCr files",
            unit="file",
            dynamic_ncols=True,
        )
        for done, future in enumerate(progress, start=1):
            file_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                tqdm.write(f"  failed {file_path}: {exc}")
                continue

            for key, arr in result.items():
                if arr.size == 0:
                    continue
                buffers.setdefault(key, []).append(arr)
                buffer_rows[key] = buffer_rows.get(key, 0) + arr.shape[0]
                if buffer_rows[key] >= args.shard_rows:
                    flush_shards(output, buffers, counters)
                    buffer_rows[key] = 0

            if args.flush_every_file:
                flush_shards(output, buffers, counters)
                buffer_rows = {}
            append_completed_file(progress_path, file_path)

            elapsed = time.time() - start
            progress.set_postfix_str(f"{done}/{len(files)} files, {elapsed / 60.0:.1f} min")

    flush_shards(output, buffers, counters)
    print(f"Done in {(time.time() - start) / 60.0:.1f} min")


def percentile_from_hist(hist: np.ndarray, percentile: float) -> np.ndarray:
    coverage = hist.sum(axis=2).astype(np.int64)
    threshold = np.maximum(1, np.ceil(coverage * (percentile / 100.0)).astype(np.int64))
    cdf = np.cumsum(hist, axis=2)
    above = cdf >= threshold[:, :, np.newaxis]
    values = np.argmax(above, axis=2).astype(np.float32)
    values[coverage == 0] = 0.0
    return values


def fill_sparse_grid(
    grid_l: np.ndarray, coverage: np.ndarray, percentile: float, max_l: int
) -> np.ndarray:
    filled_grid = grid_l.astype(np.float32, copy=True)
    for ia in range(filled_grid.shape[0]):
        valid = filled_grid[ia][coverage[ia] >= MIN_CELL]
        if len(valid):
            fill_value = float(np.percentile(valid, percentile))
            filled_grid[ia][coverage[ia] < MIN_CELL] = fill_value
    populated = filled_grid[filled_grid > 0]
    fallback = float(np.percentile(populated, percentile)) if len(populated) else float(max_l // 2)
    filled_grid[filled_grid == 0] = fallback
    return np.clip(filled_grid, 1, max_l).astype(np.float32)


def label_accuracy(accuracy: float) -> str:
    return str(int(accuracy)) if float(accuracy).is_integer() else str(accuracy).replace(".", "p")


def build_combined_model(
    input_dir: Path,
    models_dir: Path,
    block_size: int,
    scaling_factor: int,
    accuracies: list[float],
    percentile: float,
) -> None:
    files = sorted(
        input_dir.glob(f"uint8_ycbcr_features_N{block_size}_sf{scaling_factor}_part*.npy")
    )
    if not files:
        print(f"  skip combined N={block_size} sf={scaling_factor}: no feature shards")
        return

    acm_sample = []
    total_sampled = 0
    for path in files:
        data = np.load(path)
        if data.size == 0:
            continue
        take = min(len(data), SAMPLE_N - total_sampled)
        if take <= 0:
            break
        acm_sample.append(data[:take, 6].astype(np.float32))
        total_sampled += take

    if not acm_sample:
        print(f"  skip combined N={block_size} sf={scaling_factor}: no samples")
        return

    acm_arr = np.concatenate(acm_sample)
    edges_acm = np.unique(np.percentile(acm_arr, np.linspace(0, 100, N_BINS_ACM + 1))).astype(
        np.float64
    )
    if len(edges_acm) < 2:
        print(f"  skip combined N={block_size} sf={scaling_factor}: not enough acm variation")
        return

    n_acm = len(edges_acm) - 1
    n_zr = N_BINS_ZR
    max_l = block_size * block_size
    edges_zr = np.linspace(0.0, 1.0 + 1e-6, n_zr + 1, dtype=np.float64)
    hist = np.zeros((len(CHANNEL_NAMES), len(accuracies), n_acm, n_zr, max_l + 1), dtype=np.int32)

    for path in files:
        data = np.load(path)
        if data.size == 0:
            continue
        channel_ids = np.clip(data[:, 0].astype(np.int32), 0, len(CHANNEL_NAMES) - 1)
        acm = data[:, 6].astype(np.float32)
        zr = data[:, 8].astype(np.float32)
        i_acm = np.clip(np.searchsorted(edges_acm, acm, side="right") - 1, 0, n_acm - 1)
        i_zr = np.clip((zr * n_zr).astype(np.int32), 0, n_zr - 1)
        for accuracy_index in range(len(accuracies)):
            target_col = len(FEATURE_NAMES) + accuracy_index
            target_l = np.clip(data[:, target_col].astype(np.int32), 1, max_l)
            np.add.at(hist, (channel_ids, accuracy_index, i_acm, i_zr, target_l), 1)

    coverage = hist.sum(axis=4).astype(np.int32)
    grid_l = np.zeros((len(CHANNEL_NAMES), len(accuracies), n_acm, n_zr), dtype=np.float32)
    for channel_id in CHANNEL_NAMES:
        for accuracy_index in range(len(accuracies)):
            raw_grid = percentile_from_hist(hist[channel_id, accuracy_index], percentile)
            grid_l[channel_id, accuracy_index] = fill_sparse_grid(
                raw_grid,
                coverage[channel_id, accuracy_index],
                percentile,
                max_l,
            )

    out_path = (
        models_dir
        / f"lookup_uint8_ycbcr_L_N{block_size}_sf{scaling_factor}_p{int(percentile)}_model.npz"
    )
    np.savez_compressed(
        out_path,
        grid_L=grid_l,
        edges_acm=edges_acm,
        edges_zr=edges_zr,
        coverage=coverage,
        accuracy_levels=np.asarray(accuracies, dtype=np.float32),
        channel_names=np.asarray([CHANNEL_NAMES[idx] for idx in range(len(CHANNEL_NAMES))]),
        channel_ids=np.asarray(list(range(len(CHANNEL_NAMES))), dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES),
        feature_ac_abs_mean_index=np.array([FEATURE_NAMES.index("ac_abs_mean")], dtype=np.int32),
        feature_zero_ratio_index=np.array([FEATURE_NAMES.index("zero_ratio")], dtype=np.int32),
        percentile=np.array([float(percentile)], dtype=np.float32),
        min_cell=np.array([MIN_CELL], dtype=np.int32),
    )
    filled = int(np.sum(coverage >= MIN_CELL))
    total = int(np.prod(coverage.shape))
    print(
        f"  saved {out_path} grid_L={grid_l.shape} coverage={filled}/{total} "
        f"L min/med/max={grid_l.min():.0f}/{np.median(grid_l):.0f}/{grid_l.max():.0f}"
    )


def build_lookup_for(
    input_dir: Path,
    models_dir: Path,
    block_size: int,
    scaling_factor: int,
    accuracy_index: int,
    accuracy_label: str,
    percentile: float,
) -> None:
    files = sorted(
        input_dir.glob(f"uint8_ycbcr_features_N{block_size}_sf{scaling_factor}_part*.npy")
    )
    if not files:
        print(f"  skip N={block_size} sf={scaling_factor}: no feature shards")
        return

    acm_samples = {channel_id: [] for channel_id in CHANNEL_NAMES}
    total_sampled = {channel_id: 0 for channel_id in CHANNEL_NAMES}
    for path in files:
        data = np.load(path)
        for channel_id in CHANNEL_NAMES:
            rows = data[data[:, 0].astype(np.int32) == channel_id]
            if rows.size == 0 or total_sampled[channel_id] >= SAMPLE_N:
                continue
            take = min(len(rows), SAMPLE_N - total_sampled[channel_id])
            acm_samples[channel_id].append(rows[:take, 6].astype(np.float32))
            total_sampled[channel_id] += take

    max_l = block_size * block_size
    target_col = len(FEATURE_NAMES) + accuracy_index

    for channel_id, channel_name in CHANNEL_NAMES.items():
        if not acm_samples[channel_id]:
            print(f"  skip {channel_name}: no samples")
            continue
        acm_arr = np.concatenate(acm_samples[channel_id])
        edges_acm = np.unique(np.percentile(acm_arr, np.linspace(0, 100, N_BINS_ACM + 1))).astype(
            np.float64
        )
        if len(edges_acm) < 2:
            print(f"  skip {channel_name}: not enough acm variation")
            continue
        n_acm = len(edges_acm) - 1
        edges_zr = np.linspace(0.0, 1.0 + 1e-6, N_BINS_ZR + 1, dtype=np.float64)
        hist = np.zeros((n_acm, N_BINS_ZR, max_l + 1), dtype=np.int64)

        for path in files:
            data = np.load(path)
            rows = data[data[:, 0].astype(np.int32) == channel_id]
            if rows.size == 0:
                continue
            acm = rows[:, 6].astype(np.float32)
            zr = rows[:, 8].astype(np.float32)
            target_l = np.clip(rows[:, target_col].astype(np.int32), 1, max_l)
            i_acm = np.clip(np.searchsorted(edges_acm, acm, side="right") - 1, 0, n_acm - 1)
            i_zr = np.clip((zr * N_BINS_ZR).astype(np.int32), 0, N_BINS_ZR - 1)
            np.add.at(hist, (i_acm, i_zr, target_l), 1)

        coverage = hist.sum(axis=2).astype(np.int32)
        grid_l = fill_sparse_grid(
            percentile_from_hist(hist, percentile), coverage, percentile, max_l
        )

        out_path = (
            models_dir
            / f"lookup_uint8_ycbcr_L_{channel_name}_N{block_size}_sf{scaling_factor}_acc{accuracy_label}_p{int(percentile)}_grid.npz"
        )
        np.savez_compressed(
            out_path,
            # Compatibility keys used by the existing uint8 L predictor.
            grid_L=grid_l,
            edges_acm=edges_acm,
            edges_zr=edges_zr,
            coverage=coverage,
            # Extra diagnostics: safe for np.load users, ignored by old readers.
            channel=np.array([channel_name]),
            accuracy=np.array([float(accuracy_label.replace("p", "."))], dtype=np.float32),
            percentile=np.array([float(percentile)], dtype=np.float32),
            feature_names=np.array(FEATURE_NAMES),
            feature_ac_abs_mean_index=np.array(
                [FEATURE_NAMES.index("ac_abs_mean")], dtype=np.int32
            ),
            feature_zero_ratio_index=np.array([FEATURE_NAMES.index("zero_ratio")], dtype=np.int32),
            min_cell=np.array([MIN_CELL], dtype=np.int32),
        )
        print(
            f"  saved {out_path} shape={grid_l.shape} "
            f"L min/med/max={grid_l.min():.0f}/{np.median(grid_l):.0f}/{grid_l.max():.0f} "
            f"coverage={(coverage >= MIN_CELL).sum()}/{coverage.size}"
        )


def build_lookups(args: argparse.Namespace) -> None:
    input_dir = Path(args.output).expanduser()
    models_dir = Path(args.models).expanduser()
    validate_persistent_output(input_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    block_sizes = parse_int_list(args.block_sizes)
    scaling_factors = parse_int_list(args.scaling_factors)
    accuracies = parse_float_list(args.accuracies)

    for block_size in block_sizes:
        for scaling_factor in scaling_factors:
            build_combined_model(
                input_dir, models_dir, block_size, scaling_factor, accuracies, args.percentile
            )
            for accuracy_index, accuracy in enumerate(accuracies):
                label = label_accuracy(accuracy)
                build_lookup_for(
                    input_dir,
                    models_dir,
                    block_size,
                    scaling_factor,
                    accuracy_index,
                    label,
                    args.percentile,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect raw YCbCr uint8 feature shards for cm=6 RGB compression"
    )
    parser.add_argument(
        "mode",
        choices=["collect", "build", "profile"],
        help="Use collect for long training; profile diagnoses one real file",
    )
    parser.add_argument("--dataset", default="/home/infostrateg/temp/source/multi_uint_6ch")
    parser.add_argument("--output", default="results/ycbcr_features")
    parser.add_argument("--models", default="models")
    parser.add_argument("--num-files", type=int, default=0, help="0 = all files")
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--block-sizes", default="8")
    parser.add_argument("--scaling-factors", default="1,10")
    parser.add_argument("--accuracies", default="2,3")
    parser.add_argument("--fallback-multipliers", default="1.0,0.9,0.8,0.7,0.6,0.5,0.3,0.2")
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--max-blocks-per-file", type=int, default=0, help="0 = all blocks")
    parser.add_argument("--shard-rows", type=int, default=SHARD_ROWS)
    parser.add_argument(
        "--flush-every-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint raw .npy shards after each completed input file",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files already listed in uint8_ycbcr_completed_files.jsonl",
    )
    args = parser.parse_args()

    if args.mode == "collect":
        collect_features(args)
    if args.mode == "profile":
        profile_one_file(args)
    if args.mode == "build":
        build_lookups(args)


if __name__ == "__main__":
    main()
