#!/usr/bin/env python3
import rasterio
import numpy as np
import sys

file1 = "/mnt/q/temp/FIDWAC_v7/fidwac_v7/data/results/riverbottom_AUTO8_Acc0.05_tdct2_dec2_CRSepsg2180_VT.tif"
file2 = "/mnt/q/temp/FIDWAC_v7/fidwac_v7/data/results/riverbottom.tif"

try:
    print(f"Reading: {file1}")
    with rasterio.open(file1) as src1:
        data1 = src1.read()
        profile1 = src1.profile
        print(f"  - Shape: {data1.shape}, dtype: {data1.dtype}")
        print(f"  - Min: {np.nanmin(data1)}, Max: {np.nanmax(data1)}")

    print(f"\nReading: {file2}")
    with rasterio.open(file2) as src2:
        data2 = src2.read()
        profile2 = src2.profile
        print(f"  - Shape: {data2.shape}, dtype: {data2.dtype}")
        print(f"  - Min: {np.nanmin(data2)}, Max: {np.nanmax(data2)}")

    # Check if shapes match
    if data1.shape != data2.shape:
        print(f"\n⚠️  WARNING: Different shapes!")
        print(f"  File1: {data1.shape}")
        print(f"  File2: {data2.shape}")
        print("\n  Using smaller range for comparison...")
        min_rows = min(data1.shape[1], data2.shape[1])
        min_cols = min(data1.shape[2], data2.shape[2])
        min_bands = min(data1.shape[0], data2.shape[0])
        data1_crop = data1[:min_bands, :min_rows, :min_cols]
        data2_crop = data2[:min_bands, :min_rows, :min_cols]
    else:
        data1_crop = data1
        data2_crop = data2

    # Compute difference
    print(f"\nComputing difference (file1 - file2)...")
    difference = data1_crop.astype(np.float64) - data2_crop.astype(np.float64)

    # Statistics
    abs_diff = np.abs(difference)
    max_diff = np.nanmax(abs_diff)
    min_diff = np.nanmin(abs_diff)
    mean_diff = np.nanmean(abs_diff)
    std_diff = np.nanstd(abs_diff)

    print(f"\n📊 DIFFERENCE STATISTICS:")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Min difference: {min_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")
    print(f"  Std deviation: {std_diff:.10f}")

    # Pixels with difference > 0
    non_zero = np.sum(abs_diff > 0)
    total = abs_diff.size
    print(f"\n  Pixels with difference > 0: {non_zero} / {total} ({100*non_zero/total:.2f}%)")

    # Difference histogram
    print(f"\n📈 DIFFERENCE PERCENTILES:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.nanpercentile(abs_diff, p)
        print(f"  Percentile {p:3d}%: {val:.10f}")

except Exception as e:
    print(f"❌ ERROR: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    sys.exit(1)
