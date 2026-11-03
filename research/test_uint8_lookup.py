#!/usr/bin/env python3
"""
Test uint8 lookup grid after generation completes.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from predictor.predictor_uint8 import (
    load_uint8_lookup_grid,
    predict_uint8_parameters,
    predict_uint8_parameters_batch,
)


def test_single_lookup():
    """Test single-block lookup."""

    grid_path = Path(__file__).parent.parent / "models" / "lookup_uint8_grid.npz"

    if not grid_path.exists():
        print(f"❌ Grid not found: {grid_path}")
        return False

    print(f"Loading grid from {grid_path}...")
    grid_data = load_uint8_lookup_grid(str(grid_path))

    if grid_data is None:
        print("❌ Failed to load grid")
        return False

    print("✓ Grid loaded")
    print(f"  Grid shape: {grid_data['grid_sf'].shape}")
    print(f"  Edges acm: {len(grid_data['edges_acm'])} bins")
    print(f"  Edges zr: {len(grid_data['edges_zr'])} bins")

    # Test lookups at different feature values
    test_cases = [
        (5.0, 0.1),  # Low ac_abs_mean, low zero_ratio
        (10.0, 0.5),  # Medium ac_abs_mean, medium zero_ratio
        (20.0, 0.8),  # High ac_abs_mean, high zero_ratio
    ]

    print("\nSingle-block lookups:")
    for acm, zr in test_cases:
        sf, bs = predict_uint8_parameters(acm, zr, grid_data)
        print(f"  acm={acm:.1f}, zr={zr:.1f} → sf={sf}, bs={bs}")

    return True


def test_batch_lookup():
    """Test batch lookup."""

    grid_path = Path(__file__).parent.parent / "models" / "lookup_uint8_grid.npz"

    if not grid_path.exists():
        print(f"❌ Grid not found: {grid_path}")
        return False

    grid_data = load_uint8_lookup_grid(str(grid_path))

    if grid_data is None:
        print("❌ Failed to load grid")
        return False

    # Test batch lookup
    batch_size = 1000
    acm_values = np.random.uniform(1, 30, batch_size).astype(np.float32)
    zr_values = np.random.uniform(0, 1, batch_size).astype(np.float32)

    print(f"\nBatch lookup ({batch_size} samples):")
    sfs, bss = predict_uint8_parameters_batch(acm_values, zr_values, grid_data)

    print(f"  Scaling factors: {np.unique(sfs)}")
    print(f"  Block sizes: {np.unique(bss)}")
    print(f"  SF distribution: {np.bincount(sfs)}")
    print(f"  BS distribution: {np.bincount(bss)}")

    return True


def verify_grid_structure():
    """Verify grid structure matches expected format."""

    grid_path = Path(__file__).parent.parent / "models" / "lookup_uint8_grid.npz"

    if not grid_path.exists():
        print(f"❌ Grid not found: {grid_path}")
        return False

    data = np.load(grid_path)

    print("\nGrid structure:")
    expected_keys = {"grid_sf", "grid_bs", "grid_mult", "grid_n", "edges_acm", "edges_zr"}
    actual_keys = set(data.files)

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    if missing:
        print(f"  ❌ Missing keys: {missing}")
        return False

    if extra:
        print(f"  ⚠ Extra keys: {extra}")

    print("  ✓ All required keys present")

    # Check shapes
    grid_sf = data["grid_sf"]
    edges_acm = data["edges_acm"]
    edges_zr = data["edges_zr"]

    print(f"  Grid shape (sf, bs): {grid_sf.shape}")
    print(f"  Edges acm: {len(edges_acm)-1} bins ({edges_acm[0]:.4f} - {edges_acm[-1]:.4f})")
    print(f"  Edges zr: {len(edges_zr)-1} bins ({edges_zr[0]:.4f} - {edges_zr[-1]:.4f})")

    # Check value ranges
    print(f"  SF range: {grid_sf.min()} - {grid_sf.max()}")
    print(f"  BS range: {data['grid_bs'].min()} - {data['grid_bs'].max()}")

    # Check fill ratio
    filled = (data["grid_n"] > 0).sum()
    total = data["grid_n"].size
    print(f"  Grid fill: {filled}/{total} ({100*filled/total:.1f}%)")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("UINT8 LOOKUP GRID TEST")
    print("=" * 70)

    if not verify_grid_structure():
        print("\n❌ Grid structure verification failed")
        sys.exit(1)

    if not test_single_lookup():
        print("\n❌ Single lookup test failed")
        sys.exit(1)

    if not test_batch_lookup():
        print("\n❌ Batch lookup test failed")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
