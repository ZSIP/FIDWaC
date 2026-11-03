"""
Build per-block DCT length (L) prediction lookup grids from feature data.

For each (block_size N, scaling_factor sf) combination, builds a 2D quantile grid:
  (ac_abs_mean, zero_ratio) → median optimal L_acc

Uses 3D histogram accumulation (ac_abs_mean bins × zero_ratio bins × L_value)
for fast streaming without storing raw values in memory.

Output: models/lookup_uint8_L_{cfg}_grid.npz per (N, sf) combination.
"""

import numpy as np
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEAT_DIR = Path("results/features")
MODEL_DIR = Path("models")
SAMPLE_N = 120_000  # rows to sample for ac_abs_mean quantile edge estimation
N_BINS_ACM = 40  # bins for ac_abs_mean axis (quantile-based)
N_BINS_ZR = 10  # bins for zero_ratio axis (fixed [0..1])
MIN_CELL = 5  # min samples per grid cell to trust median


# ---------------------------------------------------------------------------
def _max_L_for_cfg(cfg: str) -> int:
    """Return maximum possible L value for this config (N*N)."""
    if "N8" in cfg:
        return 64
    if "N16" in cfg:
        return 256
    if "N32" in cfg:
        return 1024
    return 64


def build_L_lookup(cfg: str, n_bins_acm: int = N_BINS_ACM, n_bins_zr: int = N_BINS_ZR):
    """Build and save a (acm, zr) → median-L lookup grid for one (N, sf) config."""
    files = sorted(FEAT_DIR.glob(f"uint8_features_{cfg}_part*.npy"))
    if not files:
        print(f"  [SKIP] No files found for {cfg}")
        return None

    max_L = _max_L_for_cfg(cfg)
    print(f"\n=== Building L lookup for {cfg} ({len(files)} files, max_L={max_L}) ===")

    # ------------------------------------------------------------------
    # PASS 1: sample for ac_abs_mean quantile edges
    # ------------------------------------------------------------------
    print(f"  Pass 1: sampling up to {SAMPLE_N:,} rows for ac_abs_mean quantile edges ...")
    acm_sample = []
    total_sampled = 0
    for path in files:
        d = np.load(path)
        acm_sample.append(d[:, 5].astype(np.float32))
        total_sampled += len(d)
        if total_sampled >= SAMPLE_N:
            break

    acm_arr = np.concatenate(acm_sample)[:SAMPLE_N]
    acm_pcts = np.linspace(0, 100, n_bins_acm + 1)
    edges_acm = np.unique(np.percentile(acm_arr, acm_pcts))
    n_acm = len(edges_acm) - 1

    # Fixed edges for zero_ratio [0..1]
    edges_zr = np.linspace(0.0, 1.0 + 1e-6, n_bins_zr + 1)
    n_zr = n_bins_zr

    print(f"  Grid size: {n_acm} × {n_zr} (acm bins × zr bins)")

    # ------------------------------------------------------------------
    # PASS 2: stream all files, fill 3D histogram hist[acm_bin, zr_bin, L]
    # ------------------------------------------------------------------
    print(f"  Pass 2: streaming all files (3D histogram) ...")
    hist = np.zeros((n_acm, n_zr, max_L + 1), dtype=np.int32)  # L in [1..max_L]
    total_rows = 0
    report_every = max(1, len(files) // 20)

    for fi, path in enumerate(files):
        d = np.load(path)
        acm = d[:, 5].astype(np.float32)
        zr = d[:, 7].astype(np.float32)
        L = d[:, 16].astype(np.int32)

        # Bin indices
        i_acm = np.clip(np.searchsorted(edges_acm, acm, side="right") - 1, 0, n_acm - 1)
        i_zr = np.clip((zr * n_zr).astype(np.int32), 0, n_zr - 1)
        L_clamped = np.clip(L, 1, max_L)

        # Vectorized histogram update
        np.add.at(hist, (i_acm, i_zr, L_clamped), 1)
        total_rows += len(d)

        if fi % report_every == 0:
            print(f"    {fi + 1}/{len(files)} files, {total_rows:,} rows ...", flush=True)

    print(f"  Total rows processed: {total_rows:,}")

    # ------------------------------------------------------------------
    # Compute median L per cell from histogram
    # ------------------------------------------------------------------
    grid_L = np.zeros((n_acm, n_zr), dtype=np.float32)
    coverage = hist.sum(axis=2).astype(np.int32)  # shape (n_acm, n_zr)

    # CDF-based median: find L where cumulative count >= 50% of total
    L_values = np.arange(max_L + 1, dtype=np.float32)
    cumhist = np.cumsum(hist, axis=2)  # (n_acm, n_zr, max_L+1)
    half_total = coverage[:, :, np.newaxis] / 2.0
    # First L index where cumhist >= half_total
    above_half = cumhist >= half_total  # bool (n_acm, n_zr, max_L+1)

    for ia in range(n_acm):
        for iz in range(n_zr):
            if coverage[ia, iz] >= MIN_CELL:
                idx = np.argmax(above_half[ia, iz])  # first True
                grid_L[ia, iz] = float(max(1, idx))

    # Fill sparse cells: propagate along acm axis (row-wise median then global)
    for ia in range(n_acm):
        filled = grid_L[ia][coverage[ia] >= MIN_CELL]
        if len(filled) > 0:
            row_med = float(np.median(filled))
            for iz in range(n_zr):
                if coverage[ia, iz] < MIN_CELL:
                    grid_L[ia, iz] = row_med

    # Fill remaining zeros with global median
    populated = grid_L[grid_L > 0]
    global_med = float(np.median(populated)) if len(populated) > 0 else float(max_L // 2)
    grid_L[grid_L == 0] = global_med
    grid_L = np.clip(grid_L, 1, max_L)

    n_filled = int(np.sum(coverage >= MIN_CELL))
    pct_filled = 100.0 * n_filled / (n_acm * n_zr)
    print(f"  Grid fill rate: {n_filled}/{n_acm * n_zr} = {pct_filled:.1f}%")
    print(
        f"  L_acc in grid: min={grid_L.min():.0f}, median={np.median(grid_L):.0f}, max={grid_L.max():.0f}"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = MODEL_DIR / f"lookup_uint8_L_{cfg}_grid.npz"
    np.savez_compressed(
        out_path,
        grid_L=grid_L,
        edges_acm=edges_acm,
        edges_zr=edges_zr,
        coverage=coverage,
    )
    sz = out_path.stat().st_size / 1024
    print(f"  Saved: {out_path} ({sz:.1f} KB)")
    return out_path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)

    configs_to_build = ["N8_sf1", "N8_sf10", "N16_sf1", "N16_sf10"]

    # Allow subset via CLI args
    if len(sys.argv) > 1:
        configs_to_build = sys.argv[1:]

    results = {}
    for cfg in configs_to_build:
        path = build_L_lookup(cfg)
        results[cfg] = path

    print("\n=== Summary ===")
    for cfg, path in results.items():
        if path:
            sz = Path(path).stat().st_size / 1024
            print(f"  {cfg}: {path} ({sz:.1f} KB)")
        else:
            print(f"  {cfg}: SKIPPED")
