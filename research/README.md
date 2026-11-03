# Research / Experimental Code

This directory contains research scripts for training the heuristic lookup tables
used by FIDWAC v2's `advanced_heuristic` predictor. These scripts are **not** part
of the production build — they generate the `.npz` model files in `models/`.

## Training Pipeline — uint8 single-channel (cm=5)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `uint8_feature_collection.py` | Extract 16 DCT features per block from training GeoTIFFs. Outputs `.npy` feature shards. |
| 2a | `build_uint8_lookup_from_features.py` | Build 2D quantile grid `(ac_abs_mean, zero_ratio) → (sf, bs)` from collected features (streaming, handles 50+ GB). |
| 2b | `build_uint8_L_lookup.py` | Build per-block L prediction grids `(ac_abs_mean, zero_ratio) → L` per `(N, sf)` config. |
| 3 | `test_uint8_lookup.py` | Validate generated lookup grid: structure, single/batch lookups, fill ratio. |

## Training Pipeline — YCbCr / RGB (cm=6)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `uint8_ycbcr_L_lookup.py` | Collect per-channel Y/Cb/Cr DCT features with RGB-validated L targets. Outputs raw `.npy` shards. |
| 2 | `build_ycbcr_lookup.py` | Convert YCbCr feature shards into final `.npz` lookup grids (per accuracy, per sf). |

## Heuristic Training

| Script | Description |
|--------|-------------|
| `heuristic_uint8_trainer.py` | Train heuristic rules for optimal `(sf, bs)` prediction from channel-level statistics. |
| `heuristic_uint8.py` | Auto-generated heuristic rules (output of `heuristic_uint8_trainer.py`). |

## Usage

All scripts should be run from the project root directory:

```bash
# --- uint8 single-channel pipeline ---
# Step 1: Collect features
python3 research/uint8_feature_collection.py collect

# Step 2: Build lookup grid
python3 research/build_uint8_lookup_from_features.py

# Step 3: Validate
python3 research/test_uint8_lookup.py

# --- YCbCr RGB pipeline (cm=6) ---
# Step 1: Collect YCbCr features
python3 research/uint8_ycbcr_L_lookup.py collect \
    --dataset /path/to/dataset \
    --output results/ycbcr_features \
    --block-sizes 8 \
    --scaling-factors 1,10 \
    --accuracies 2,3,5,10,20,30

# Step 2: Build YCbCr lookup grids
python3 research/build_ycbcr_lookup.py \
    --features /abs/path/results/ycbcr_features \
    --models /abs/path/models \
    --block-sizes 8 \
    --scaling-factors 1,10 \
    --accuracies 2,3,5,10,20,30 \
    --percentile 90
```

## Output

Generated `.npz` files go to `models/` (configured via `directories.models` in `config.json`).
The production predictor (`predictor/predictor.py`, `predictor/predictor_uint8.py`) loads
them automatically at runtime when `advanced_heuristic = true`.

## tif_compare — GeoTIFF Comparison Tool

`tif_compare/` contains tools for comparing original and decompressed GeoTIFF files
to verify compression accuracy:

| Script | Description |
|--------|-------------|
| `tif_compare/compare_tif.py` | CLI tool — compares two GeoTIFFs pixel-by-pixel, reports max error, mean error, and shape/dtype differences. |
| `tif_compare/compare_tif_gui.py` | GUI version — visual comparison with difference maps, histograms, and side-by-side preview using matplotlib. |

### Usage

```bash
# CLI
python3 research/tif_compare/compare_tif.py original.tif restored.tif

# GUI
python3 research/tif_compare/compare_tif_gui.py
```
