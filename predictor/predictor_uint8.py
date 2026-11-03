"""
Uint8 parameter predictor using lookup grid (sf, bs) from features.

Maps (ac_abs_mean, zero_ratio) → (scaling_factor, block_size) using 2D quantile grid.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_uint8_lookup_grid(grid_file: str = "models/lookup_uint8_grid.npz"):
    """Load uint8 lookup grid (sf, bs) from NPZ file."""

    grid_file = Path(grid_file)

    if not grid_file.exists():
        return None  # Fallback to defaults

    try:
        data = np.load(grid_file)
        return {
            "grid_sf": data["grid_sf"],  # (n_acm_bins, n_zr_bins) uint8
            "grid_bs": data["grid_bs"],  # (n_acm_bins, n_zr_bins) uint8
            "grid_mult": data["grid_mult"],  # (n_acm_bins, n_zr_bins) float32
            "edges_acm": data["edges_acm"],  # (n_acm_bins+1,) float32
            "edges_zr": data["edges_zr"],  # (n_zr_bins+1,) float32
        }
    except Exception as e:
        print(f"Warning: Failed to load uint8 lookup grid: {e}")
        return None


def predict_uint8_parameters(ac_abs_mean: float, zero_ratio: float, grid_data: dict) -> tuple:
    """
    Single-block lookup: (ac_abs_mean, zero_ratio) → (sf, bs)

    Returns:
        (sf, bs): scaling_factor, block_size
        Falls back to (1, 16) if grid is None or lookup fails.
    """

    if grid_data is None:
        return 1, 16  # Defaults

    try:
        grid_sf = grid_data["grid_sf"]
        grid_bs = grid_data["grid_bs"]
        edges_acm = grid_data["edges_acm"]
        edges_zr = grid_data["edges_zr"]

        # Find bin indices using quantile edges
        i = np.clip(np.searchsorted(edges_acm[1:-1], ac_abs_mean), 0, grid_sf.shape[0] - 1)
        j = np.clip(np.searchsorted(edges_zr[1:-1], zero_ratio), 0, grid_sf.shape[1] - 1)

        sf = int(grid_sf[i, j])
        bs = int(grid_bs[i, j])

        return sf, bs

    except Exception as e:
        print(f"Warning: Lookup failed for ({ac_abs_mean}, {zero_ratio}): {e}")
        return 1, 16


def predict_uint8_parameters_batch(
    ac_abs_means: np.ndarray, zero_ratios: np.ndarray, grid_data: dict
) -> tuple:
    """
    Batch lookup: (n,) arrays → (n,) arrays of (sf, bs)

    Parameters
    ----------
    ac_abs_means : np.ndarray, shape (n,), float32
        AC absolute mean values
    zero_ratios : np.ndarray, shape (n,), float32
        Zero ratio values
    grid_data : dict
        Grid lookup data from load_uint8_lookup_grid()

    Returns
    -------
    (sfs, bss) : tuple of np.ndarray, shape (n,) uint8
        Scaling factors and block sizes for each sample
    """

    if grid_data is None:
        n = len(ac_abs_means)
        return np.ones(n, dtype=np.uint8), np.full(n, 16, dtype=np.uint8)

    try:
        grid_sf = grid_data["grid_sf"]
        grid_bs = grid_data["grid_bs"]
        edges_acm = grid_data["edges_acm"]
        edges_zr = grid_data["edges_zr"]

        # Vectorized bin lookup
        i = np.clip(np.searchsorted(edges_acm[1:-1], ac_abs_means), 0, grid_sf.shape[0] - 1)
        j = np.clip(np.searchsorted(edges_zr[1:-1], zero_ratios), 0, grid_sf.shape[1] - 1)

        sfs = grid_sf[i, j].astype(np.uint8)
        bss = grid_bs[i, j].astype(np.uint8)

        return sfs, bss

    except Exception as e:
        print(f"Warning: Batch lookup failed: {e}")
        n = len(ac_abs_means)
        return np.ones(n, dtype=np.uint8), np.full(n, 16, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Per-block DCT length (L) predictor
# ---------------------------------------------------------------------------


def load_uint8_L_grid(
    nloc: int, sf: int, model_dir: str = "models", is_ycbcr: bool = False, accuracy: float = None
) -> dict | None:
    """
    Load the per-block L prediction lookup grid for given block size and scaling factor.

    Parameters
    ----------
    nloc : int
        Block size (8, 16, 32).
    sf : int
        Uint8 scaling factor (1, 10, ...).
    model_dir : str
        Directory containing lookup files.
    is_ycbcr : bool
        True if grid should be loaded specifically for YCbCr (RGB) per-block mode.
    accuracy : float, optional
        Target accuracy (max error in pixel values, e.g., 3.0, 5.0) to load exact grid.

    Returns
    -------
    dict with keys: grid_L, edges_acm, edges_zr
        or None if file not found.
    """
    # Convert model_dir to absolute path
    model_dir = str(Path(model_dir).resolve())
    # Predefined accuracies
    predefined_accuracies = [2, 3, 5, 10, 20, 30]

    # 1. Try accuracy-specific combined YCbCr file (grid_L_Y / grid_L_Cb / grid_L_Cr)
    if is_ycbcr and accuracy is not None:
        # First try exact accuracy
        acc_label = (
            str(int(accuracy)) if float(accuracy).is_integer() else str(accuracy).replace(".", "p")
        )
        path = Path(model_dir) / f"lookup_uint8_ycbcr_L_N{nloc}_sf{sf}_acc{acc_label}_grid.npz"

        # If exact accuracy not found, try closest smaller predefined accuracy
        if not path.exists():
            smaller_accuracies = [a for a in predefined_accuracies if a <= accuracy]
            if smaller_accuracies:
                closest_smaller = max(smaller_accuracies)
                acc_label = str(closest_smaller)
                path = (
                    Path(model_dir) / f"lookup_uint8_ycbcr_L_N{nloc}_sf{sf}_acc{acc_label}_grid.npz"
                )
                if path.exists():
                    print(
                        f"Using closest smaller predefined accuracy {closest_smaller} "
                        f"for requested accuracy {accuracy}"
                    )

        if path.exists():
            try:
                data = np.load(path)
                if "grid_L_Y" in data:
                    return {
                        "grid_L_Y": data["grid_L_Y"].astype(np.float32),
                        "grid_L_Cb": data["grid_L_Cb"].astype(np.float32),
                        "grid_L_Cr": data["grid_L_Cr"].astype(np.float32),
                        "edges_acm": data["edges_acm"].astype(np.float32),
                        "edges_zr": data["edges_zr"].astype(np.float32),
                        "max_L": int(nloc * nloc),
                    }
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

    # 2. Fallback: generic YCbCr or standard uint8 grid (single grid_L)
    if is_ycbcr:
        path = Path(model_dir) / f"lookup_uint8_ycbcr_L_N{nloc}_sf{sf}_grid.npz"
        if not path.exists():
            path = Path(model_dir) / f"lookup_uint8_L_N{nloc}_sf{sf}_grid.npz"
    else:
        path = Path(model_dir) / f"lookup_uint8_L_N{nloc}_sf{sf}_grid.npz"
        if not path.exists():
            path = Path(model_dir) / f"lookup_uint8_ycbcr_L_N{nloc}_sf{sf}_grid.npz"

    if not path.exists():
        return None
    try:
        data = np.load(path)
        grid_L = data["grid_L"] if "grid_L" in data else (data["grid"] if "grid" in data else None)
        if grid_L is None:
            return None
        return {
            "grid_L": grid_L.astype(np.float32),
            "edges_acm": data["edges_acm"].astype(np.float32),
            "edges_zr": data["edges_zr"].astype(np.float32),
            "max_L": int(nloc * nloc),
        }
    except Exception as e:
        print(f"Warning: Failed to load L lookup grid {path}: {e}")
        return None


def predict_uint8_L(
    ac_abs_mean: float, zero_ratio: float, grid_data: dict | None, default_L: int | None = None
) -> int:
    """
    Predict optimal DCT start length L for a single block.

    Parameters
    ----------
    ac_abs_mean : float
        Mean absolute value of AC coefficients.
    zero_ratio : float
        Fraction of AC coefficients that are exactly zero.
    grid_data : dict or None
        Grid loaded by load_uint8_L_grid(). Falls back to default_L if None.
    default_L : int or None
        Fallback L if grid is None. If None, uses max_L // 2 from grid or 32.

    Returns
    -------
    int : predicted L (>= 1)
    """
    if grid_data is None:
        return default_L if default_L is not None else 32

    grid_L = grid_data["grid_L"]
    edges_acm = grid_data["edges_acm"]
    n_acm, n_zr = grid_L.shape

    i = int(np.clip(np.searchsorted(edges_acm[1:-1], ac_abs_mean), 0, n_acm - 1))
    j = int(np.clip(zero_ratio * n_zr, 0, n_zr - 1))

    return max(1, int(grid_L[i, j]))


def predict_uint8_L_batch(
    ac_abs_means: np.ndarray, zero_ratios: np.ndarray, grid_data: dict | None, default_L: int = 32
) -> np.ndarray:
    """
    Batch predict optimal DCT start length L for multiple blocks.

    Parameters
    ----------
    ac_abs_means : np.ndarray, shape (n,)
    zero_ratios  : np.ndarray, shape (n,)
    grid_data    : dict or None
    default_L    : int  fallback if grid is None

    Returns
    -------
    np.ndarray, shape (n,), dtype int32
    """
    n = len(ac_abs_means)
    if grid_data is None:
        return np.full(n, default_L, dtype=np.int32)

    grid_L = grid_data["grid_L"]
    edges_acm = grid_data["edges_acm"]
    n_acm, n_zr = grid_L.shape

    i = np.clip(np.searchsorted(edges_acm[1:-1], ac_abs_means.astype(np.float32)), 0, n_acm - 1)
    j = np.clip((zero_ratios.astype(np.float32) * n_zr).astype(np.int32), 0, n_zr - 1)

    return np.maximum(1, grid_L[i, j].astype(np.int32))


# =============================================================================
# UINT8 PREDICTOR CLASS (compatible with float predictor interface)
# =============================================================================


class Uint8Predictor:
    """
    Uint8-specific predictor using lookup grids for L prediction.

    Compatible with float predictor interface (HeuristicPredictor, AdvancedHeuristicPredictor).
    Uses (ac_abs_mean, zero_ratio) → L lookup from trained grids.
    """

    def __init__(
        self,
        accuracy: float = 5.0,
        block_size: int = 16,
        models_dir: str = "models",
        config=None,
        is_ycbcr: bool = False,
    ):
        """
        Initialize Uint8Predictor.

        Parameters
        ----------
        accuracy : float
            Target accuracy (max error in pixel values, e.g., 5.0)
        block_size : int
            Block size (8, 16, 32)
        models_dir : str
            Directory containing lookup tables
        config : Config
            Configuration object
        is_ycbcr : bool
            True if grid should be loaded specifically for YCbCr (RGB) per-block mode.
        """
        self.accuracy = accuracy
        self.block_size = block_size
        self.models_dir = models_dir
        self.config = config
        self.is_ycbcr = is_ycbcr

        # Get scaling factor from config
        if config is not None:
            sf_override = getattr(config, "_scaling_factor_override", None)
            sf_cfg = getattr(config.compression, "uint8_scaling_factor", [1])
            if sf_override is not None:
                self.scaling_factor = int(sf_override)
            elif isinstance(sf_cfg, (list, tuple)):
                self.scaling_factor = int(sf_cfg[0]) if sf_cfg else 1
            else:
                self.scaling_factor = int(sf_cfg)
        else:
            self.scaling_factor = 1

        self.has_channel_grids = False
        combined = load_uint8_L_grid(
            block_size, self.scaling_factor, models_dir, is_ycbcr=is_ycbcr, accuracy=accuracy
        )
        if combined is not None and "grid_L_Y" in combined:
            # Accuracy-specific combined file with separate Y/Cb/Cr grids
            edges = {
                "edges_acm": combined["edges_acm"],
                "edges_zr": combined["edges_zr"],
                "max_L": combined["max_L"],
            }
            self._l_grid_Y = {**edges, "grid_L": combined["grid_L_Y"]}
            self._l_grid_Cb = {**edges, "grid_L": combined["grid_L_Cb"]}
            self._l_grid_Cr = {**edges, "grid_L": combined["grid_L_Cr"]}
            self._l_grid = self._l_grid_Y  # fallback for non-batch paths
            self.has_channel_grids = True
        else:
            # Generic single-grid fallback
            self._l_grid = combined
            if self._l_grid is None:
                print(
                    f"WARNING: Uint8Predictor could not load L grid for "
                    f"N={block_size}, sf={self.scaling_factor}"
                )

    def predict_dct_length(self, block: np.ndarray) -> int:
        """
        Predict optimal DCT length for a single block.

        Parameters
        ----------
        block : np.ndarray
            Input block (uint8 or uint16, will be centered to [-128, 127])

        Returns
        -------
        int : Predicted L (DCT coefficient count)
        """
        if self._l_grid is None:
            # Fallback: use half of total coefficients
            return (self.block_size * self.block_size) // 2

        # Center block for DCT (0-255 -> -128..127)
        centered = block.astype(np.float32) - 128.0

        # Compute DCT and zigzag
        from core.dct import dct2, to_zigzag

        dct_block = dct2(centered, dct_type=2)
        dct_block = np.round(dct_block, 2)
        zigzag = to_zigzag(dct_block).astype(np.float64)

        # Extract features
        ac = zigzag[1:]  # skip DC
        ac_abs_mean = float(np.mean(np.abs(ac)))
        zero_ratio = float(np.sum(ac == 0)) / max(1, len(ac))

        return predict_uint8_L(ac_abs_mean, zero_ratio, self._l_grid)

    def predict_dct_length_batch(self, blocks: List[np.ndarray]) -> np.ndarray:
        """
        Predict optimal DCT lengths for a batch of blocks.

        Parameters
        ----------
        blocks : List[np.ndarray]
            List of input blocks

        Returns
        -------
        np.ndarray : Predicted L values for each block
        """
        if self._l_grid is None:
            n_blocks = len(blocks)
            return np.full(n_blocks, (self.block_size * self.block_size) // 2, dtype=np.int32)

        # Batch centering
        blocks_arr = np.array([b.astype(np.float32) - 128.0 for b in blocks])

        # Batch DCT
        from core.dct import _get_zigzag_indices
        from scipy.fft import dct as scipy_dct

        dct_batch = scipy_dct(scipy_dct(blocks_arr, axis=1, norm="ortho"), axis=2, norm="ortho")
        dct_batch = np.round(dct_batch, 2)

        # Zigzag conversion
        B = len(blocks)
        n = self.block_size
        idx_z = _get_zigzag_indices(n)
        zigzag_batch = dct_batch.reshape(B, -1)[:, idx_z]

        # Extract features batch
        ac_abs_means = np.empty(B, dtype=np.float32)
        zero_ratios = np.empty(B, dtype=np.float32)

        for k in range(B):
            ac = zigzag_batch[k][1:]  # skip DC
            ac_abs_means[k] = float(np.mean(np.abs(ac)))
            zero_ratios[k] = float(np.sum(ac == 0)) / max(1, len(ac))

        # Predict L batch
        predictions = predict_uint8_L_batch(ac_abs_means, zero_ratios, self._l_grid)

        return predictions

    def predict_from_dct_batch(self, dct_zigzags: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Predict L from pre-computed DCT zigzag coefficients (batch).

        Parameters
        ----------
        dct_zigzags : np.ndarray
            DCT coefficients in zigzag format, shape (batch, n²)

        Returns
        -------
        Tuple[np.ndarray, List]
            (predictions: np.ndarray[int32], skip_queues: list)
        """
        batch_size = len(dct_zigzags)

        if self._l_grid is None and not self.has_channel_grids:
            default_L = (self.block_size * self.block_size) // 2
            return np.full(batch_size, default_L, dtype=np.int32), [[] for _ in range(batch_size)]

        # features from the zigzag coefficients, vectorised over the batch
        ac = np.asarray(dct_zigzags)[:, 1:]  # skip DC
        ac_abs_means = np.mean(np.abs(ac), axis=1).astype(np.float32)
        zero_ratios = (np.sum(ac == 0, axis=1) / max(1, ac.shape[1])).astype(np.float32)

        # Use channel-specific grids when available (accuracy-specific combined file)
        if self.has_channel_grids:
            gn = batch_size // 3
            preds_Y = predict_uint8_L_batch(ac_abs_means[:gn], zero_ratios[:gn], self._l_grid_Y)
            preds_Cb = predict_uint8_L_batch(
                ac_abs_means[gn : 2 * gn], zero_ratios[gn : 2 * gn], self._l_grid_Cb
            )
            preds_Cr = predict_uint8_L_batch(
                ac_abs_means[2 * gn :], zero_ratios[2 * gn :], self._l_grid_Cr
            )
            predictions = np.concatenate([preds_Y, preds_Cb, preds_Cr])
        else:
            predictions = predict_uint8_L_batch(ac_abs_means, zero_ratios, self._l_grid)

        return predictions, [[] for _ in range(batch_size)]


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "load_uint8_lookup_grid",
    "predict_uint8_parameters",
    "predict_uint8_parameters_batch",
    "load_uint8_L_grid",
    "predict_uint8_L",
    "predict_uint8_L_batch",
    "Uint8Predictor",
]
