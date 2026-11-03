"""
FIDWAC v2 - DCT Length Predictors
==================================

Backends:
1. Heuristic - fast variance-based heuristic
2. Advanced Heuristic - 2D quantile grid lookup table (ac_abs_mean, zero_ratio)
"""

import os
from typing import Optional, List

import numpy as np
import numba as nb

from core.dct import dct2, to_zigzag
from .predictor_npy import predict_from_multi_lookup_npy

# ---------------------------------------------------------------------------
# Numba JIT — fast extraction of 4 features from DCT zigzag in a single pass.
# Uses Parseval's theorem (DCT ortho): std_dev = sqrt(sum(ac²)) / N
# Features: ac_abs_mean (#1 Spearman+MI), zero_ratio (#1 RF 87.9%),
#           ac_std (#3 Spearman), std_dev (#3 Spearman, from Parseval without orig block).
# ---------------------------------------------------------------------------


@nb.njit(cache=True, fastmath=True)
def _nb_extract_features(dct_zigzag: np.ndarray, block_n: int) -> tuple:
    """
    Extract 4 features in a single pass over AC zigzag.

    Returns (ac_abs_mean, zero_ratio, ac_std, std_dev).
    std_dev from Parseval's theorem: std_dev = sqrt(sum(ac²)) / N
    (exact for DCT with norm='ortho').
    """
    n_total = len(dct_zigzag)
    n_ac = n_total - 1  # skip DC (index 0)

    s = 0.0  # sum of AC
    s2 = 0.0  # sum of AC²
    s_abs = 0.0  # sum of |AC|
    n_zeros = 0  # count of exact zeros in AC

    for k in range(1, n_total):
        v = dct_zigzag[k]
        s += v
        s2 += v * v
        if v < 0.0:
            s_abs -= v
        else:
            s_abs += v
        if v == 0.0:
            n_zeros += 1

    ac_abs_mean = s_abs / n_ac
    ac_mean = s / n_ac
    variance = s2 / n_ac - ac_mean * ac_mean
    ac_std = variance**0.5 if variance > 0.0 else 0.0
    zero_ratio = n_zeros / n_ac
    std_dev = s2**0.5 / block_n  # Parseval: block std_dev = sqrt(sum(ac²)) / N

    return ac_abs_mean, zero_ratio, ac_std, std_dev


@nb.njit(cache=True, fastmath=True, parallel=True)
def _nb_extract_features_batch(dct_zigzags: np.ndarray) -> tuple:
    """
    Batch extraction of (ac_abs_mean, zero_ratio) in a single pass.

    One pass over data, zero temporary allocations, parallel across batch.
    Used by predict_from_dct_batch() and predict_dct_length_batch().

    Parameters
    ----------
    dct_zigzags : np.ndarray shape (batch, n²)
        DCT in zigzag format (float64 or float32), column 0 = DC

    Returns
    -------
    (ac_abs_mean, zero_ratio) : tuple of float32 arrays, shape (batch,)
    """
    batch_size = dct_zigzags.shape[0]
    n_total = dct_zigzags.shape[1]
    n_ac = n_total - 1

    ac_abs_mean = np.empty(batch_size, dtype=np.float32)
    zero_ratio = np.empty(batch_size, dtype=np.float32)

    for b in nb.prange(batch_size):  # pylint: disable=not-an-iterable
        s_abs = 0.0
        n_zeros = 0
        for k in range(1, n_total):
            v = dct_zigzags[b, k]
            if v < 0.0:
                s_abs -= v
            else:
                s_abs += v
            if v == 0.0:
                n_zeros += 1
        ac_abs_mean[b] = s_abs / n_ac
        zero_ratio[b] = n_zeros / n_ac

    return ac_abs_mean, zero_ratio


# Warm-up JIT on import (8×8 dummy block)
_dz = np.zeros(64, dtype=np.float64)
_nb_extract_features(_dz, 8)
_dz_batch = np.zeros((1, 64), dtype=np.float64)
_nb_extract_features_batch(_dz_batch)
del _dz, _dz_batch


# =============================================================================
# HEURISTIC PREDICTOR
# =============================================================================


class HeuristicPredictor:
    """
    Fast heuristic for DCT chain length prediction.

    Based on simple block statistics (variance).
    Automatically adapts to different block sizes.
    """

    def __init__(self, accuracy: float = 0.05, block_size: int = 8):
        self.accuracy = accuracy
        self.block_size = block_size
        self.total_coeffs = block_size * block_size

        # Accuracy -> coefficient PERCENTAGE mapping
        self.accuracy_to_ratio = {
            0.01: 0.75,  # very high quality
            0.05: 0.50,  # high quality
            0.1: 0.37,  # medium quality
            0.5: 0.25,  # low quality
            1.0: 0.19,  # very low quality
        }

    def predict_dct_length(self, block: np.ndarray) -> int:
        """Predict optimal DCT length for a block."""
        # Auto-detect block size
        if hasattr(block, "shape"):
            detected_size = block.shape[0]
            if detected_size != self.block_size:
                self.block_size = detected_size
                self.total_coeffs = detected_size * detected_size

        # Baseline based on accuracy
        base_ratio = self._get_base_ratio(self.accuracy)
        base_length = int(base_ratio * self.total_coeffs)

        # Variance-based correction
        variance = float(np.var(block))

        # Dynamic thresholds scaled with block size
        size_scale = (self.block_size / 8.0) ** 2
        high_threshold = 1000 * size_scale
        mid_threshold = 100 * size_scale
        low_threshold = 10 * size_scale

        if variance > high_threshold:
            correction = 1.3
        elif variance > mid_threshold:
            correction = 1.1
        elif variance < low_threshold:
            correction = 0.8
        else:
            correction = 1.0

        predicted = int(base_length * correction)
        return max(4, min(predicted, self.total_coeffs))

    def predict_dct_length_batch(self, blocks: List[np.ndarray]) -> np.ndarray:
        """Batch prediction for a list of blocks."""
        return np.array([self.predict_dct_length(block) for block in blocks], dtype=np.int32)

    def _get_base_ratio(self, accuracy: float) -> float:
        """Map accuracy to base coefficient percentage."""
        accuracies = sorted(self.accuracy_to_ratio.keys())
        closest = min(accuracies, key=lambda x: abs(x - accuracy))
        return self.accuracy_to_ratio[closest]


# =============================================================================
# ADVANCED HEURISTIC PREDICTOR (Lookup Table)
# =============================================================================


class AdvancedHeuristicPredictor:
    """
    Advanced heuristic based on 2D quantile grid lookup table.

    Uses (ac_abs_mean, zero_ratio) — top-2 features from feature analysis —
    with a 2D quantile grid (64×64 = 4096 cells) to predict L_optimal.
    Much more accurate than simple heuristic, but requires pre-generated tables.
    """

    def __init__(
        self, accuracy: float = 0.05, block_size: int = 8, lookup_dir: Optional[str] = None
    ):
        self.accuracy = accuracy
        self.block_size = block_size
        self.total_coeffs = block_size * block_size
        self.lookup_dir = lookup_dir

        self._multi_lookup_array = None
        self._top_features = None
        self._use_npy_format = False

        self._load_lookup_tables()

        if not self._use_npy_format:
            print(
                f"WARNING: AdvancedHeuristicPredictor requires lookup tables. "
                f"Expected file: lookup_N{self.block_size}_acc{self.accuracy:.2f}_grid.npz "
                f"in directory: {self.lookup_dir}. "
                f"Falling back to binary search."
            )

    def _load_lookup_tables(self):
        """Load grid NPZ."""
        if self.lookup_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            for d in [
                os.path.join(current_dir, "models"),
                os.path.join(os.path.dirname(current_dir), "models"),
            ]:
                if os.path.exists(d):
                    self.lookup_dir = d
                    break

        if self.lookup_dir is None:
            return

        grid_file = os.path.join(
            self.lookup_dir, f"lookup_N{self.block_size}_acc{self.accuracy:.2f}_grid.npz"
        )
        if os.path.exists(grid_file):
            try:
                data = np.load(grid_file)
                self._grid = data["grid"]
                self._grid_edges_acm = data["edges_acm"]
                self._grid_edges_zr = data["edges_zr"]
                self._use_grid = True
                self._use_kdtree = False
                self._top_features = ["ac_abs_mean", "zero_ratio"]
                self._use_npy_format = True
                if "acm_lookup" in data and "zr_lookup" in data:
                    self._acm_lookup = data["acm_lookup"]
                    self._zr_lookup = data["zr_lookup"]
                    self._use_precomputed_lookup = True
                    self._acm_log_min = float(np.log1p(data["edges_acm"][0]))
                    self._acm_log_max = float(np.log1p(data["edges_acm"][-1]))
                    self._lookup_resolution = len(data["acm_lookup"])
                else:
                    self._use_precomputed_lookup = False
            except Exception:
                pass

    def _extract_features_optimized(self, block: np.ndarray) -> np.ndarray:
        """
        Extract 4 features via Numba JIT (one pass over AC zigzag).

        features[0] = ac_abs_mean  (#1 Spearman+MI)
        features[1] = zero_ratio   (#1 RF 87.9%)
        features[2] = ac_std       (#3 Spearman)
        features[3] = std_dev      (#3 Spearman, Parseval from DCT)
        """
        dct_block = dct2(block, dct_type=2)
        dct_block = np.round(dct_block, 2)
        dct_zigzag = to_zigzag(dct_block).astype(np.float64)
        acm, zr, acs, sdv = _nb_extract_features(dct_zigzag, block.shape[0])
        features = np.empty(4, dtype=np.float32)
        features[0] = acm
        features[1] = zr
        features[2] = acs
        features[3] = sdv
        return features

    def predict_dct_length(self, block: np.ndarray) -> int:
        """Predict optimal DCT length using lookup table."""
        # Auto-detect block size
        if hasattr(block, "shape"):
            detected_size = block.shape[0]
            if detected_size != self.block_size:
                self.block_size = detected_size
                self.total_coeffs = detected_size * detected_size
                self._load_lookup_tables()  # Reload for new size

        # Extract 4 features via Numba (one pass over AC zigzag)
        features = self._extract_features_optimized(block)
        pred = self._predict_from_multi_lookup(features)
        if pred is not None:
            return pred

        print("WARNING: Failed to predict from lookup table. Falling back to binary search.")
        return None

    def predict_dct_length_batch(self, blocks: List[np.ndarray]) -> tuple:
        """Batch prediction for a list of blocks (vectorized). Returns (preds, queues)."""
        if not blocks:
            return np.array([]), []

        if not getattr(self, "_use_npy_format", False):
            print(
                "WARNING: AdvancedHeuristicPredictor requires lookup tables to be loaded. "
                "Falling back to binary search."
            )
            # Return None for each block - will trigger binary search in refine.py
            preds = np.array([None] * len(blocks), dtype=np.int32)
            return preds, [[] for _ in blocks]

        # Vectorized extraction of 4 features from DCT (without original blocks)
        blocks_array = np.stack(blocks, axis=0)  # (batch, n, n)
        batch_size = blocks_array.shape[0]

        from scipy.fft import dct as scipy_dct

        dct_batch = scipy_dct(scipy_dct(blocks_array, axis=1, norm="ortho"), axis=2, norm="ortho")
        dct_flat = np.round(dct_batch.reshape(batch_size, -1), 2)

        # Features — one pass Numba (zero temporary allocations)
        ac_abs_mean, zero_ratio = _nb_extract_features_batch(
            dct_flat.astype(np.float64, copy=False)
        )
        queues = [[] for _ in range(batch_size)]

        # Batch query to grid
        if getattr(self, "_use_npy_format", False) and (
            getattr(self, "_use_kdtree", False) or getattr(self, "_use_grid", False)
        ):
            from .predictor_npy import predict_batch_from_npy

            # query_points shape: (batch_size, 2) — [ac_abs_mean, zero_ratio]
            query_points = np.stack([ac_abs_mean, zero_ratio], axis=1).astype(np.float32)
            results = predict_batch_from_npy(self, query_points)
        else:
            raise ValueError("AdvancedHeuristicPredictor requires grid or kdtree to be loaded")

        return results, queues

    def predict_from_dct_batch(self, dct_zigzags: np.ndarray) -> tuple:
        """
        Optimized batch prediction — features from dct_zigzags (already computed).

        Returns tuple (predictions_array, skip_queues_list).

        Parameters
        ----------
        dct_zigzags : np.ndarray
            DCT in zigzag format, shape (batch, n²)

        Returns
        -------
        tuple
            (predictions: np.ndarray[int32], skip_queues: list)
        """
        batch_size = len(dct_zigzags)

        # Features from AC zigzag — one pass Numba (zero temporary allocations)
        dct_arr = dct_zigzags if isinstance(dct_zigzags, np.ndarray) else np.asarray(dct_zigzags)
        ac_abs_mean, zero_ratio = _nb_extract_features_batch(dct_arr.astype(np.float64, copy=False))

        queues = [[] for _ in range(batch_size)]

        # Batch query to grid / cKDTree - single query instead of loop
        if getattr(self, "_use_npy_format", False) and (
            getattr(self, "_use_kdtree", False) or getattr(self, "_use_grid", False)
        ):
            from .predictor_npy import predict_batch_from_npy

            # query_points: (batch, 2) — [ac_abs_mean, zero_ratio]
            query_points = np.stack([ac_abs_mean, zero_ratio], axis=1).astype(np.float32)
            results = predict_batch_from_npy(self, query_points)
        else:
            print(
                "WARNING: AdvancedHeuristicPredictor requires grid or kdtree to be loaded. "
                "Falling back to binary search."
            )
            # Return default L value (half of max coefficients) instead of None
            default_L = (self.block_size * self.block_size) // 2
            results = np.full(batch_size, default_L, dtype=np.int32)

        return results, queues

    def _predict_from_multi_lookup(self, features: np.ndarray):
        """Prediction from lookup table (NPY grid). Returns int or None."""
        if getattr(self, "_use_npy_format", False):
            result = predict_from_multi_lookup_npy(self, features)
            if isinstance(result, tuple):
                pred, _ = result
                return pred
            return result
        return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_predictor_cache: dict = {}


def get_predictor(
    backend="heuristic",
    advanced_heuristic=False,
    accuracy=0.05,
    block_size=16,
    config=None,
    src_dtype=None,
    is_ycbcr=False,
):
    """Get predictor from cache

    Parameters
    ----------
    backend : str
        Predictor backend ("heuristic", "binary")
    advanced_heuristic : bool
        Use advanced heuristic with lookup tables
    accuracy : float
        Target accuracy
    block_size : int
        Block size
    config : Config, optional
        Configuration object
    src_dtype : str, optional
        Source data type (e.g., "uint8", "float32"). If uint8, returns Uint8Predictor.
    is_ycbcr : bool, optional
        True if building predictor specifically for YCbCr (RGB) channels (cm=6)

    Returns
    -------
    Predictor instance
    """
    # Load config if not provided
    if config is None:
        from config import load_config

        config = load_config()

    sf_override = getattr(config, "_scaling_factor_override", None)
    sf_cfg = getattr(config.compression, "uint8_scaling_factor", 1)
    if sf_override is not None:
        effective_sf = int(sf_override)
    elif isinstance(sf_cfg, (list, tuple)):
        effective_sf = int(sf_cfg[0]) if sf_cfg else 1
    else:
        effective_sf = int(sf_cfg)
    uint8_l_scales = tuple(
        float(x) for x in getattr(config.model, "uint8_L_prediction_scales", [1.5])
    )

    # Cache key includes uint8 sf/scales and is_ycbcr because Uint8Predictor
    # loads sf-specific L grids.
    cache_key = (
        backend,
        advanced_heuristic,
        accuracy,
        block_size,
        src_dtype,
        effective_sf,
        uint8_l_scales,
        is_ycbcr,
    )
    if cache_key in _predictor_cache:
        return _predictor_cache[cache_key]

    # Use paths from config
    models_dir = config.models_dir

    predictor = None

    # For uint8 data, use uint8-specific predictor if available
    is_uint8 = src_dtype in ("uint8", "int8", "uint16", "int16")

    if is_uint8 and config.model.uint8_use_L_prediction:
        try:
            from .predictor_uint8 import Uint8Predictor

            predictor = Uint8Predictor(
                accuracy=accuracy,
                block_size=block_size,
                models_dir=models_dir,
                config=config,
                is_ycbcr=is_ycbcr,
            )
            _predictor_cache[cache_key] = predictor
            return predictor
        except Exception:
            # Fallback to standard predictor if uint8 predictor fails
            pass

    if backend == "heuristic":
        if advanced_heuristic:
            # Advanced heuristic with lookup table
            predictor = AdvancedHeuristicPredictor(
                accuracy=accuracy, block_size=block_size, lookup_dir=models_dir
            )
        else:
            # Simple heuristic (std)
            predictor = HeuristicPredictor(accuracy=accuracy, block_size=block_size)

    else:
        predictor = HeuristicPredictor(accuracy=accuracy, block_size=block_size)

    _predictor_cache[cache_key] = predictor
    return predictor


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "HeuristicPredictor",
    "AdvancedHeuristicPredictor",
    "get_predictor",
]
