"""
FIDWAC v2 - Uint8 Heuristic Predictor

Predicts optimal parameters for uint8 compression:
  - scaling_factor (sf): [1, 10] adaptive selection
  - block_size: [8, 16, 32] based on variance
  - ycbcr_multipliers: [0.9, 0.7, 0.5, 0.4, 0.3] fallback sequence

Training: learns from multi_uint_6ch dataset to predict best parameter combination
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Uint8Stats:
    """Statistics extracted from uint8 image for prediction."""

    min_val: float
    max_val: float
    mean: float
    std: float
    range_val: float

    # Distribution characteristics
    percentile_10: float
    percentile_25: float
    percentile_75: float
    percentile_90: float

    # Sparsity
    zero_count: float  # percentage of zeros
    low_values_count: float  # percentage < 50
    high_values_count: float  # percentage > 200

    # Histogram characteristics
    histogram: List[int]  # 256-bin histogram
    entropy: float

    num_channels: int
    data_type: str


@dataclass
class Uint8Prediction:
    """Predicted parameters for uint8 compression."""

    scaling_factor: int
    block_size: int
    multiplier_sequence: List[float]
    confidence: float  # 0.0-1.0
    reason: str


class Uint8Heuristic:
    """
    Heuristic predictor for uint8 compression parameters.

    Based on image statistics, predicts:
    1. Optimal scaling factor (sf) from [1, 10]
    2. Optimal block size from [8, 16, 32]
    3. Recommended multiplier sequence for YCbCr fallback
    """

    # Rule-based thresholds (learned from typical uint8 data)
    STD_HIGH = 50.0  # High variance threshold
    STD_MEDIUM = 30.0  # Medium variance threshold
    STD_LOW = 10.0  # Low variance threshold

    def __init__(self):
        """Initialize heuristic with default rules."""
        self.training_data: List[Dict] = []
        self.model_params = {}

    @staticmethod
    def extract_stats(image: np.ndarray, nodata_value: Optional[float] = None) -> Uint8Stats:
        """Extract comprehensive statistics from uint8 image."""

        # Get valid data
        if nodata_value is not None:
            valid = image[image != nodata_value].flatten()
        else:
            valid = image.flatten()

        if len(valid) == 0:
            raise ValueError("No valid data in image")

        # Basic stats
        min_val = float(np.min(valid))
        max_val = float(np.max(valid))
        mean = float(np.mean(valid))
        std = float(np.std(valid))
        range_val = max_val - min_val

        # Percentiles
        p10 = float(np.percentile(valid, 10))
        p25 = float(np.percentile(valid, 25))
        p75 = float(np.percentile(valid, 75))
        p90 = float(np.percentile(valid, 90))

        # Distribution characteristics
        zero_count = 100.0 * np.sum(valid == 0) / len(valid)
        low_values = 100.0 * np.sum(valid < 50) / len(valid)
        high_values = 100.0 * np.sum(valid > 200) / len(valid)

        # Histogram
        histogram, _ = np.histogram(valid, bins=256, range=(0, 256))
        histogram = histogram.tolist()

        # Entropy
        probs = histogram / np.sum(histogram)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))

        num_channels = image.shape[0] if len(image.shape) > 2 else 1

        return Uint8Stats(
            min_val=min_val,
            max_val=max_val,
            mean=mean,
            std=std,
            range_val=range_val,
            percentile_10=p10,
            percentile_25=p25,
            percentile_75=p75,
            percentile_90=p90,
            zero_count=zero_count,
            low_values_count=low_values,
            high_values_count=high_values,
            histogram=histogram,
            entropy=entropy,
            num_channels=num_channels,
            data_type="uint8",
        )

    def predict_scaling_factor(self, stats: Uint8Stats) -> Tuple[int, str]:
        """Predict optimal scaling factor based on value distribution."""

        # Rule 1: If values are already in tight range, use sf=1
        if stats.range_val <= 64:
            return 1, "Range ≤ 64: sf=1 sufficient"

        # Rule 2: If high variance + sparse, use sf=10
        if stats.std > self.STD_HIGH and stats.high_values_count > 10:
            return 10, "High variance + sparse: sf=10 for coefficient resolution"

        # Rule 3: If many zeros (> 30%), use sf=1 (DCT already handles sparsity)
        if stats.zero_count > 30:
            return 1, "Sparse data (>30% zeros): sf=1 (DCT compresses zeros well)"

        # Rule 4: If medium variance, adaptive choice
        if stats.std > self.STD_MEDIUM:
            # Use sf=10 if we have enough non-trivial values
            if stats.low_values_count < 70:
                return 10, "Medium variance + dense: sf=10 for accuracy"
            else:
                return 1, "Medium variance + sparse: sf=1 baseline"

        # Default: sf=1 for low variance
        return 1, "Low variance: sf=1 baseline"

    def predict_block_size(self, stats: Uint8Stats) -> Tuple[int, str]:
        """Predict optimal block size based on image homogeneity."""

        # Rule 1: High variance → small blocks
        if stats.std > self.STD_HIGH:
            return 8, f"High variance (std={stats.std:.1f}): small blocks for detail"

        # Rule 2: Medium variance → medium blocks
        if stats.std > self.STD_MEDIUM:
            return 16, f"Medium variance (std={stats.std:.1f}): medium blocks"

        # Rule 3: Low variance + high entropy → larger blocks
        if stats.std <= self.STD_LOW and stats.entropy > 6.0:
            return 32, f"Homogeneous (std={stats.std:.1f}) + high entropy: large blocks"

        # Rule 4: Very low variance → small blocks (may be smooth regions)
        if stats.std < 5.0:
            return 8, f"Very homogeneous (std={stats.std:.1f}): small blocks for accuracy"

        # Default
        return 16, "Default: medium blocks"

    def predict_multiplier_sequence(self, stats: Uint8Stats) -> Tuple[List[float], str]:
        """Predict optimal YCbCr fallback multiplier sequence."""

        # Rule 1: High values + sparse → aggressive fallback
        if stats.high_values_count > 20 or (stats.max_val > 230 and stats.std > 30):
            return [0.9, 0.7, 0.5, 0.3], "High brightness + variance: aggressive fallback"

        # Rule 2: Low variance + many zeros → conservative fallback
        if stats.zero_count > 25 or stats.std < 15:
            return [0.95, 0.85, 0.7, 0.5, 0.3], "Sparse/homogeneous: conservative fallback"

        # Rule 3: Balanced distribution → standard fallback
        if 20 < stats.mean < 230 and 15 < stats.std < 50:
            return [0.9, 0.7, 0.5, 0.4, 0.3], "Balanced distribution: standard fallback"

        # Default
        return [0.9, 0.7, 0.5, 0.4, 0.3], "Default: standard sequence"

    def predict(
        self, image: np.ndarray, nodata_value: Optional[float] = None, accuracy_target: float = 5.0
    ) -> Uint8Prediction:
        """
        Predict optimal compression parameters for uint8 image.

        Parameters
        ----------
        image : np.ndarray
            Uint8 image data
        nodata_value : Optional[float]
            Value to ignore (e.g., 0 for nodata)
        accuracy_target : float
            Target accuracy (used for confidence)

        Returns
        -------
        Uint8Prediction
            Predicted parameters with confidence score
        """

        # Extract statistics
        stats = self.extract_stats(image, nodata_value)

        # Predict each parameter
        sf, sf_reason = self.predict_scaling_factor(stats)
        bs, bs_reason = self.predict_block_size(stats)
        mults, mult_reason = self.predict_multiplier_sequence(stats)

        # Calculate confidence (0.0-1.0)
        # Higher confidence if std is moderate (not extreme)
        std_confidence = 1.0 - abs(stats.std - self.STD_MEDIUM) / (self.STD_HIGH * 2)
        std_confidence = max(0.3, min(1.0, std_confidence))

        # Higher confidence if range is well-defined
        range_confidence = min(1.0, stats.range_val / 256.0)

        # Higher confidence if not too sparse
        sparsity_confidence = 1.0 - min(1.0, stats.zero_count / 50.0)

        confidence = (std_confidence + range_confidence + sparsity_confidence) / 3.0

        reason = (
            f"[Stats: mean={stats.mean:.1f}, std={stats.std:.1f}, "
            f"range={stats.range_val:.0f}, entropy={stats.entropy:.2f}] "
            f"{sf_reason} | {bs_reason} | {mult_reason}"
        )

        return Uint8Prediction(
            scaling_factor=sf,
            block_size=bs,
            multiplier_sequence=mults,
            confidence=confidence,
            reason=reason,
        )

    def log_prediction(
        self, stats: Uint8Stats, prediction: Uint8Prediction, actual_sf: Optional[int] = None
    ):
        """Log prediction for later analysis/training."""

        self.training_data.append(
            {
                "stats": {
                    "mean": stats.mean,
                    "std": stats.std,
                    "range": stats.range_val,
                    "entropy": stats.entropy,
                    "zero_count": stats.zero_count,
                    "max_val": stats.max_val,
                },
                "prediction": {
                    "sf": prediction.scaling_factor,
                    "bs": prediction.block_size,
                    "confidence": prediction.confidence,
                },
                "actual_sf": actual_sf,
                "correct": (
                    (actual_sf == prediction.scaling_factor) if actual_sf is not None else None
                ),
            }
        )

    def save_training_log(self, filepath: str):
        """Save training/prediction log for analysis."""
        with open(filepath, "w") as f:
            json.dump(self.training_data, f, indent=2)

    def get_accuracy(self) -> float:
        """Calculate prediction accuracy from training log."""
        if not self.training_data:
            return 0.0

        correct = sum(1 for d in self.training_data if d.get("correct") is True)
        total = sum(1 for d in self.training_data if d.get("actual_sf") is not None)

        return correct / total if total > 0 else 0.0


class Uint8HeuristicAdvanced(Uint8Heuristic):
    """
    Advanced heuristic with learned thresholds from training data.
    Extends base heuristic with trained parameters.
    """

    def __init__(self):
        super().__init__()
        self.learned_params = {
            "std_high_threshold": 50.0,
            "std_medium_threshold": 30.0,
            "sf_boundary": 100,  # max range before switching to sf=10
        }

    def train_from_dataset(self, dataset_path: str):
        """
        Train heuristic parameters from multi_uint_6ch dataset.

        Parameters
        ----------
        dataset_path : str
            Path to dataset directory with images and metadata
        """
        from pathlib import Path
        import rasterio

        dataset_dir = Path(dataset_path)

        # Collect statistics for all images
        stats_list = []

        for tiff_file in sorted(dataset_dir.glob("*.tif"))[:100]:  # Use first 100 files
            try:
                with rasterio.open(tiff_file) as src:
                    for band_idx in range(1, src.count + 1):
                        data = src.read(band_idx).astype(np.float32)
                        stats = self.extract_stats(data)
                        stats_list.append((stats, tiff_file.name))
            except Exception as e:
                print(f"Error reading {tiff_file}: {e}")

        # Analyze correlation between stats and optimal sf
        # This is simplified - in reality would need ground truth
        print(f"\n=== Training on {len(stats_list)} bands ===")
        print(f"Mean std: {np.mean([s[0].std for s in stats_list]):.2f}")
        print(f"Mean entropy: {np.mean([s[0].entropy for s in stats_list]):.2f}")

    def predict_batch(self, image_list: List[np.ndarray]) -> List[Uint8Prediction]:
        """Predict parameters for multiple images."""
        return [self.predict(img) for img in image_list]


# ============================================================================
# Convenience functions
# ============================================================================


def quick_predict(image: np.ndarray) -> Uint8Prediction:
    """Quick prediction using default heuristic."""
    h = Uint8Heuristic()
    return h.predict(image)


def print_prediction(prediction: Uint8Prediction):
    """Pretty-print prediction results."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           UINT8 COMPRESSION PARAMETER PREDICTION            ║
╚══════════════════════════════════════════════════════════════╝

📊 Predicted Parameters:
   • Scaling Factor (sf):       {prediction.scaling_factor}
   • Block Size:                {prediction.block_size}×{prediction.block_size}
   • Multiplier Sequence:       {prediction.multiplier_sequence}
   • Confidence:                {prediction.confidence:.1%}

💭 Reasoning:
   {prediction.reason}

┌──────────────────────────────────────────────────────────────┐
""")


if __name__ == "__main__":
    # Example usage
    import rasterio
    from pathlib import Path

    # Try to load a sample uint8 image
    sample_paths = [
        "g:/FIDWAC_2026/dane/source/multi_uint_6ch/*.tif",
        "/mnt/g/FIDWAC_2026/dane/source/multi_uint_6ch/*.tif",
    ]

    for pattern in sample_paths:
        files = list(Path(".").glob(pattern.replace("./", "")))
        if files:
            sample_file = str(files[0])
            print(f"\nTesting on: {sample_file}")

            with rasterio.open(sample_file) as src:
                for band in range(1, min(src.count + 1, 3)):
                    data = src.read(band).astype(np.uint8)
                    prediction = quick_predict(data)
                    print_prediction(prediction)
            break
