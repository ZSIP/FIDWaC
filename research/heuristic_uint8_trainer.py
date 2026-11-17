"""
FIDWAC v2 - Uint8 Heuristic Training System

Training on large-scale datasets (100k-200k+ files, multi-channel)
Builds learned model for optimal parameter prediction.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
from dataclasses import dataclass, asdict
from collections import defaultdict
import rasterio
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ChannelStats:
    """Statistics for a single channel from a single file."""

    file_id: str
    channel_idx: int
    min_val: float
    max_val: float
    mean: float
    std: float
    entropy: float
    zero_pct: float

    # Distribution bins (256 for uint8)
    histogram: List[int]  # Will be stored separately to save space

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding histogram for storage."""
        d = asdict(self)
        del d["histogram"]  # Save separately
        return d


@dataclass
class TrainingExample:
    """Single training example with ground truth."""

    stats: Dict  # ChannelStats as dict
    optimal_sf: int  # Ground truth: which sf actually worked
    optimal_bs: int  # Ground truth block size (if available)
    optimal_mults: List[float]  # Ground truth multipliers (if available)
    compression_error: float  # Actual compression error achieved
    file_id: str
    channel_idx: int


class Uint8TrainingDataCollector:
    """
    Collect training data from large dataset of uint8 raster files.
    Handles 100k-200k files with 6+ channels each.
    """

    def __init__(self, output_dir: str = "./training_data_uint8"):
        """
        Initialize collector.

        Parameters
        ----------
        output_dir : str
            Directory to save training data and statistics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Storage
        self.channel_stats: List[ChannelStats] = []
        self.training_examples: List[TrainingExample] = []
        self.stats_by_file: Dict[str, List[ChannelStats]] = defaultdict(list)

        # Aggregates
        self.histogram_data = {}  # filename -> channel -> histogram array

        logger.info(f"Output directory: {self.output_dir}")

    @staticmethod
    def extract_channel_stats(data: np.ndarray, file_id: str, channel_idx: int) -> ChannelStats:
        """Extract comprehensive statistics from single channel."""

        valid = data.flatten()
        if len(valid) == 0:
            raise ValueError("Empty channel data")

        # Basic statistics
        min_val = float(np.min(valid))
        max_val = float(np.max(valid))
        mean = float(np.mean(valid))
        std = float(np.std(valid))

        # Histogram
        histogram, _ = np.histogram(valid, bins=256, range=(0, 256))
        histogram = histogram.tolist()

        # Entropy
        probs = histogram / np.sum(histogram)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))

        # Sparsity
        zero_pct = 100.0 * np.sum(valid == 0) / len(valid)

        return ChannelStats(
            file_id=file_id,
            channel_idx=channel_idx,
            min_val=min_val,
            max_val=max_val,
            mean=mean,
            std=std,
            entropy=entropy,
            zero_pct=zero_pct,
            histogram=histogram,
        )

    def process_file(self, filepath: str, file_id: str = None) -> int:
        """
        Process single raster file and extract all channel statistics.

        Parameters
        ----------
        filepath : str
            Path to raster file
        file_id : str, optional
            Identifier for file (default: filename)

        Returns
        -------
        int
            Number of channels processed
        """

        if file_id is None:
            file_id = Path(filepath).stem

        try:
            with rasterio.open(filepath) as src:
                num_channels = src.count

                for band_idx in range(1, num_channels + 1):
                    try:
                        # Read band as uint8
                        data = src.read(band_idx)
                        if data.dtype != np.uint8:
                            data = np.clip(data, 0, 255).astype(np.uint8)

                        # Extract statistics
                        stats = self.extract_channel_stats(data, file_id, band_idx - 1)

                        self.channel_stats.append(stats)
                        self.stats_by_file[file_id].append(stats)

                        # Store histogram separately
                        key = f"{file_id}_ch{band_idx}"
                        self.histogram_data[key] = stats.histogram

                    except Exception as e:
                        logger.warning(f"  Error processing band {band_idx} of {file_id}: {e}")
                        continue

                return num_channels

        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return 0

    def process_dataset(
        self, dataset_path: str, max_files: Optional[int] = None, pattern: str = "*.tif"
    ) -> int:
        """
        Process all files in dataset directory.

        Parameters
        ----------
        dataset_path : str
            Path to directory containing raster files
        max_files : Optional[int]
            Maximum number of files to process (for testing)
        pattern : str
            File pattern to search for

        Returns
        -------
        int
            Total number of channels processed
        """

        dataset_dir = Path(dataset_path)
        files = sorted(list(dataset_dir.glob(f"**/{pattern}")))

        if max_files:
            files = files[:max_files]

        logger.info(f"Found {len(files)} files in {dataset_path}")

        total_channels = 0
        failed_files = 0

        with tqdm(files, desc="Processing files") as pbar:
            for filepath in pbar:
                channels = self.process_file(str(filepath))
                if channels > 0:
                    total_channels += channels
                    pbar.set_description(f"Processing files ({total_channels} channels)")
                else:
                    failed_files += 1

        logger.info(f"✓ Processed {len(files)} files ({total_channels} channels)")
        logger.info(f"✗ Failed: {failed_files} files")

        return total_channels

    def get_statistics_summary(self) -> Dict:
        """Get summary statistics from collected data."""

        if not self.channel_stats:
            return {}

        stats_array = np.array(
            [
                [s.min_val, s.max_val, s.mean, s.std, s.entropy, s.zero_pct]
                for s in self.channel_stats
            ]
        )

        return {
            "total_channels": len(self.channel_stats),
            "total_files": len(self.stats_by_file),
            "mean_val": {
                "min": float(np.mean(stats_array[:, 0])),
                "max": float(np.mean(stats_array[:, 1])),
                "mean": float(np.mean(stats_array[:, 2])),
                "std": float(np.mean(stats_array[:, 3])),
                "entropy": float(np.mean(stats_array[:, 4])),
                "zero_pct": float(np.mean(stats_array[:, 5])),
            },
            "percentiles": {
                "std_p10": float(np.percentile(stats_array[:, 3], 10)),
                "std_p25": float(np.percentile(stats_array[:, 3], 25)),
                "std_p50": float(np.percentile(stats_array[:, 3], 50)),
                "std_p75": float(np.percentile(stats_array[:, 3], 75)),
                "std_p90": float(np.percentile(stats_array[:, 3], 90)),
                "entropy_p10": float(np.percentile(stats_array[:, 4], 10)),
                "entropy_p25": float(np.percentile(stats_array[:, 4], 25)),
                "entropy_p50": float(np.percentile(stats_array[:, 4], 50)),
                "entropy_p75": float(np.percentile(stats_array[:, 4], 75)),
                "entropy_p90": float(np.percentile(stats_array[:, 4], 90)),
            },
        }

    def analyze_scaling_factor_distribution(self) -> Dict:
        """
        Analyze which std/entropy ranges should use sf=1 vs sf=10.

        This will be refined later with actual compression results,
        but provides initial clustering.
        """

        if not self.channel_stats:
            return {}

        # Cluster by std threshold
        stats_array = np.array([[s.std, s.entropy, s.mean, s.zero_pct] for s in self.channel_stats])

        # Define clusters based on observation
        low_std = stats_array[:, 0] < 30
        high_std = stats_array[:, 0] >= 30
        high_entropy = stats_array[:, 1] > 6.0
        low_entropy = stats_array[:, 1] <= 6.0

        return {
            "low_std_high_entropy": {
                "count": int(np.sum(low_std & high_entropy)),
                "typical_sf": 1,
                "reason": "Low variance but complex texture → sf=1",
            },
            "low_std_low_entropy": {
                "count": int(np.sum(low_std & low_entropy)),
                "typical_sf": 1,
                "reason": "Smooth homogeneous areas → sf=1",
            },
            "high_std_high_entropy": {
                "count": int(np.sum(high_std & high_entropy)),
                "typical_sf": 10,
                "reason": "High variance and complex → sf=10",
            },
            "high_std_low_entropy": {
                "count": int(np.sum(high_std & low_entropy)),
                "typical_sf": "mixed",
                "reason": "High variance but uniform → try both",
            },
        }

    def save_statistics(self, filename: str = "channel_statistics.json"):
        """Save collected statistics to JSON."""

        stats_list = []
        for stat in self.channel_stats:
            d = stat.to_dict()
            stats_list.append(d)

        output_file = self.output_dir / filename
        with open(output_file, "w") as f:
            json.dump(stats_list, f, indent=2)

        logger.info(f"✓ Saved statistics: {output_file}")
        return output_file

    def save_histograms(self, filename: str = "histograms.pkl"):
        """Save histograms (pickle for efficiency)."""

        output_file = self.output_dir / filename
        with open(output_file, "wb") as f:
            pickle.dump(self.histogram_data, f)

        logger.info(f"✓ Saved histograms: {output_file}")
        return output_file

    def save_summary(self, filename: str = "training_summary.json"):
        """Save summary statistics and analysis."""

        summary = {
            "dataset_statistics": self.get_statistics_summary(),
            "scaling_factor_analysis": self.analyze_scaling_factor_distribution(),
        }

        output_file = self.output_dir / filename
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Saved summary: {output_file}")
        return output_file

    def export_for_analysis(self) -> Dict:
        """Export all collected data for external analysis (R, pandas, etc.)."""

        stats_csv_path = self.output_dir / "channel_statistics.csv"

        # Write CSV header
        with open(stats_csv_path, "w") as f:
            f.write("file_id,channel_idx,min,max,mean,std,entropy,zero_pct\n")
            for stat in self.channel_stats:
                f.write(
                    f"{stat.file_id},{stat.channel_idx},"
                    f"{stat.min_val},{stat.max_val},{stat.mean},{stat.std},"
                    f"{stat.entropy},{stat.zero_pct}\n"
                )

        logger.info(f"✓ Saved CSV: {stats_csv_path}")

        # Save numpy lookup table
        lookup_npz = self.save_lookup_table_npz()

        return {
            "statistics_json": str(self.save_statistics()),
            "histograms_pkl": str(self.save_histograms()),
            "summary_json": str(self.save_summary()),
            "statistics_csv": str(stats_csv_path),
            "lookup_table_npz": str(lookup_npz),
        }

    def build_numpy_lookup_table(self, num_bins_std: int = 64, num_bins_entropy: int = 29) -> Dict:
        """
        Build numpy lookup table from collected statistics.

        Similar to float32 lookup tables (grid, edges, lookups).
        Creates multi-dimensional index for fast parameter prediction.

        Parameters
        ----------
        num_bins_std : int
            Number of bins for std threshold (default 64)
        num_bins_entropy : int
            Number of bins for entropy threshold (default 29)

        Returns
        -------
        Dict
            Dictionary with arrays: 'grid', 'edges_std', 'edges_entropy'
        """

        if not self.channel_stats:
            logger.warning("No channel statistics collected")
            return {}

        # Extract feature arrays
        std_vals = np.array([s.std for s in self.channel_stats])
        entropy_vals = np.array([s.entropy for s in self.channel_stats])

        # Create bin edges (quantile-based for better distribution)
        edges_std = np.percentile(std_vals, np.linspace(0, 100, num_bins_std + 1))
        edges_entropy = np.percentile(entropy_vals, np.linspace(0, 100, num_bins_entropy + 1))

        # Remove duplicates from edges
        edges_std = np.unique(edges_std)
        edges_entropy = np.unique(edges_entropy)

        logger.info(f"Creating lookup grid: {len(edges_std)-1} × {len(edges_entropy)-1} bins")

        # Initialize grid to store optimal block_size for each (std, entropy) bin
        grid = np.zeros((len(edges_std) - 1, len(edges_entropy) - 1), dtype=np.uint8)

        # Fill grid by counting which block_size is most appropriate
        # For each bin, find most common optimal block_size
        for i, stat in enumerate(self.channel_stats):
            # Find which bin this sample belongs to
            std_bin = np.searchsorted(edges_std, stat.std, side="right") - 1
            entropy_bin = np.searchsorted(edges_entropy, stat.entropy, side="right") - 1

            # Clamp to valid range
            std_bin = np.clip(std_bin, 0, len(edges_std) - 2)
            entropy_bin = np.clip(entropy_bin, 0, len(edges_entropy) - 2)

            # Map (std, entropy) → optimal BS via simple heuristic
            # From accuracy sweep results: BS=8 is generally optimal
            optimal_bs = 8
            if stat.std < np.percentile(std_vals, 25) and stat.entropy > np.percentile(
                entropy_vals, 75
            ):
                optimal_bs = 32
            elif stat.std > np.percentile(std_vals, 75):
                optimal_bs = 8
            else:
                optimal_bs = 16

            # Store most common value (simple: just take last/first applicable)
            grid[std_bin, entropy_bin] = optimal_bs

        return {
            "grid": grid,
            "edges_std": edges_std,
            "edges_entropy": edges_entropy,
            "num_channels": len(self.channel_stats),
        }

    def save_lookup_table_npz(
        self, accuracy: Optional[float] = None, filename: Optional[str] = None
    ) -> Path:
        """
        Save numpy lookup table as NPZ file (like float32 lookup tables).

        Parameters
        ----------
        accuracy : Optional[float]
            Accuracy level (e.g., 0.05 for 5%) - used in filename
        filename : Optional[str]
            Custom filename (if None, auto-generate)

        Returns
        -------
        Path
            Path to saved NPZ file
        """

        lookup_data = self.build_numpy_lookup_table()

        if not lookup_data:
            raise ValueError("Could not build lookup table")

        # Generate filename
        if filename is None:
            if accuracy is not None:
                filename = f"lookup_uint8_acc{accuracy:.2f}_grid.npz"
            else:
                filename = "lookup_uint8_grid.npz"

        output_file = self.output_dir / filename

        np.savez_compressed(
            output_file,
            grid=lookup_data["grid"],
            edges_std=lookup_data["edges_std"],
            edges_entropy=lookup_data["edges_entropy"],
            metadata=np.array([lookup_data["num_channels"]], dtype=np.uint32),
        )

        logger.info(
            f"✓ Saved lookup table NPZ: {output_file} ({output_file.stat().st_size / 1024:.1f} KB)"
        )
        return output_file


class Uint8ModelTrainer:
    """
    Train heuristic model from collected statistics.
    Uses clustering and regression to optimize thresholds.
    """

    def __init__(self, statistics_path: str):
        """
        Initialize trainer from saved statistics.

        Parameters
        ----------
        statistics_path : str
            Path to JSON file with channel statistics
        """

        with open(statistics_path, "r") as f:
            self.stats_list = json.load(f)

        logger.info(f"Loaded {len(self.stats_list)} channel statistics")

    def optimize_thresholds(self) -> Dict:
        """
        Optimize decision thresholds for sf, bs, multipliers.

        Uses statistical clustering to find optimal breakpoints.
        """

        # Extract feature vectors
        std_vals = np.array([s["std"] for s in self.stats_list])
        entropy_vals = np.array([s["entropy"] for s in self.stats_list])
        mean_vals = np.array([s["mean"] for s in self.stats_list])
        zero_pct_vals = np.array([s["zero_pct"] for s in self.stats_list])

        # Find optimal thresholds via percentiles
        optimized = {
            "std_thresholds": {
                "low": float(np.percentile(std_vals, 33)),  # Bottom third
                "medium": float(np.percentile(std_vals, 66)),  # Middle third
                "high": float(np.percentile(std_vals, 95)),  # Top 5%
            },
            "entropy_thresholds": {
                "low": float(np.percentile(entropy_vals, 25)),
                "high": float(np.percentile(entropy_vals, 75)),
            },
            "scaling_factor_rules": self._derive_sf_rules(std_vals, entropy_vals, zero_pct_vals),
            "block_size_rules": self._derive_bs_rules(std_vals, entropy_vals),
        }

        return optimized

    def _derive_sf_rules(self, std_vals, entropy_vals, zero_pct_vals) -> Dict:
        """Derive optimal scaling factor rules."""

        rules = {}

        # Rule 1: High std + high entropy → sf=10
        mask_10 = (std_vals > np.percentile(std_vals, 66)) & (
            entropy_vals > np.percentile(entropy_vals, 50)
        )
        rules["sf_10_criteria"] = {
            "std_gt": float(np.percentile(std_vals, 66)),
            "entropy_gt": float(np.percentile(entropy_vals, 50)),
            "match_count": int(np.sum(mask_10)),
            "match_pct": float(100 * np.sum(mask_10) / len(std_vals)),
        }

        # Rule 2: Very low variance → sf=1
        mask_1_smooth = std_vals < np.percentile(std_vals, 25)
        rules["sf_1_smooth"] = {
            "std_lt": float(np.percentile(std_vals, 25)),
            "match_count": int(np.sum(mask_1_smooth)),
            "match_pct": float(100 * np.sum(mask_1_smooth) / len(std_vals)),
        }

        # Rule 3: Sparse data → sf=1
        mask_1_sparse = zero_pct_vals > 30
        rules["sf_1_sparse"] = {
            "zero_pct_gt": 30.0,
            "match_count": int(np.sum(mask_1_sparse)),
            "match_pct": float(100 * np.sum(mask_1_sparse) / len(std_vals)),
        }

        return rules

    def _derive_bs_rules(self, std_vals, entropy_vals) -> Dict:
        """Derive optimal block size rules."""

        rules = {}

        # Rule 1: High variance → bs=8
        mask_bs8 = std_vals > np.percentile(std_vals, 75)
        rules["bs_8_high_var"] = {
            "std_gt": float(np.percentile(std_vals, 75)),
            "match_count": int(np.sum(mask_bs8)),
            "match_pct": float(100 * np.sum(mask_bs8) / len(std_vals)),
        }

        # Rule 2: Low variance, high entropy → bs=32
        mask_bs32 = (std_vals < np.percentile(std_vals, 50)) & (
            entropy_vals > np.percentile(entropy_vals, 75)
        )
        rules["bs_32_smooth_complex"] = {
            "std_lt": float(np.percentile(std_vals, 50)),
            "entropy_gt": float(np.percentile(entropy_vals, 75)),
            "match_count": int(np.sum(mask_bs32)),
            "match_pct": float(100 * np.sum(mask_bs32) / len(std_vals)),
        }

        # Rule 3: Default → bs=16
        rules["bs_16_default"] = {
            "coverage": "remaining channels",
            "typical_pct": float(
                100
                - 100 * np.sum(mask_bs8) / len(std_vals)
                - 100 * np.sum(mask_bs32) / len(std_vals)
            ),
        }

        return rules

    def generate_learned_heuristic_code(self, output_file: str = None) -> str:
        """
        Generate Python code with learned parameters for use in heuristic.
        """

        optimized = self.optimize_thresholds()

        code = f'''"""
AUTO-GENERATED: Learned Uint8 Heuristic Parameters
Generated from {len(self.stats_list)} channel statistics
"""

# Learned thresholds
STD_LOW_THRESHOLD = {optimized['std_thresholds']['low']:.2f}
STD_MEDIUM_THRESHOLD = {optimized['std_thresholds']['medium']:.2f}
STD_HIGH_THRESHOLD = {optimized['std_thresholds']['high']:.2f}

ENTROPY_LOW_THRESHOLD = {optimized['entropy_thresholds']['low']:.2f}
ENTROPY_HIGH_THRESHOLD = {optimized['entropy_thresholds']['high']:.2f}

# Scaling factor rules (from {len(self.stats_list)} channels)
SF_10_RULE = {{
    "description": "Use sf=10 if: std > {optimized['std_thresholds']['medium']:.2f} AND entropy > {optimized['entropy_thresholds']['low']:.2f}",
    "std_threshold": {optimized['std_thresholds']['medium']:.2f},
    "entropy_threshold": {optimized['entropy_thresholds']['low']:.2f},
    "expected_coverage": {optimized['scaling_factor_rules']['sf_10_criteria']['match_pct']:.1f}
}}

SF_1_SMOOTH_RULE = {{
    "description": "Use sf=1 if: std < {optimized['std_thresholds']['low']:.2f}",
    "std_threshold": {optimized['std_thresholds']['low']:.2f},
    "expected_coverage": {optimized['scaling_factor_rules']['sf_1_smooth']['match_pct']:.1f}
}}

SF_1_SPARSE_RULE = {{
    "description": "Use sf=1 if: zero_pct > 30%",
    "zero_pct_threshold": 30.0,
    "expected_coverage": {optimized['scaling_factor_rules']['sf_1_sparse']['match_pct']:.1f}
}}

# Block size rules
BS_8_RULE = {{
    "description": "Use bs=8 if: std > {optimized['std_thresholds']['high']:.2f}",
    "std_threshold": {optimized['std_thresholds']['high']:.2f},
    "expected_coverage": {optimized['block_size_rules']['bs_8_high_var']['match_pct']:.1f}
}}

BS_32_RULE = {{
    "description": "Use bs=32 if: std < {optimized['std_thresholds']['medium']:.2f} AND entropy > {optimized['entropy_thresholds']['high']:.2f}",
    "std_threshold": {optimized['std_thresholds']['medium']:.2f},
    "entropy_threshold": {optimized['entropy_thresholds']['high']:.2f},
    "expected_coverage": {optimized['block_size_rules']['bs_32_smooth_complex']['match_pct']:.1f}
}}

BS_16_DEFAULT = {{
    "description": "Use bs=16 for all other cases",
    "expected_coverage": {optimized['block_size_rules']['bs_16_default']['typical_pct']:.1f}
}}
'''

        if output_file:
            with open(output_file, "w") as f:
                f.write(code)
            logger.info(f"✓ Generated heuristic code: {output_file}")

        return code


# ============================================================================
# Quick start functions
# ============================================================================


def collect_training_data(
    dataset_path: str, max_files: Optional[int] = None, output_dir: str = "./training_data_uint8"
) -> Path:
    """
    Collect training data from dataset.

    Parameters
    ----------
    dataset_path : str
        Path to directory with raster files
    max_files : Optional[int]
        Max files to process (for testing)
    output_dir : str
        Output directory for training data

    Returns
    -------
    Path
        Path to output directory with saved training data
    """

    collector = Uint8TrainingDataCollector(output_dir)
    collector.process_dataset(dataset_path, max_files=max_files)

    exported_paths = collector.export_for_analysis()

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING DATA COLLECTION COMPLETE")
    logger.info("=" * 70)
    for key, path in exported_paths.items():
        logger.info(f"  {key}: {path}")

    return collector.output_dir


def train_heuristic(statistics_path: str, output_file: str = None):
    """
    Train heuristic from collected statistics.

    Parameters
    ----------
    statistics_path : str
        Path to channel_statistics.json
    output_file : str, optional
        Where to save generated heuristic code

    Returns
    -------
    str
        Generated Python code with learned parameters
    """

    trainer = Uint8ModelTrainer(statistics_path)

    if output_file is None:
        output_file = Path(statistics_path).parent / "learned_heuristic_parameters.py"

    code = trainer.generate_learned_heuristic_code(output_file)

    logger.info("\n" + "=" * 70)
    logger.info("HEURISTIC TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Generated: {output_file}")

    return code


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python heuristic_uint8_trainer.py collect <dataset_path> [max_files]")
        print("  python heuristic_uint8_trainer.py train <statistics_json>")
        print("\nExample:")
        print(
            "  python heuristic_uint8_trainer.py collect /mnt/g/FIDWAC_2026/dane/source/multi_uint_6ch 1000"
        )
        print(
            "  python heuristic_uint8_trainer.py train ./training_data_uint8/channel_statistics.json"
        )
        sys.exit(1)

    command = sys.argv[1]

    if command == "collect":
        dataset_path = sys.argv[2]
        max_files = int(sys.argv[3]) if len(sys.argv) > 3 else None
        collect_training_data(dataset_path, max_files)

    elif command == "train":
        stats_path = sys.argv[2]
        train_heuristic(stats_path)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
