"""
FIDWAC v2 - DCT compression for geospatial data.

Modules: core, compression, decompression, predictor, refine, blocks, utils, config
Prediction backends: heuristic (variance-based), advanced heuristic (lookup table)
"""

from core.dct import dct2, idct2, to_zigzag, from_zigzag
from compress.compression import compress_image
from compress.decompression import decompress_image, decompress_file, load_compressed_data
from config import load_config, Config
from predictor.predictor import get_predictor, HeuristicPredictor, AdvancedHeuristicPredictor

__version__ = "2.0.0"
__all__ = [
    # Core
    "dct2",
    "idct2",
    "to_zigzag",
    "from_zigzag",
    # Compression/Decompression
    "compress_image",
    "decompress_image",
    "decompress_file",
    "load_compressed_data",
    # Config
    "load_config",
    "Config",
    # Predictor
    "get_predictor",
    "HeuristicPredictor",
    "AdvancedHeuristicPredictor",
]
