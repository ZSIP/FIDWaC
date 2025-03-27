# Standard libraries
import os
import time
import json
import pickle
import tempfile
import glob
import shutil
import multiprocessing
from functools import partial
import sys

# External libraries
import numpy as np
import py7zr
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import griddata
import rasterio
float = np.float32 

# Local modules
from compression.compress_function import *

# Loading configuration
with open(r"./compression/config.json", "r") as file:
    config_data = json.load(file)

# Parameters from configuration
result_dir = config_data.get("results_directory")
source_dir = config_data.get("source_directory")

if __name__ == "__main__":
    import sys

    file_path = None
    output_dir = None

    # Command line arguments handling
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    # Run the main function with the provided arguments
    main(file_path, output_dir)
