# FIDWaC: Fast Inverse Distance Weighting and Compression

**FIDWaC (Fast Inverse Distance Weighting and Compression)** is a Python toolkit for:
- **Fast creation of continuous surfaces** (rasters) from scattered measurement points using the IDW (Inverse Distance Weighting) method
- **Lossy compression** of raster data (GeoTIFF) using **Discrete Cosine Transform (DCT)** and zigzag encoding

The package is designed for efficient processing and storage of geospatial data in fields such as:
- Bathymetry analysis
- Digital terrain modeling
- Shoreline monitoring
- Hydrology, climatology, and other applications requiring large spatial datasets

## Table of Contents
1. [Features](#features)
2. [Tool Structure](#structure)
   1. [interpolation_FIT.py](#interpolation)
   2. [compress_function.py](#compression)
3. [Operation Diagrams](#schemes)
4. [System Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration and Usage](#configuration)
   1. [JSON Configuration System](#json-config)
   2. [Interpolation Configuration](#interp-config)
   3. [Compression Configuration](#comp-config)
   4. [Usage Examples](#usage)
   5. [Batch Processing](#batch)
7. [Multi-threading Capabilities](#multi-threading)
8. [Key Functions](#key-functions)
9. [Examples](#examples)
10. [Advantages and Applications](#benefits)

---

<a name="features"></a>
## 1. Features

### High-Performance IDW Interpolation

- **Dual Processing Options**:
  - NumPy + numexpr for standard datasets
  - Dask for parallel processing of large datasets
- **Efficient Spatial Search** with `scipy.spatial.cKDTree`
- **Flexible Parameter Selection**:
  - Maximum search radius (`RMAX`)
  - Maximum number of neighbors (`NMAX`)
  - Minimum number of observations (`NMIN`)
- **Multiple Input Formats**: `.shp`, `.las`, `.laz`, `.txt`, `.csv`
- **Multiple Output Formats**: GeoTIFF, CSV, Surfer GRD
- **Multi-threaded Processing**: Utilizes all available CPU cores for spatial queries

### Advanced Raster Compression

- **2D-DCT** (Discrete Cosine Transform) with zigzag coefficient encoding
- **Quality-Controlled Compression** with iterative coefficient pruning to achieve preset error thresholds
- **Intelligent Handling of NoData Values** (-9999) and zero values using RLE representation
- **Optimized Storage** through serialization (RLE) with .json file and 7z archiving
- **Parallel Block Processing**: Multi-threaded compression of image blocks

---

<a name="structure"></a>
## 2. Tool Structure

<a name="interpolation"></a>
### 2.1 `interpolation_FIT.py`

Script for advanced spatial data interpolation using IDW method and KDTree optimization.

**Key Components**:
- Configuration management through `config.json`
- Data preprocessing (NaN and -9999 removal)
- Meshgrid generation based on data boundaries
- KDTree implementation for efficient neighbor search
- Dual interpolation engines (NumPy/numexpr and Dask)
- Multi-format export functionality
- Multi-threaded KDTree queries with `workers=-1` parameter

**Core Functions**:
```python
def calculate_idw(distance: np.ndarray) -> np.ndarray:
    """Calculate IDW interpolation using NumExpr for optimized performance"""
    w = ne.evaluate(f"{weights}/(distance)**2") 
    a = np.nansum(w * value_data, axis=1)
    b = np.nansum(w, axis=1)
    idw = ne.evaluate("a/b")
    return idw

def calculate_idw_dask(distance: np.ndarray, weights: float, value_data: np.ndarray) -> np.ndarray:
    """Calculate IDW interpolation using Dask for parallel processing"""
    distance_dask = da.from_array(distance, chunks='auto')
    value_data_dask = da.from_array(value_data, chunks='auto')        
    
    # Handle NaN values
    distance_dask = da.nan_to_num(distance_dask)
    value_data_dask = da.nan_to_num(value_data_dask)
    no_nan = 1e-10  # Epsilon value to avoid division by zero
    
    # Calculate IDW
    w = weights/(distance_dask ** 2 + no_nan)
    a = da.nansum(w * value_data_dask, axis=1)
    b = da.nansum(w, axis=1)
    idw = a/b  
    return idw.compute()
```

**Processing Workflow**:
1. Data loading and cleaning
2. Grid preparation based on data extents
3. KDTree construction for spatial indexing
4. Nearest neighbors search for each grid point (multi-threaded)
5. Distance-based weight calculation
6. IDW interpolation via optimized numerical processing (parallel with Dask)
7. Output generation in selected formats

<a name="compression"></a>
### 2.2 `compress_function.py`

Script for lossy compression of raster data using DCT and zigzag encoding with quality control.

**Key Components**:
- Configuration management through `config.json`
- Special value detection and handling (0, -9999, NaN)
- N×N block-based processing
- 2D-DCT transformation with coefficient optimization
- RLE (Run-Length Encoding) for masks
- Multi-threaded block processing with Python's `multiprocessing` module

**Core Functions**:
```python
def dct2(a: np.ndarray) -> np.ndarray:
    """Performs a two-dimensional Discrete Cosine Transform (DCT)"""
    return dct(dct(a.T, norm="ortho", type=type_dct).T, norm="ortho", type=type_dct)

def idct2(a: np.ndarray) -> np.ndarray:
    """Performs an inverse two-dimensional Discrete Cosine Transform (IDCT)"""
    return idct(idct(a.T, norm="ortho", type=type_dct).T, norm="ortho", type=type_dct)

def to_zigzag(matrix: np.ndarray) -> np.ndarray:
    """Converts a 2D matrix to a 1D vector using zig-zag scanning"""
    # Implementation processes the matrix in diagonal traversal pattern
    # to concentrate energy in the early coefficients
    
def refine_dct_array(org_dct_zigzag, accuracy, agt, max_value, split_point, original_matrix):
    """Optimizes the number of DCT coefficients to achieve desired accuracy"""
    # Uses search to find optimal number of coefficients
    # while maintaining error below the specified threshold
```

**Processing Workflow**:
1. Raster data loading
2. Special value identification and masking
3. Division into N×N processing blocks
4. Parallel processing of blocks using multiple CPU cores:
   - Special value-only blocks: by RLE encoding
   - Mixed blocks: DCT with special value handling
   - Data-only blocks: full DCT processing
5. Zigzag encoding of DCT coefficients
6. Iterative coefficient pruning for quality control
7. RLE serialization with create json file and 7z compression

---

<a name="schemes"></a>
## 3. Operation Diagrams

### 3.1 Interpolation Workflow

```mermaid
graph TB
    A[Load input data] --> B[Remove invalid data]
    B --> C[Prepare the interpolation grid]
    C --> D[Construct KDTree]
    D --> E[Search for N nearest neighbors]
    E --> F{Select IDW method}
    F -->|NumPy + numexpr| G1[Standard IDW interpolation]
    F -->|Dask| G2[Parallel IDW interpolation]
    G1 --> H[Save results]
    G2 --> H
    H --> I[Export to GeoTIFF]
    H --> J[Export to CSV]
    H --> K[Export to Surfer GRD]
```

### 3.2 Compression Workflow

```mermaid
graph TB
    A[Load GeoTIFF raster] --> B[Detect special values<br>0, -9999, NaN]
    B --> C[Divide into N×N blocks]
    C --> D[Distribute blocks to<br>multiple CPU cores]
    D --> E{Block type classification}
    E -->|Special values only <br> no data, 0 | F1[RLE encoding]
    E -->|Contains data| F2[2D-DCT + zigzag encoding]
    F2 --> G[Iterative coefficient<br>pruning for accuracy]
    G --> H[Quality verification<br>error checking]
    F1 --> I[Merge processed blocks]
    H --> I
    I --> J[serialization<br>with RLE]
    J --> K[Additional 7z compression]
```

---

<a name="requirements"></a>
## 4. System Requirements

### Dependencies

- **Python** 3.7+
- **Core Libraries**:
  - NumPy, SciPy: Array operations, KDTree, DCT
  - numexpr: Fast numerical expressions
  - Dask: Parallel computing
  - rasterio: GeoTIFF handling
- **File Format Support**:
  - Pandas: Tabular data
  - Shapefile: Vector data
  - laspy: LiDAR point clouds
- **Additional Tools**:
  - py7zr: 7z compression
  - tqdm: Progress visualization

### Hardware Recommendations

- **CPU**: Multi-core processor (recommended for Dask mode and parallel compression)
- **RAM**: 8GB minimum, 16GB+ recommended for larger datasets
- **Storage**: Sufficient space for input/output data
- **Platform**: Compatible with Windows and Linux

---

<a name="installation"></a>
## 5. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FIDWaC.git
cd FIDWaC

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

<a name="configuration"></a>
## 6. Configuration and Usage

<a name="json-config"></a>
### 6.1 JSON Configuration System

FIDWaC uses JSON format for configuration because:

- **Readability**: Simple human-readable format that's easy to edit
- **Structured**: Hierarchical structure for organizing related parameters
- **Language-Independent**: Can be processed by various programming languages
- **Validation**: Easy to validate against schemas
- **No Code Changes**: Allows changing parameters without modifying code
- **Portability**: Easily shareable between different environments
- **Versatility**: Supports various data types (numbers, strings, booleans, arrays)

<a name="interp-config"></a>
### 6.2 Interpolation Configuration

Configuration for the interpolation module (`interpolation_FIT.py`):

```json
{
  "results_directory": "./results/",
  "z_field_name": "z",
  "N": 12,
  "resolution": 0.5,
  "max_distance": 50,
  "leafsize": 16,
  "weights": 1,
  "rasterCrs": "EPSG:2180",
  "save_data_to_shp": false,
  "idw_dask": true,
  "idw_numpy": false,
  "knn_calculate": false,
  "knn_image": false,
  "interpolation_image_create": true,
  "csv_result": true,
  "surfer_grd": false
}
```
    results_directory: Directory where the processing results will be saved.
    z_field_name: Name of the field representing Z values (elevations) in the input data.
    N: Number of neighbors used in the IDW (Inverse Distance Weighting) method.
    resolution: Resolution of the interpolation grid.
    max_distance: Maximum search distance for neighbors.
    leafsize: Leaf size for the KDTree structure.
    weights: Weights used in IDW calculations.
    rasterCrs: Coordinate reference system for the output raster.
    save_data_to_shp: Flag indicating whether to save data to a Shapefile.
    idw_dask: Flag indicating whether to use Dask for IDW calculations.
    idw_numpy: Flag indicating whether to use NumPy for IDW calculations.
    knn_calculate: Flag indicating whether to calculate KNN (K-nearest neighbors).
    knn_image: Flag indicating whether to create a KNN image.
    interpolation_image_create: Flag indicating whether to create an interpolation image.
    csv_result: Flag indicating whether to save results to a CSV file.
    surfer_grd: Flag indicating whether to save results in Surfer grid format.


<a name="comp-config"></a>
### 6.3 Compression Configuration

Configuration for the compression module (`compress_function.py`):

```json
{
  "results_directory": "./results_compression/",
  "source_directory": "./source/",
  "accuracy": 0.01,
  "matrix": 8,
  "decimal": 2,
  "type_dct": 2
}
```

    results_directory: Directory where the processing results will be saved.
    source_directory: Directory where the source data is located.
    accuracy: Accuracy of the compression algorithm. Setting accuracy to 0 allows for saving data with the maximum possible accuracy relative to the original, but does not guarantee lossless compression.
    matrix: Matrix size used in the compression process.
    decimal: Number of decimal places to use in the output.
    type_dct: Type of Discrete Cosine Transform (DCT) to use.



<a name="usage"></a>
### 6.4 Usage Examples

## Download Data

Click the link below to download the data:

[SAMPLE_DATA](https://zutedupl-my.sharepoint.com/:f:/g/personal/alysko_zut_edu_pl/Et_gh1jMHc1FpyUV54IvqbIBSMft-kS_2VjHdPQXj-GtOg?e=8sZVvT))

#### Using interpolation_FIT.py

```bash
# Basic usage with a shapefile
python interpolation_FIT.py path/to/your/data.shp

# Using with a LAS/LAZ file
python interpolation_FIT.py path/to/lidar.las

# Using with a CSV file
python interpolation_FIT.py path/to/points.csv
```

#### Using compress_function.py

```bash
# For compression of a GeoTIFF file
python compress_function.py path/to/your/geotiff.tif

# For decompression (automatically detected by .7z extension)
python compress_function.py path/to/your/compressed_file.7z
```

<a name="batch"></a>
### 6.5 Batch Processing

FIDWaC supports batch processing of multiple files using tools like `find` and `parallel`:

#### Batch Interpolation

```bash
# Process all shapefiles in a directory
find ./data -name "*.shp" | parallel -j 4 python interpolation_FIT.py {}

# Process all LAS files with a specific naming pattern
find ./lidar -name "*_2023*.las" | parallel python interpolation_FIT.py {}
```

#### Batch Compression

```bash
# Compress all GeoTIFF files in the results directory
find ./results -name "*.tif" | parallel -j 8 python compress_function.py {}

# Decompress all compressed files
find ./compressed -name "*.7z" | parallel python compress_function.py {}
```

Reference: Tange, O. (2022, November 22). GNU Parallel 20221122 ('Херсо́н'). Zenodo. https://doi.org/10.5281/zenodo.7347980

---

<a name="multi-threading"></a>
## 7. Multi-threading Capabilities

FIDWaC is designed to take advantage of modern multi-core processors for improved performance:

### Interpolation Multi-threading

- **KDTree Queries**: Uses all available CPU cores for neighbor searches with `workers=-1` parameter
- **Dask Parallel Processing**: Automatically distributes interpolation calculations across multiple cores
- **Load Balancing**: Dask automatically balances workloads across available CPU resources
- **Memory Management**: Chunks large arrays to process data larger than available RAM

### Compression Multi-threading

- **Block-Level Parallelism**: Processes multiple N×N blocks simultaneously
- **Python Multiprocessing**: Uses the `multiprocessing` module to distribute work
- **Configurable Thread Count**: Can specify number of processes or use all available cores
- **Independent Block Processing**: Perfect for parallel execution as blocks can be processed independently

Both modules automatically detect the number of available CPU cores and adjust their parallelism accordingly, making efficient use of your hardware without manual configuration.

---

<a name="key-functions"></a>
## 8. Key Functions

### Interpolation

- **near_divided()**: Rounds values to nearest divisible by resolution
- **calculate_idw()**: Optimized IDW using numexpr
- **calculate_idw_dask()**: Parallel IDW using Dask

### Compression

- **dct2()** / **idct2()**: 2D DCT and inverse DCT
- **to_zigzag()** / **from_zigzag()**: Zigzag encoding/decoding
- **refine_dct_array()**: Coefficient optimization for quality control
- **encode_mask_rle()** / **decode_mask_rle()**: Run-length encoding for RLE masks

---

<a name="examples"></a>
## 9. Examples

### Interpolation Example

```python
# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Prepare data
data = np.loadtxt('sample_points.csv', delimiter=',', skiprows=1)
# Clean data (remove NaN and -9999)
data = data[~np.any(np.isnan(data) | (data == -9999), axis=1)]
x, y, z = data[:,0], data[:,1], data[:,2]

# Create KDTree
tree = spatial.cKDTree(data[:,:2], leafsize=config['leafsize'])

# Query KDTree for nearest neighbors (multi-threaded)
distance, index = tree.query(grid_points, 
                            k=config['N'],
                            workers=-1,  # Use all CPU cores
                            distance_upper_bound=config['max_distance'])

# Perform IDW interpolation
if config['idw_dask']:
    result = calculate_idw_dask(distance, config['weights'], value_data)
else:
    result = calculate_idw(distance)
```

### Compression Example

```python
# Load a GeoTIFF file
with rasterio.open('input.tif') as src:
    image = src.read(1)
    transform = src.transform
    crs = src.crs

# Divide into blocks
blocks = sliding_window_view(image, (N, N))

# Set up multiprocessing pool
num_processes = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=num_processes)

# Process blocks in parallel
block_data = [(i, block, nodata_value) for i, block in enumerate(blocks)]
results = pool.map(process_block, block_data)
pool.close()
pool.join()

# Assemble results
compressed_blocks = {}
for block_idx, compressed_data in results:
    compressed_blocks[block_idx] = compressed_data
```

---

<a name="benefits"></a>
## 10. Advantages and Applications

### Advantages

- **High Performance**: Optimized algorithms for processing large geospatial datasets
- **Memory Efficiency**: Dask-based processing for datasets larger than available RAM
- **Quality Control**: Precise control over compression accuracy
- **Format Flexibility**: Support for multiple input/output formats
- **Compression Ratio**: Significant file size reduction while maintaining quality
- **Multi-threaded Processing**: Efficient use of all available CPU cores
- **Configurable Precision**: Adjustable trade-off between quality and compression ratio

### Applications

- **Bathymetry**: Processing and storing sonar data from water bodies
- **Terrain Modeling**: Creating DEM/DTM from LiDAR or photogrammetry
- **Environmental Monitoring**: Processing time-series data for change detection
- **Hydrology**: Managing flood mapping and hydrological modeling data
- **GIS Data Management**: Efficient storage of large raster catalogs

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this software in your research, please cite:

```
@software{fidwac2025,
  author = {Andrzej Łysko, Wojciech Maleika, Witold Maćków, Malwina Bondarewicz, Jakub Śledziowski2, Paweł Terefenko}
  title = {FIDWaC: Fast Inverse Distance Weighting and Compression},
  year = {2025},
  url = {https://github.com/yourusername/FIDWaC}
}
