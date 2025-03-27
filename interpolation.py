# Standard library imports
import time
import os
import sys
import json

# Third-party library imports
import numpy as np
float = np.float32 
import scipy
from scipy import spatial
import rasterio
from rasterio.enums import Compression
import pandas as pd
import geopandas as gpd
import numexpr as ne
import dask.array as da
import shapefile
import laspy
from rasterio.transform import Affine
from typing import Tuple, Union, List, Optional

def near_divided(number: np.ndarray, resolution: float) -> np.ndarray:
    """
    Round up a number to the nearest multiple of resolution.
    
    Parameters
    ----------
    number : np.ndarray
        Input array of numbers to be rounded
    resolution : float
        Resolution value to round to
        
    Returns
    -------
    np.ndarray
        Array with numbers rounded up to nearest multiple of resolution
    """
    remainder = number % resolution
    nearest_value = number + remainder
    return nearest_value.astype(np.int64)

def calculate_idw(distance: np.ndarray) -> np.ndarray:
    """
    Calculate Inverse Distance Weighting interpolation using NumExpr.
    
    Parameters
    ----------
    distance : np.ndarray
        Array of distances between points
        
    Returns
    -------
    np.ndarray
        Interpolated values using IDW method
    
    Notes
    -----
    Uses global variables 'weights' and 'value_data' in the calculation
    """
    w = ne.evaluate(f"{weights}/(distance)**2") 
    a=np.nansum(w * value_data, axis=1)
    b=np.nansum(w, axis=1)
    idw = ne.evaluate("a/b")
    return idw

def calculate_idw_dask(distance: np.ndarray, weights: float, value_data: np.ndarray) -> np.ndarray:
    """
    Calculate Inverse Distance Weighting interpolation using Dask for parallel processing.
    
    Parameters
    ----------
    distance : np.ndarray
        Array of distances between points
    weights : float
        Weight factor for IDW calculation
    value_data : np.ndarray
        Array of values to be interpolated
        
    Returns
    -------
    np.ndarray
        Interpolated values using IDW method with Dask computation
    """
    # Convert numpy arrays to dask arrays for parallel processing
    distance_dask = da.from_array(distance, chunks='auto')
    value_data_dask = da.from_array(value_data, chunks='auto')        
    
    # Handle NaN values
    distance_dask = da.nan_to_num(distance_dask)
    value_data_dask = da.nan_to_num(value_data_dask)
    no_nan = 1e-10  # epsilon value to avoid division by zero
    
    # Calculate IDW
    w = weights/(distance_dask ** 2 + no_nan)
    a = da.nansum(w * value_data_dask, axis=1)
    b = da.nansum(w, axis=1)
    idw = a/b  
    return idw.compute()

# Parse command line arguments
file_full_path = sys.argv[1]
print(f"Script path: {file_full_path}")

# Load configuration
with open(r'./interpolation/config.json', 'r') as file:
    config_data = json.load(file)    

# Extract configuration parameters
results_directory = config_data.get("results_directory")
z_field_name = config_data.get("z_field_name")
filter_las = config_data.get("filter_las")
filter_classes = config_data.get("filter_classes")
smooth_result = config_data.get("smooth_result")
smooth_level = config_data.get("smooth_level")
geoid_correction = config_data.get("geoid_correction")
geoid_correction_file = config_data.get("geoid_correction_file")
geoidCrs = config_data.get("geoidCrs")
sourceCrs = config_data.get("sourceCrs")
base_name, file_extension = os.path.splitext(os.path.basename(file_full_path))

# KDTree and IDW configuration parameters
N = config_data.get("N")  # Number of neighbors
resolution = config_data.get("resolution")  # Grid resolution
max_distance = config_data.get("max_distance")  # Maximum search distance
leafsize = config_data.get("leafsize")  # Leaf size for KDTree
weights = config_data.get("weights")  # Weights for IDW

# Construct output filename
file_parameter = base_name + '-N-' + str(N) + '-R-' + str(resolution) + '-dist-' + str(max_distance)

# Feature flags from configuration
save_data_to_shp = config_data.get("save_data_to_shp")  # Save data to shapefile
idw_dask = config_data.get("idw_dask")  # Use Dask for IDW calculation
idw_numpy = config_data.get("idw_numpy")  # Use NumPy for IDW calculation
knn_calculate = config_data.get("knn_calculate")  # Calculate KNN
interpolation_image_create = config_data.get("interpolation_image_create")  # Create interpolation image
csv_result = config_data.get("csv_result")  # Save results to CSV
surfer_grd = config_data.get("surfer_grd")  # Save results to Surfer grid format

# Print configuration information
print(f'File name: {file_full_path}')
print(f'File result directory: {results_directory}')
print('\nConfiguration:')
print(f'Search N neighbours: {N}')
print(f'Resolution grid: {resolution}')
print(f'Max distance search: {max_distance}')

# Read input data based on file extension
print("\nReading input file...")
start = time.time() 

if file_extension == '.shp':    
    print('Reading Shapefile...')
    sf = shapefile.Reader(file_full_path)
    fields = [field[0] for field in sf.fields[1:]]
    records = [list(record.record) for record in sf.iterShapeRecords()]
    data = pd.DataFrame(records, columns=fields).to_numpy()
    # Remove no data values (-9999) and NaN
    data = data[~np.any(np.isnan(data) | (data == -9999), axis=1)]    
    del sf, fields, records # Free up memory
    

elif file_extension in ['.las', '.laz']:
    print('Reading LAS/LAZ file...')    
    with laspy.open(file_full_path) as lasfile:
        las = lasfile.read()
    if filter_las:
        las= las[las.classification == filter_classes]
    x, y, z = las.x, las.y, las.z
    data = np.stack((x, y, z), axis=-1)
    # Remove no data values (-9999) and NaN
    data = data[~np.any(np.isnan(data) | (data == -9999), axis=1)]
    del las, x, y, z# Free up memory
        
elif file_extension in ['.txt', '.csv']:  
    print('Reading CSV/TXT file...')
    data = np.loadtxt(file_full_path, delimiter=',', skiprows=1)
    # Remove no data values (-9999) and NaN
    data = data[~np.any(np.isnan(data) | (data == -9999), axis=1)]
    #x, y, z = data[:,0], data[:,1], data[:,2]

# Optionally save point data to shapefile
if save_data_to_shp:
    print('Saving points to Shapefile...')
    w = shapefile.Writer(f'{file_full_path}.shp', shapeType=shapefile.POINT)
    w.field('x', 'F', decimal=10)
    w.field('y', 'F', decimal=10)
    w.field('z', 'F', decimal=10)
    for xt, yt, zt in data:
        w.point(xt, yt)
        w.record(xt, yt, zt)
    w.close()
    del w # Free up memory

"""
Apply geoid correction to elevation data.

This functionality adjusts elevation values from ellipsoidal heights (typically measured by GPS)
to orthometric heights (heights above mean sea level) by subtracting the geoid undulation.
Geoid undulation is the height difference between the reference ellipsoid and geoid at any given point.

The process includes:
1. Loading geoid model data from a specified file
2. Transforming coordinates between source CRS and geoid CRS if needed
3. Creating a KDTree for efficient spatial querying of the geoid model
4. Finding the nearest geoid height value for each input point
5. Adjusting elevation values by subtracting the geoid undulation

Parameters used from configuration:
- geoid_correction_file: Path to file containing geoid model data
- geoidCrs: Coordinate reference system of the geoid model
- sourceCrs: Coordinate reference system of the input data

The correction is critical for:
- Converting GPS-derived heights to heights usable in engineering applications
- Ensuring accurate representation of terrain relative to sea level
- Properly integrating datasets with different vertical reference systems

Note:
The correction only applies to points within the specified distance (max_distance_geoid)
of a geoid model point. Points outside this range remain unchanged.
"""
if geoid_correction:
    import chardet    
    print('Applying geoid correction...')
    # prepare geoid correction    
    max_distance_geoid = 2000
    #check file
    with open(geoid_correction_file, 'rb') as f:
        rawdata = f.read(100000)
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        print(f"Encoding: {encoding}")

    df = pd.read_csv(geoid_correction_file, sep=' ', names=['x', 'y', 'z'],  header=None, encoding=encoding)
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df["x"], df["y"])).set_crs(geoidCrs)
    try:
        gdf=gdf.to_crs(sourceCrs)
    except:
        print(f'geoidCrs:{geoidCrs} and sourceCrs:{sourceCrs} are not compatible or this same')
    #x_geoid, y_geoid, z_geoid = df['x'], df['y'], df['z']
    # prepare data for geoid correction 
    print('Preparing data for geoid correction...')    
    gdf_data = gpd.GeoDataFrame({'x': data[:,0], 'y': data[:,1], 'z': data[:,2]}, geometry=gpd.points_from_xy(data[:,0], data[:,1])).set_crs(sourceCrs)
    # Create KD-Tree for geoid correction
    print('Creating KD-Tree for geoid correction...')
    tree_geoid = spatial.cKDTree(np.column_stack((gdf.geometry.x, gdf.geometry.y)), leafsize=leafsize)  
    distance_geoid, index_geoid = tree_geoid.query(
    np.column_stack((data[:,0], data[:,1])),
    workers=-1,
    k=1,
    distance_upper_bound=max_distance_geoid
    )    
    valid_mask = np.isfinite(distance_geoid)
    z_geoid_nearest = np.full(data.shape[0], np.nan)
    z_geoid_vals = gdf['z'].to_numpy()
    z_geoid_nearest[valid_mask] = z_geoid_vals[index_geoid[valid_mask]]
    # uppdate data z
    data[valid_mask, 2] = data[valid_mask, 2] - z_geoid_nearest[valid_mask]
    print("Geoid correction applied to", valid_mask.sum(), "points.")
    del gdf, gdf_data, tree_geoid, distance_geoid, index_geoid, valid_mask, z_geoid_nearest, z_geoid_vals # Free up memory

# Print data statistics
try:    
    print('\nData Statistics:')
    print(f'Source min x: {np.min(data[:,0]):.2f}')
    print(f'Source max x: {np.max(data[:,0]):.2f}')
    print(f'Source min y: {np.min(data[:,1]):.2f}')
    print(f'Source max y: {np.max(data[:,1]):.2f}')
    print(f'Source min z: {np.min(data[:,2]):.2f}')
    print(f'Source max z: {np.max(data[:,2]):.2f}')
    print(f'Source points: {len(data[:,1]):.0f}') 
except:
    print('0 points readed. Mayby LAS filter not find poinsts or file damaged!!!')
    sys.exit(1)

end = time.time()
print(f'Time elapsed: {end - start:.2f} seconds')

# Prepare grid extent
xmin = near_divided(np.min(data[:,0]), resolution).astype(int)
xmax = near_divided(np.max(data[:,0]), resolution).astype(int)
ymin = near_divided(np.min(data[:,1]), resolution).astype(int)
ymax = near_divided(np.max(data[:,1]), resolution).astype(int)

print('\nGrid Extent:')
print(f'min x: {xmin}')
print(f'max x: {xmax}')
print(f'min y: {ymin}')
print(f'max y: {ymax}')

# Calculate grid dimensions
spacex = ((xmax-xmin)/resolution).astype(int)  # shape x
spacey = ((ymax-ymin)/resolution).astype(int)  # shape y

print(f'Grid dimensions: {spacex} x {spacey}')

# Create interpolation grid
xx = np.linspace(xmin, xmax, spacex+1)
yy = np.linspace(ymin, ymax, spacey+1)
zz = np.nan

# Create empty meshgrid
# Adjust grid cell centers by half resolution
X, Y, Z = np.meshgrid(xx+resolution/2, yy-resolution/2, zz)
data2 = np.array((X.ravel(), Y.ravel(), Z.ravel())).T

# Calculate KD-Tree for nearest neighbor search
print('\nCreating KD-Tree...')
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
tree = spatial.cKDTree(data[:,:2], 
                      leafsize=leafsize,
                      balanced_tree=True,
                      compact_nodes=True)

# Query KD-Tree for nearest neighbors
distance, index = tree.query(data2[:,:2],
                           k=N,
                           workers=-1,
                           distance_upper_bound=max_distance, 
                           eps=0,
                           p=2)  # Euclidean distance

end = time.time()
print(f'Time elapsed: {end - start:.2f} seconds')

print('Adjusting zero distances...')
# Replace zero distances with half of minimum non-zero distance per row
min_value_greater_than_zero = np.min(np.where(distance > 0, distance, np.max(distance)), axis=1)
distance = np.where(distance == 0, min_value_greater_than_zero[:, np.newaxis] / 2, distance)

# Create mask for points beyond max_distance
m = np.array(distance == np.inf)
# Reset invalid indices
index[index == len(data)] = 0

end = time.time()
print(f'Time elapsed: {end - start:.2f} seconds')

print('Preparing interpolation values...')
# Get z values for each neighbor point
value_data = np.take(data[:,2], index.ravel())
value_data = value_data.reshape(np.shape(index)[0], np.shape(index)[1])

# Convert arrays to float for NaN handling
value_data = value_data.astype(float)
distance = distance.astype(float)

# Mask out points beyond max_distance
value_data[m] = np.nan
distance[m] = np.nan

end = time.time()
print(f'Time elapsed: {end - start:.2f} seconds')

print('Starting interpolation...')
if idw_dask:
    print('Using Dask for parallel processing...')
    start_idw = time.time()
    idw=calculate_idw_dask(distance,weights,value_data)
    idw[idw == 0] = np.nan
    end = time.time()
    print(f'Time elapsed: {end - start_idw:.2f} seconds')
    
if idw_numpy:
    print('Using NumPy for interpolation...')
    start_numpy = time.time()
    idw=calculate_idw(distance)
    end = time.time()
    print(f'Time elapsed: {end - start_numpy:.2f} seconds')
    
if knn_calculate:
    start_knn = time.time()
    print('Calculating KNN...')
    knn=np.nanmean(value_data, axis=1)
    end = time.time()
    print(f'Time elapsed: {end - start_knn:.2f} seconds')

# Generate result array
print('Generating result array...')
spacex=np.shape(X)[0] # shapex
spacey=np.shape(Y)[1] # shapey

# Calculate Knn or IDW
if knn_calculate:
    print('Calculate KNN...')
    result=np.column_stack((data2[:,:2], knn))  # xyz to grid
    grid_result=result[:,2].reshape(spacex,spacey)
else:
    if idw_numpy or idw_dask:
        print('Calculate IDW...')
    result=np.column_stack((data2[:,:2], idw)) # xyz to grid
    grid_result=result[:,2].reshape(spacex,spacey)
if smooth_result: # smooth by Gauss
    grid_result = scipy.ndimage.gaussian_filter(grid_result, sigma=smooth_level)
grid_result = np.where(np.isnan(grid_result) | (grid_result < -9995), -9999, grid_result)
end = time.time()
print(f'Time elapsed: {end - start:.2f} seconds')

# Save CSV
if csv_result:
    print('Saving data to CSV...')
    df = pd.DataFrame(result, columns = ['x','y','value'])
    df.to_csv(f'{results_directory}'+file_parameter+'result.csv',sep=',') #save results to csv
    end = time.time()
    print(f'Time elapsed: {end - start:.2f} seconds')

# Generate geotif
if interpolation_image_create:
    print('Generating geotif...')
    transform = Affine.translation(xmin, ymin-resolution)* Affine.scale(resolution, resolution)
    interpRaster = rasterio.open(results_directory+file_parameter+'_geotif.tif',
                                    'w',
                                    driver='GTiff',
                                    height=grid_result.shape[0],
                                    width=grid_result.shape[1],
                                    count=1,
                                    crs=sourceCrs,
                                    transform=transform,
                                    nodata=-9999,
                                    dtype=rasterio.float32,
                                    compress=Compression.deflate,
                                    predictor=2,
                                    tiled=True,
                                    blockxsize=256,
                                    blockysize=256                                 
                                    )
    interpRaster.write(grid_result,1)
    interpRaster.close()

# Generate surfer grid
if surfer_grd:
    print('Generating Surfer grid...')
    ascii_grid_header='ascii_grid_header.grid'
    
    header="{a}\n{b}\n{c}\n{d}\n{e}\n".format(
        a='DSAA',
        b=str(spacey)+' '+str(spacex),
        c=str(xmin+resolution/2)+' '+str(xmax+resolution/2),
        d=str(ymin-resolution/2)+' '+str(ymax-resolution/2),
        e=str(resolution)+' '+str(resolution))
    print('grid surfer header',header)

    grid_result[np.isnan(grid_result)]=-9999
    with open(f'{results_directory}'+file_parameter+'_grid.grd', 'w')as file_grid:
        file_grid.write(header)
        np.savetxt(file_grid, grid_result, fmt='%f', delimiter=' ', newline=os.linesep)
        file_grid.close()    
    

end = time.time()
print(f'END time: {end - start:.2f} seconds')
end_string_idw = "{:.2f}".format(end - start)
print('end calculate')
