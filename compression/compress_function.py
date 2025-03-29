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
import msgpack

# External libraries
import numpy as np
import py7zr
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import griddata
from scipy.fftpack import dct, idct
import rasterio
from rasterio.enums import Compression
from typing import List, Tuple, Dict, Any, Optional, Union

# Load configuration
with open(r"./compression/config.json", "r") as file:
    config_data = json.load(file)

# Parameters from configuration
result_dir = config_data.get("results_directory")
source_dir = config_data.get("source_directory")
accuracy = config_data.get("accuracy")
N = config_data.get("matrix")
sourceCrs_force_declare = config_data.get("sourceCrs_force_declare")
global decimal
decimal = config_data.get("decimal")
type_dct = config_data.get("type_dct")


# Generate zero matrix
def generate_zeros(number_of_zeros):
    return "0" * number_of_zeros


# Set compression accuracy
global scaling_factor
if decimal != 0:
    scaling_factor = int(str("1" + generate_zeros(decimal)))
else:
    scaling_factor = 1


# 2D DCT implementation
def dct2(a: np.ndarray) -> np.ndarray:
    """
    Performs a two-dimensional Discrete Cosine Transform (DCT).

    The Discrete Cosine Transform is similar to the Fourier Transform but uses only
    cosine functions and produces real coefficients. It is commonly used
    in image compression (e.g., JPEG).

    Parameters:
    ----------
    a : np.ndarray
        Input data block (2D matrix)

    Returns:
    -------
    np.ndarray
        Result of the DCT - matrix of coefficients in the frequency domain
    """
    return dct(dct(a.T, norm="ortho", type=type_dct).T, norm="ortho", type=type_dct)


# 2D IDCT implementation
def idct2(a: np.ndarray) -> np.ndarray:
    """
    Performs an inverse two-dimensional Discrete Cosine Transform (IDCT).

    The inverse Discrete Cosine Transform converts coefficients from the frequency domain
    back to the spatial domain, reconstructing the original signal.

    Parameters:
    ----------
    a : np.ndarray
        Input data block (in the frequency domain)

    Returns:
    -------
    np.ndarray
        Result of the inverse DCT - matrix in the spatial domain
    """
    return idct(idct(a.T, norm="ortho", type=type_dct).T, norm="ortho", type=type_dct)


# ===== ZIGZAG FUNCTIONS =====
def to_zigzag(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a 2D matrix to a 1D vector using zig-zag scanning.

    Zig-zag scanning is a technique used in image compression (e.g., JPEG),
    which transforms a 2D matrix into a 1D vector in such a way that low-frequency
    coefficients are placed at the beginning of the vector.

    Parameters:
    ----------
    matrix : np.ndarray
        Input 2D matrix

    Returns:
    -------
    np.ndarray
        1D vector containing the matrix elements in zig-zag order
    """
    matrix = np.array(matrix)
    rows, cols = matrix.shape
    vector = np.empty(rows * cols, dtype=matrix.dtype)
    index = 0
    for i in range(rows + cols - 1):
        if i % 2 == 0:
            start_row = min(i, rows - 1)
            start_col = i - start_row
            count = min(start_row + 1, cols - start_col)
            vector[index : index + count] = matrix[
                start_row - np.arange(count), start_col + np.arange(count)
            ]
            index += count
        else:
            start_col = min(i, cols - 1)
            start_row = i - start_col
            count = min(start_col + 1, rows - start_row)
            vector[index : index + count] = matrix[
                start_row + np.arange(count), start_col - np.arange(count)
            ]
            index += count
    return vector


def from_zigzag(vector: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Converts a 1D vector back to a 2D matrix using inverse zig-zag scanning.

    Parameters:
    ----------
    vector : np.ndarray
        Input 1D vector
    rows : int
        Number of rows in the output matrix
    cols : int
        Number of columns in the output matrix

    Returns:
    -------
    np.ndarray
        2D matrix reconstructed from the vector
    """
    matrix = np.zeros((rows, cols), dtype=np.float64)
    index = 0
    for i in range(rows + cols - 1):
        if i % 2 == 0:
            start_row = min(i, rows - 1)
            start_col = i - start_row
            count = min(start_row + 1, cols - start_col)
            matrix[start_row - np.arange(count), start_col + np.arange(count)] = vector[
                index : index + count
            ]
            index += count
        else:
            start_col = min(i, cols - 1)
            start_row = i - start_col
            count = min(start_col + 1, rows - start_row)
            matrix[start_row + np.arange(count), start_col - np.arange(count)] = vector[
                index : index + count
            ]
            index += count
    return matrix

# ===== COMPRESSION ACCURACY CHECK FUNCTIONS =====
def check_precision_decompression_dct(
    original_matrix: np.ndarray, idct_reconstructed: np.ndarray
) -> Tuple[float, float]:
    """
    Checks the accuracy of matrix reconstruction after DCT compression.

    Parameters:
    ----------
    original_matrix : np.ndarray
        Original matrix before compression
    idct_reconstructed : np.ndarray
        Matrix reconstructed after compression

    Returns:
    -------
    Tuple[float, float]
        (mean_difference, max_difference) - reconstruction error metrics
    """
    difference = original_matrix - idct_reconstructed
    mean_value = np.mean(
        np.abs(difference)
    )  # Check the mean value of the matrix difference
    max_value = np.max(
        np.abs(difference)
    )  # Check the maximum value of the matrix difference
    return mean_value, max_value


# Inverse DCT function
def undct(
    split: np.ndarray, org_dct_zigzag: np.ndarray, original_matrix: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Performs an inverse DCT based on a partial vector of coefficients.

    The function reconstructs a matrix from a partial vector of DCT coefficients,
    filling in the missing coefficients with zeros, and then checks the accuracy of the reconstruction.

    Parameters:
    ----------
    split : np.ndarray
        Partial vector of DCT coefficients (first n elements)
    org_dct_zigzag : np.ndarray
        Full original vector of DCT coefficients in zig-zag order
    original_matrix : np.ndarray
        Original matrix before compression

    Returns:
    -------
    Tuple[np.ndarray, float, np.ndarray]
        (partial_vector, max_error, reconstructed_matrix)
    """
    array = np.zeros(
        len(org_dct_zigzag), dtype=np.float32
    )  # Empty zero array with length of org_dct_zigzag
    array[: len(split)] = split  # Adding split values to the empty array
    reconstructed_matrix = np.array(from_zigzag(array, N, N))
    idct_reconstructed = idct2(reconstructed_matrix)
    mean_value, max_value = check_precision_decompression_dct(
        original_matrix, idct_reconstructed
    )
    return split, max_value, idct_reconstructed


# Function to refine the DCT array
def refine_dct_array(
    org_dct_zigzag: np.ndarray,
    accuracy: float,
    agt: int,
    max_value: float,
    split_point: int,
    original_matrix: np.ndarray,
    max_iterations: Optional[int] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Optimizes the number of DCT coefficients needed to achieve the desired accuracy.

    The function uses a binary search algorithm to find the optimal number
    of DCT coefficients to retain in order to achieve the desired reconstruction accuracy.

    Parameters:
    ----------
    org_dct_zigzag : np.ndarray
        Original vector of DCT coefficients in zig-zag order
    accuracy : float
        Required reconstruction accuracy (maximum allowable error)
    agt : int
        Initial binary search step
    max_value : float
        Initial maximum reconstruction error
    split_point : int
        Initial split point (number of coefficients to retain)
    original_matrix : np.ndarray
        Original matrix before compression
    max_iterations : Optional[int]
        Maximum number of iterations of the algorithm. If None, it will be calculated automatically.

    Returns:
    -------
    Tuple[np.ndarray, float, np.ndarray]
        (optimized_vector_of_coefficients, max_error, reconstructed_matrix)
    """
    # Dynamic determination of the maximum number of iterations based on the size of the matrix
    if max_iterations is None:
        # Use logarithm to determine a reasonable number of iterations depending on the size of the matrix
        # The larger the matrix, the more iterations, but the increase is logarithmic
        max_iterations = int(20 * np.log2(len(org_dct_zigzag)) + 10)

    array_iteration = (
        []
    )  # All iterations to the list and then select the best iteration by index
    array_accuracy = []
    array_split_point = []

    iteration_count = 0

    while True:
        # Increment the iteration counter
        iteration_count += 1

        # Block of array division always in half depending on max_value
        agt = agt // 2  # I had for array and not at the beginning change
        split_point = (
            (split_point - agt) if max_value <= accuracy else (split_point + agt)
        )
        array, max_value, idct_reconstructed = undct(
            org_dct_zigzag[:split_point], org_dct_zigzag, original_matrix
        )

        # End of division
        array_iteration.append(array)  # Iteration arrays
        array_accuracy.append(max_value)
        array_split_point.append(split_point)

        # Check if the maximum number of iterations has been exceeded
        if iteration_count >= max_iterations:
            # Check if any result meets the accuracy criterion
            min_accuracy_index = np.argmin(array_accuracy)
            best_accuracy = array_accuracy[min_accuracy_index]

            if best_accuracy <= accuracy:
                # If we have a result that meets the accuracy requirements, use it
                result = np.array(array_iteration[min_accuracy_index]) * scaling_factor
                return np.round(result).astype(int), best_accuracy, idct_reconstructed
            else:
                # If no result meets the accuracy requirements,
                # increase the number of DCT coefficients until the required accuracy is achieved

                # Start with the best result so far
                current_split = array_split_point[min_accuracy_index]

                # Check if increasing the split improves accuracy
                while current_split < len(org_dct_zigzag):
                    current_split += 1
                    test_array, test_max_value, test_idct = undct(
                        org_dct_zigzag[:current_split], org_dct_zigzag, original_matrix
                    )

                    if test_max_value <= accuracy:
                        # Sufficient accuracy found
                        result = np.array(test_array) * scaling_factor
                        return np.round(result).astype(int), test_max_value, test_idct

                # If the required accuracy is still not achieved, use the entire original DCT
                full_array, full_max_value, full_idct = undct(
                    org_dct_zigzag, org_dct_zigzag, original_matrix
                )
                result = (
                    np.array(org_dct_zigzag) * scaling_factor
                )  # Use the entire original DCT
                return np.round(result).astype(int), full_max_value, full_idct

        if (
            agt == 0 and max_value > accuracy
        ):  # Iterations do not always yield a result below the accuracy; add one more element to the array until the condition is met
            agt = agt + 2

        if agt <= 1 and max_value <= accuracy:
            min_split_point = np.array(
                list(zip(array_accuracy, array_split_point, array_iteration)),
                dtype=object,
            )  # Combine arrays with results
            filtered_values = min_split_point[
                min_split_point[:, 0] <= accuracy
            ]  # Filter data by accuracy

            if len(filtered_values) > 0:
                sorted_indices = np.argsort(filtered_values[:, 1])  # Sort data
                sorted_data = filtered_values[sorted_indices]  # Get indices from data
                # The first row is the best, add to dcv_compress, multiply by scaling_factor for conversion to integer
                result = np.array(sorted_data[0, 2]) * scaling_factor
                return np.round(result).astype(int), max_value, idct_reconstructed
            else:
                # If no values meet the accuracy criterion
                # Use the last iteration
                result = np.array(array) * scaling_factor
                return np.round(result).astype(int), max_value, idct_reconstructed

        if agt <= 0 and max_value > accuracy:
            # If we can no longer divide and have not achieved the required accuracy
            result = np.array(array) * scaling_factor
            return np.round(result).astype(int), max_value, idct_reconstructed

    # in some cases if may not be met then result will take default data
    result = np.array(array) * scaling_factor
    return result.tolist(), max_value, idct_reconstructed


# ===== MATRIX RECONSTRUCTION FUNCTIONS =====


def reconstructed_matrix_function(
    element: Union[List, np.ndarray, int, float],
    reconstructed_zero_matrix: np.ndarray,
    array: np.ndarray,
    scaling_factor: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Function for reconstructing the matrix from compressed data.

    Parameters:
    ----------
    element : Union[List, np.ndarray, int, float]
        Compressed element (list or single value)
    reconstructed_zero_matrix : np.ndarray
        Matrix with marked special values (NoData, zeros)
    array : np.ndarray
        Empty array to fill
    scaling_factor : int
        Scaling factor

    Returns:
    -------
    Tuple[np.ndarray, int, int]
        (reconstructed_matrix, number_of_NoData_values, number_of_zero_values)
    """
    N = int(np.sqrt(len(array)))  # Block size

    # Handling the case when element is an integer or float, not a list
    if isinstance(element, (int, float)):
        # For an integer/float, set only the first value in the array
        array[0] = element / scaling_factor
    else:
        # For a list, set all values
        array[: len(element)] = (
            np.array(element) / scaling_factor
        )  # Adding data to the empty array and reverse by scaling factor

    reconstructed_matrix = np.array(from_zigzag(array, N, N))  # Reconstruct 3D models
    idct_reconstructed = idct2(reconstructed_matrix)

    # Restore special values based on reconstructed_zero_matrix
    nodata_positions_manual = []
    zero_positions_manual = []

    for i in range(reconstructed_zero_matrix.shape[0]):
        for j in range(reconstructed_zero_matrix.shape[1]):
            if reconstructed_zero_matrix[i, j] == -9999:
                nodata_positions_manual.append((i, j))
            elif reconstructed_zero_matrix[i, j] == 2:  # Changed from 1 to 2
                zero_positions_manual.append((i, j))

    # Restoring special values - first zeros, then NoData
    for pos in zero_positions_manual:
        idct_reconstructed[pos] = 0

    # Restoring NoData values
    for pos in nodata_positions_manual:
        idct_reconstructed[pos] = -9999

    # Counting special values in the final matrix
    final_nodata_count = np.sum(idct_reconstructed == -9999)
    final_zero_count = np.sum(idct_reconstructed == 0)

    return idct_reconstructed, final_nodata_count, final_zero_count


# ===== ERROR HANDLING =====


def handle_error(
    error_type: str,
    message: str,
    exception: Optional[Exception] = None,
    exit_program: bool = False,
) -> None:
    """
    Unified function for error handling in the application.

    Parameters:
    ----------
    error_type : str
        Error type (e.g., 'IO', 'Format', 'Compression')
    message : str
        Error message
    exception : Optional[Exception]
        Exception object, if available
    exit_program : bool
        Whether to terminate the program after the error
    """
    error_msg = f"[ERROR {error_type}] {message}"
    if exception:
        error_msg += f"\nDetails: {str(exception)}"

    print(error_msg)

    if exit_program:
        print("The program will be terminated due to a critical error.")
        sys.exit(1)


# ===== MASK ENCODING AND DECODING =====


def encode_mask_rle(mask: np.ndarray) -> List[int]:
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    RLE is a compression technique that replaces sequences of repeating
    values with a single value and the count of its repetitions.

    Parameters:
    ----------
    mask : np.ndarray
        Binary 2D mask (True/False or 1/0 values)

    Returns:
    -------
    List[int]
        List of integers where odd indices are the starts of series,
        and even indices are the lengths
    """
    # Flattening the mask to 1D
    flat_mask = mask.flatten()

    # Initialization
    rle = []
    current_val = False
    start_idx = 0

    # Iteration through the flattened mask
    for i in range(len(flat_mask) + 1):
        # If we've reached the end of the mask or the value has changed
        if i == len(flat_mask) or flat_mask[i] != current_val:
            if current_val:  # We only record series of ones (True)
                rle.append(start_idx)  # Start of series
                rle.append(i - start_idx)  # Length of series

            if i < len(flat_mask):
                current_val = flat_mask[i]
                start_idx = i

    return rle


def decode_mask_rle(rle: List[int], shape: Tuple[int, int]) -> np.ndarray:
    """
    Decodes a mask from RLE format.

    Parameters:
    ----------
    rle : List[int]
        List of integers where odd indices are the starts of series,
        and even indices are the lengths
    shape : Tuple[int, int]
        Shape of the original mask (height, width)

    Returns:
    -------
    np.ndarray
        Decoded binary 2D mask
    """
    # Initialize empty mask
    flat_mask = np.zeros(shape[0] * shape[1], dtype=bool)

    # Set True values for each series
    for i in range(0, len(rle), 2):
        if i + 1 < len(rle):  # Make sure we have a pair (start, length)
            start = rle[i]
            length = rle[i + 1]
            flat_mask[start : start + length] = True

    # Transform back to 2D
    return flat_mask.reshape(shape)


# ===== ZERO VALUE INTERPOLATION =====


def interpolate_zeros(
    matrix_for_compression: np.ndarray, nonzero_mask: np.ndarray
) -> np.ndarray:
    """
    Interpolates zero values using KNN method.

    Parameters:
    ----------
    matrix_for_compression : np.ndarray
        Matrix with values to interpolate
    nonzero_mask : np.ndarray
        Mask indicating non-zero positions

    Returns:
    -------
    np.ndarray
        Matrix with interpolated values
    """
    # Creating a copy for modification
    compression_copy = matrix_for_compression.copy()

    # If there are no non-zero values, there's nothing to interpolate
    if not np.any(nonzero_mask):
        return compression_copy

    # Creating a grid of points
    y_indices, x_indices = np.indices(matrix_for_compression.shape)

    # Coordinates of non-zero points
    points = np.column_stack((y_indices[nonzero_mask], x_indices[nonzero_mask]))

    # Values at non-zero points
    values = matrix_for_compression[nonzero_mask]

    # Coordinates of zero points that we want to interpolate
    y_zeros, x_zeros = np.indices(matrix_for_compression.shape)
    zero_mask = ~nonzero_mask

    # If there are no points to interpolate, return the original matrix
    if not np.any(zero_mask) or len(points) == 0:
        return compression_copy

    grid_y = y_zeros[zero_mask]
    grid_x = x_zeros[zero_mask]
    grid_points = np.column_stack((grid_y, grid_x))

    try:
        # KNN interpolation - we use k=min(3, number of non-zero points)
        k = min(3, len(points))
        interpolated = griddata(points, values, grid_points, method="nearest")

        # Set interpolated values
        compression_copy[zero_mask] = interpolated
    except Exception as e:
        # In case of interpolation error, use the mean
        mean_value = np.mean(values) if len(values) > 0 else 0
        compression_copy[zero_mask] = mean_value
        print(f"Interpolation error: {str(e)}")

    return compression_copy


# Function to process a single block - will be used in multi-threaded processing
def process_block(
    block_data: Tuple[int, np.ndarray, Optional[float]],
) -> Tuple[int, List, float]:
    """
    Processes a single image block.

    Parameters:
    ----------
    block_data : tuple
        Tuple containing (block_index, block_data, nodata_value)

    Returns:
    -------
    tuple
        (block_index, compressed_data, max_error)
    """
    idx, original_matrix, nodata_value = block_data
    max_error = 0.0  # Initialize max error for this block

    # Fast path for zero blocks
    if np.all(original_matrix == 0):
        return idx, [0], 0.0  # 0 indicates an all-zero block (special code)

    # Fast path for NoData blocks
    if nodata_value is not None and np.all(original_matrix == nodata_value):
        return idx, [-1], 0.0  # -1 indicates an all-NoData block (special code)

    # Create a copy of the original matrix for modification
    matrix_for_compression = original_matrix.copy()

    # Create masks
    has_special_values = False
    masks = []

    # Handle NoData values
    if nodata_value is not None:
        nodata_mask = original_matrix == nodata_value
        if np.any(nodata_mask):
            has_special_values = True
            # Encode NoData mask using RLE
            nodata_rle = encode_mask_rle(nodata_mask)
            # Store as [1, ...RLE data...], where 1 is the mask type (NoData)
            masks.append([1] + nodata_rle)

            # Replace NoData values with 0 for compression purposes (-9999 values would make DCT compression difficult)
            matrix_for_compression[nodata_mask] = 0

    # Handle zero values (but not those that were NoData)
    zero_mask = original_matrix == 0
    if nodata_value is not None:
        # Exclude values that were NoData
        zero_mask = zero_mask & ~(original_matrix == nodata_value)

    if np.any(zero_mask):
        has_special_values = True
        # Encode zero mask using RLE
        zero_rle = encode_mask_rle(zero_mask)
        # Store as [2, ...RLE data...], where 2 is the mask type (zeros)
        masks.append([2] + zero_rle)

    # If we have special values, add masks to the data
    if has_special_values:
        # KNN implementation instead of simple mean
        # First create a copy for compression
        compression_copy = matrix_for_compression.copy()

        # Find positions of non-zero values (those that are neither NoData nor 0)
        nonzero_mask = matrix_for_compression != 0

        if np.any(nonzero_mask):
            # Interpolation of zero values
            compression_copy = interpolate_zeros(compression_copy, nonzero_mask)

        # Calculate DCT
        orginal_dct = dct2(compression_copy)
        orginal_dct = np.round(orginal_dct, decimal)
        org_dct_zigzag = np.array(to_zigzag(orginal_dct))

        agt = len(org_dct_zigzag) // 2

        # Compression
        compressed_array, max_error, _ = refine_dct_array(
            org_dct_zigzag, accuracy, agt, accuracy + 1, 0, compression_copy
        )
        # Return index, masks, compressed data, and max error
        result = [idx]
        for mask in masks:
            result.append(mask)
        result.append(compressed_array)
        return idx, result, max_error
    else:
        # Standard compression without special values
        orginal_dct = dct2(matrix_for_compression)
        orginal_dct = np.round(orginal_dct, decimal)
        org_dct_zigzag = np.array(to_zigzag(orginal_dct))

        agt = len(org_dct_zigzag) // 2

        # Compression
        compressed_array, max_error, _ = refine_dct_array(
            org_dct_zigzag, accuracy, agt, accuracy + 1, 0, matrix_for_compression
        )
        return idx, compressed_array, max_error


def compress_image(file_path, num_processes=None):
    """
    Function for compressing an image using N x N blocks and DCT transform.

    Parameters:
    ----------
    file_path : str
        Path to the image file
    num_processes : int, optional
        Number of processes to use for compression.
        If not specified, uses all available cores.

    Returns:
    -------
    tuple
        (compressed_data, image, transformation, rasterCrs, padded_shape, max_error)
    """
    print(f"Loading image: {file_path}")

    # Load image
    with rasterio.open(file_path) as image:
        transform = image.transform
        global rasterCrs
        rasterCrs = image.crs
        bounds = image.bounds

        # Display image parameters
        print(f"Image dimensions: {image.shape}")
        X, Y = image.shape
        pixelSizeX, pixelSizeY = image.res
        x_min, y_min, x_max, y_max = bounds

        # Load data
        image_array = image.read()
        image_array = np.where(np.isnan(image_array), -9999, image_array) #-9999 when NaN in source

    # Display value range
    z_min, z_max = np.min(image_array[image_array > -9999]), np.max(image_array)

    print(f"Source min x: {x_min}")
    print(f"Source max x: {x_max}")
    print(f"Source min y: {y_min}")
    print(f"Source max y: {y_max}")
    print(f"Source min z: {z_min}")
    print(f"Source max z: {z_max}")
    print(f"Resolution: {pixelSizeX} x {pixelSizeY}")

    # Identify NoData value
    nodata_value = image.nodata
    print(f"NoData value: {nodata_value}")

    # Shape after expansion to multiples of N
    expanded_x = (X + N - 1) // N * N
    expanded_y = (Y + N - 1) // N * N

    print(f"Original size: {X} x {Y}")
    print(f"Expanded size: {expanded_x} x {expanded_y}")

    # Padding the matrix with zeros to dimensions divisible by N
    padded_shape = (expanded_x, expanded_y)
    frame = np.zeros(padded_shape, dtype=np.float32)

    print("Preparing data")

    # Channel [0] - by default compressed data always has only channel [0]
    frame[:X, :Y] = image_array[0, :, :]

    # Definition of N x N blocks
    stride = N
    blocks = []
    block_idx = 0
    for i in range(0, frame.shape[0], stride):
        for j in range(0, frame.shape[1], stride):
            blocks.append(
                (block_idx, frame[i : i + stride, j : j + stride].copy(), nodata_value)
            )
            block_idx += 1

    print(f"Compressing data ({len(blocks)} blocks)")

    # Initialize results in blocks
    results_buffer = [None] * len(blocks)
    # Track maximum error
    max_error_global = 0.0

    # Set the number of processes for threads
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Initialize process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance
        for result in tqdm(
            pool.imap_unordered(process_block, blocks), total=len(blocks)
        ):
            idx, compressed_data, max_error = result
            results_buffer[idx] = compressed_data  # Store only the compressed data

            # Update max error
            if max_error > max_error_global:
                max_error_global = max_error
    if max_error_global >= accuracy:
        print(f"Compression failed. Maximum error higher than accuracy: {max_error_global:.6f} > {accuracy:.6f}")
        valid='F'
    else:
        print(f"Compression completed. Maximum error: {max_error_global:.6f}")
        valid='T'

    # Prepare data for saving
    crs_str = rasterCrs.to_string() if rasterCrs else sourceCrs_force_declare
    max_error_global = round(max_error_global, decimal+2)
    dcv_compress = [
        N,  # Block size
        accuracy,  # Compression accuracy
        image.height,  # Original image size
        image.width,  # Original image size
        expanded_x,  # Size of transformed matrix enlarged to size divisible by block N
        expanded_y,  # Size of transformed matrix enlarged to size divisible by block N
        type_dct,
        decimal,
        # Adding information about geometric transformation
        transform.a,  # terrain resolution of pixel in x axis
        transform.b,  # 0
        transform.c,  # x coordinate
        transform.d,  # 0
        transform.e,  # terrain resolution of pixel in y axis
        transform.f,  # y coordinate
        crs_str,  # CRS info
        max_error_global,  # Maximum error
    ]

    # Add compressed data
    dcv_compress.extend([data for data in results_buffer])

    # Save compressed data
    outfile = os.path.basename(file_path)
    outfilename = os.path.splitext(outfile)[0]

    # Update file parameters with CRS information
    crs_info = "none"
    if rasterCrs is not None:
        try:
            # Attempt to get EPSG code
            epsg_code = rasterCrs.to_epsg()
            if epsg_code is not None:
                crs_info = f"epsg{epsg_code}"
            else:
                # If no EPSG code, use short form of CRS representation
                crs_info = str(rasterCrs).replace(":", "_").replace(" ", "_")[:20]
        except:
            # In case of failure, use 'none'
            crs_info = "none"

    # Prepare file parameters for filename
    file_parameters = f"_N{N}_Acc{accuracy}_tdct{type_dct}_dec{decimal}_CRS{crs_info}_V{valid}"

    # Prepare file paths
    json_path = f"{result_dir}/{outfilename}{file_parameters}.json"

    # Helper function to convert numpy types to JSON-serializable types
    def numpy_to_python(obj: Any) -> Any:
        """
        Converts numpy data types to standard Python types that are JSON-serializable.

        Parameters:
        ----------
        obj : Any
            Object to convert, can be a number, numpy array or list

        Returns:
        -------
        Any
            Converted object compatible with JSON
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        else:
            return obj

    # Convert numpy types to JSON-serializable types
    json_friendly_data = numpy_to_python(dcv_compress)

    # Pack to MessagePack
    msgpack_data = msgpack.packb(json_friendly_data)

    # Save to temporary file
    temp_msgpack_file = f"{result_dir}/temp_{outfilename}_content.msgpack"
    with open(temp_msgpack_file, "wb") as f:
        f.write(msgpack_data)

    # Add to 7z archive
    archive_name = f"{result_dir}/{outfilename}{file_parameters}.7z"
    with py7zr.SevenZipFile(archive_name, "w") as archive:
        archive.write(temp_msgpack_file, arcname=f"msgpack_obj_{outfilename}")

    print(f"✅ Data saved to 7z file: {archive_name}")

    # Delete temporary file
    if config_data.get("delete_temp_files", True):
        if os.path.exists(temp_msgpack_file):
            os.remove(temp_msgpack_file)
            print(f"🗑️ Deleted temporary file: {temp_msgpack_file}")

    print("✅ Compression completed")
    return dcv_compress, image, transform, rasterCrs, padded_shape, max_error_global


def decompress_image(dcv_compress, image, transform, rasterCrs, padded_shape):
    """
    Function for decompressing an image.

    Parameters:
    ----------
    dcv_compress : list
        Compressed data
    image
        Image object
    transform
        Geometric transformation
    rasterCrs
        Raster coordinate system
    padded_shape : tuple
        Dimensions of the padded matrix

    Returns:
    -------
    np.ndarray
        Decompressed image matrix
    """
    print("Starting decompression")

    print("Decompressing data")

    # Reconstruction parameters
    N = int(dcv_compress[0])  # Block size
    accuracy = float(dcv_compress[1])  # Accuracy
    img_x = int(dcv_compress[2])  # Original image size
    img_y = int(dcv_compress[3])
    pix_x = int(
        dcv_compress[4]
    )  # Size of transformed matrix enlarged to size divisible by block N
    pix_y = int(dcv_compress[5])
    type_dct = int(dcv_compress[6])
    decimal = int(dcv_compress[7])
    transform_a = float(dcv_compress[8])
    transform_b = float(dcv_compress[9])
    transform_c = float(dcv_compress[10])
    transform_d = float(dcv_compress[11])
    transform_e = float(dcv_compress[12])
    transform_f = float(dcv_compress[13])
    rasterCrs = str(dcv_compress[14]) # Crs
    max_error_global = str(dcv_compress[15]) # decoded maximum error
    
    # Scaling factor for restoring floating-point numbers
    print(
        f"Decompression parameters: N={N}, accuracy={accuracy}, size={img_x}x{img_y}, scaling factor={scaling_factor}"
    )
    print(f"Maximum recorded compression error: {max_error_global}")
    print(f"CRS: {rasterCrs}")

    # List to store decompressed blocks
    idct_table = []

    print("Reconstructing original matrix")
    pbar = tqdm(
        total=len(dcv_compress) - 16
    )  # Counting from the first 15 elements with parameters
    i = 16

    while i < len(dcv_compress):
        array = np.zeros(N * N, dtype=np.float32)  # Empty matrix
        current_data = dcv_compress[i]
        # Check if current_data is a list containing masks and compressed array
        if (
            isinstance(current_data, list)
            and len(current_data) > 1
            and any(isinstance(x, list) for x in current_data) == True
        ):
            # Format [idx, (mask_type1, mask1), (mask_type2, mask2), ..., compressed_array]
            masks = current_data[1:-1]
            element = current_data[-1]

            # Creating a zero matrix to store information about special values
            reconstructed_zero_matrix = np.zeros((N, N), dtype=np.int32)

            # Setting values according to masks
            a = 1
            for mask_data in masks:
                mask_type = mask_data[0]  # First element is the mask type
                mask_rle = mask_data[1:]  # The rest is RLE data

                if mask_type == 1:  # 1 means NoData mask
                    # Decoding NoData mask from RLE format
                    nodata_mask = decode_mask_rle(mask_rle, (N, N))
                    reconstructed_zero_matrix[nodata_mask] = (
                        -9999
                    )  # Mark NoData values with code -9999
                elif mask_type == 2:  # 2 means zero mask
                    # Decoding zero mask from RLE format
                    zero_mask = decode_mask_rle(mask_rle, (N, N))
                    reconstructed_zero_matrix[zero_mask] = (
                        2  # Mark zero values with code 2 (previously 1)
                    )

            idct_reconstructed, _, _ = reconstructed_matrix_function(
                element, reconstructed_zero_matrix, array, scaling_factor
            )
            idct_table.append(idct_reconstructed)
            i += 1
        # Check if this is special code -1 for NoData block
        elif (
            isinstance(current_data, list)
            and len(current_data) == 1
            and current_data[0] == -1
        ):
            idct_reconstructed = np.full((N, N), image.nodata)
            idct_table.append(idct_reconstructed)
            i += 1
        # Check if this is special code 0 indicating an all-zero block
        elif (
            isinstance(current_data, list)
            and len(current_data) == 1
            and current_data[0] == 0
        ):
            idct_reconstructed = np.zeros((N, N))
            idct_table.append(idct_reconstructed)
            i += 1
        # Otherwise it's a standard compressed array
        else:
            element = current_data
            reconstructed_zero_matrix = np.zeros((N, N), dtype=np.int32)
            idct_reconstructed, _, _ = reconstructed_matrix_function(
                element, reconstructed_zero_matrix, array, scaling_factor
            )
            idct_table.append(idct_reconstructed)
            i += 1

        pbar.update(1)

    # Vectorize the list of idct_table arrays
    original_matrix = np.stack(idct_table)

    # Empty matrix with source parameters
    final_matrix = np.zeros((pix_x, pix_y), dtype=np.float32)
    blocks_per_row = int(pix_y / N)  # Calculate blocks per row

    print("Creating final matrix")
    pbar = tqdm(total=len(original_matrix))

    # Reconstruction of the source matrix - can be optimized by vectorization
    for idx, block in enumerate(original_matrix):
        # Calculate row and column in target matrix based on block index
        row = (idx // blocks_per_row) * N
        col = (idx % blocks_per_row) * N
        # Place block in appropriate position in target matrix
        final_matrix[row : row + N, col : col + N] = block
        pbar.update(1)

    # Cropping the final matrix to original size
    final_matrix = final_matrix[:img_x, :img_y]

    # Save final reconstruction
    outfile = os.path.basename(image.name)
    with rasterio.open(
        f"{result_dir}/{outfile}.tif",
        "w",
        driver="GTiff",
        height=final_matrix.shape[0],
        width=final_matrix.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=rasterCrs,
        transform=rasterio.transform.Affine(
            transform_a, transform_b, transform_c, transform_d, transform_e, transform_f
        ),
        nodata=-9999,
        compress=Compression.deflate,
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256
    ) as interpRaster:  # Setting NoData value for output file
        interpRaster.write(final_matrix, 1)

    print("Decompression completed")
    return final_matrix


def load_compressed_data(outfilename: str, file_parameters: str, result_dir: str) -> dict:
    """
    Loads compressed data from a 7z archive containing MessagePack.

    Parameters
    ----------
    outfilename : str
        Output filename without extension
    file_parameters : str
        File parameters used in compressed filename
    result_dir : str
        Path to directory with compressed files

    Returns
    -------
    dict
        Dictionary containing decompressed data or None if failed
    """
    try:
        print("📦 Reading compressed data from 7z file")
        archive_path = os.path.join(result_dir, f"{outfilename}{file_parameters}.7z")

        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            temp_dir = tempfile.mkdtemp()
            try:
                z.extractall(temp_dir)
                extracted_files = glob.glob(os.path.join(temp_dir, "*"))
                if extracted_files:
                    extracted_path = extracted_files[0]
                    with open(extracted_path, "rb") as f:
                        try:
                            raw = f.read()
                            data = msgpack.unpackb(raw, raw=False)
                            print("✅ Data loaded and decoded from MessagePack inside 7z")
                            return data
                        except Exception as decode_err:
                            print(f"❌ Failed to decode MessagePack: {decode_err}")
                else:
                    print(f"⚠️ No files found in extracted archive in {temp_dir}")
            finally:
                shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"❌ An error occurred while reading 7z file: {str(e)}")
        return None


def main(file_path=None, output_dir=None):
    """
    Main function performing image compression and decompression.
    If a file with a .7z extension is provided as an argument, decompression is performed,
    otherwise, image compression is performed.

    Parameters:
    ----------
    file_path : str, optional
        Path to the input file. If not provided, uses the default value.
    output_dir : str, optional
        Path to the output directory. If not provided, uses the default value.
    """
    # Load configuration
    with open(r"./compression/config.json", "r") as file:
        config_data = json.load(file)

    global result_dir

    # Get default directories from configuration
    default_result_dir = config_data.get("results_directory")
    source_dir = config_data.get("source_directory")

    # Set file parameters based on configuration
    global file_parameters
    file_parameters = f"_acc_{accuracy}_{N}_dec_{decimal}_dct_{type_dct}_"

    # file_parameters = f'_acc_{accuracy}_{N}_dec_{decimal}_dct_{type_dct}_'
    print(
        f"Compression parameters: {source_dir}, {result_dir}, {accuracy}, {N}, {decimal}, {type_dct}"
    )
    print(f"File parameters: {file_parameters}")

    # If an output directory is provided, use it instead of the default
    if output_dir is not None:
        result_dir = output_dir
    else:
        result_dir = default_result_dir

    print(f"Using output directory: {result_dir}")

    # Ensure the directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created directory: {result_dir}")

    # If no path is provided, use the default file
    if file_path is None:
        file_path = os.path.join(source_dir, "urban_buildings_very_small.tif")

    # If the provided file has a .7z extension, perform decompression
    if file_path.lower().endswith(".7z"):
        print(f"Decompressing from file: {file_path}")
        # Determine outfilename (without extension)
        outfilename = os.path.splitext(os.path.basename(file_path))[0]

        # Get parameters from the file name
        file_parts = outfilename.split("_")

        # Extract CRS information
        # crs_info = None
        # for part in file_parts:
        #     if part.startswith("CRS"):
        #         crs_info = part[3:]  # Remove 'CRS' from the beginning
        #         break

        # if crs_info and crs_info.startswith("epsg"):
        #     try:
        #         # Attempt to get EPSG code
        #         epsg_code = int(
        #             crs_info[4:]
        #         )  # Remove 'epsg' from the beginning and convert to int
        #         rasterCrs = rasterio.crs.CRS.from_epsg(epsg_code)
        #         print(f"Read CRS from file name: EPSG:{epsg_code}")
        #     except ValueError:
        #         print(f'Error: Cannot read EPSG code from "{crs_info}"')
        #         return
        # else:
        #     print("Warning: No CRS information found in file name")
        #     rasterCrs = None

        # Extract block size N and accuracy Acc information
        N_value = None
        acc_value = None

        for part in file_parts:
            if part.startswith("N") and part[1:].isdigit():
                N_value = int(part[1:])
            if part.startswith("Acc"):
                try:
                    acc_value = float(part[3:])
                except ValueError:
                    pass

        if N_value is not None:
            print(f"Read block size from file name: N={N_value}")
        else:
            print(f"Warning: Using default block size: N={N}")
            N_value = N

        if acc_value is not None:
            print(f"Read accuracy from file name: Acc={acc_value}")
        else:
            print(f"Warning: Using default accuracy: Acc={accuracy}")
            acc_value = accuracy

        # Read compressed data from .7z file
        # Note that we now use file_parameters from the configuration - we can also extract them from the file name
        dcv_compress = load_compressed_data(outfilename, "", result_dir)
        if dcv_compress is None:
            print("Error: Failed to read data from archive.")
            return

        # Get image sizes from dcv_compress
        try:
            img_x = dcv_compress[2]  # Original size X
            img_y = dcv_compress[3]  # Original size Y
            padded_x = dcv_compress[4]  # Expanded size X
            padded_y = dcv_compress[5]  # Expanded size Y
            padded_shape = (padded_x, padded_y)
            rasterCrs = dcv_compress[14]

            print(f"Image size: {img_x}x{img_y}")
            print(f"Expanded size: {padded_x}x{padded_y}")
        except Exception as e:
            print(f"Error: Cannot read image size information: {str(e)}")
            return

        # Geometric transformation of the image
        transform = rasterio.transform.Affine(
            dcv_compress[8],
            dcv_compress[9],
            dcv_compress[10],
            dcv_compress[11],
            dcv_compress[12],
            dcv_compress[13],
        )

        # Dummy object that has only the necessary attributes of a geotif
        class DummyImage:
            def __init__(
                self,
                height: int,
                width: int,
                transform: rasterio.transform.Affine,
                crs: rasterio.crs.CRS,
                filename: str = "file_name.tif",
            ):
                """
                Initializes a dummy image object with basic attributes.

                Parameters:
                ----------
                height : int
                    Image height in pixels
                width : int
                    Image width in pixels
                transform : rasterio.transform.Affine
                    Georeferencing transformation
                crs : rasterio.crs.CRS
                    Raster coordinate system
                filename : str, optional
                    File name.
                """
                self.height = height
                self.width = width
                self.transform = transform
                self.crs = crs
                self.name = filename
                self.nodata = -9999

            def close(self):
                pass

        image = DummyImage(img_x, img_y, transform, rasterCrs, outfilename)

        # Decompression
        final_matrix = decompress_image(
            dcv_compress, image, transform, rasterCrs, padded_shape
        )

        print("Decompression completed")
        return
    # Otherwise, perform the compression process
    else:
        print(f"Compressing image: {file_path}")
        dcv_compress, image, transform, rasterCrs, padded_shape, max_error_global = compress_image(
            file_path
        )

        print("Pocess completed")
