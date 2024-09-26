import numpy as np
import pandas as pd
from itertools import product

def stack_rotated_matrices(matrices: pd.Series, max_rotation: int) -> np.ndarray:
    """
    Creates a vertically stacked array of flattened, rotated matrices.

    Parameters:
    ----------
    matrices : pd.Series
        A series of matrices to be rotated and stacked.
    
    max_rotation : int
        The maximum number of shifts (both positive and negative) applied to each matrix.

    Returns:
    -------
    np.ndarray
        A vertically stacked array of rotated matrices, where each matrix is flattened 
        into a 1D array. The resulting shape is (len(matrices) * (2 * max_rotation + 1), N),
        where N is the number of elements in the original matrix.
    """
    return np.vstack([
        np.roll(matrix, shift, axis=0).flatten()
        for matrix, shift in product(matrices, range(-max_rotation, max_rotation + 1))
    ])

def get_pairwise_min_dist_across_rotations(iris_matrices: pd.Series, mask_matrices: pd.Series, max_rotation: int) -> np.ndarray:
    """
    Computes the pairwise minimum Hamming distance across rotations between iris matrices,
    taking into account the mask matrices.

    Parameters:
    ----------
    iris_matrices : pd.Series
        A series of binary matrices representing iris patterns.
    
    mask_matrices : pd.Series
        A series of binary mask matrices indicating the valid positions 
        in the iris matrices (True for valid, False for invalid).
    
    max_rotation : int
        The maximum number of shifts (both positive and negative) applied to each matrix.

    Returns:
    -------
    np.ndarray
        A 1D array containing the minimum pairwise Hamming distances for each pair of 
        original matrices (considering all rotations). The length of the array is 
        len(iris_matrices) * (len(iris_matrices) - 1) / 2, corresponding to all unique 
        pairs without self-comparisons.
    """    
    # Create rotated matrices and masks
    rotated_matrices = stack_rotated_matrices(iris_matrices, max_rotation)
    rotated_masks = stack_rotated_matrices(mask_matrices, max_rotation)
    
    # Calculate pairwise Hamming distances considering only True values in the mask
    valid_positions = np.expand_dims(rotated_masks, axis=1) & np.expand_dims(rotated_masks, axis=0)
    differences = np.expand_dims(rotated_matrices, axis=1) != np.expand_dims(rotated_matrices, axis=0)
    hamming_distances = np.sum(differences & valid_positions, axis=-1) / np.sum(valid_positions, axis=-1)
    
    # Mask self-comparisons with np.inf
    matrix_indices = np.arange(len(iris_matrices)).repeat(2 * max_rotation + 1)
    hamming_distances[matrix_indices[:, None] == matrix_indices[None, :]] = np.inf
    
    # Reshape and find minimum distances
    reshaped_distances = hamming_distances.reshape(len(iris_matrices), 2 * max_rotation + 1, len(iris_matrices), 2 * max_rotation + 1)
    min_distances_per_matrix = np.min(reshaped_distances, axis=(1, 3))

    # Extract only the lower triangle (excluding the diagonal)
    return min_distances_per_matrix[np.tril_indices(len(iris_matrices), k=-1)]
