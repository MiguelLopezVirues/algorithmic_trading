import numpy as np
import polars as pl
import pandas as pd
from typping import Tuple

from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

def compute_normalized_mi_with_regression(dataset: pl.DataFrame, n_jobs: int = -1) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Computes the normalized mutual information matrix and joint entropy matrix for continuous variables using mutual_info_regression.

    Args:
        dataset (pl.DataFrame): Input dataset with continuous data.
        n_jobs (int): Number of parallel jobs (-1 uses all available CPUs).

    Returns:
        tuple: Two Polars DataFrames, one for normalized mutual information and one for joint entropy values.
    """
    # Convert to numpy for efficient processing
    dataset_numpy = dataset.to_numpy()
    num_cols = dataset.shape[1]
    column_names = dataset.columns

    # Initialize symmetric matrices with zeros
    normalized_mi_matrix = np.zeros((num_cols, num_cols))
    joint_entropy_matrix = np.zeros((num_cols, num_cols))

    # Function to calculate normalized mutual information and joint entropy for a pair of columns
    def compute_normalized_mi_and_joint_entropy(pair):
        i, j = pair
        col_i = dataset_numpy[:, i]
        col_j = dataset_numpy[:, j]

        # Calculate mutual information using mutual_info_regression for continuous data
        mi = mutual_info_regression(col_i.reshape(-1, 1), col_j, n_neighbors=20)[0]  # We reshape to 2D as expected by sklearn

        # Calculate joint entropy
        joint_hist, _, _ = np.histogram2d(col_i, col_j, bins=20, density=True)  # Joint histogram
        joint_probs = joint_hist / joint_hist.sum()  # Normalize to get probabilities

        # Ensure joint probabilities sum to 1
        assert np.isclose(joint_probs.sum(), 1), f"Joint probability sum is {joint_probs.sum()}"

        # Remove zero probabilities to avoid log(0)
        joint_probs = joint_probs[joint_probs > 0]
        joint_entropy = -np.sum(joint_probs * np.log(joint_probs))  # Joint entropy

        # Normalize mutual information by joint entropy
        normalized_mi = mi / joint_entropy if joint_entropy > 0 else 0

        return i, j, normalized_mi, joint_entropy, np.sqrt(1 - np.exp(-2 * mi))

    # Generate combinations of column indices
    column_pairs = combinations(range(num_cols), 2)

    # Compute mutual information and joint entropy in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_normalized_mi_and_joint_entropy)(pair) for pair in column_pairs
    )

    # Fill in the matrices
    for i, j, normalized_mi, joint_entropy, r in results:
        normalized_mi_matrix[i, j] = normalized_mi
        normalized_mi_matrix[j, i] = normalized_mi
        joint_entropy_matrix[i, j] = joint_entropy
        joint_entropy_matrix[j, i] = joint_entropy

    # Create Polars DataFrames from the symmetric matrices
    normalized_mi_df = pd.DataFrame(normalized_mi_matrix, columns=column_names, index=column_names)
    joint_entropy_df = pd.DataFrame(joint_entropy_matrix, columns=column_names, index=column_names)

    return normalized_mi_df, joint_entropy_df
