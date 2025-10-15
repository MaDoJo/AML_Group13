from utils.processing import flatten_data, get_pattern_mean
from utils.visualizeData import visualize_data_point

import numpy as np
from typing import Tuple


def normalize(data: np.ndarray) -> None:
    """
    Normalizes the data by subtracting the pattern mean from each data point.

    Args:
        data (np.ndarray): list of data points. Each data point is a time series 
        with 12 channels of cepstrum coefficients.
    """

    # should be 29, using the normal training and testing data
    n_time_steps = data.shape[1]

    pattern_mean = get_pattern_mean(data)

    normalized_data = np.zeros(data.shape)
    for idx, data_point in enumerate(data):
        for time_step in range(n_time_steps):

            # only subtract the mean for non-padded time steps
            if np.count_nonzero(data_point[time_step]) > 0:
                normalized_data[idx, time_step] = data_point[time_step] - pattern_mean[time_step]

    return normalized_data


def SVD(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the singular value decomposition of the normalized data matrix (X)
    by first obtaining the covariance matrix (C = 1/N X'X).

    Args:
        data (np.ndarray): list of flattend data points. Each data point is a 
        time series with 12 concatenated channels of cepstrum coefficients.

    Returns:
        Tuple[np.ndarray, np.ndarray]: a list of principal component vectors (U)
        and a list of principal component variances (Σ).
    """

    data = normalize(data)
    data = flatten_data(data)

    Cov_matrix = (1 / data.shape[0]) * np.matmul(data.transpose(), data)

    # only return U and Σ (and not U', since it is redundant)
    return np.linalg.svd(Cov_matrix)[:2]


def determine_cutoff(variance_vector: np.ndarray, wanted_variance: float) -> int:
    """
    Determines the number of Principal Components (PCs) needed, given a desired 
    percentage of variance wanted to preserve.

    Args:
        variance_vector (np.ndarray): PC variance vector (variances are in 
        descending order).
        wanted_variance (float): the percentage of desired variance to be 
        preserved.

    Returns:
        int: number of PCs to be kept to preserve wanted_variance percent of 
        variance.
    """

    if wanted_variance < 0 or wanted_variance > 100:
        print(f"expected wanted_variance to be a percentage between 0-100, \
              instead got {wanted_variance}.")
        return

    cutoff = 1
    while (sum(variance_vector[:cutoff]) / sum(variance_vector)) * 100 < wanted_variance:
        cutoff += 1

    return cutoff


def reduce_PCs(feature_variances, principal_components, wanted_variance):
    cutoff = determine_cutoff(feature_variances, wanted_variance)
    return principal_components[:cutoff]

def get_feature_vectors(data, PCs_reduced):
    data = normalize(data)
    data = flatten_data(data)
    return np.matmul(data, PCs_reduced.transpose())
