from typing import Tuple

import numpy as np

from src.utils.processing import flatten_data


def normalize(data: np.ndarray, pattern_mean: np.ndarray) -> np.ndarray:
    """
    Normalizes the data by subtracting the pattern mean from each data point.

    Args:
        data (np.ndarray): array of data points. Each data point is a time
        series with 12 channels of cepstrum coefficients.
        pattern_mean (np.ndarray): the pattern mean to substract from every
        data point.

    Returns:
        np.ndarray: the array of normalized data points
    """

    # should be 29, using the normal training and testing data
    n_time_steps = data.shape[1]

    normalized_data = np.zeros(data.shape)
    for idx, data_point in enumerate(data):
        for time_step in range(n_time_steps):

            # only subtract the mean for non-padded time steps
            if np.count_nonzero(data_point[time_step]) > 0:
                normalized_data[idx, time_step] = (
                    data_point[time_step] - pattern_mean[time_step]
                )

    return normalized_data


def SVD(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the singular value decomposition of the normalized data matrix (X)
    by first obtaining the covariance matrix (C = 1/N X'X).

    Args:
        data (np.ndarray): array of normalized data points. Each data point is
        a time series with 12 channels of cepstrum coefficients.

    Returns:
        Tuple[np.ndarray, np.ndarray]: an array of principal component vectors
        (U) and an array of principal component variances (Σ).
    """

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
        print(
            f"expected wanted_variance to be a percentage between 0-100, \
              instead got {wanted_variance}."
        )
        return

    cutoff = 1
    while (
        sum(variance_vector[:cutoff]) / sum(variance_vector)
    ) * 100 < wanted_variance:
        cutoff += 1

    return cutoff


def reduce_PCs(
    feature_variances: np.ndarray,
    principal_components: np.ndarray,
    wanted_variance: float,
) -> np.ndarray:
    """
    Reduces the array of principal components (PCs) to the number of principal
    components that perserve the wanted variance.

    Args:
        feature_variances (np.ndarray): the array with feature variances (in
        descending order) corresponding to the array of PCs.
        principal_components (np.ndarray): the array of PCs.
        wanted_variance (float): the percentage of variance desired to be kept
        in the PCs.

    Returns:
        np.ndarray: array of the first n PCs that keep the given wanted
        variance.
    """

    cutoff = determine_cutoff(feature_variances, wanted_variance)
    return principal_components[:cutoff]


def get_feature_vectors(data: np.ndarray, PCs_reduced: np.ndarray) -> np.ndarray:
    """
    Obtains the feature vectors by projecting the normalized data onto the
    selected Principal Components (PCs).

    Args:
        data (np.ndarray): array of normalized data points. Each data point is
        a time series with 12 channels of cepstrum coefficients.
        PCs_reduced (np.ndarray): array of the selected PCs.

    Returns:
        np.ndarray: array of the resulting feature fectors, obtained from the
        projection of the normalized data onto the PCs.
    """

    data = flatten_data(data)
    return np.matmul(data, PCs_reduced.transpose())
