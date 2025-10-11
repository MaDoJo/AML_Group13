from loadData import TEST_DATA_POINTS, TRAIN_DATA_POINTS, N_CLASSES
from typing import Tuple
import numpy as np

def get_time_steps(data: np.ndarray) -> np.ndarray:
    """
    Generates a list of how many times each time step is used in the given data.

    Args:
        data (np.ndarray): list of data points. Each data point is a time series 
        with 12 channels of cepstrum coefficients.

    Returns:
        np.ndarray: a list with how often each of the TEST_DATA_POINTS time steps
        are used in the given data.
    """

    time_steps = np.zeros(TEST_DATA_POINTS)

    for data_point in data:
        # list of length max-time-steps lenght with 1's for non-padded time steps
        # and 0's for all-0-valued (padded) time steps
        mask = [1 if np.count_nonzero(time_step) > 0 else 0 for time_step in data_point]
        time_steps += mask

    return time_steps


def get_pattern_mean(data: np.ndarray) -> np.ndarray:
    """
    Calculates the mean data point (time series) of all data points in data. Because 
    not every time series has the same length, the average of each time step is only
    taken over the non-padded instances of that time step. The data points consist of
    12 individual time series (channels), so the resulting mean data point consists of
    the 12 averaged channels.

    Args:
        data (np.ndarray): list of data points. Each data point is a time series 
        with 12 channels of cepstrum coefficients.

    Returns:
        np.ndarray: the mean data point, consisting of the 12 averaged time series
    """

    # get the frequency of each time step (how often each time step was not padded)
    time_steps = get_time_steps(data)

    # to avoid division by 0
    time_steps = [1 if time_step == 0 else time_step for time_step in time_steps]

    # sum the values per channel
    summed_data = np.sum(data, axis=0)

    # calculate the mean per channel
    pattern_mean = (summed_data.transpose() / time_steps).transpose()
    return pattern_mean


def normalize(data: np.ndarray, pattern_mean: np.ndarray) -> None:
    """
    Normalizes the data by subtracting the pattern mean from each data point.

    Args:
        data (np.ndarray): list of data points. Each data point is a time series 
        with 12 channels of cepstrum coefficients.
        pattern_mean (np.ndarray): mean data point, consisting of the 12 averaged
        channels.
    """

    # should be 29, using the normal training and testing data
    n_time_steps = data.shape[1]

    for data_point in data:
        for time_step in range(n_time_steps):

            # only subtract the mean for non-padded time steps
            if np.count_nonzero(data_point[time_step]) > 0:
                data_point[time_step] -= pattern_mean[time_step]


def flatten_data(data: np.ndarray) -> np.ndarray:
    """
    Restructures the data points in the data array to vectors of length 
    [channels x time steps] instead of matrices with [time steps] rows and
    [channels] columns.

    Args:
        data (np.ndarray): list of data points. Each data point is a time series 
        with 12 channels of cepstrum coefficients.

    Returns:
        np.ndarray: same data array, but with flattened data points.
    """
    return np.array([data[idx].flatten() for idx in range(data.shape[0])])


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


def generate_class_matrix(n_data_points: int, n_classes: int) -> np.ndarray:
    """
    Generates the class matrix corresponding to the training data. The original
    training data consists 270 datapoints and 9 classes, where the first 30 data
    points are from class 1, the next 30 from class 2, etc. The class matrix is
    a list of one-hot encoded class-labels.

    Args:
        n_data_points (int): number of data points
        n_classes (int): number of classes

    Returns:
        np.ndarray: the class matrix, a list of one-hot encoded class-labels.
    """

    # should be 30 for the original training data
    points_per_class = n_data_points // n_classes

    return np.array(
        [[1 if data_point < (class_n + 1) * points_per_class and 
               data_point >= (class_n) * points_per_class 
            else 0 
            for class_n in range(n_classes)] 
            for data_point in range(n_data_points)]
        )

def generate_test_class_matrix(n_data_points: int, n_classes: int) -> np.ndarray:
    """
    Generates the class matrix corresponding to the test data. The distribution
    of classes in the test data set is found on 
    https://archive.ics.uci.edu/dataset/128/japanese+vowels.
    The class matrix is a list of one-hot encoded class-labels.

    Args:
        n_data_points (int): number of data points
        n_classes (int): number of classes

    Returns:
        np.ndarray: the class matrix, a list of one-hot encoded class-labels.
    """

    # data points per class (in order) in test data
    points_per_class = [31, 35, 88, 44, 29, 24, 40, 50, 29]

    return np.array(
        [[1 if data_point < sum(points_per_class[:idx + 1]) and 
               data_point >= sum(points_per_class[:idx]) 
            else 0 
            for idx in range(n_classes)] 
            for data_point in range(n_data_points)])


def compute_regression_classifier(feature_vectors: np.ndarray, class_matrix: np.ndarray) -> np.ndarray:
    """
    Computes a linear regression classifier (W), given a set of feature vectors (F) 
    and a class matrix (V) according to the function W = (F'F)^-1 (F'V).

    Args:
        feature_vectors (np.ndarray): a list of feature vectors (F).
        class_matrix (np.ndarray): a list of one-hot encoded class-label vectors (V).

    Returns:
        np.ndarray: a linear regression classifier (W).
    """
    return np.matmul(np.linalg.inv(np.matmul(feature_vectors.transpose(), feature_vectors)), np.matmul(feature_vectors.transpose(), class_matrix))


def compute_MSE(regression_classifier: np.ndarray, feature_vectors: np.ndarray, class_matrix: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) given a linear regression classifier,
    a set of feature vectors and a class matrix, based on the error between the
    class matrix and the product of the regression classifier and feature vectors.

    Args:
        regression_classifier (np.ndarray): linear regression classifier.
        feature_vectors (np.ndarray): a list of feature vectors.
        class_matrix (np.ndarray): a list of one-hot encoded class-label vectors.

    Returns:
        float: the MSE.
    """
    return sum([np.linalg.norm(class_matrix[idx] - np.matmul(regression_classifier.transpose(), feature_vector))**2 for idx, feature_vector in enumerate(feature_vectors)]) / len(feature_vectors)


def compute_mismatch(regression_classifier, feature_vectors, class_matrix):
    """
    Computes the fraction of mis-classified data points. Classification is done
    by taking the index of the highest value in the output vector per datapoint.
    Output vectors are generated by multiplying the regression classifier and the
    feature vectors. Misclassification is when this index is not the same as the
    index of the 1 in the one-hot encoded vector corresponding to this data point.

    Args:
        regression_classifier (np.ndarray): linear regression classifier.
        feature_vectors (np.ndarray): a list of feature vectors.
        class_matrix (np.ndarray): a list of one-hot encoded class-label vectors.

    Returns:
        float: the fraction of mis-classified data points.
    """
    return sum(
        [0 if np.argmax(class_matrix[idx]) == np.argmax(np.matmul(regression_classifier.transpose(), feature_vector)) 
                else 1 
                for idx, feature_vector in enumerate(feature_vectors)]
        ) / len(feature_vectors)
