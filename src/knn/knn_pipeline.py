from typing import List

import numpy as np
from dtaidistance.dtw import distance_fast

from src.utils.load_data import N_CHANNELS, N_CLASSES
from src.utils.metrics import compute_accuracy
from src.utils.processing import remove_padding


def independent_dtw(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Calculates the independent Dynamic Time Warping distance (DTW_i). This is
    done by summing the DTW distances of all channels.

    Args:
        signal1 (np.ndarray): first signal used to compare DTW_i distance with.
        signal2 (np.ndarray): second signal used to compare DTW_i distance with.

    Returns:
        float: independent Dynamic Time Warping distance.
    """

    signal1 = remove_padding(signal1)
    signal2 = remove_padding(signal2)

    dtw_i = 0
    for channel in range(N_CHANNELS):
        dtw_i += distance_fast(signal1[channel], signal2[channel])

    return dtw_i


def get_distances(data_point: np.ndarray, training_data: np.ndarray) -> List[float]:
    """
    Collects the Dynamic Time Warping (DTW) distances from a data point to all
    other data points in the training data.

    Args:
        data_point (np.ndarray): the data point to measure the DTW distances of
        all training data points with.
        training_data (np.ndarray): the training data points.

    Returns:
        List[float]: a list of DTW distances from data_point to all data points
        in training_data
    """
    return [
        independent_dtw(data_point, training_data[idx])
        for idx in range(len(training_data))
    ]


def predict(k_nn: int, distances: List[float], labels: np.ndarray) -> int:
    """
    Predicts the class of a data point, given the Dynamic Time Warp (DTW)
    distances to all points in the training data, from that data point. The
    labels of the k nearest neighbors are collected, and the most frequently
    collected class-label is returned as the prediction. Ties are broken in
    favor of the class with the lowest index in the class vector.

    Args:
        k_nn (int): the number of nearest neighbors to determine the class of.
        distances (List[float]): the DTW distances between one data point and
        all data points in the training data.
        labels (np.ndarray): list of all labels (class-vectors), corresponding
        to the training data.

    Returns:
        int: the most frequently observed class (index in a one-hot encoded
        class-vector) from the (k_nn) nearest neighbors.
    """

    nearest_neighbors = np.zeros(N_CLASSES)
    for _ in range(k_nn):
        # select the (index of the) nearest neighbor
        nearest_neighbor = np.argmin(distances)

        # add the class of this data point to the collection of observed classes
        nearest_neighbors += labels[nearest_neighbor]

        # "remove" this data point from the training data
        distances[nearest_neighbor] = np.inf

    # return the most frequently observed class
    return np.argmax(nearest_neighbors)


def concat_folds(folds: np.ndarray, validation_fold: int, k_folds: int) -> np.ndarray:
    """
    Concatenates the folds used for training.

    Args:
        folds (np.ndarray): array of all (k_folds) folds.
        validation_fold (int): the fold used for validation.
        k_folds (int): the number of folds.

    Returns:
        np.ndarray: an array with all folds used for training.
    """
    return np.concatenate(
        [folds[idx] for idx in range(k_folds) if idx != validation_fold]
    )


def get_knn_predictions(
    k_nn: int, test_data: np.ndarray, train_data: np.ndarray, train_labels: np.ndarray
) -> np.ndarray:
    """
    Collects the predictions of the test data, given the training data and labels.

    Args:
        k_nn (int): the number of nearest neighbors to use.
        test_data (np.ndarray): the test data.
        train_data (np.ndarray): the training data.
        train_labels (np.ndarray): the labels of the training
        data.

    Returns:
        np.ndarray: list of predictions
    """

    predictions = []
    for data_point in test_data:
        distances = get_distances(data_point, train_data)
        prediction = predict(k_nn, distances, train_labels)
        predictions.append(prediction)

    return np.array(predictions)


def knn_hyperparameter_search(
    data: np.ndarray, labels: np.ndarray, k_to_search: range, k_folds: int
) -> dict:
    """
    Performs hyperparameter search on the number of nearest neighbors to use.
    Results are stored in a text file at "results_location". Validation is
    done with k-fold cross validation.

    Args:
        data (np.ndarray): list of data points. Each data point is a time
        series with 12 channels of cepstrum coefficients.
        labels (np.ndarray): list of one-hot encoded class-labels.
        k_to_search (range): the range of k-nearest-neighbors (hyperparameter)
        to search.
        k_folds (int): the number of folds used for the k-fold cross validation.

    Returns:
        dict: results of the hyperparameter search with the structure
        {k_nn: accuracy}.
    """

    idxs = np.arange(len(data))
    accuracies = {}
    for k_nn in k_to_search:
        # randomize the order of the data
        np.random.shuffle(idxs)
        shuffeled_data = data[idxs]
        shuffeled_labels = labels[idxs]

        # split the data into k-folds
        data_folds = np.array(np.split(shuffeled_data, k_folds))
        label_folds = np.array(np.split(shuffeled_labels, k_folds))

        # get the cross-validation accuracy
        for val_fold in range(k_folds):
            training_data = concat_folds(data_folds, val_fold, k_folds)
            training_labels = concat_folds(label_folds, val_fold, k_folds)

            validation_data = data_folds[val_fold]
            validation_labels = label_folds[val_fold]

            predictions = get_knn_predictions(
                k_nn, validation_data, training_data, training_labels
            )
            accuracy = compute_accuracy(
                predictions, np.argmax(validation_labels, axis=1)
            )
            accuracies.update({k_nn: accuracy})

    return accuracies
