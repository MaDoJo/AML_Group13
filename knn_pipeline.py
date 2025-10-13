import numpy as np
from typing import List
from loadData import load_data, TRAIN_DATA_POINTS, TEST_DATA_POINTS, N_CLASSES
from processing import generate_class_matrix
from DynamicTimeWarping import DTW


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
    return [DTW(data_point, training_data[idx]) for idx in range(len(training_data))]


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
    return np.concatenate([folds[idx] for idx in range(k_folds) if idx != validation_fold])


def get_knn_misclasses(
        k_nn: int, 
        test_data: np.ndarray, 
        test_labels: np.ndarray, 
        train_data: np.ndarray, 
        train_labels: np.ndarray) -> int:
    """
    Counts the number of misclassifications of a given validation fold.

    Args:
        k_nn (int): the number of nearest neighbors to use.
        test_data (np.ndarray): the test data.
        test_labels (np.ndarray): the labels of the test data.
        train_data (np.ndarray): the training data.
        train_labels (np.ndarray): the labels of the training
        data.
    """

    misclassifications = 0
    for data_point, class_vect in zip(test_data, test_labels):
        distances = get_distances(data_point, train_data)
        prediction = predict(k_nn, distances, train_labels)

        # if the predicted class is not the same as the true class
        if prediction != np.argmax(class_vect):
            misclassifications += 1

    return misclassifications


def get_knn_accuracy(
        k_nn: int, 
        k_folds: int, 
        data_folds: np.ndarray, 
        label_folds: np.ndarray, 
        printing: bool) -> float:
    """
    Calculates the accuracy, using k-fold cross validation for a given number
    of nearest neighbors.

    Args:
        k_nn (int): the number of nearest neighbors.
        k_folds (int): the number of folds in the k-fold cross validation.
        data_folds (np.ndarray): the k folds containing the data.
        label_folds (np.ndarray): the k folds containing the labels.
        printing (bool): prints the accuracy of each fold, if True.

    Returns:
        float: the accuracy for the given number of nearest neigbors.
    """
    
    misclasses = 0
    for val_fold in range(k_folds):
        training_data = concat_folds(data_folds, val_fold, k_folds)
        training_labels = concat_folds(label_folds, val_fold, k_folds)

        validation_data = data_folds[val_fold]
        validation_labels = label_folds[val_fold]

        fold_misclasses = get_knn_misclasses(k_nn, validation_data, validation_labels, training_data, training_labels)
        misclasses += fold_misclasses

        if printing:
            print(f"accuracy on fold {val_fold + 1}:\t {100 - (fold_misclasses / data_folds.shape[1]) * 100}%")

    n_data_points = data_folds.shape[0] * data_folds.shape[1]
    return 100 - (misclasses / n_data_points) * 100



def save_results(accuracies: dict, path: str) -> None:
    """
    Writes results to a file.

    Args:
        accuracies (dict): dictionary with the structure {k_nn: accuracy}.
        path (str): path where the results will be stored.
    """

    with open(path, "w") as f:
        for knn, acc in accuracies.items():
            f.write(f"{knn} nearest neighbors generated {acc}% accuracy\n")

        f.write(str(accuracies))


def knn_hyperparameter_search(
        data: np.ndarray, 
        labels: np.ndarray, 
        k_to_search: range, 
        k_folds: int, 
        results_location: str) -> None:
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
        results_location (str): path where the results of the hyperparameter
        results will be stored.
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
        accuracy = get_knn_accuracy(k_nn, k_folds, data_folds, label_folds, printing=False)
        accuracies.update({k_nn: accuracy})
        print(f"validation accuracy for k = {k_nn}:\t {accuracy}%")

    save_results(accuracies, results_location)


def knn_accuracy(k_nn: int, 
        test_data: np.ndarray, 
        test_labels: np.ndarray, 
        train_data: np.ndarray, 
        train_labels: np.ndarray) -> int:
    """
    Calculates the accuracy of the k-nearest neighbors procedure for a test set
    and a training set.

    Args:
        k_nn (int): the number of nearest neighbors to use.
        test_data (np.ndarray): the test data.
        test_labels (np.ndarray): the labels of the test data.
        train_data (np.ndarray): the training data.
        train_labels (np.ndarray): the labels of the training
        data.
    """

    misclasses = get_knn_misclasses(k_nn, test_data, test_labels, train_data, train_labels)
    return 100 - (misclasses / len(test_data)) * 100

if __name__ == "__main__":
    train_data = load_data("data/ae.train", num_data_points=TRAIN_DATA_POINTS)
    class_matrix = generate_class_matrix(TRAIN_DATA_POINTS, N_CLASSES)
    k_to_search = range(1, 11)
    path = "search_results5.txt"

    # leave-one-out cross validation
    knn_hyperparameter_search(
        data=train_data, 
        labels=class_matrix, 
        k_to_search=k_to_search, 
        k_folds=TRAIN_DATA_POINTS, 
        results_location=path)
