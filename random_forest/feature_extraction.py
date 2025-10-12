import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Optional

from random_forest.load_data import load_data, TRAIN_DATA_POINTS, TEST_DATA_POINTS, MAX_LENGTH, N_CLASSES

TRAIN_DATA_POINTS = 270
TEST_DATA_POINTS = 370
N_CLASSES = 9
MAX_LENGTH = 29


def get_time_steps(data):
    """
    Count number of non-padded time steps at each position
    """
    time_steps = np.zeros(MAX_LENGTH)

    for data_point in data:
        mask = [1 if np.count_nonzero(time_step) > 0 else 0 for time_step in data_point]
        time_steps += mask

    return time_steps


def get_pattern_mean(data):
    """
    Calculate mean pattern across all data points, accounting for padding
    """
    summed_data = np.sum(data, axis=0)
    time_steps = get_time_steps(data)

    time_steps = [1 if time_step == 0 else time_step for time_step in time_steps]
    pattern_mean = (summed_data.transpose() / time_steps).transpose()
    return pattern_mean


def normalize(data, pattern_mean):
    """
    Normalize data by subtracting the pattern mean
    Modifies data in-place
    """
    n_time_steps = data.shape[1]

    for data_point in data:
        for time_step in range(n_time_steps):
            if np.count_nonzero(data_point[time_step]) > 0:
                data_point[time_step] -= pattern_mean[time_step]


def flatten_data(data):
    """
    Flatten each data point from (max_length, 12) to (max_length * 12,).
    """
    return np.array([data[idx].flatten() for idx in range(data.shape[0])])


def SVD(data):
    """
    Uses SVD on the data covariance matrix.
    """
    C = (1 / data.shape[0]) * np.matmul(data.transpose(), data)
    return np.linalg.svd(C)[:2]


def determine_cutoff(variance_vector, wanted_variance):
    """
    Determine number of PC needed to capture needed variance.
    """
    cutoff = 1
    while (sum(variance_vector[:cutoff]) / sum(variance_vector)) * 100 < wanted_variance:
        cutoff += 1

    return cutoff


def generate_class_matrix(n_data_points, n_classes):
    """
    Generate one-hot encoded class matrix for training data.
    Assumes first 30 points are class 0, next 30 are class 1, etc.
    """
    return np.array([
        [1 if data_point < (class_n + 1) * 30 and data_point >= (class_n) * 30 else 0 
         for class_n in range(n_classes)] 
        for data_point in range(n_data_points)
    ])


def generate_test_class_matrix(n_data_points, n_classes):
    """
    Generate one-hot encoded class matrix for test data.
    Test data has unequal class sizes: [31, 35, 88, 44, 29, 24, 40, 50, 29].
    """
    points_per_class = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    return np.array([
        [1 if data_point < sum(points_per_class[:idx + 1]) and data_point >= sum(points_per_class[:idx]) else 0 
         for idx in range(n_classes)] 
        for data_point in range(n_data_points)
    ])


def extract_features_for_random_forest(
    train_data_path: str,
    test_data_path: str,
    variance_threshold: float = 95.0
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Complete feature extraction pipeline for Random Forest following the
    linear regression approach: normalize, flatten, and apply PCA.
    """
    import os
    
    train_data_path = os.path.abspath(train_data_path)
    test_data_path = os.path.abspath(test_data_path)
    
    # load training data
    train_data = load_data(train_data_path, TRAIN_DATA_POINTS, padding_value=0.0)
    if train_data is None:
        raise ValueError("Failed to load training data")
    # load test data
    test_data = load_data(test_data_path, TEST_DATA_POINTS, padding_value=0.0)
    if test_data is None:
        raise ValueError("Failed to load test data")
    
    # normalize data (subtract pattern mean)
    pattern_mean = get_pattern_mean(train_data)
    normalize(train_data, pattern_mean)
    normalize(test_data, pattern_mean)
    
    # flatten
    train_data_flat = flatten_data(train_data)
    test_data_flat = flatten_data(test_data)
    # print(f"training shape after flattening: {train_data_flat.shape}")
    # print(f"test shape after flattening: {test_data_flat.shape}")
    
    U, S = SVD(train_data_flat)
    
    # cutoff
    n_components = determine_cutoff(S, variance_threshold)
    
    # project data onto PCs
    U_reduced = U[:, :n_components]
    X_train = np.matmul(train_data_flat, U_reduced)
    X_test = np.matmul(test_data_flat, U_reduced)
    
    # print(f"training features shape after PCA: {X_train.shape}")
    # print(f"test features shape after PCA: {X_test.shape}")
    
    # labels
    train_class_matrix = generate_class_matrix(TRAIN_DATA_POINTS, N_CLASSES)
    test_class_matrix = generate_test_class_matrix(TEST_DATA_POINTS, N_CLASSES)
    
    y_train = np.argmax(train_class_matrix, axis=1)
    y_test = np.argmax(test_class_matrix, axis=1)
    
    # print(f"Training labels shape: {y_train.shape}")
    # print(f"Test labels shape: {y_test.shape}")
    # print(f"Number of classes: {len(np.unique(y_train))}")
    
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = extract_features_for_random_forest(
        train_data_path="ae.train",
        test_data_path="ae.test",
        variance_threshold=95.0
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Classes: {len(torch.unique(y_train))}")
    print("\nClass distribution (training):")
    for class_idx in range(N_CLASSES):
        count = (y_train == class_idx).sum().item()
        print(f"  Class {class_idx}: {count} samples")
    print("\nClass distribution (test):")
    for class_idx in range(N_CLASSES):
        count = (y_test == class_idx).sum().item()
        print(f"  Class {class_idx}: {count} samples")