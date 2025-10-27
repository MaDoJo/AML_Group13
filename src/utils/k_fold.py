from src.utils.config import N_CLASSES

from typing import List, Tuple

import numpy as np


def get_k_folds(
    X: np.ndarray, y: np.ndarray, k: int = 5, augmentation: bool = False
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform stratified k-fold cross-validation on training data.
    Ensures each fold has roughly equal class proportions.

    Args:
        X (np.ndarray): Training features.
        y (np.ndarray): Training labels.
        k (int): Number of folds.
        augmentation (bool): Whether the data has been augmented.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            Each tuple contains:
            (X_train, y_train, X_val, y_val)
    """
    if k == len(X): # Leave-One-Out (no need to stratify)
        folds = []
        for i in range(len(X)):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            X_val = X[i:i+1]
            y_val = y[i:i+1]
            folds.append((X_train, y_train, X_val, y_val))
        return folds
    # Handle one-hot labels

    if y.ndim > 1 and y.shape[1] > 1:
        if augmentation:
            y_classes = np.argmax(y, axis=2)
            y_classes = y_classes[:, 0]  # Take the class of the original sample
        else:
            y_classes = np.argmax(y, axis=1)
    else:
        y_classes = y

    unique_classes = np.unique(y_classes)
    fold_indices = [[] for _ in range(k)]

    # Distribute samples of each class evenly across folds
    for class_label in unique_classes:
        class_indices = np.flatnonzero(y_classes == class_label)
        np.random.shuffle(class_indices)
        for i, idx in enumerate(class_indices):
            fold_indices[i % k].append(int(idx))  # ensure int

    # Shuffle indices within each fold
    for f in range(k):
        np.random.shuffle(fold_indices[f])
        fold_indices[f] = np.array(fold_indices[f], dtype=int)

    # Build folds
    folds = []
    for fold in range(k):
        val_idx = fold_indices[fold]
        train_idx = np.concatenate(
            [fold_indices[i] for i in range(k) if i != fold], dtype=int
        )
        folds.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))

    return folds


def reshape_fold(fold, n_samples: int, data_length: int) -> np.ndarray:
    return np.reshape(fold, (len(fold) * (n_samples + 1), data_length))


def reshape_folds(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_samples: int,
    feature_vector_length: int
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Reshape the folds to account for data augmentation.

    Args:
        folds (List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
            Original folds from k-fold cross-validation.
    """
    new_folds = []
    for fold in folds:
        train_folds, train_labels, val_folds, val_labels = fold
        train_folds = reshape_fold(train_folds, n_samples, feature_vector_length)
        train_labels = reshape_fold(train_labels, n_samples, N_CLASSES)
        val_folds = reshape_fold(val_folds, n_samples, feature_vector_length)
        val_labels = reshape_fold(val_labels, n_samples, N_CLASSES)
        new_folds.append((train_folds, train_labels, val_folds, val_labels))

    return new_folds