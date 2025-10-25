from typing import List, Tuple

import numpy as np


def get_k_folds(
    X: np.ndarray, y: np.ndarray, k: int = 5
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform stratified k-fold cross-validation on training data.
    Ensures each fold has roughly equal class proportions.

    Args:
        X (np.ndarray): Training features.
        y (np.ndarray): Training labels.
        k (int): Number of folds.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            Each tuple contains:
            (X_train, y_train, X_val, y_val)
    """
    # Handle one-hot labels
    if y.ndim > 1 and y.shape[1] > 1:
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