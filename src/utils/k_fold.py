import numpy as np
from typing import Tuple, List

def get_k_folds(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5
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
    unique_classes = np.unique(y)
    fold_indices = [[] for _ in range(k)]

    # Distribute samples of each class across folds
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        np.random.shuffle(class_indices)
        
        for i, idx in enumerate(class_indices):
            fold_indices[i % k].append(idx)

    # Shuffle indices within each fold
    for f in range(k):
        np.random.shuffle(fold_indices[f])

    folds = []
    for fold in range(k):
        val_indices = fold_indices[fold]
        train_indices = np.concatenate([fold_indices[i] for i in range(k) if i != fold])

        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]

        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))

    return folds
