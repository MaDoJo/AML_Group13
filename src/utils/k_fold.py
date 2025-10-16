import torch
import numpy as np
from typing import Tuple, List

def get_k_folds(X: torch.Tensor, y: torch.Tensor, k: int = 5) -> List[Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
    """
    Perform stratified k-fold cross-validation on the training data, due to the class imbalances
    and already ordered data. Ensures that each fold has an equal proportion of each class.    
    Args:
        X (torch.Tensor): training features.
        y (torch.Tensor): training labels.
        n_trees (int): num of trees in the forest.
        k (int): number of folds for cross-validation.
    
    Returns:
        (List[Tuple[torch.Tensor]]): For each fold a tuple containing:
            - X train data
            - y train data
            - X validation data
            - y validation data
    """
    # creates stratified folds
    unique_classes = torch.unique(y)
    fold_indices = [[] for _ in range(k)]
    
    # distribute samples evenly across folds for each class
    for class_label in unique_classes:
        class_indices = torch.where(y == class_label)[0].tolist()
        np.random.shuffle(class_indices)  # Shuffle to randomize which samples go to which fold
        
        # distribute class samples across folds
        for i, idx in enumerate(class_indices):
            fold_indices[i % k].append(idx)
    
    # shuffles indices within each fold
    for fold_idx in fold_indices:
        np.random.shuffle(fold_idx)
    
    folds = []
    # perform cross-validation
    for fold in range(k):
        val_indices = fold_indices[fold]
        
        train_indices = []
        for f in range(k):
            if f != fold:
                train_indices.extend(fold_indices[f])
        
        # split data
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        folds.append(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold
        )

    return folds