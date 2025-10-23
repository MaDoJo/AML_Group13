#####################################################################################################################################

from typing import Optional, Tuple, Union

import numpy as np
import torch

from src.random_forest.random_forest import RandomForest
from src.utils.config import N_CLASSES


def prepare_tensors_for_training(
    x_train: Union[torch.Tensor, np.ndarray],
    y_train: Union[torch.Tensor, np.ndarray],
    x_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
    y_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
    dtype_x: torch.dtype = torch.float32,
    dtype_y: torch.dtype = torch.long,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Converts NumPy arrays or Torch tensors into properly typed PyTorch tensors.
    Converts one-hot encoded labels to class indices.

    Args:
        x_train (Union[torch.Tensor, np.ndarray]): Training features.
        y_train (Union[torch.Tensor, np.ndarray]): Training labels (NumPy or Torch tensor, may be one-hot)
        x_val (Optional[Union[torch.Tensor, np.ndarray]]): Optional validation features.
        y_val (Optional[Union[torch.Tensor, np.ndarray]]): Optional validation labels.
        dtype_x (torch.dtype, default=torch.float32): Feature tensor dtype.
        dtype_y (torch.dtype, default=torch.long): Label tensor dtype.

    Returns:
        (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]):
            - X_train_tensor
            - y_train_tensor
            - X_val_tensor
            - y_val_tensor
    """

    def to_tensor(x, dtype):
        # Converts data to tensor with specified dtype
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=dtype)
        elif isinstance(x, torch.Tensor):
            return x.to(dtype)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    def handle_labels(y):
        # Converts one-hot encoded labels to class indices
        if isinstance(y, np.ndarray):
            if y.ndim > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
        elif isinstance(y, torch.Tensor):
            if y.ndim > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
        else:
            raise TypeError(f"Unsupported label type: {type(y)}")
        return y

    # Prepare train tensors
    y_train = handle_labels(y_train)
    X_train_tensor = to_tensor(x_train, dtype_x)
    y_train_tensor = to_tensor(y_train, dtype_y)

    # Prepare val tensors
    X_val_tensor = None
    y_val_tensor = None
    if x_val is not None and y_val is not None:
        y_val = handle_labels(y_val)
        X_val_tensor = to_tensor(x_val, dtype_x)
        y_val_tensor = to_tensor(y_val, dtype_y)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): True labels.

    Returns:
        float: Accuracy as a decimal (0-1).
    """
    return (predictions == labels).float().mean().item()


def compute_per_class_metrics(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int
):
    """
    Compute per-class and overall precision, recall, F1 score, and accuracy.

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): True labels.
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary with per-class metrics and overall metrics (including accuracy).
    """
    metrics = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for class_idx in range(num_classes):
        true_positives = (
            ((predictions == class_idx) & (labels == class_idx)).sum().item()
        )
        false_positives = (
            ((predictions == class_idx) & (labels != class_idx)).sum().item()
        )
        false_negatives = (
            ((predictions != class_idx) & (labels == class_idx)).sum().item()
        )

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = (labels == class_idx).sum().item()

        metrics[class_idx] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    total_support = sum(supports)
    total_precision = sum(precisions) / num_classes
    total_recall = sum(recalls) / num_classes
    total_f1 = sum(f1s) / num_classes

    accuracy = (predictions == labels).float().mean().item()

    metrics["overall"] = {
        "accuracy": accuracy,
        "precision": total_precision,
        "recall": total_recall,
        "f1": total_f1,
    }

    return metrics


def train_and_evaluate_random_forest(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_trees: int,
    max_depth: int,
):
    """
    Complete pipeline to train and evaluate Random Forest on speaker classification.

    Args:
        train_data_path (str): Path to training data file.
        test_data_path (str): Path to test data file.
        n_trees (int): Number of trees in the forest.
        variance_threshold (float): Percentage of variance to retain in PCA.
        use_cross_validation (bool): Whether to perform cross-validation.
        k_folds (int): Number of folds for cross-validation.
        visualize (bool): Whether to show visualization plots.

    Returns:
        tuple: (model, train_accuracy, test_accuracy, cv_results)
    """
    model = RandomForest(n_trees=n_trees, max_depth=max_depth)
    model.train_forest(X_train, y_train)

    # Eval on test
    with torch.no_grad():
        test_predictions = model(X_test)
        test_accuracy = compute_accuracy(test_predictions, y_test)

    # compute per-class metrics
    metrics = compute_per_class_metrics(test_predictions, y_test, N_CLASSES)

    return model, test_accuracy, metrics
