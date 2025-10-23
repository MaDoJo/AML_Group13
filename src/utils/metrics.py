from typing import Union

import numpy as np
import torch


def convert_to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert input to a PyTorch tensor.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input data.

    Returns:
        torch.Tensor: The data as a PyTorch tensor.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")


def compute_accuracy(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions (Union[np.ndarray, torch.Tensor]): Model predictions.
        labels (Union[np.ndarray, torch.Tensor]): True labels.

    Returns:
        (float): Accuracy as a decimal (0-1).
    """
    predictions = convert_to_tensor(predictions)
    labels = convert_to_tensor(labels)

    return (predictions == labels).float().mean().item()


def compute_per_class_metrics(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int
):
    """
    Compute per-class precision, recall, and F1 score.

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): True labels.
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary with per-class metrics.
    """
    predictions = convert_to_tensor(predictions)
    labels = convert_to_tensor(labels)

    metrics = {}

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

        metrics[class_idx] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": (labels == class_idx).sum().item(),
        }

    return metrics
