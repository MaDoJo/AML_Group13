from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.types import Number


def compute_entropy(y: Tensor) -> Number:
    """
    Compute entropy of data.

    Args:
        y (Tensor): Data.

    Returns:
        (Number): Computed entropy.
    """
    N = y.numel()

    if N == 0:
        return 0.0

    probs = torch.bincount(y).float() / N
    non_zero_probs = probs[probs > 0]

    entropy = -(non_zero_probs * torch.log2(non_zero_probs)).sum().item()

    return entropy


def split_data(
    X: Tensor, y: Tensor, feature_idx: int, threshold: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split data on feature by threshold.

    Args:
        X (Tensor): Data containing the features.
        y (Tensor): Label data.
        feature_idx (int): Feature index to split on.
        threshold (float): Threshold to split on.

    Returns:
        (Tuple[Tensor, Tensor, Tensor, Tensor]):
            - X data with feature <= threshold
            - y data with feature <= threshold
            - X data with feature > threshold
            - y data with feature > threshold
    """
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]


def compute_split_gain(
    X: Tensor, y: Tensor, feature_idx: int, threshold: float
) -> Number:
    """
    Compute split gain of a split.

    Args:
        X (Tensor): Data containing the features.
        y (Tensor): Label data.
        feature_idx (int): Feature index to split on.
        threshold (float): Threshold to split on.

    Returns:
        (Number): Computed split gain.
    """
    parent_entropy = compute_entropy(y)
    N = len(y)

    _, y_left, _, y_right = split_data(X, y, feature_idx, threshold=threshold)

    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    info_gain = parent_entropy - (
        len(y_left) / N * left_entropy + len(y_right) / N * right_entropy
    )

    return info_gain


def find_best_feature_split(
    X: Tensor, y: Tensor, feature_idx: int
) -> Tuple[Number, Number]:
    """
    Find best split of data on a given feature.

    Args:
        X (Tensor): Data containing the features.
        y (Tensor): Label data.
        feature_idx (int): Feature index to evaluate.

    Returns:
        (Tuple[Number, Number]):
            - Best threshold
            - Best split gain
    """
    values, order = X[:, feature_idx].sort()
    y_sorted = y[order]

    best_threshold = None
    best_split_gain = -float("inf")
    N = len(y)

    for i in range(1, N):
        if y_sorted[i] != y_sorted[i - 1]:
            threshold = ((values[i] + values[i - 1]) / 2).item()

            split_gain = compute_split_gain(X, y, feature_idx, threshold=threshold)
            if split_gain > best_split_gain:
                best_split_gain = split_gain
                best_threshold = threshold

    return best_threshold, best_split_gain


def find_best_split(
    X: Tensor, y: Tensor, feature_indices: List[int]
) -> Tuple[int, Number, Number]:
    """
    Find best split of data on given features

    Args:
        X (Tensor): Data containing the features.
        y (Tensor): Label data.
        feature_indices (List[int]): Indices of features to evaluate.

    Returns:
        (Tuple[int, Number, Number]):
            - Best feature index
            - Best threshold
            - Best split gain
    """
    best_gain = -float("inf")
    best_threshold = None
    best_feature_idx = None

    for feature_idx in feature_indices:
        threshold, gain = find_best_feature_split(X, y, feature_idx=feature_idx)

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_feature_idx = feature_idx.item()

    return best_feature_idx, best_threshold, best_gain

def bootstrap_sample(X: Tensor, y: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Return a bootstrap sample of X (and y if provided).

    Args:
        X (Tensor): Data containing features.
        y (Optional[Tensor]): Label data.

    Returns:
        (Tuple[Tensor, Optional[Tensor]]):
            - Sample from X
            - Sample from y (if provided)
    """
    n_samples = X.shape[0]
    indices = torch.randint(0, n_samples, (n_samples,))
    X_sample = X[indices]

    y_sample = y[indices] if y is not None else None

    return X_sample, y_sample
