from typing import Optional

import torch
from torch import Tensor, nn
from torch.types import Number

import random_forest.utils as utl



class DecisionTree(nn.Module):
    def __init__(self, n_features_eval: Optional[int] = None, max_depth: Optional[int] = None):
        """
        Initialise a decision tree.

        Args:
            n_features_eval (Optional[int]): Number of features to evaluate on each split.
            max_depth (Optional[int]): Maximum depth of the tree.
        """
        super().__init__()
        self.n_features_eval = n_features_eval
        self.tree = None
    
    class TreeNode:
        def __init__(
            self,
            feature_idx: Optional[int] = None,
            threshold: Optional[float] = None,
            left: Optional["TreeNode"] = None,
            right: Optional["TreeNode"] = None,
            value: Optional[Number] = None,
        ):
            """
            Initialise a TreeNode.

            Args:
                feature_idx (Optional[int]): Index of the feature split in this node.
                threshold (Optional[float]): Split threshold of this node.
                left (Optional[TreeNode]): Left child node.
                right (Optional[TreeNode]): Right child node.
                value (Optional[Number]): Value of the node. Specified for leaves.
            """
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X: Tensor, y: Tensor, depth: int = 0) -> TreeNode:
        """
        Fit a decision tree for given data.

        Args:
            X (Tensor): Data containing the features.
            y (Tensor): Label data.
            depth (int, default=0): Depth of the tree at the current node.

        Returns
            (TreeNode): Tree node fit to the data.
        """
        if len(y.unique()) == 1:  # Leaf if all labels the same
            return self.TreeNode(value=y[0].item())
    
        if self.max_depth is not None and depth >= self.max_depth:  # If max_depth reached
            return self.TreeNode(value=y[0].item())

        num_features_data = X.shape[1]
        shuffled_feature_indices = torch.randperm(num_features_data)

        n_features_eval = self.n_features_eval or num_features_data
        feature_indices_eval = shuffled_feature_indices[:n_features_eval]

        feature_idx, threshold, _ = utl.find_best_split(
            X, y, feature_indices=feature_indices_eval
        )

        X_left, y_left, X_right, y_right = utl.split_data(
            X, y, feature_idx=feature_idx, threshold=threshold
        )

        # recurse
        left_child = self.fit(X_left, y_left, depth + 1)
        right_child = self.fit(X_right, y_right, depth + 1)

        return self.TreeNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def train_tree(self, X: Tensor, y: Tensor) -> None:
        """
        Train a decision tree.

        Args:
            X (Tensor): Data containing the features.
            y (Tensor): Label data.
        """
        self.tree = self.fit(X, y)

    def predict(self, x: Tensor, node: Optional[TreeNode] = None) -> Number:
        """
        Predict on a single data point.

        Args:
            x (Tensor): Data point.
            node (Optional[TreeNode]): Tree node to use for prediction.
                If `None`, start at root.

        Returns:
            (Number): Prediction.
        """
        if node is None:
            node = self.tree

        if node.value is not None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self.predict(x, node.left)
        else:
            return self.predict(x, node.right)

    def forward(self, X: Tensor) -> Tensor:
        """
        Predict on data.

        Args:
            X (Tensor): Data.

        Returns:
            (Tensor): Predictions.
        """
        return torch.tensor([self.predict(x) for x in X])
