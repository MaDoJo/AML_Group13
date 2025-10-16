import torch
from torch import nn, Tensor
from random_forest.utils import bootstrap_sample
from random_forest import DecisionTree

class RandomForest(nn.Module):
    def __init__(self, n_trees: int, *args, **kwargs):
        """
        Initialise the Random Forest.

        Args:
            n_trees (int): Number of trees to train in the forest.
        """
        super().__init__(*args, **kwargs)

        self.n_trees = n_trees

        self.trees = nn.ModuleList(
            [
                DecisionTree() for _ in range(self.n_trees)
            ]
        )
    
    def train_forest(self, X: Tensor, y: Tensor) -> None:
        """
        Train the Random Forest.

        Args:
            X (Tensor): Data containing the features.
            y (Tensor): Label data.
        """
        # Use sqrt(d) for features to eval in each tree
        n_features_eval = torch.sqrt(X.shape[1]).item()

        for tree in self.trees:
            tree.n_features_eval = n_features_eval
            X_bag, y_bag = bootstrap_sample(X, y)
            tree.train_tree(X_bag, y_bag)

    
    def forward(self, X: Tensor) -> Tensor:
        """
        Predict on data.
        Tree predictions are aggregated through majority voting.

        Args:
            X (Tensor): Data.

        Returns:
            (Tensor): Predictions.
        """
        all_preds = torch.stack(
            [tree(X) for tree in self.trees]
        ).T  # Transpose to have (samples, preds)

        # Majority vote
        final_preds = torch.mode(all_preds, dim=1).values  

        return final_preds
