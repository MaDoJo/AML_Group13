import torch
from torch import nn, Tensor
from random_forest import DecisionTree

class RandomForest(nn.Module):
    def __init__(self, n_trees: int, n_features_eval: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_trees = n_trees
        self.n_features_eval = n_features_eval

        self.trees = nn.ModuleList(
            [
                DecisionTree(n_features_eval=self.n_features_eval)
                for _ in range(self.n_trees)
            ]
        )

    
    def train_forest(self, X: Tensor, y: Tensor) -> None:
        for tree in self.trees:
            tree.train_tree(X, y)
    
    def forward(self, X: Tensor):
        all_preds = torch.stack(
            [tree(X) for tree in self.trees]
        )

        return all_preds