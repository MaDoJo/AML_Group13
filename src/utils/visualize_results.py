import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
from typing import Dict, Tuple, List


def plot_metrics(results: Dict[str, Dict[str, float]]) -> None:
    """
    Create a grouped bar plot for multiple metrics per method.
    Args:
        results (Dict[str, Dict[str, float]]): method names, metrics and values.
    """
    methods = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    all_methods, all_metrics, all_values = [], [], []
    for metric in metrics:
        for m in methods:
            all_methods.append(m)
            all_metrics.append(metric)
            all_values.append(results[m][metric])

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=all_methods, y=all_values, hue=all_metrics, palette="tab10", edgecolor="none")

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f'{height:.2f}', 
            (patch.get_x() + patch.get_width() / 2, height), 
            ha='center', 
            va='bottom',
            fontsize=9,
            rotation=0
        )

    # Automatically adjust y-axis.
    min_val = min(all_values)
    max_val = max(all_values)
    diff = max_val - min_val
    if diff < 1e-6:
        diff = 1e-6  # avoid zero range
    padding = diff * 0.4
    plt.ylim(min_val - padding, max_val + padding)

    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Metric Value")
    plt.title("Performance Comparison")
    plt.legend(title="Metric")
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_first_2_PCs(PCs: np.ndarray, labels: np.ndarray) -> None:
    """
    Plots the first two principal components and colors points by provided labels.

    Args:
        PCs (np.ndarray): array of principal components.
        labels (np.ndarray, optional): array of labels for each data point.
    """

    # Extract first two PCs for visualization
    pc1 = PCs[:, 0]
    pc2 = PCs[:, 1]

    # Plot the first two PCs
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pc1, pc2, c=labels, cmap="tab10", alpha=0.8, edgecolors="k")
    plt.legend(*scatter.legend_elements(), title="Labels", loc="best")
    plt.title("Visualization of First Two Principal Components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def plot_cm(
    title: str, true_labels: np.ndarray, predictions: np.ndarray, save_path: str
) -> None:
    """
    Plots and saves the confusion matrix.
    Args:
        title (str): Title for the plot.
        true_labels (np.ndarray): True class labels.
        predictions (np.ndarray): Predicted class labels.
        save_path (str): File path to save the confusion matrix plot.
    """

    cm = confusion_matrix(true_labels, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()

    plt.close()


def plot_cv_heatmap(
    results: Dict[Tuple[int, int], float],
    n_trees_options: List[int],
    max_depth_options: List[int],
    save_path: str = "hyperparameter_rf.png"
) -> None:
    """
    Plots and saves a heatmap of mean CV accuracies for random forest hyperparameter search.
    
    Argds:
    - results: dict with keys as (n_trees, max_depth) and values as mean accuracies
    - n_trees_options: list of tree counts used in CV
    - max_depth_options: list of depths used in CV
    - save_path: file path where the heatmap will be saved
    """
    # Create accuracy matrix
    acc_matrix = np.zeros((len(n_trees_options), len(max_depth_options)))

    for i, n_trees in enumerate(n_trees_options):
        for j, max_depth in enumerate(max_depth_options):
            acc_matrix[i, j] = results[(n_trees, max_depth)]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        acc_matrix,
        annot=True, fmt=".3f", cmap="viridis",
        xticklabels=max_depth_options,
        yticklabels=n_trees_options
    )
    plt.title("Random Forest CV Mean Accuracy")
    plt.xlabel("Max Depth")
    plt.ylabel("Number of Trees")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {save_path}")

def show_lambda_search_results(results: Dict[float, float], file_name: str = "results/hyperparameter_ridge.png") -> None:
    """
    Visualizes the results of the ridge regression lambda hyperparameter search.
    Plots the mean validation accuracy for each tested lambda value.

    Args:
        results (Dict[float, float]): Dictionary mapping lambda values to mean accuracies.
        file_name (str): File path to save the resulting plot.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o')

    plt.title("Ridge Regression Hyperparameter Search", fontsize=14)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.xlabel("Lambda", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()