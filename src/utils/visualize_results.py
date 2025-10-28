import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
from typing import Dict, Tuple, List


def plot_metrics(results: Dict[str, Dict[str, float]], file_name: str) -> None:
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
    ax = sns.barplot(x=all_methods, y=all_values, hue=all_metrics, hue_order=metrics, palette="Set1", edgecolor="none")

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f'{height:.3f}', 
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

    plt.xticks(rotation=0, ha="center")
    plt.ylabel("Metric Value")
    plt.title("Performance Comparison")
    plt.legend(title="Metric")
    sns.despine()
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()


    print(f"Metrics comparison saved to {file_name}")


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

    print(f"Confusion matrix saved to {save_path}")

def plot_per_class_metric(
    per_class_metrics_list: list, 
    model_names: list, 
    metric: str = "f1",
    num_classes: int = 9, 
    file_name: str = 'results/per_class_model_comparison.png'
) -> None:
    """
    Plot per-class scores for a specified metric across multiple models.

    Args:
        per_class_metrics_list (list[dict]): One per-class metrics dict per model.
        model_names (list[str]): List of model names.
        metric (str): Metric to plot ('precision', 'recall', or 'f1').
        num_classes (int): Number of classes.
        file_name (str): Path to save the figure.
    """
    metric = metric.lower()
    if metric not in ["precision", "recall", "f1"]:
        raise ValueError("Metric must be one of 'precision', 'recall', or 'f1'.")

    all_classes, all_models, all_values = [], [], []

    for model_name, metrics in zip(model_names, per_class_metrics_list):
        for class_idx in range(num_classes):
            all_classes.append(f"Class {class_idx}")
            all_models.append(model_name)
            all_values.append(metrics[class_idx][metric])

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=all_classes,
        y=all_values,
        hue=all_models,
        palette="Set1",
        edgecolor="none",
        ci=None
    )

    # Add value labels on top of each bar
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f'{height:.2f}', 
            (patch.get_x() + patch.get_width() / 2, height), 
            ha='center', 
            va='bottom',
            fontsize=9
        )

    plt.title(f"Per-Class {metric.capitalize()} Comparison Across Models", fontsize=14, weight="bold")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    # Automatically adjust y-axis
    min_val = min(all_values)
    max_val = max(all_values)
    diff = max_val - min_val
    if diff < 1e-6:
        diff = 1e-6
    padding = diff * 0.6
    plt.ylim(min_val - padding, 1.2)

    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model")
    sns.despine()
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()

    print(f"Per-class {metric} metrics saved to {file_name}")


def plot_cv_heatmap(
    results: Dict[Tuple[int, int], float],
    n_trees_options: List[int],
    max_depth_options: List[int],
    save_path: str = "results/hyperparameter_rf.png"
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

    plt.xscale('log')
    
    plt.title("Ridge Regression Hyperparameter Search", fontsize=14)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.xlabel("Lambda", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    print(f"Lambda search saved to {file_name}")

def show_knn_search_results(results: Dict[float, float], file_name: str = "results/hyperparameter_knn.png") -> None:
    """
    Visualizes the results of the knn k hyperparameter search.
    Plots the mean validation accuracy for each tested k value.

    Args:
        results (Dict[float, float]): Dictionary mapping k values to mean accuracies.
        file_name (str): File path to save the resulting plot.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    
    plt.title("K-NN Hyperparameter Search", fontsize=14)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.xlabel("K value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    print(f"K-NN search saved to {file_name}")