import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_metrics(results):
    """
    Create a grouped bar plot for multiple metrics per method.
    Automatically adjusts y-axis to emphasize small differences.
    """
    methods = list(results.keys())
    metrics = list(next(iter(results.values())).keys())
    n_metrics = len(metrics)
    x = np.arange(len(methods))
    width = 0.8 / n_metrics  # space bars evenly within each method group

    all_values = []

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in methods]
        all_values.extend(values)
        plt.bar(x + i*width, values, width, label=metric)

    # Compute y-limits to zoom in if differences are small
    min_val = min(all_values)
    max_val = max(all_values)
    diff = max_val - min_val

    if diff < 1e-6:
        diff = 1e-6  # avoid zero range

    padding = diff * 0.4  # small padding for clarity
    plt.ylim(min_val - padding, max_val + padding)

    plt.xticks(x + width*(n_metrics-1)/2, methods, rotation=20, ha='right')
    plt.ylabel("Metric Value")
    plt.title("Performance Comparison")
    plt.legend()
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
    scatter = plt.scatter(pc1, pc2, c=labels, cmap='tab10', alpha=0.8, edgecolors='k')
    plt.legend(*scatter.legend_elements(), title="Labels", loc="best")
    plt.title("Visualization of First Two Principal Components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_cm(title: str, true_labels: np.ndarray, predictions: np.ndarray) -> None:

    cm = confusion_matrix(true_labels, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.show()