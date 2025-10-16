#####################################################################################################################################

import torch
import numpy as np
from random_forest.random_forest import RandomForest
from random_forest.feature_extraction import extract_features_for_random_forest
from random_forest.load_data import N_CLASSES


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


def compute_per_class_metrics(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    Compute per-class precision, recall, and F1 score.
    
    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): True labels.
        num_classes (int): Number of classes.
    
    Returns:
        dict: Dictionary with per-class metrics.
    """
    metrics = {}
    
    for class_idx in range(num_classes):
        true_positives = ((predictions == class_idx) & (labels == class_idx)).sum().item()
        false_positives = ((predictions == class_idx) & (labels != class_idx)).sum().item()
        false_negatives = ((predictions != class_idx) & (labels == class_idx)).sum().item()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[class_idx] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': (labels == class_idx).sum().item()
        }
    
    return metrics

def k_fold_cross_validation(X: torch.Tensor, y: torch.Tensor, n_trees: int, k: int = 5):
    """
    Perform stratified k-fold cross-validation on the training data, due to the class imbalances
    and already ordered data. Ensures that each fold has an equal proportion of each class.    
    Args:
        X (torch.Tensor): training features.
        y (torch.Tensor): training labels.
        n_trees (int): num of trees in the forest.
        k (int): number of folds for cross-validation.
    
    Returns:
        dict: Cross-validation results including mean and std of accuracies.
    """
    n_samples = X.shape[0]
    fold_accuracies = []
    
    print(f"\n{k}-Fold stratified cross-calidation:")
    
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
        
        # train model on this fold
        model = RandomForest(n_trees=n_trees)
        model.train_forest(X_train_fold, y_train_fold)
        
        # eval on validation fold
        with torch.no_grad():
            val_predictions = model(X_val_fold)
            val_accuracy = compute_accuracy(val_predictions, y_val_fold)
        
        fold_accuracies.append(val_accuracy)
        
        # shows class distribution in each fold
        unique_val, counts_val = torch.unique(y_val_fold, return_counts=True)
        print(f"Fold {fold + 1}: Val Acc = {val_accuracy * 100:.2f}% ")
    
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    print(f"Cross-Validation Mean Acc: {mean_accuracy * 100:.2f}% (+/- {std_accuracy * 100:.2f}%)")
    
    return {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    }

def train_and_evaluate_random_forest(
    train_data_path: str = "ae.train",
    test_data_path: str = "ae.test",
    n_trees: int = 100,
    variance_threshold: float = 95.0, # uses 35 features
    use_cross_validation: bool = True,
    k_folds: int = 5,
    visualize: bool = False
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
        
    # extract features
    X_train, y_train, X_test, y_test = extract_features_for_random_forest(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        variance_threshold=variance_threshold
    )
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Number of speakers: {N_CLASSES}")
    
    # cross-validation
    cv_results = None
    if use_cross_validation:
        cv_results = k_fold_cross_validation(X_train, y_train, n_trees, k=k_folds)
    
    # train final model
    print("TRAINING FINAL MODEL")
    print(f"Number of trees: {n_trees}")
    
    model = RandomForest(n_trees=n_trees)
    model.train_forest(X_train, y_train)
    print("Training complete...")
    
    # eval on training
    print("EVALUATION")    
    with torch.no_grad():
        train_predictions = model(X_train)
        train_accuracy = compute_accuracy(train_predictions, y_train)
    
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    
    # eval on test
    with torch.no_grad():
        test_predictions = model(X_test)
        test_accuracy = compute_accuracy(test_predictions, y_test)
    
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # compute per-class metrics
    print("PER-CLASS METRICS (ON TEST SET)")
    metrics = compute_per_class_metrics(test_predictions, y_test, N_CLASSES)
    
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    for class_idx in range(N_CLASSES):
        m = metrics[class_idx]
        print(f"{class_idx:<8} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}")
    
    avg_precision = np.mean([metrics[i]['precision'] for i in range(N_CLASSES)])
    avg_recall = np.mean([metrics[i]['recall'] for i in range(N_CLASSES)])
    avg_f1 = np.mean([metrics[i]['f1'] for i in range(N_CLASSES)])
    
    print(f"{'Average':<8} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")
    
    print("FINAL SUMMARY")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    if cv_results:
        print(f"Cross-Validation Accuracy: {cv_results['mean_accuracy'] * 100:.2f}% (+/- {cv_results['std_accuracy'] * 100:.2f}%)")
    print(f"Number of trees: {n_trees}")
    print(f"Features used: {X_train.shape[1]} (from {variance_threshold}% variance)")
    
    # if visualize:
    #     print("\nGenerating visualizations...")
        
    #     # confusion matrix
    #     confusion_mat = compute_confusion_matrix(test_predictions, y_test, N_CLASSES)
    #     visualize_confusion_matrix(confusion_mat)
        
    #     # per-class accuracy (precision, recall, f1-score)
    #     visualize_per_class_accuracy(test_predictions, y_test, N_CLASSES)
        
    
    return model, train_accuracy, test_accuracy, cv_results


# run to check which hyperparameters (n_trees & variance_threshold) are best to use
# main is commented out, if needs to be used uncomment.
def hyperparameter_search(
    train_data_path: str = "ae.train",
    test_data_path: str = "ae.test",
    n_trees_options: list = [50, 100, 200],
    variance_options: list = [90.0, 95.0, 99.0],
    k_folds: int = 5
):
    """
    Perform a simple grid search over hyperparameters using cross-validation.
    
    Args:
        train_data_path (str): Path to training data file.
        test_data_path (str): Path to test data file.
        n_trees_options (list): List of n_trees values to try.
        variance_options (list): List of variance thresholds to try.
        k_folds (int): Number of folds for cross-validation.
    
    Returns:
        dict: Best hyperparameters and their performance.
    """
    print("HYPERPARAMETER SEARCH")
    
    best_score = -1
    best_params = {}
    results = []
    
    for variance in variance_options:
        # Extract features with this variance threshold
        X_train, y_train, _, _ = extract_features_for_random_forest(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            variance_threshold=variance
        )
        
        for n_trees in n_trees_options:
            print(f"\nTrying: n_trees={n_trees}, variance_threshold={variance}%")
            
            # Cross-validation
            cv_results = k_fold_cross_validation(X_train, y_train, n_trees, k=k_folds)
            mean_acc = cv_results['mean_accuracy']
            
            results.append({
                'n_trees': n_trees,
                'variance_threshold': variance,
                'cv_accuracy': mean_acc,
                'cv_std': cv_results['std_accuracy']
            })
            
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = {
                    'n_trees': n_trees,
                    'variance_threshold': variance
                }
    
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'N Trees':<10} {'Variance %':<12} {'CV Accuracy':<15} {'Std':<10}")
    for r in results:
        print(f"{r['n_trees']:<10} {r['variance_threshold']:<12} {r['cv_accuracy']*100:<15.2f} {r['cv_std']*100:<10.2f}")
    
    print("BEST HYPERPARAMETERS")
    print(f"Number of trees: {best_params['n_trees']}")
    print(f"Variance threshold: {best_params['variance_threshold']}%")
    print(f"Cross-validation accuracy: {best_score * 100:.2f}%")
    
    return best_params, results


if __name__ == "__main__":
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    train_path = os.path.join(project_root, "ae.train")
    test_path = os.path.join(project_root, "ae.test")
    
    # train
    model, train_acc, test_acc, cv_results = train_and_evaluate_random_forest(
        train_data_path=train_path,
        test_data_path=test_path,
        n_trees=100,
        variance_threshold=95.0,
        use_cross_validation=True,
        k_folds=5,
        visualize=False
    )
    
    # # Hyperparameter search (uncomment to use)
    # best_params, search_results = hyperparameter_search(
    #     train_data_path=train_path,
    #     test_data_path=test_path,
    #     n_trees_options=[50, 100, 200],
    #     variance_options=[90.0, 95.0, 99.0],
    #     k_folds=5
    # )

    # # train final model with best parameters
    # model, train_acc, test_acc, _ = train_and_evaluate_random_forest(
    #     train_data_path=train_path,
    #     test_data_path=test_path,
    #     n_trees=best_params['n_trees'],
    #     variance_threshold=best_params['variance_threshold'],
    #     use_cross_validation=False,
    #     visualize=True
    # )