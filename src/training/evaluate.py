"""
Evaluate image classification model.

Includes:
- Accuracy (overall and per-class)
- Confusion Matrix
- Classification Report (Precision, Recall, F1)
"""

import torch
import numpy as np


def compute_accuracy(model, dataloader, device='cpu'):
    """
    Compute accuracy on entire dataset.
    
    Args:
        model: Trained model
        dataloader: ImageDataLoader
        device: 'cpu' or 'cuda'
    
    Returns:
        accuracy: float (0.0 → 1.0)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / max(total, 1)
    return accuracy


def compute_confusion_matrix(model, dataloader, num_classes, device='cpu'):
    """
    Compute confusion matrix.
    
    confusion_matrix[i][j] = number of samples actually class i but predicted as class j
    
    Args:
        model: Trained model
        dataloader: ImageDataLoader
        num_classes: Number of classes
        device: 'cpu' or 'cuda'
    
    Returns:
        confusion_matrix: numpy array shape (num_classes, num_classes)
    """
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            for true_label, pred_label in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                matrix[true_label][pred_label] += 1

    return matrix


def classification_report(confusion_matrix, class_names=None):
    """
    Compute Precision, Recall, F1 for each class from confusion matrix.
    
    Formula:
        Precision = TP / (TP + FP)    — In predictions of class X, how many % are correct
        Recall = TP / (TP + FN)       — In actual class X, how many % are recognized
        F1 = 2 * Precision * Recall / (Precision + Recall)
    
    Args:
        confusion_matrix: numpy array (num_classes, num_classes)
        class_names: list of class names
    
    Returns:
        report: dict with per-class information
    """
    num_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    report = {}
    total_correct = 0
    total_samples = 0

    for i in range(num_classes):
        tp = confusion_matrix[i][i]
        fp = confusion_matrix[:, i].sum() - tp  # Cột i trừ TP
        fn = confusion_matrix[i, :].sum() - tp  # Hàng i trừ TP

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        support = confusion_matrix[i, :].sum()

        report[class_names[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

        total_correct += tp
        total_samples += support

    # Overall accuracy
    report["accuracy"] = total_correct / max(total_samples, 1)

    # Macro average (trung bình không trọng số)
    avg_precision = np.mean([report[c]["precision"] for c in class_names])
    avg_recall = np.mean([report[c]["recall"] for c in class_names])
    avg_f1 = np.mean([report[c]["f1"] for c in class_names])

    report["macro_avg"] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }

    return report


def print_classification_report(report, class_names):
    """Print classification report nicely."""
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)

    for name in class_names:
        r = report[name]
        print(f"{name:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['support']:>10}")

    print("-" * 60)
    print(f"{'Accuracy':<15} {'':>10} {'':>10} {report['accuracy']:>10.4f}")

    m = report['macro_avg']
    print(f"{'Macro Avg':<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
