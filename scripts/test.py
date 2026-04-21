"""
Model evaluation script for trained CIFAR-10.
Displays: Accuracy, Confusion Matrix, Classification Report.

Usage:
    python scripts/test.py --model cnn
    python scripts/test.py --model mlp
"""

import json
import os
import argparse
import random
import numpy as np
import torch

from src.data.transforms import ToTensor, Normalize, Compose
from src.data.dataset import ImageDataset, load_cifar10
from src.data.dataloader import get_dataloader

from src.models.cnn import build_cnn

from src.training.evaluate import (
    compute_accuracy,
    compute_confusion_matrix,
    classification_report,
    print_classification_report,
)
from src.training.visualize import (
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 model")
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_weights.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Load data =====
    print("=" * 60)
    print(f"  EVALUATING MODEL {args.model.upper()} ON CIFAR-10")
    print("=" * 60)

    _, _, test_images, test_labels, class_names = load_cifar10("data")

    stats_path = os.path.join(os.path.dirname(args.checkpoint), "normalization_stats.json")
    with open(stats_path, "r") as f:
        norm_stats = json.load(f)

    cifar10_mean = tuple(norm_stats["mean"])
    cifar10_std = tuple(norm_stats["std"])

    test_transform = Compose([
        ToTensor(),
        Normalize(cifar10_mean, cifar10_std),
    ])

    test_dataset = ImageDataset(test_images, test_labels, transform=test_transform)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ===== Build & Load model =====

    model = build_cnn(in_channels=3, num_classes=10)

    model = model.to(device)

    if not os.path.exists(args.checkpoint):
        print(f"[!] Not found: {args.checkpoint}")
        print("[!] Please run `python scripts/train.py` first.")
        return

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    print(f"[+] Loaded: {args.checkpoint}")

    # ===== Đánh giá =====
    print("\n--- ACCURACY ---")
    acc = compute_accuracy(model, test_loader, device=device)
    print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    print("\n--- CONFUSION MATRIX ---")
    cm = compute_confusion_matrix(model, test_loader, num_classes=10, device=device)

    print("\n--- CLASSIFICATION REPORT ---")
    report = classification_report(cm, class_names)
    print_classification_report(report, class_names)

    # ===== Trực quan hóa =====
    os.makedirs(args.save_dir, exist_ok=True)

    # Training history
    plot_training_history(
        history_path="checkpoints/train_history.json",
        save_path=os.path.join(args.save_dir, "training_history.png"),
        show=False,
    )

    # Confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(args.save_dir, "confusion_matrix.png"),
        show=False,
    )

    # Sample predictions
    plot_sample_predictions(
        model, test_loader, class_names,
        num_samples=16, device=device,
        save_path=os.path.join(args.save_dir, "sample_predictions.png"),
        show=False,
    )

    print(f"\n[✓] Results saved in: {args.save_dir}/")


if __name__ == "__main__":
    main()
