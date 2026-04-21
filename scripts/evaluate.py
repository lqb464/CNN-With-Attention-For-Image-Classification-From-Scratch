"""
Evaluation script for trained CNN checkpoints.
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from src.data.dataset import load_cifar10
from src.factories import build_loaders, build_training_components, build_transforms
from src.training.evaluate import (
    classification_report,
    compute_accuracy,
    compute_confusion_matrix,
    print_classification_report,
)
from src.training.visualize import (
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_training_history,
)
from src.utils.config import deep_merge, load_yaml, parse_overrides


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CNN on CIFAR-10")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def _apply_cli_overrides(config, args):
    updates = {}
    if args.output_dir is not None:
        updates.setdefault("project", {})["output_dir"] = args.output_dir
    if args.batch_size is not None:
        updates.setdefault("data", {})["batch_size"] = args.batch_size
    if args.checkpoint is not None:
        updates.setdefault("evaluation", {})["checkpoint"] = args.checkpoint
    if updates:
        config = deep_merge(config, updates)
    if args.override:
        config = deep_merge(config, parse_overrides(args.override))
    return config


def main():
    args = parse_args()
    config = _apply_cli_overrides(load_yaml(args.config), args)

    project_cfg = config.get("project", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    evaluation_cfg = config.get("evaluation", {})

    set_seed(int(project_cfg.get("seed", 42)))
    output_dir = str(project_cfg.get("output_dir", "outputs/cnn_baseline"))
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    checkpoint_name = str(evaluation_cfg.get("checkpoint", "best_checkpoint.pt"))
    checkpoint_path = (
        checkpoint_name
        if os.path.isabs(checkpoint_name)
        else os.path.join(checkpoint_dir, checkpoint_name)
    )
    stats_path = os.path.join(checkpoint_dir, "normalization_stats.json")
    history_path = os.path.join(checkpoint_dir, "train_history.json")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as file:
        norm_stats = json.load(file)
    cifar10_mean = tuple(norm_stats["mean"])
    cifar10_std = tuple(norm_stats["std"])

    _, _, test_images, test_labels, class_names = load_cifar10(
        str(project_cfg.get("data_dir", "data"))
    )

    _, eval_transform = build_transforms(cifar10_mean, cifar10_std, data_cfg)
    _, test_loader = build_loaders(
        train_images=test_images,
        train_labels=test_labels,
        test_images=test_images,
        test_labels=test_labels,
        train_transform=eval_transform,
        eval_transform=eval_transform,
        batch_size=int(data_cfg.get("batch_size", 64)),
    )

    model, _, _ = build_training_components(model_cfg, {"optimizer": "adam", "lr": 0.001})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    acc = compute_accuracy(model, test_loader, device=device)
    cm = compute_confusion_matrix(model, test_loader, num_classes=10, device=device)
    report = classification_report(cm, class_names)
    print_classification_report(report, class_names)
    print(f"\nAccuracy: {acc:.4f}")

    metrics = {
        "accuracy": float(acc),
        "checkpoint": checkpoint_path,
        "macro_avg": report["macro_avg"],
    }
    with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with open(os.path.join(eval_dir, "classification_report.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name in class_names:
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"], row["f1"], row["support"]])

    if bool(evaluation_cfg.get("save_plots", True)):
        plot_confusion_matrix(
            cm,
            class_names,
            save_path=os.path.join(eval_dir, "confusion_matrix.png"),
            show=False,
        )
        plot_sample_predictions(
            model,
            test_loader,
            class_names,
            num_samples=int(evaluation_cfg.get("num_prediction_samples", 16)),
            device=device,
            save_path=os.path.join(eval_dir, "sample_predictions.png"),
            show=False,
        )
        plot_training_history(
            history_path=history_path,
            save_path=os.path.join(eval_dir, "training_history.png"),
            show=False,
        )

    print(f"[✓] Evaluation artifacts saved in: {eval_dir}")


if __name__ == "__main__":
    main()
