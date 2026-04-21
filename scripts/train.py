"""
Training script for CNN on CIFAR-10 (config-driven).
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import load_cifar10
from src.factories import build_loaders, build_training_components, build_transforms
from src.training.evaluate import compute_accuracy
from src.training.trainer import Trainer
from src.utils.config import deep_merge, load_yaml, parse_overrides, save_resolved_config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values. Example: --override training.epochs=5",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _apply_cli_overrides(config, args):
    cli_updates = {}
    if args.output_dir is not None:
        cli_updates.setdefault("project", {})["output_dir"] = args.output_dir
    if args.epochs is not None:
        cli_updates.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cli_updates.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cli_updates.setdefault("training", {})["lr"] = args.lr
    if args.optimizer is not None:
        cli_updates.setdefault("training", {})["optimizer"] = args.optimizer
    if args.num_samples is not None:
        cli_updates.setdefault("data", {})["num_samples"] = args.num_samples
    if args.attention:
        cli_updates.setdefault("model", {})["use_attention"] = True
    if args.resume:
        cli_updates.setdefault("training", {})["resume"] = True
    if cli_updates:
        config = deep_merge(config, cli_updates)
    if args.override:
        config = deep_merge(config, parse_overrides(args.override))
    return config


def main():
    args = parse_args()
    config = load_yaml(args.config)
    config = _apply_cli_overrides(config, args)

    project_cfg = config.get("project", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    seed = int(project_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = str(project_cfg.get("output_dir", "outputs/cnn_baseline"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== STEP 1: LOAD DATA =====
    print("=" * 60)
    print(
        f"  IMAGE CLASSIFICATION CIFAR-10 | Model: CNN "
        f"{'+ Attention' if model_cfg.get('use_attention', False) else '(Baseline)'}"
    )
    print("=" * 60)

    print("\n=== STEP 1: LOAD AND PREPROCESS DATA ===")
    train_images, train_labels, test_images, test_labels, class_names = load_cifar10(
        str(project_cfg.get("data_dir", "data"))
    )

    num_samples = data_cfg.get("num_samples", None)
    if num_samples is not None:
        train_images = train_images[: int(num_samples)]
        train_labels = train_labels[: int(num_samples)]

    print(f"[+] Train: {len(train_images)} images")
    print(f"[+] Test:  {len(test_images)} images")
    print(f"[+] Classes: {class_names}")

    # CIFAR-10 mean & std
    def compute_mean_std(images):
        """
        Compute mean/std per channel from numpy array (N, H, W, C)
        """
        images = images.astype(np.float32) / 255.0
        mean = images.mean(axis=(0, 1, 2))
        std = images.std(axis=(0, 1, 2))
        return tuple(mean.tolist()), tuple(std.tolist())

    cifar10_mean, cifar10_std = compute_mean_std(train_images)

    norm_stats = {
        "mean": list(cifar10_mean),
        "std": list(cifar10_std),
    }
    with open(os.path.join(checkpoint_dir, "normalization_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    train_transform, test_transform = build_transforms(cifar10_mean, cifar10_std, data_cfg)
    batch_size = int(data_cfg.get("batch_size", 64))
    train_loader, test_loader = build_loaders(
        train_images,
        train_labels,
        test_images,
        test_labels,
        train_transform,
        test_transform,
        batch_size=batch_size,
    )

    print(f"[+] Train batches: {len(train_loader)}")
    print(f"[+] Test batches:  {len(test_loader)}")

    # ===== STEP 2: INITIALIZE MODEL (CNN) =====
    print(f"\n=== STEP 2: INITIALIZE MODEL (CNN) ===")

    model, optimizer, criterion = build_training_components(model_cfg, training_cfg)

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"[+] Model: CNN "
        f"{'+ Attention' if model_cfg.get('use_attention', False) else '(Baseline)'}"
    )
    print(f"[+] Parameters: {num_params:,}")
    print(f"[+] Device: {device}")

    # ===== BƯỚC 3: HUẤN LUYỆN =====
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=int(training_cfg.get("epochs", 20)),
        device=device,
        save_dir=checkpoint_dir,
        log_every=int(training_cfg.get("log_every", 100)),
    )

    start_epoch = 1
    last_ckpt = os.path.join(checkpoint_dir, "last_checkpoint.pt")

    should_resume = bool(training_cfg.get("resume", False))
    if should_resume and os.path.exists(last_ckpt):
        print(f"\n[+] Resume from: {last_ckpt}")
        start_epoch = trainer.load_checkpoint(last_ckpt)
        print(f"[+] Starting from epoch {start_epoch}")

        if start_epoch > int(training_cfg.get("epochs", 20)):
            print("[!] Training already completed. No further training needed.")
        else:
            trainer.train(start_epoch=start_epoch)
    else:
        trainer.train(start_epoch=1)

    # ===== STEP 4: EVALUATE ON TEST SET =====
    print("\n=== STEP 4: EVALUATE ON TEST SET ===")

    # Load best model
    best_weights = os.path.join(checkpoint_dir, "best_model_weights.pt")
    if os.path.exists(best_weights):
        model.load_state_dict(torch.load(best_weights, map_location=device))
        print("[+] Loaded best_model_weights.pt")

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f"[+] Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    print(f"\n[✓] Checkpoint saved in: {checkpoint_dir}/")


if __name__ == "__main__":
    main()