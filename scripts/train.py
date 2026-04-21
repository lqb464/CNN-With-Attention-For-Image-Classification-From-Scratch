"""
Training script for MLP or CNN on CIFAR-10.

Usage:
    python scripts/train.py                # Train CNN (baseline)
    python scripts/train.py --attention    # Train CNN + attention
"""

import json
import os
import argparse
import random
import numpy as np
import torch

from src.data.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, Compose
from src.data.dataset import ImageDataset, load_cifar10
from src.data.dataloader import get_dataloader

from src.models.cnn import build_cnn

from src.training.losses import get_loss_function
from src.training.optimizers import get_optimizer
from src.training.trainer import Trainer
from src.training.evaluate import compute_accuracy


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")

    parser.add_argument("--attention", action="store_true",
                        help="Enable attention (default: False = baseline)")

    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Optimizer (default: adam)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of training samples (None = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--checkpoints_path", type=str, default="checkpoints",
                        help="Path to checkpoint directory (default: checkpoints)")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = args.checkpoints_path

    # ===== STEP 1: LOAD DATA =====
    print("=" * 60)
    print(f"  IMAGE CLASSIFICATION CIFAR-10 | Model: CNN {'+ Attention' if args.attention else '(Baseline)'}")
    print("=" * 60)

    print("\n=== STEP 1: LOAD AND PREPROCESS DATA ===")
    train_images, train_labels, test_images, test_labels, class_names = load_cifar10("data")

    if args.num_samples is not None:
        train_images = train_images[:args.num_samples]
        train_labels = train_labels[:args.num_samples]

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
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "normalization_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    # Transforms
    train_transform = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomCrop(32, padding=4),
        Normalize(cifar10_mean, cifar10_std),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize(cifar10_mean, cifar10_std),
    ])

    # Dataset & DataLoader
    train_dataset = ImageDataset(train_images, train_labels, transform=train_transform)
    test_dataset = ImageDataset(test_images, test_labels, transform=test_transform)

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[+] Train batches: {len(train_loader)}")
    print(f"[+] Test batches:  {len(test_loader)}")

    # ===== STEP 2: INITIALIZE MODEL (CNN) =====
    print(f"\n=== STEP 2: INITIALIZE MODEL (CNN) ===")

    model = build_cnn(
        in_channels=3,
        num_classes=10,
        dropout_rate=0.5,
        use_attention=args.attention,
    )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Model: CNN {'+ Attention' if args.attention else '(Baseline)'}")
    print(f"[+] Parameters: {num_params:,}")
    print(f"[+] Device: {device}")

    # Optimizer & Loss
    optimizer = get_optimizer(model, lr=args.lr, opt_type=args.optimizer)
    criterion = get_loss_function()

    # ===== BƯỚC 3: HUẤN LUYỆN =====
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        device=device,
        save_dir=checkpoint_dir,
        log_every=100,
    )

    start_epoch = 1
    last_ckpt = os.path.join(checkpoint_dir, "last_checkpoint.pt")

    if args.resume and os.path.exists(last_ckpt):
        print(f"\n[+] Resume from: {last_ckpt}")
        start_epoch = trainer.load_checkpoint(last_ckpt)
        print(f"[+] Starting from epoch {start_epoch}")

        if start_epoch > args.epochs:
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