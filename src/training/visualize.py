"""
Visualize training and evaluation results.

Includes:
- Training/validation loss & accuracy plots
- Confusion matrix heatmap
- Display sample predictions
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history_path="checkpoints/train_history.json",
                          save_path=None, show=True):
    """
    Plot training loss and accuracy across epochs.
    """
    if not os.path.exists(history_path):
        print(f"[!] Not found: {history_path}")
        return None

    with open(history_path, "r") as f:
        history = json.load(f)

    if not history:
        print("[!] History is empty")
        return None

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    has_val = "val_loss" in history[0]

    sns.set_theme(style="whitegrid", palette="muted")

    if has_val:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        val_loss = [h["val_loss"] for h in history]
        val_acc = [h["val_acc"] for h in history]

        # Loss
        axes[0].plot(epochs, train_loss, 'o-', color='#e74c3c', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_loss, 's-', color='#3498db', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, train_acc, 'o-', color='#e74c3c', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, val_acc, 's-', color='#2ecc71', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Đánh dấu best
        best_idx = val_acc.index(max(val_acc))
        axes[1].annotate(
            f'Best: {val_acc[best_idx]:.4f}',
            xy=(epochs[best_idx], val_acc[best_idx]),
            xytext=(10, -15), textcoords='offset points',
            fontsize=10, color='#2ecc71',
            arrowprops=dict(arrowstyle='->', color='#2ecc71'),
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2, label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Saved: {save_path}")

    if show:
        plt.show()
    return fig


def plot_confusion_matrix(confusion_matrix, class_names, save_path=None, show=True):
    """
    Plot confusion matrix as heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        confusion_matrix,
        annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Saved: {save_path}")

    if show:
        plt.show()
    return fig


def plot_sample_predictions(model, dataloader, class_names, num_samples=16,
                            device='cpu', save_path=None, show=True):
    """
    Hiển thị một số ảnh cùng với nhãn thật và nhãn dự đoán.
    """
    import torch

    model.eval()

    # Lấy 1 batch
    images, labels = next(iter(dataloader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    with torch.no_grad():
        logits = model(images)
        predictions = torch.argmax(logits, dim=1).cpu()

    # Vẽ
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            ax.axis('off')
            continue

        img = images[i].cpu()
        # (C, H, W) → (H, W, C) cho matplotlib
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            # Denormalize (ước lượng)
            img = img * 0.25 + 0.5
            img = img.clamp(0, 1)
        elif img.shape[0] == 1:
            img = img.squeeze(0)

        ax.imshow(img.numpy() if img.dim() == 3 else img.numpy(), cmap='gray' if img.dim() == 2 else None)

        true_label = class_names[labels[i].item()]
        pred_label = class_names[predictions[i].item()]
        correct = labels[i].item() == predictions[i].item()

        color = 'green' if correct else 'red'
        ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=9, color=color)
        ax.axis('off')

    fig.suptitle('🖼️ Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    return fig
