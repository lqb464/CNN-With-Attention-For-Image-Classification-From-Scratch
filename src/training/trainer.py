"""
Training loop for image classification.
Supports:
- Train epoch-by-epoch
- Gradient clipping
- Validation accuracy per epoch
- Save/load checkpoint
- Training history (loss, accuracy)
"""

import os
import json
import csv
import torch


def clip_grad_norm(parameters, max_norm):
    """Gradient clipping by norm."""
    params = list(parameters)
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += (p.grad ** 2).sum().item()
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)

    return total_norm


class Trainer:
    """
    Trainer for image classification task.
    
    Args:
        model: MLP or CNN
        optimizer: SGD or Adam (custom-written)
        criterion: CrossEntropyLoss (custom-written)
        train_loader: ImageDataLoader for training
        val_loader: ImageDataLoader for validation (optional)
        epochs: Number of epochs
        device: 'cpu' or 'cuda'
        save_dir: Directory to save checkpoints
        log_every: Log every N batches
    """

    def __init__(self, model, optimizer, criterion, train_loader,
                 val_loader=None, epochs=20, device='cpu',
                 save_dir='checkpoints', log_every=100):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = save_dir
        self.log_every = log_every

        self.history = []
        self.best_val_acc = 0.0
        self.best_loss = float('inf')

        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        """Train 1 epoch."""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward
            loss.backward()

            # Gradient clipping
            clip_grad_norm(self.model.parameters(), max_norm=5.0)

            # Update weights
            self.optimizer.step()

            # Thống kê
            epoch_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Log
            if self.log_every and (batch_idx + 1) % self.log_every == 0:
                print(
                    f"  Epoch {epoch:02d} | "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(self.train_loader)
        train_acc = correct / max(total, 1)
        return avg_loss, train_acc

    def validate(self):
        """Evaluate on validation set."""
        if self.val_loader is None:
            return None

        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = val_loss / max(len(self.val_loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, train_loss, val_acc=None, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_acc": float(val_acc) if val_acc is not None else None,
            "best_val_acc": float(self.best_val_acc),
            "best_loss": float(self.best_loss),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if hasattr(self.optimizer, "state_dict") else None
            ),
            "history": self.history,
        }

        # Luôn lưu last
        torch.save(checkpoint, os.path.join(self.save_dir, "last_checkpoint.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_model_weights.pt"))

        # Lưu best nếu cần
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best_checkpoint.pt"))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model_weights.pt"))

        # Lưu history JSON
        history_path = os.path.join(self.save_dir, "train_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(self.save_dir, "train_history.csv")
        if self.history:
            fieldnames = sorted(self.history[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.history)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint để resume training."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        opt_state = checkpoint.get("optimizer_state_dict", None)
        if opt_state is not None and hasattr(self.optimizer, "load_state_dict"):
            self.optimizer.load_state_dict(opt_state)

        self.history = checkpoint.get("history", [])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))

        start_epoch = checkpoint.get("epoch", 0) + 1
        return start_epoch

    def train(self, start_epoch=1):
        """Main training loop."""
        print("\n=== START TRAINING ===")

        for epoch in range(start_epoch, self.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_result = self.validate()

            if val_result is not None:
                val_loss, val_acc = val_result
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                if train_loss < self.best_loss:
                    self.best_loss = train_loss

                log_item = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "is_best": bool(is_best),
                }

                print(
                    f"Epoch {epoch:02d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                    f"{' ★' if is_best else ''}"
                )
            else:
                is_best = train_loss < self.best_loss
                if is_best:
                    self.best_loss = train_loss

                log_item = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "is_best": bool(is_best),
                }

                print(
                    f"Epoch {epoch:02d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                    f"{' ★' if is_best else ''}"
                )

            self.history.append(log_item)

            self.save_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                val_acc=val_result[1] if val_result else None,
                is_best=is_best,
            )

        print(f"\n[✓] COMPLETED! Best Val Acc: {self.best_val_acc:.4f}")
        return self.history
