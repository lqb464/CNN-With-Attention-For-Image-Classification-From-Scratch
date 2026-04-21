"""
Custom-written loss functions from mathematical formulas.
Does not use torch.nn.CrossEntropyLoss.
"""

import torch


class CrossEntropyLoss:
    """
    Custom Cross Entropy Loss.
    
    Formula:
        L = -1/N * Σ log(softmax(logits)[target])
        
    Combines log-softmax and negative log likelihood:
        1. Compute log-softmax (numerically stable)
        2. Select value at target index
        3. Take average
    
    Args:
        ignore_index: Index to ignore when computing loss (e.g., PAD)
    """

    def __init__(self, ignore_index=-100):
        self.ignore_index = ignore_index

    def __call__(self, logits, targets):
        """
        Args:
            logits: tensor shape (N, num_classes) — chưa qua softmax
            targets: tensor shape (N,) — nhãn đúng
        Returns:
            loss: scalar tensor
        """
        N = logits.shape[0]

        # Bước 1: Log-softmax (ổn định bằng cách trừ max)
        max_val = logits.max(dim=1, keepdim=True)[0]
        shifted = logits - max_val
        log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1))
        log_softmax = shifted - log_sum_exp.unsqueeze(1)

        # Bước 2: Lấy log probability tại target index
        mask = (targets != self.ignore_index)
        valid_count = mask.sum().item()

        if valid_count == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        # Bước 3: Negative log likelihood
        loss = 0.0
        for i in range(N):
            if mask[i]:
                target_idx = targets[i].item()
                loss = loss + (-log_softmax[i, target_idx])

        loss = loss / valid_count
        return loss


def get_loss_function():
    """Trả về Cross Entropy Loss."""
    return CrossEntropyLoss(ignore_index=-100)
