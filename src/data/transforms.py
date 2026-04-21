"""
Custom-written transforms for image processing from scratch.
Does not use torchvision.transforms.

Supports:
- ToTensor: Convert numpy array → torch tensor, scale [0,255] → [0,1]
- Normalize: Normalize using dataset mean and std
- Flatten: Flatten image into 1D vector (for MLP)
- RandomHorizontalFlip: Random horizontal flip (data augmentation)
- Compose: Combine multiple transforms
"""

import torch
import numpy as np
import random


class ToTensor:
    """
    Convert numpy array (H, W, C) or (H, W) → float32 tensor.
    Scale pixel values from [0, 255] → [0.0, 1.0].
    Output shape: (C, H, W) — PyTorch standard.
    """

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return image.float()

        if isinstance(image, np.ndarray):
            # If grayscale image (H, W) → add channel dimension
            if image.ndim == 2:
                image = image[:, :, np.newaxis]

            # (H, W, C) → (C, H, W)
            image = image.transpose(2, 0, 1)

            # Scale [0, 255] → [0, 1]
            tensor = torch.from_numpy(image.copy()).float() / 255.0
            return tensor

        raise TypeError(f"Unsupported type: {type(image)}")


class Normalize:
    """
    Normalize tensor using mean and std for each channel.
    
    Formula: output[c] = (input[c] - mean[c]) / std[c]
    
    Args:
        mean: tuple/list of mean per channel, e.g. (0.4914, 0.4822, 0.4465) for CIFAR-10
        std: tuple/list of std per channel, e.g. (0.2470, 0.2435, 0.2616) for CIFAR-10
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).float().view(-1, 1, 1)  # (C, 1, 1)
        self.std = torch.tensor(std).float().view(-1, 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor: shape (C, H, W), already in float [0, 1]
        Returns:
            normalized tensor
        """
        return (tensor - self.mean) / self.std


class Flatten:
    """
    Flatten tensor (C, H, W) → 1D vector (C*H*W,).
    Used for MLP (requires 1D input).
    """

    def __call__(self, tensor):
        return tensor.view(-1)


class RandomHorizontalFlip:
    """
    Horizontally flip the image with probability p.
    Simple but effective data augmentation.
    
    Args:
        p: flip probability (default 0.5)
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        """
        Args:
            tensor: shape (C, H, W)
        """
        if random.random() < self.p:
            # Flip along width (dim=-1)
            return tensor.flip(-1)
        return tensor


class RandomCrop:
    """
    Randomly crop the image with padding applied first.
    
    Args:
        size: Output size (H, W) or int (H=W)
        padding: Number of pixels padded on each side
    """

    def __init__(self, size, padding=4):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding

    def __call__(self, tensor):
        """
        Args:
            tensor: shape (C, H, W)
        """
        c, h, w = tensor.shape
        target_h, target_w = self.size

        if self.padding > 0:
            # Zero-padding
            padded = torch.zeros(c, h + 2 * self.padding, w + 2 * self.padding)
            padded[:, self.padding:self.padding + h, self.padding:self.padding + w] = tensor
            tensor = padded
            h += 2 * self.padding
            w += 2 * self.padding

        # Random crop
        top = random.randint(0, h - target_h)
        left = random.randint(0, w - target_w)

        return tensor[:, top:top + target_h, left:left + target_w]


class Compose:
    """
    Combine multiple transforms into one pipeline.
    
    Example:
        transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,)),
        ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image