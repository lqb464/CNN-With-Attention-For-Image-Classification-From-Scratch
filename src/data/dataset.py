"""
Custom dataset for image classification.
Does not inherit from torch.utils.data.Dataset.

Features:
- Load CIFAR-10 from torchvision (only for data download)
- Apply custom transforms
- Index-based access
"""

import torch
import numpy as np


class ImageDataset:
    """
    Image classification dataset, stores image + label.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array shape (N, H, W, C) or (N, H, W) - pixel values [0, 255]
            labels: numpy array or list shape (N,) - class labels
            transform: Compose or callable - image transformation
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Apply transforms (ToTensor, Normalize, ...)
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to float tensor
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]
                image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0

        # Label to tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return image, label


def load_cifar10(data_dir="data"):
    """
    Load CIFAR-10 using torchvision (only for downloading).
    Returns raw numpy arrays to use with custom Dataset.
    
    CIFAR-10:
    - 60,000 color images 32x32
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - 50,000 train + 10,000 test
    
    Returns:
        train_images: (50000, 32, 32, 3) uint8
        train_labels: (50000,) int
        test_images: (10000, 32, 32, 3) uint8
        test_labels: (10000,) int
        class_names: list of 10 class names
    """
    import torchvision

    # Download CIFAR-10 (only use torchvision for downloading)
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True
    )

    # Get raw numpy arrays
    train_images = np.array(train_set.data)        # (50000, 32, 32, 3)
    train_labels = np.array(train_set.targets)      # (50000,)
    test_images = np.array(test_set.data)            # (10000, 32, 32, 3)
    test_labels = np.array(test_set.targets)         # (10000,)

    class_names = train_set.classes  # ['airplane', 'automobile', ...]

    return train_images, train_labels, test_images, test_labels, class_names