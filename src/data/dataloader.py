"""
Custom DataLoader written from scratch for image classification.
Without using torch.utils.data.DataLoader.

Features:
- Shuffle: randomly shuffle data order each epoch
- Batching: divide data into batches
- Stacking: combine images/labels in batch into tensor
- Iterator: support for-loop through each batch
"""

import torch
import random


class ImageDataLoader:
    """
    Custom DataLoader for images.
    Each batch returns:
        - images: tensor shape (batch_size, C, H, W)
        - labels: tensor shape (batch_size,)
    """

    def __init__(self, dataset, batch_size=32, shuffle=True):
        """
        Args:
            dataset: ImageDataset object
            batch_size: number of images in one batch
            shuffle: shuffle data each epoch?
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Return the number of batches (rounded up)."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Create iterator to traverse through each batch.
        Each time __iter__ is called (each epoch), shuffle again if needed.
        """
        # Create list of indices
        indices = list(range(len(self.dataset)))

        # Shuffle: random shuffle
        if self.shuffle:
            random.shuffle(indices)

        # Divide into batches
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]

            # Get data from dataset
            images = []
            labels = []
            for idx in batch_indices:
                img, lbl = self.dataset[idx]
                images.append(img)
                labels.append(lbl)

            # Stack into batch tensor
            # images: list of (C, H, W) → (batch_size, C, H, W)
            images_batch = torch.stack(images, dim=0)
            labels_batch = torch.stack(labels, dim=0)

            yield images_batch, labels_batch


def get_dataloader(dataset, batch_size=32, shuffle=True):
    """Utility function: create DataLoader from dataset."""
    return ImageDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
