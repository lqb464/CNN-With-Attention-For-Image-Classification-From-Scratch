"""
Custom Convolutional Neural Network (CNN) built from scratch.
Uses custom layers: Conv2d, MaxPool2d, BatchNorm2d, Dropout, Linear.

Default architecture for CIFAR-10:
    Conv(3→32, 3x3) → BN → ReLU → Conv(32→32, 3x3) → BN → ReLU → MaxPool
    Conv(32→64, 3x3) → BN → ReLU → Conv(64→64, 3x3) → BN → ReLU → MaxPool
    Conv(64→128, 3x3) → BN → ReLU → Conv(128→128, 3x3) → BN → ReLU → MaxPool
    Flatten → Linear(2048→256) → ReLU → Dropout → Linear(256→10)

If use_attention=True:
    after each block, SE attention is added to reweight channels.
"""

import torch
import torch.nn as nn
from src.models.layers import (
    Conv2d, MaxPool2d, BatchNorm2d, BatchNorm1d,
    Linear, Dropout, FlattenLayer
)
from src.models.activations import relu


class ConvBlock(nn.Module):
    """
    Block: Conv2d → BatchNorm2d → ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = relu(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Input:  (N, C, H, W)
    Output: (N, C, H, W)

    Idea:
    1. Global average pooling -> (N, C)
    2. FC reduce dimension
    3. ReLU
    4. FC expand dimension
    5. Sigmoid -> attention weights per channel
    6. Multiply with original feature map
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_dim = max(channels // reduction, 4)

        self.fc1 = Linear(channels, hidden_dim)
        self.fc2 = Linear(hidden_dim, channels)

    def forward(self, x):
        # x: (N, C, H, W)
        # Squeeze: global average pooling
        s = x.mean(dim=(2, 3))              # (N, C)

        # Excitation
        s = self.fc1(s)                     # (N, hidden_dim)
        s = relu(s)
        s = self.fc2(s)                     # (N, C)
        s = torch.sigmoid(s)                # (N, C)

        # Reshape to broadcast back to (N, C, H, W)
        s = s.view(s.size(0), s.size(1), 1, 1)

        # Reweight channels
        return x * s


class CNN(nn.Module):
    """
    Custom CNN for image classification.

    Args:
        in_channels: number of input channels
        num_classes: number of classes
        dropout_rate: dropout in classifier
        use_attention: whether to enable SE attention
        attention_reduction: reduction ratio in SE block
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        dropout_rate=0.5,
        use_attention=False,
        attention_reduction=16,
    ):
        super().__init__()

        self.use_attention = use_attention

        # ===== Feature Extractor =====
        # Block 1: 3 -> 32
        self.conv1_1 = ConvBlock(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = ConvBlock(32, 32, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = Dropout(0.25)
        self.attn1 = SEBlock(32, reduction=attention_reduction) if use_attention else None

        # Block 2: 32 -> 64
        self.conv2_1 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = Dropout(0.25)
        self.attn2 = SEBlock(64, reduction=attention_reduction) if use_attention else None

        # Block 3: 64 -> 128
        self.conv3_1 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = ConvBlock(128, 128, kernel_size=3, padding=1)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = Dropout(0.25)
        self.attn3 = SEBlock(128, reduction=attention_reduction) if use_attention else None

        # ===== Classifier =====
        self.flatten = FlattenLayer()
        # 32x32 -> pooled 3 times -> 4x4, so 128*4*4 = 2048
        self.fc1 = Linear(128 * 4 * 4, 256)
        self.bn_fc1 = BatchNorm1d(256)
        self.drop_fc = Dropout(dropout_rate)
        self.fc2 = Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        if self.attn1 is not None:
            x = self.attn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        if self.attn2 is not None:
            x = self.attn2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        if self.attn3 is not None:
            x = self.attn3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = relu(x)
        x = self.drop_fc(x)
        x = self.fc2(x)

        return x


def build_cnn(
    in_channels=3,
    num_classes=10,
    dropout_rate=0.5,
    use_attention=False,
    attention_reduction=16,
):
    """Utility function to create CNN."""
    return CNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_attention=use_attention,
        attention_reduction=attention_reduction,
    )