"""
Basic layers implemented from scratch for MLP and CNN.
Does not use torch.nn.Linear, torch.nn.Conv2d, etc.
Only uses nn.Parameter to store weights.

Includes:
- Linear: Fully connected layer
- Conv2d: 2D convolution (using unfold/im2col)
- MaxPool2d: 2D max pooling
- BatchNorm1d: Batch normalization for vectors
- BatchNorm2d: Batch normalization for feature maps
- Dropout: Regularization
- Flatten: Reshape (C, H, W) → (C*H*W,)
"""

import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    Custom Fully Connected Layer.
    
    Formula: y = x @ W^T + b
    
    Args:
        in_features: Input size
        out_features: Output size
        bias: Whether to use bias
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with Kaiming Uniform (suitable for ReLU)
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-bound, bound)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        """
        Args:
            x: tensor shape (..., in_features)
        Returns:
            tensor shape (..., out_features)
        """
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output


class Conv2d(nn.Module):
    """
    Custom 2D convolution using the im2col (unfold) method.
    
    Instead of looping through each pixel, we "unfold" the input into a 2D matrix,
    then multiply the matrix with the kernel → much more efficient.
    
    Formula:
        output[n, c_out, h, w] = Σ kernel[c_out, c_in, kh, kw] * input[n, c_in, h+kh, w+kw] + bias[c_out]
    
    Args:
        in_channels: Number of input channels (e.g. 3 for RGB)
        out_channels: Number of filters (number of output channels)
        kernel_size: Kernel size (int or tuple)
        stride: Step size
        padding: Number of pixels padded on each side
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        kh, kw = self.kernel_size
        # Kaiming initialization
        fan_in = in_channels * kh * kw
        bound = 1.0 / math.sqrt(fan_in)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kh, kw).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        Args:
            x: tensor shape (N, C_in, H, W)
        Returns:
            tensor shape (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # Step 1: Padding
        if ph > 0 or pw > 0:
            x = torch.nn.functional.pad(x, (pw, pw, ph, ph), mode='constant', value=0)
            _, _, H, W = x.shape

        # Step 2: Compute output size
        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1

        # Step 3: Im2col — "unfold" input into a matrix
        # Use unfold to extract patches
        # x_unfold shape: (N, C_in * kh * kw, H_out * W_out)
        x_unfold = x.unfold(2, kh, sh).unfold(3, kw, sw)  # (N, C_in, H_out, W_out, kh, kw)
        x_unfold = x_unfold.contiguous().view(N, C_in * kh * kw, H_out * W_out)

        # Step 4: Matrix multiplication with kernel
        # weight reshape: (C_out, C_in * kh * kw)
        weight_flat = self.weight.view(self.out_channels, -1)

        # output: (N, C_out, H_out * W_out)
        output = weight_flat @ x_unfold

        # Step 5: Add bias
        output = output + self.bias.view(1, -1, 1)

        # Step 6: Reshape back
        output = output.view(N, self.out_channels, H_out, W_out)

        return output


class MaxPool2d(nn.Module):
    """
    Custom 2D Max Pooling.
    
    Reduces the feature map size by taking the maximum value
    in each kernel_size x kernel_size region.
    
    Args:
        kernel_size: Pooling region size
        stride: Step size (default = kernel_size)
    """

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

    def forward(self, x):
        """
        Args:
            x: tensor shape (N, C, H, W)
        Returns:
            tensor shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1

        # Use unfold to extract regions
        x_unfold = x.unfold(2, kh, sh).unfold(3, kw, sw)  # (N, C, H_out, W_out, kh, kw)
        x_unfold = x_unfold.contiguous().view(N, C, H_out, W_out, -1)  # (N, C, H_out, W_out, kh*kw)

        # Take max in each region
        output, _ = x_unfold.max(dim=-1)  # (N, C, H_out, W_out)

        return output


class AvgPool2d(nn.Module):
    """
    Custom 2D Average Pooling.
    Similar to MaxPool, but takes the average instead of the max.
    """

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1

        x_unfold = x.unfold(2, kh, sh).unfold(3, kw, sw)
        x_unfold = x_unfold.contiguous().view(N, C, H_out, W_out, -1)

        output = x_unfold.mean(dim=-1)

        return output


class BatchNorm2d(nn.Module):
    """
    Batch Normalization for 2D feature maps (after Conv2d).
    
    Formula:
        y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    Where mean and var are computed over the batch (dims N, H, W),
    while running mean/var are kept for inference.
    
    Args:
        num_features: Number of channels (C)
        eps: Small constant to avoid division by zero
        momentum: Running stats update factor
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running statistics (not parameters, no gradients needed)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        """
        Args:
            x: tensor shape (N, C, H, W)
        """
        if self.training:
            # Compute mean and var over the batch (dims 0, 2, 3 — keep dim C)
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            # Update running stats
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        # Reshape gamma, beta, mean, var into (1, C, 1, 1)
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return gamma * x_norm + beta


class BatchNorm1d(nn.Module):
    """
    Batch Normalization for 1D vectors (after Linear).
    
    Similar to BatchNorm2d but with input shape (N, features).
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        """
        Args:
            x: tensor shape (N, features)
        """
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class Dropout(nn.Module):
    """
    Custom Dropout.
    
    During training: randomly disables each neuron with probability p.
    During inference: does nothing (already scaled during training).
    
    Inverted Dropout:
        - Training: output = mask * x / (1 - p)
        - Inference: output = x
    
    Args:
        p: Probability of disabling a neuron (default 0.5)
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # Create mask: 1 = keep, 0 = disable
        mask = (torch.rand_like(x.float()) > self.p).float()

        # Inverted dropout: scale up to keep the expected value unchanged
        return mask * x / (1.0 - self.p)


class FlattenLayer(nn.Module):
    """
    Flatten tensor from (N, C, H, W) → (N, C*H*W).
    Used when moving from Conv layers to Linear layers.
    """

    def forward(self, x):
        return x.view(x.size(0), -1)