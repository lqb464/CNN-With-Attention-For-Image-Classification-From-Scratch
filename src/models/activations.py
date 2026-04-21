"""
Custom activation functions implemented from mathematical formulas.
Does not use torch.nn.functional.

Supports: Sigmoid, ReLU, LeakyReLU, Tanh, Softmax, GELU
"""

import torch
import math


def sigmoid(x):
    """
    Sigmoid: σ(x) = 1 / (1 + e^(-x))
    Squashes values into the range (0, 1).
    """
    return 1.0 / (1.0 + torch.exp(-x))


def relu(x):
    """
    ReLU: f(x) = max(0, x)
    Most commonly used activation in deep learning.
    """
    return torch.clamp(x, min=0)


def leaky_relu(x, negative_slope=0.01):
    """
    Leaky ReLU: f(x) = x if x > 0, else negative_slope * x
    Addresses the "dying ReLU" problem.
    """
    return torch.where(x > 0, x, negative_slope * x)


def tanh(x):
    """
    Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Squashes values into the range (-1, 1).
    """
    exp_pos = torch.exp(x)
    exp_neg = torch.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def softmax(x, dim=-1):
    """
    Softmax: f(x_i) = e^(x_i) / Σ e^(x_j)
    Converts logits into a probability distribution.
    Subtracts max for numerical stability (log-sum-exp trick).
    """
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)


def gelu(x):
    """
    GELU: f(x) = x * Φ(x)
    Where Φ(x) is the CDF of the normal distribution.
    Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


def log_softmax(x, dim=-1):
    """
    Log-Softmax: log(softmax(x))
    Computed directly for numerical stability.
    """
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    shifted = x - max_val
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    return shifted - log_sum_exp