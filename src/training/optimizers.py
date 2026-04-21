"""
Custom-written optimizers from mathematical formulas.
Does not use torch.optim.

Supports:
- SGD: Stochastic Gradient Descent (with momentum)
- Adam: Adaptive Moment Estimation
"""

import torch


class SGD:
    """
    Custom SGD with Momentum.
    
    Formula:
        v_t = momentum * v_{t-1} + lr * grad
        param = param - v_t
    
    Args:
        parameters: Iterator of parameters to update
        lr: Learning rate
        momentum: Momentum coefficient (0 = pure SGD)
        weight_decay: L2 regularization
    """

    def __init__(self, parameters, lr=0.01, momentum=0.9, weight_decay=0):
        self.params = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Velocity cho momentum
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                grad = param.grad

                # L2 regularization (weight decay)
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param

                # Momentum
                self.velocities[i] = self.momentum * self.velocities[i] + self.lr * grad

                # Update parameter
                param -= self.velocities[i]

    def state_dict(self):
        return {
            "type": "sgd",
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get("lr", self.lr)
        self.momentum = state_dict.get("momentum", self.momentum)
        self.weight_decay = state_dict.get("weight_decay", self.weight_decay)


class Adam:
    """
    Custom Adam optimizer.
    
    Formula:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t          (first moment)
        v_t = β2 * v_{t-1} + (1 - β2) * g_t²         (second moment)
        m̂_t = m_t / (1 - β1^t)                       (bias correction)
        v̂_t = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
    
    Args:
        parameters: Iterator of parameters
        lr: Learning rate
        beta1: Decay rate for first moment (default 0.9)
        beta2: Decay rate for second moment (default 0.999)
        eps: Small constant to avoid division by zero
        weight_decay: L2 regularization
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0):
        self.params = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        self.t += 1

        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                g = param.grad

                # Weight decay
                if self.weight_decay != 0:
                    g = g + self.weight_decay * param

                # Update biased first & second moment
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Update parameter
                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {
            "type": "adam",
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t,
            "m": [tensor.clone() for tensor in self.m],
            "v": [tensor.clone() for tensor in self.v],
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get("lr", self.lr)
        self.beta1 = state_dict.get("beta1", self.beta1)
        self.beta2 = state_dict.get("beta2", self.beta2)
        self.eps = state_dict.get("eps", self.eps)
        self.weight_decay = state_dict.get("weight_decay", self.weight_decay)
        self.t = state_dict.get("t", self.t)

        loaded_m = state_dict.get("m", None)
        loaded_v = state_dict.get("v", None)

        if loaded_m is not None and len(loaded_m) == len(self.params):
            self.m = [loaded_m[i].to(self.params[i].device) for i in range(len(self.params))]

        if loaded_v is not None and len(loaded_v) == len(self.params):
            self.v = [loaded_v[i].to(self.params[i].device) for i in range(len(self.params))]


def get_optimizer(model, lr=0.001, opt_type='adam', **kwargs):
    """
    Tạo optimizer tự viết.
    
    Args:
        model: nn.Module
        lr: Learning rate
        opt_type: 'adam' hoặc 'sgd'
    """
    if opt_type.lower() == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=kwargs.get('weight_decay', 0))
    elif opt_type.lower() == 'sgd':
        return SGD(
            model.parameters(), lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Optimizer '{opt_type}' chưa được hỗ trợ. Chọn 'adam' hoặc 'sgd'.")
