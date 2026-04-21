# MLP & CNN for Image Classification — From Scratch

This project builds a **Multi-Layer Perceptron (MLP)** and **Convolutional Neural Network (CNN)** completely from scratch using PyTorch, without using pre-built modules like `torch.nn.Linear`, `torch.nn.Conv2d`, etc. The goal is to classify images on the **CIFAR-10** dataset.

## Directory Structure

```
├── data/                          # CIFAR-10 (auto-download)
├── scripts/
│   ├── main.py                    # Train MLP or CNN
│   └── test.py                    # Evaluation + visualization
├── src/
│   ├── data/
│   │   ├── transforms.py          # ToTensor, Normalize, RandomFlip, RandomCrop
│   │   ├── dataset.py             # ImageDataset + CIFAR-10 loader
│   │   └── dataloader.py          # DataLoader (batch + shuffle)
│   ├── models/
│   │   ├── activations.py         # Sigmoid, ReLU, LeakyReLU, Tanh, Softmax, GELU
│   │   ├── layers.py              # Linear, Conv2d, MaxPool2d, BatchNorm, Dropout
│   │   ├── mlp.py                 # Multi-Layer Perceptron
│   │   └── cnn.py                 # CNN (VGG-style)
│   └── training/
│       ├── losses.py              # CrossEntropyLoss
│       ├── optimizers.py          # SGD (momentum), Adam
│       ├── trainer.py             # Training loop + checkpoint
│       ├── evaluate.py            # Accuracy, Confusion Matrix, F1
│       └── visualize.py           # Training curves, confusion matrix, predictions
├── checkpoints/
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Train CNN (default)
python scripts/train.py --model cnn

# Train MLP
python scripts/train.py --model mlp

# Customize
python scripts/train.py --model cnn --epochs 30 --lr 0.001 --batch_size 128 --optimizer adam
```

## Evaluation

```bash
python scripts/test.py --model cnn
```

## Custom-Built Components

| Component | File | Description |
|-----------|------|-------------|
| ToTensor | `transforms.py` | numpy → tensor, scale [0,255] → [0,1] |
| Normalize | `transforms.py` | Normalize by mean/std |
| RandomHorizontalFlip | `transforms.py` | Data augmentation |
| RandomCrop | `transforms.py` | Random crop with padding |
| Linear | `layers.py` | y = xW^T + b |
| Conv2d | `layers.py` | Convolution 2D (im2col/unfold) |
| MaxPool2d | `layers.py` | Max pooling |
| BatchNorm1d/2d | `layers.py` | Batch normalization |
| Dropout | `layers.py` | Inverted dropout |
| CrossEntropyLoss | `losses.py` | Log-softmax + NLL |
| SGD | `optimizers.py` | SGD with momentum |
| Adam | `optimizers.py` | Adaptive moment estimation |