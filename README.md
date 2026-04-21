# CNN Architectures and Image Classification Implementation

This project follows a learning-first path:

1. Build core CNN blocks from scratch.
2. Assemble a full image classification pipeline on CIFAR-10.
3. Add practical experimentation and evaluation workflows on top of the core understanding.

The application layer is intentionally secondary. The primary goal is to understand and reproduce CNN behavior.

## Name options

If you want to rename this repository later, suggested alternatives are:

- `CNN-Foundations-to-Application` (current README title)
- `Professional-CNN-From-Scratch`
- `CNN-Learning-First-ImageLab`
- `CNN-Core-and-Practice`
- `CNN-Architecture-Lab`

## Learning progression

### 1) Core layers (from scratch)
- `src/models/layers.py`
- `src/models/activations.py`

### 2) CNN architecture assembly
- `src/models/cnn.py`

### 3) Data and augmentation pipeline
- `src/data/dataset.py`
- `src/data/dataloader.py`
- `src/data/transforms.py`

### 4) Training and evaluation stack
- `src/training/trainer.py`
- `src/training/evaluate.py`
- `src/training/visualize.py`
- `src/factories.py`

### 5) Scripted experiments and outputs
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/test.py` (smoke test)
- `configs/default.yaml`

## Repository structure

```text
.
├── configs/
│   └── default.yaml
├── data/                            # CIFAR-10 auto-download location
├── outputs/                         # Experiment artifacts
├── scripts/
│   ├── train.py                     # Train using config + CLI overrides
│   ├── evaluate.py                  # Metrics, reports, and plots
│   └── test.py                      # Smoke test pipeline
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── utils/
│   │   └── config.py
│   └── factories.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

### Train

```bash
# Train with defaults in configs/default.yaml
python scripts/train.py

# Train with explicit config path
python scripts/train.py --config configs/default.yaml

# Override selected options directly from CLI
python scripts/train.py --epochs 10 --batch_size 32 --lr 0.0005 --attention

# Generic deep override
python scripts/train.py --override training.optimizer=sgd --override project.output_dir=outputs/sgd_run
```

### Evaluate

```bash
# Evaluate using output_dir/checkpoints/best_checkpoint.pt
python scripts/evaluate.py

# Evaluate a specific checkpoint file
python scripts/evaluate.py --checkpoint last_checkpoint.pt --output_dir outputs/cnn_baseline
```

### Smoke test

```bash
python scripts/test.py --epochs 1 --num_samples 256 --batch_size 32 --output_dir outputs/smoke_test
```

## Configuration

Main settings live in `configs/default.yaml`:

- `project`: experiment identity, output directory, random seed, data directory
- `model`: attention, dropout, reduction ratio, class count
- `data`: batch size, optional sample cap, augmentation options
- `training`: optimizer, learning rate, epochs, resume, logging interval
- `evaluation`: checkpoint selection and plotting behavior

## Outputs

Training saves to `project.output_dir`:

- `resolved_config.json`
- `checkpoints/normalization_stats.json`
- `checkpoints/last_checkpoint.pt`
- `checkpoints/best_checkpoint.pt`
- `checkpoints/train_history.json`
- `checkpoints/train_history.csv`

Evaluation saves to `project.output_dir/evaluation`:

- `metrics.json`
- `classification_report.csv`
- `confusion_matrix.png`
- `sample_predictions.png`
- `training_history.png`

## Reproducibility checklist

- Fix random seed in config (`project.seed`).
- Keep `resolved_config.json` with every run.
- Store checkpoints and history artifacts per experiment directory.
- Prefer one output directory per experiment to avoid artifact mixing.