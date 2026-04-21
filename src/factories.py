from typing import Dict, Tuple

from src.data.dataloader import get_dataloader
from src.data.dataset import ImageDataset
from src.data.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
from src.models.cnn import build_cnn
from src.training.losses import get_loss_function
from src.training.optimizers import get_optimizer


def build_transforms(cifar10_mean, cifar10_std, data_cfg: Dict):
    augment_cfg = data_cfg.get("augment", {})
    flip_prob = float(augment_cfg.get("random_flip_prob", 0.5))
    crop_padding = int(augment_cfg.get("random_crop_padding", 4))

    train_transform = Compose(
        [
            ToTensor(),
            RandomHorizontalFlip(p=flip_prob),
            RandomCrop(32, padding=crop_padding),
            Normalize(cifar10_mean, cifar10_std),
        ]
    )
    eval_transform = Compose([ToTensor(), Normalize(cifar10_mean, cifar10_std)])
    return train_transform, eval_transform


def build_loaders(
    train_images,
    train_labels,
    test_images,
    test_labels,
    train_transform,
    eval_transform,
    batch_size: int,
):
    train_dataset = ImageDataset(train_images, train_labels, transform=train_transform)
    test_dataset = ImageDataset(test_images, test_labels, transform=eval_transform)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_training_components(model_cfg: Dict, training_cfg: Dict):
    model = build_cnn(
        in_channels=3,
        num_classes=int(model_cfg.get("num_classes", 10)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.5)),
        use_attention=bool(model_cfg.get("use_attention", False)),
        attention_reduction=int(model_cfg.get("attention_reduction", 16)),
    )
    optimizer = get_optimizer(
        model,
        lr=float(training_cfg.get("lr", 0.001)),
        opt_type=str(training_cfg.get("optimizer", "adam")),
    )
    criterion = get_loss_function()
    return model, optimizer, criterion
