"""
CIFAR-10 dataset loading, deterministic train/validation splitting, preprocessing,
and sample visualization utilities.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def get_transforms(use_augmentation: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create preprocessing pipelines for training and evaluation.

    Args:
        use_augmentation: If True, applies RandomCrop and RandomHorizontalFlip to training images.

    Returns:
        Tuple of (train_transform, eval_transform).
    """
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    use_augmentation: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    smoke_test: bool = False,
    smoke_subset_size: int = 128,
) -> Dict[str, DataLoader]:
    """
    Build deterministic train, validation, and test DataLoaders for CIFAR-10.

    The official test set is strictly isolated and never used during training or validation.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

    train_transform, eval_transform = get_transforms(use_augmentation=use_augmentation)

    try:
        train_base = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        val_base = datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=eval_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=eval_transform
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download or load CIFAR-10 in '{data_dir}'. "
            "Check your internet connection or manually place 'cifar-10-python.tar.gz' "
            f"into '{Path(data_dir).resolve()}'. Original error: {exc}"
        ) from exc

    num_total_train = len(train_base)
    indices = list(range(num_total_train))

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_idx = int(np.floor(val_split * num_total_train))
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]

    if smoke_test:
        train_indices = train_indices[:smoke_subset_size]
        val_indices = val_indices[: max(32, smoke_subset_size // 2)]
        test_indices = list(range(min(len(test_dataset), max(32, smoke_subset_size // 2))))
        test_dataset = Subset(test_dataset, test_indices)

    train_subset = Subset(train_base, train_indices)
    val_subset = Subset(val_base, val_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def inspect_batch(dataloader: DataLoader) -> Dict[str, object]:
    """Extract and validate a single batch from a DataLoader."""
    images, labels = next(iter(dataloader))
    return {
        "image_shape": tuple(images.shape),
        "image_dtype": str(images.dtype),
        "label_shape": tuple(labels.shape),
        "label_dtype": str(labels.dtype),
        "min_label": int(labels.min().item()),
        "max_label": int(labels.max().item()),
        "min_pixel": float(images.min().item()),
        "max_pixel": float(images.max().item()),
    }


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized (C, H, W) tensor back to an (H, W, C) float array in [0, 1]."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(CIFAR10_MEAN)
    std = np.array(CIFAR10_STD)
    img = std * img + mean
    return np.clip(img, 0.0, 1.0)


def save_sample_grid(
    dataloader: DataLoader,
    output_path: str | Path,
    num_images: int = 16,
) -> None:
    """Save a grid of sample images with their ground-truth class labels."""
    images, labels = next(iter(dataloader))
    count = min(num_images, images.size(0))
    cols = 4
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes_flat = np.array(axes).reshape(-1)

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < count:
            img_np = denormalize_image(images[idx])
            label_name = CIFAR10_CLASSES[int(labels[idx].item())]
            ax.imshow(img_np)
            ax.set_title(f"True: {label_name}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
