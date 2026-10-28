"""
Unit tests for dataset transformations and image denormalization.
"""

from PIL import Image
import numpy as np
import torch

from src.dataset import CIFAR10_CLASSES, denormalize_image, get_transforms


def test_cifar10_class_list() -> None:
    assert len(CIFAR10_CLASSES) == 10
    assert "airplane" in CIFAR10_CLASSES
    assert "truck" in CIFAR10_CLASSES


def test_transforms_output_shape_and_dtype() -> None:
    dummy_pil = Image.fromarray(np.uint8(np.random.randint(0, 255, (32, 32, 3))))
    train_tf, eval_tf = get_transforms(use_augmentation=True)

    tensor_train = train_tf(dummy_pil)
    tensor_eval = eval_tf(dummy_pil)

    assert tensor_train.shape == (3, 32, 32)
    assert tensor_eval.shape == (3, 32, 32)
    assert tensor_train.dtype == torch.float32
    assert tensor_eval.dtype == torch.float32


def test_denormalize_image_range() -> None:
    normalized_tensor = torch.randn(3, 32, 32)
    img_np = denormalize_image(normalized_tensor)
    assert img_np.shape == (32, 32, 3)
    assert img_np.min() >= 0.0
    assert img_np.max() <= 1.0
