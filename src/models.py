"""Two convolutional classifiers trained from scratch on 32 × 32 RGB images."""

from typing import Dict

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Two pooling stages produce 64 × 8 × 8 features for the logit head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits of shape [batch_size, num_classes]."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        logits = self.fc2(x)
        return logits


class ImprovedCNN(nn.Module):
    """A third pooling stage reduces the logit head to 128 × 4 × 4 inputs."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits of shape [batch_size, num_classes]."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """Construct the named architecture with freshly initialized weights."""
    model_key = model_name.strip().lower()
    if model_key == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_key == "improved":
        return ImprovedCNN(num_classes=num_classes)
    raise ValueError(
        f"Unknown model_name '{model_name}'. Supported options: 'baseline', 'improved'."
    )


def get_model_summary(model: nn.Module) -> Dict[str, object]:
    """Compute total and trainable parameter counts and layer-wise breakdown."""
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_parameters_by_name = {
        name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        "model_class": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "layer_parameter_breakdown": trainable_parameters_by_name,
    }
