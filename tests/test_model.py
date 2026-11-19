"""
Unit tests for BaselineCNN and ImprovedCNN forward pass, gradients, and checkpoint I/O.
"""

from pathlib import Path
import torch
import torch.nn as nn

from src.models import BaselineCNN, ImprovedCNN, build_model, get_model_summary


def test_baseline_cnn_forward_and_params() -> None:
    model = BaselineCNN(num_classes=10)
    summary = get_model_summary(model)
    assert summary["trainable_params"] == 1_070_794

    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_improved_cnn_forward_and_params() -> None:
    model = ImprovedCNN(num_classes=10)
    summary = get_model_summary(model)
    assert summary["trainable_params"] == 620_586

    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_backward_and_parameter_update(tmp_path: Path) -> None:
    model = build_model("baseline", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    x = torch.randn(6, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

    before_w = model.fc2.weight.clone().detach()
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    after_w = model.fc2.weight.clone().detach()

    assert not torch.equal(before_w, after_w)

    ckpt_file = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt_file)
    reloaded = build_model("baseline", num_classes=10)
    reloaded.load_state_dict(torch.load(ckpt_file, weights_only=True))
    assert torch.equal(reloaded.fc2.weight, after_w)
