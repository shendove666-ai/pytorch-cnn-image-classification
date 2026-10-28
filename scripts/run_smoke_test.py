"""
Fast CPU-compatible end-to-end smoke test.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_all_experiments
from src.models import BaselineCNN, ImprovedCNN, get_model_summary
from src.utils import collect_environment_info, get_device, load_config, set_seed, setup_logger


def verify_core_mechanics(logger) -> None:
    """Verify forward shape, finite loss, valid gradients, and parameter updates."""
    set_seed(42)
    device = get_device(prefer_cuda=False)
    logger.info(f"Running synthetic mechanics verification on device: {device}")

    for model_cls in (BaselineCNN, ImprovedCNN):
        model = model_cls(num_classes=10).to(device)
        summary = get_model_summary(model)
        logger.info(
            f"Initialized {summary['model_class']} with {summary['trainable_params']:,} parameters."
        )

        dummy_inputs = torch.randn(8, 3, 32, 32, device=device)
        dummy_labels = torch.randint(0, 10, (8,), device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        initial_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        model.train()
        optimizer.zero_grad()
        logits = model(dummy_inputs)

        assert logits.shape == (8, 10), f"Expected output shape (8, 10), got {logits.shape}"
        loss = criterion(logits, dummy_labels)
        assert torch.isfinite(loss).item(), "Loss is not finite."

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"
                assert torch.isfinite(param.grad).all().item(), f"Non-finite gradient in {name}"

        optimizer.step()
        updated_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        any_changed = any(
            not torch.equal(p0, p1) for p0, p1 in zip(initial_params, updated_params)
        )
        assert any_changed, f"Parameters did not update for {summary['model_class']}"
        logger.info(
            f"[PASS] {summary['model_class']}: Output shape (8, 10), "
            f"Loss={loss.item():.4f}, Gradients finite, Parameters updated."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast smoke test.")
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="Also run a 2-epoch mini-subset CIFAR-10 pipeline to verify data I/O and plots.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    logger = setup_logger("smoke_test", log_file="outputs/smoke_test.log")
    env_info = collect_environment_info()
    logger.info(f"Environment Summary: PyTorch {env_info['torch_version']} on {env_info['active_device']}")

    verify_core_mechanics(logger)

    if args.with_data:
        config = load_config(args.config)
        df = run_all_experiments(config, logger, smoke_test=True)
        logger.info("Smoke Test Data Pipeline Summary:\n" + df.to_string(index=False))

    logger.info("All smoke test checks completed successfully.")


if __name__ == "__main__":
    main()
