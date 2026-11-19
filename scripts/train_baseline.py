"""
Entry point to train and evaluate only Experiment A (Baseline CNN).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import EXPERIMENT_DEFINITIONS, run_single_experiment
from src.utils import load_config, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline CNN on CIFAR-10.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a tiny 128-sample subset for 2 epochs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    logger = setup_logger("baseline_runner", log_file="outputs/train_baseline.log")
    baseline_spec = EXPERIMENT_DEFINITIONS[0]
    summary = run_single_experiment(
        baseline_spec, config, logger, smoke_test=args.smoke_test
    )
    logger.info(f"Baseline Experiment Result: {summary}")


if __name__ == "__main__":
    main()
