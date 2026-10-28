"""
Unified entry point to execute all three controlled experiments (Experiments A, B, and C).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_all_experiments
from src.utils import load_config, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled CIFAR-10 CNN experiments (A, B, and C)."
    )
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
        help="Optional override for number of training epochs per experiment.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 2-epoch smoke test on 128 samples per experiment.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    logger = setup_logger("experiments_runner", log_file="outputs/experiments.log")
    df_results = run_all_experiments(config, logger, smoke_test=args.smoke_test)
    logger.info("\nFinal Comparison Table:\n" + df_results.to_string(index=False))


if __name__ == "__main__":
    main()
