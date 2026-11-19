"""Run Experiments A-C over multiple seeds and aggregate their metrics."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_all_experiments
from src.reporting import summarize_repeated_results
from src.utils import load_config, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the controlled CIFAR-10 experiments over multiple seeds."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional CIFAR-10 directory override; useful when the data already exists elsewhere.",
    )
    parser.add_argument("--output-root", default="outputs/repeated")
    parser.add_argument("--figures-root", default="reports/figures/repeated")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use the small two-epoch smoke-test dataset for pipeline validation only.",
    )
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    base_config = load_config(args.config)
    output_root = Path(args.output_root)
    figures_root = Path(args.figures_root)
    per_seed_tables = []

    for seed in args.seeds:
        config = copy.deepcopy(base_config)
        config["seed"] = seed
        if args.data_dir is not None:
            config["data"]["data_dir"] = args.data_dir
        config["training"]["epochs"] = args.epochs
        config["training"]["save_dir"] = str(output_root / f"seed_{seed}")
        config["training"]["figures_dir"] = str(figures_root / f"seed_{seed}")

        logger = setup_logger(
            f"repeated_experiments_seed_{seed}",
            log_file=output_root / f"seed_{seed}" / "experiments.log",
        )
        seed_results = run_all_experiments(
            config, logger, smoke_test=args.smoke_test
        )
        per_seed_tables.append(seed_results)

    all_seed_results = pd.concat(per_seed_tables, ignore_index=True)
    aggregate_metrics = summarize_repeated_results(all_seed_results)
    output_root.mkdir(parents=True, exist_ok=True)
    all_seed_results.to_csv(output_root / "all_seed_results.csv", index=False)
    aggregate_metrics.to_csv(output_root / "aggregate_summary.csv", index=False)
    print("\nRepeated-run summary (mean and sample standard deviation):")
    print(aggregate_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
