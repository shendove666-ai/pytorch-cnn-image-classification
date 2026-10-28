"""
Utility functions for reproducibility, environment inspection, logging, and file I/O.
"""

import json
import logging
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import yaml


def setup_logger(name: str = "cnn_study", log_file: str | None = None) -> logging.Logger:
    """Configure and return a standard console and optional file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds across Python, NumPy, and PyTorch for reproducibility.

    Note: Complete bitwise determinism across different hardware architectures
    (e.g., CPU vs. different CUDA GPUs) is not guaranteed by random seeds alone.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the active computation device, defaulting safely to CPU if CUDA is unavailable."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_environment_info() -> Dict[str, Any]:
    """Gather exact software versions and hardware availability."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "scikit_learn_version": sklearn.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else "N/A",
        "active_device": "cuda" if cuda_available else "cpu",
        "device_name": device_name,
    }


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at '{path.resolve()}'. "
            "Verify that you are running the command from the project root directory."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], output_path: str | Path) -> None:
    """Save a dictionary as a formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
