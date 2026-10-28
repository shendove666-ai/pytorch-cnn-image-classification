"""
Environment inspection script.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import collect_environment_info, setup_logger


def main() -> None:
    logger = setup_logger("env_check")
    info = collect_environment_info()
    logger.info("Environment and Dependency Inspection Report:")
    for key, value in info.items():
        logger.info(f"  {key:<24}: {value}")


if __name__ == "__main__":
    main()
