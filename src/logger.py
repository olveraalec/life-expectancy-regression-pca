import logging
from pathlib import Path

from src.config import load_config, get_project_path


def setup_logger(name: str = "life_expectancy_project") -> logging.Logger:
    """
    Create and configure a project logger.

    Logs are written to:
    - terminal/console
    - logs/project.log
    """

    config = load_config()

    log_dir = get_project_path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "project.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
