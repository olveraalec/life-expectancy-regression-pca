from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """
    Load project configuration from a YAML file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def get_project_path(relative_path: str) -> Path:
    """
    Convert a project-relative path from the config file into an absolute Path.
    """
    return PROJECT_ROOT / relative_path
