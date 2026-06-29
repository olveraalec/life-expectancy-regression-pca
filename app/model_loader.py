import json
from pathlib import Path

import joblib

from src.config import load_config, get_project_path


def get_model_paths():
    """
    Build model artifact and metadata paths from the project config.
    """
    config = load_config()

    model_dir = get_project_path(config["paths"]["model_dir"])
    model_path = model_dir / "life_expectancy_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    return model_path, metadata_path


def load_model_bundle():
    """
    Load the saved model bundle from disk.

    The model bundle contains:
    - fitted regression model
    - fitted preprocessing objects
    - selected features
    - log-transform settings
    - zero-to-NaN settings
    - processed feature names
    - target name
    """
    model_path, _ = get_model_paths()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run python -m src.train_model first."
        )

    model_bundle = joblib.load(model_path)

    required_keys = [
        "model",
        "preprocessing_objects",
        "selected_features",
        "log_features",
        "zero_to_nan_features",
        "processed_feature_names",
        "target",
    ]

    missing_keys = [key for key in required_keys if key not in model_bundle]

    if missing_keys:
        raise ValueError(f"Model bundle is missing required keys: {missing_keys}")

    return model_bundle


def load_model_metadata():
    """
    Load model metadata from disk.
    """
    _, metadata_path = get_model_paths()

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {metadata_path}. "
            "Run python -m src.train_model first."
        )

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    return metadata


def load_model_and_metadata():
    """
    Load both the model bundle and model metadata.
    """
    model_bundle = load_model_bundle()
    metadata = load_model_metadata()

    return model_bundle, metadata
