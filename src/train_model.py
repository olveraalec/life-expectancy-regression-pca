from datetime import datetime
import json

import joblib
import duckdb

from sklearn.model_selection import train_test_split

from src.config import load_config, get_project_path
from src.logger import setup_logger
from src.modeling import train_regression_model
from src.preprocessing import preprocess_features


logger = setup_logger(__name__)


def load_modeling_data(database_path):
    """
    Load the official SQL modeling view from DuckDB.
    """
    logger.info(f"Loading modeling data from DuckDB database: {database_path}")

    conn = duckdb.connect(database_path)

    try:
        modeling_df = conn.execute("""
            SELECT *
            FROM modeling_life_expectancy;
        """).fetchdf()
    finally:
        conn.close()

    logger.info(f"Loaded modeling data with shape: {modeling_df.shape}")

    return modeling_df


def train_and_save_model() -> None:
    """
    Train and persist the Version 3 deployable model using the same
    preprocessing and modeling workflow from Version 2.
    """

    logger.info("Starting model training and persistence workflow.")

    config = load_config()

    database_path = get_project_path(config["paths"]["database"])
    model_dir = get_project_path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    if not database_path.exists():
        raise FileNotFoundError(
            f"Database not found at {database_path}. Run python run_pipeline.py first."
        )

    modeling_df = load_modeling_data(database_path)

    target = "life_expectancy"

    X = modeling_df.drop(columns=[target]).copy()
    y = modeling_df[target].copy()

    selected_features = config["model"]["selected_features"]
    log_features = config["model"]["log_features"]
    zero_to_nan_features = config["model"]["zero_to_nan_features"]

    logger.info(f"Selected raw features: {selected_features}")
    logger.info(f"Log-transform features: {log_features}")
    logger.info(f"Zero-to-NaN features: {zero_to_nan_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )

    logger.info(
        f"Created train/test split: X_train={X_train.shape}, X_test={X_test.shape}"
    )

    final_result = train_regression_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selected_features=selected_features,
        log_features=log_features,
        zero_to_nan_features=zero_to_nan_features,
        model_type="linear",
        alpha=1.0,
        scale=True,
        use_pca=False,
    )

    X_train_prep, X_test_prep, preprocessing_objects = preprocess_features(
        X_train=X_train,
        X_test=X_test,
        selected_features=selected_features,
        log_features=log_features,
        zero_to_nan_features=zero_to_nan_features,
        scale=True,
        use_pca=False,
    )

    processed_feature_names = list(X_train_prep.columns)

    logger.info(f"Processed feature names: {processed_feature_names}")

    model_bundle = {
        "model": final_result["model"],
        "preprocessing_objects": preprocessing_objects,
        "selected_features": selected_features,
        "log_features": log_features,
        "zero_to_nan_features": zero_to_nan_features,
        "processed_feature_names": processed_feature_names,
        "target": target,
    }

    model_path = model_dir / "life_expectancy_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    joblib.dump(model_bundle, model_path)

    logger.info(f"Saved model bundle to: {model_path}")

    metadata = {
        "model_name": config["api"]["model_name"],
        "model_version": config["model"]["model_version"],
        "model_type": "LinearRegression",
        "target": target,
        "selected_features": selected_features,
        "processed_feature_names": processed_feature_names,
        "log_features": log_features,
        "zero_to_nan_features": zero_to_nan_features,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_size": config["model"]["test_size"],
        "random_state": config["model"]["random_state"],
        "train_rmse": round(float(final_result["train_rmse"]), 4),
        "test_rmse": round(float(final_result["test_rmse"]), 4),
        "train_r2": round(float(final_result["train_r2"]), 4),
        "test_r2": round(float(final_result["test_r2"]), 4),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_artifact": "models/life_expectancy_model.joblib",
    }

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    logger.info(f"Saved model metadata to: {metadata_path}")
    logger.info(
        f"Model training complete. Test RMSE={metadata['test_rmse']}, "
        f"Test R2={metadata['test_r2']}"
    )

    print("Model training complete.")
    print(f"Saved model bundle to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Processed features: {processed_feature_names}")
    print(f"Test RMSE: {metadata['test_rmse']}")
    print(f"Test R2: {metadata['test_r2']}")


if __name__ == "__main__":
    train_and_save_model()
