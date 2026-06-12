"""
export_model_outputs.py

Purpose:
Train and evaluate regression models, tune Ridge/Lasso alpha values,
and export model outputs for documentation and Power BI.

This script:
1. Loads the SQL-generated modeling dataset from DuckDB.
2. Creates a train/test split.
3. Tunes Ridge and Lasso alpha values using cross-validation.
4. Compares Linear Regression, Ridge, Lasso, and PCA + Linear Regression.
5. Saves model comparison and tuning outputs.
6. Saves final model predictions for Power BI.
"""

# Import numpy for numeric operations.
import numpy as np

# Import pandas for DataFrame handling.
import pandas as pd

# Import DuckDB to query the local database.
import duckdb

# Import Path for clean file path handling.
from pathlib import Path

# Import train_test_split to create the modeling split.
from sklearn.model_selection import train_test_split

# Import modeling functions from our project modules.
from src.modeling import (
    train_regression_model,
    cross_validate_regression_model,
    evaluate_model_workflow
)


# Define the project root.
# This file lives in src/, so parents[1] moves up to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the database path.
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "life_expectancy.duckdb"

# Define the processed data output folder.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# Define the optimized 6-feature set from the original selected-feature model.
OPTIMIZED_FEATURES_6 = [
    "adult_mortality",
    "income_composition_of_resources",
    "schooling",
    "bmi",
    "gdp",
    "hiv_aids"
]


# Define the ordered 10-feature set used for PCA modeling.
ORDERED_FEATURES = [
    "adult_mortality",
    "income_composition_of_resources",
    "schooling",
    "bmi",
    "gdp",
    "hiv_aids",
    "percentage_expenditure",
    "diphtheria",
    "polio",
    "alcohol"
]


# Define features where zero should be treated as missing.
ZERO_TO_NAN_FEATURES = [
    "gdp",
    "percentage_expenditure"
]


# Define skewed features to log-transform.
LOG_FEATURES = [
    "adult_mortality",
    "gdp",
    "hiv_aids",
    "percentage_expenditure"
]


def load_modeling_data():
    """
    Load the official modeling dataset from the DuckDB modeling view.

    Returns
    -------
    pandas.DataFrame:
        Modeling dataset from SQL.
    """

    # Connect to the local DuckDB database.
    conn = duckdb.connect(DATABASE_PATH)

    try:
        # Pull the modeling view into pandas.
        modeling_df = conn.execute("""
            SELECT *
            FROM modeling_life_expectancy;
        """).fetchdf()

    finally:
        # Close the database connection.
        conn.close()

    # Return the modeling dataset.
    return modeling_df


def tune_alpha_grid(X_train, y_train, model_type, alpha_grid):
    """
    Tune alpha values for Ridge or Lasso using repeated K-fold CV.

    Parameters
    ----------
    X_train:
        Training predictors.

    y_train:
        Training target values.

    model_type:
        "ridge" or "lasso".

    alpha_grid:
        List of alpha values to test.

    Returns
    -------
    pandas.DataFrame:
        Alpha tuning results sorted by CV RMSE.
    """

    # Create an empty list to store tuning results.
    tuning_results = []

    # Loop through each alpha value.
    for alpha in alpha_grid:

        # Cross-validate the model with the current alpha.
        cv_result = cross_validate_regression_model(
            X=X_train,
            y=y_train,
            selected_features=OPTIMIZED_FEATURES_6,
            log_features=LOG_FEATURES,
            zero_to_nan_features=ZERO_TO_NAN_FEATURES,
            model_type=model_type,
            alpha=alpha,
            scale=True,
            use_pca=False,
            k=5,
            n_repeats=3,
            random_state=42
        )

        # Store the current alpha and CV results.
        tuning_results.append({
            "model": model_type,
            "alpha": alpha,
            "cv_rmse_mean": cv_result["cv_rmse_mean"],
            "cv_rmse_std": cv_result["cv_rmse_std"],
            "cv_r2_mean": cv_result["cv_r2_mean"],
            "cv_r2_std": cv_result["cv_r2_std"]
        })

    # Convert results into a DataFrame.
    tuning_df = pd.DataFrame(tuning_results)

    # Sort by CV RMSE because lower RMSE is better.
    tuning_df = tuning_df.sort_values("cv_rmse_mean").reset_index(drop=True)

    # Return the sorted tuning results.
    return tuning_df


def create_predictions_table(X_test, y_test, y_test_pred):
    """
    Create a dashboard-ready predictions table

    Parameters
    ----------
    X_test:
        Original test predictor DataFrame

    y_test:
        Actual target values

    y_test_pred:
        Predicted target values

    Returns
    -------
    pandas.DataFrame:
        Predictions table with actuals, predictions, residuals, and absolute errors
    """

    # Keep context columns for dashboard filtering
    predictions_df = X_test[["country", "year", "status"]].copy()

    # Add actual life expectancy
    predictions_df["actual_life_expectancy"] = y_test.values

    # Add predicted life expectancy
    predictions_df["predicted_life_expectancy"] = y_test_pred

    # Calculate residuals
    predictions_df["residual"] = (
        predictions_df["actual_life_expectancy"]
        - predictions_df["predicted_life_expectancy"]
    )

    # Calculate absolute error
    predictions_df["absolute_error"] = predictions_df["residual"].abs()

    # Round numeric columns for cleaner dashboard display
    numeric_cols = [
        "actual_life_expectancy",
        "predicted_life_expectancy",
        "residual",
        "absolute_error"
    ]

    for col in numeric_cols:
        predictions_df[col] = predictions_df[col].round(2)

    # Return the predictions table
    return predictions_df


def export_model_outputs():
    """
    Run the full model-output export workflow
    """

    # Make sure the processed data folder exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load modeling data from SQL
    modeling_df = load_modeling_data()

    # Separate predictors and target
    X = modeling_df.drop(columns=["life_expectancy"]).copy()
    y = modeling_df["life_expectancy"].copy()

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=42
    )

    # Define alpha grids
    ridge_alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    lasso_alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]

    # Tune Ridge
    ridge_tuning_df = tune_alpha_grid(
        X_train=X_train,
        y_train=y_train,
        model_type="ridge",
        alpha_grid=ridge_alphas
    )

    # Tune Lasso
    lasso_tuning_df = tune_alpha_grid(
        X_train=X_train,
        y_train=y_train,
        model_type="lasso",
        alpha_grid=lasso_alphas
    )

    # Save tuning outputs
    ridge_tuning_df.to_csv(PROCESSED_DIR / "ridge_alpha_tuning.csv", index=False)
    lasso_tuning_df.to_csv(PROCESSED_DIR / "lasso_alpha_tuning.csv", index=False)

    # Select best alpha values
    best_ridge_alpha = ridge_tuning_df.iloc[0]["alpha"]
    best_lasso_alpha = lasso_tuning_df.iloc[0]["alpha"]

    # Create final model comparison results.
    model_results = []

    # Linear Regression
    model_results.append(
        evaluate_model_workflow(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_cv=X_train,
            y_cv=y_train,
            selected_features=OPTIMIZED_FEATURES_6,
            log_features=LOG_FEATURES,
            zero_to_nan_features=ZERO_TO_NAN_FEATURES,
            model_name="Linear Regression",
            model_type="linear",
            alpha=1.0,
            scale=True,
            use_pca=False,
            setup_description="6 selected features"
        )
    )

    # Ridge Regression
    model_results.append(
        evaluate_model_workflow(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_cv=X_train,
            y_cv=y_train,
            selected_features=OPTIMIZED_FEATURES_6,
            log_features=LOG_FEATURES,
            zero_to_nan_features=ZERO_TO_NAN_FEATURES,
            model_name="Ridge Regression",
            model_type="ridge",
            alpha=best_ridge_alpha,
            scale=True,
            use_pca=False,
            setup_description=f"6 selected features, tuned alpha={best_ridge_alpha}"
        )
    )

    # Lasso Regression
    model_results.append(
        evaluate_model_workflow(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_cv=X_train,
            y_cv=y_train,
            selected_features=OPTIMIZED_FEATURES_6,
            log_features=LOG_FEATURES,
            zero_to_nan_features=ZERO_TO_NAN_FEATURES,
            model_name="Lasso Regression",
            model_type="lasso",
            alpha=best_lasso_alpha,
            scale=True,
            use_pca=False,
            setup_description=f"6 selected features, tuned alpha={best_lasso_alpha}"
        )
    )

    # PCA + Linear Regression
    model_results.append(
        evaluate_model_workflow(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_cv=X_train,
            y_cv=y_train,
            selected_features=ORDERED_FEATURES,
            log_features=LOG_FEATURES,
            zero_to_nan_features=ZERO_TO_NAN_FEATURES,
            model_name="PCA + Linear Regression",
            model_type="linear",
            alpha=1.0,
            scale=True,
            use_pca=True,
            n_components=3,
            setup_description="10 selected features reduced to 3 PCA components"
        )
    )

    # Convert final results to a DataFrame
    model_comparison_df = pd.DataFrame(model_results)

    # Round numeric columns for readability
    numeric_cols = [
        "alpha",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_r2_mean",
        "cv_r2_std",
        "test_rmse",
        "test_r2",
        "train_rmse",
        "train_r2"
    ]

    for col in numeric_cols:
        if col in model_comparison_df.columns:
            model_comparison_df[col] = model_comparison_df[col].round(4)

    # Save final model comparison
    model_comparison_df.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    # Train final selected model: Linear Regression
    final_model_result = train_regression_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selected_features=OPTIMIZED_FEATURES_6,
        log_features=LOG_FEATURES,
        zero_to_nan_features=ZERO_TO_NAN_FEATURES,
        model_type="linear",
        alpha=1.0,
        scale=True,
        use_pca=False
    )

    # Get the fitted final model
    final_model = final_model_result["model"]

    # Recreate processed X_test using the final model preprocessing objects
    from src.preprocessing import preprocess_features

    X_train_prep, X_test_prep, _ = preprocess_features(
        X_train=X_train,
        X_test=X_test,
        selected_features=OPTIMIZED_FEATURES_6,
        log_features=LOG_FEATURES,
        zero_to_nan_features=ZERO_TO_NAN_FEATURES,
        scale=True,
        use_pca=False
    )

    # Generate test predictions
    y_test_pred = final_model.predict(X_test_prep)

    # Create prediction table
    predictions_df = create_predictions_table(
        X_test=X_test,
        y_test=y_test,
        y_test_pred=y_test_pred
    )

    # Save prediction output
    predictions_df.to_csv(PROCESSED_DIR / "predictions.csv", index=False)

    # Print a clean summary
    print("Model outputs exported successfully.")
    print(f"Best Ridge alpha: {best_ridge_alpha}")
    print(f"Best Lasso alpha: {best_lasso_alpha}")
    print(f"Saved: {PROCESSED_DIR / 'ridge_alpha_tuning.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'lasso_alpha_tuning.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'model_comparison.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'predictions.csv'}")


# This block runs only when the script is executed directly
if __name__ == "__main__":
    export_model_outputs()