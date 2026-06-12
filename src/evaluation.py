"""
evaluation.py

Purpose:
Reusable evaluation functions for regression models

This module contains helper functions for:
1. Calculating RMSE
2. Calculating standard regression metrics
3. Creating clean model comparison tables
"""

# Import numpy
import numpy as np

# Import pandas
import pandas as pd

# Import scikit
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error

    RMSE measures the average size of prediction errors
    Lower RMSE means better predictive accuracy

    Parameters
    ----------
    y_true:
        Actual target values

    y_pred:
        Predicted target values

    Returns
    -------
    float:
        Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    """
    Calculate common regression metrics

    Parameters
    ----------
    y_true:
        Actual target values

    y_pred:
        Predicted target values

    Returns
    -------
    dict:
        Dictionary containing RMSE and R-squared
    """

    # Calculate RMSE using the helper function above
    model_rmse = rmse(y_true, y_pred)

    # Calculate R-squared using scikit-learn
    # R-squared measures the proportion of variance explained by the model
    model_r2 = r2_score(y_true, y_pred)

    # Return both metrics
    return {
        "rmse": model_rmse,
        "r2": model_r2
    }


def train_test_metrics(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a fitted model on both training and test data.

    Parameters
    ----------
    model:
        A fitted scikit-learn regression model.

    X_train:
        Training predictors.

    X_test:
        Test predictors.

    y_train:
        Training target values.

    y_test:
        Test target values.

    Returns
    -------
    dict:
        Dictionary containing train and test RMSE/R².
    """

    # Generate predictions on the training set
    y_train_pred = model.predict(X_train)

    # Generate predictions on the test set
    y_test_pred = model.predict(X_test)

    # Calculate training metrics
    train_results = regression_metrics(y_train, y_train_pred)

    # Calculate test metrics
    test_results = regression_metrics(y_test, y_test_pred)

    # Return all results
    return {
        "train_rmse": train_results["rmse"],
        "test_rmse": test_results["rmse"],
        "train_r2": train_results["r2"],
        "test_r2": test_results["r2"]
    }


def create_model_comparison_table(results):
    """
    Create a clean model comparison table from a list of result dictionaries.

    Parameters
    ----------
    results:
        List of dictionaries. Each dictionary should contain model name,
        setup description, CV metrics, test metrics, and model notes.

    Returns
    -------
    pandas.DataFrame:
        Clean comparison table.
    """

    # Convert the list of dictionaries into a pandas DataFrame
    comparison_df = pd.DataFrame(results)

    # Define numeric columns that are rounded
    numeric_cols = [
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_r2_mean",
        "cv_r2_std",
        "test_rmse",
        "test_r2"
    ]

    # Round numeric columns only if they exist in the table
    for col in numeric_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(4)

    # Return the cleaned comparison table
    return comparison_df