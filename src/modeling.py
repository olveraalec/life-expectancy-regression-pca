"""
modeling.py

Purpose:
Reusable model training and validation functions for the life expectancy project

This module handles:
1. Training standard linear regression
2. Training Ridge regression
3. Training Lasso regression
4. Training PCA + linear regression
5. Running repeated K-fold cross-validation
6. Creating model result dictionaries for comparison
"""

# Imports
import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import RepeatedKFold

from src.preprocessing import preprocess_features

from src.evaluation import train_test_metrics


def get_model(model_type="linear", alpha=1.0, max_iter=10000):
    """
    Create a regression model based on the requested model type

    Parameters
    ----------
    model_type:
        Type of regression model to create
        Options:
        - "linear"
        - "ridge"
        - "lasso"

    alpha:
        Regularization strength for Ridge and Lasso
        Larger alpha means stronger regularization

    max_iter:
        Maximum number of iterations for Lasso
        Lasso sometimes needs more iterations to converge

    Returns
    -------
    scikit-learn model:
        Unfitted regression model.
    """

    # Standard linear regression has no regularization
    if model_type == "linear":
        return LinearRegression()

    # Ridge regression uses L2 regularization
    # This shrinks coefficients but usually does not set them exactly to zero
    elif model_type == "ridge":
        return Ridge(alpha=alpha)

    # Lasso regression uses L1 regularization
    # This can shrink some coefficients exactly to zero, acting like feature selection.
    elif model_type == "lasso":
        return Lasso(alpha=alpha, max_iter=max_iter)

    # If the user gives an unsupported model type, raise a clear error
    else:
        raise ValueError("model_type must be one of: 'linear', 'ridge', or 'lasso'")


def train_regression_model(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_features,
    log_features,
    zero_to_nan_features,
    model_type="linear",
    alpha=1.0,
    scale=True,
    use_pca=False,
    n_components=None
):
    """
    Train one regression model using the shared preprocessing pipeline

    Parameters
    ----------
    X_train, X_test:
        Training and test predictor DataFrames

    y_train, y_test:
        Training and test target values

    selected_features:
        List of features to use

    log_features:
        List of features to log-transform

    zero_to_nan_features:
        List of features where zero should be treated as missing.

    model_type:
        Regression model type: "linear", "ridge", or "lasso".

    alpha:
        Regularization strength for Ridge/Lasso

    scale:
        Whether to standardize features

    use_pca:
        Whether to apply PCA after preprocessing

    n_components:
        Number of PCA components if use_pca=True

    Returns
    -------
    dict:
        Dictionary containing model, preprocessing objects, metrics, and setup information
    """

    # Preprocess the training and test predictors
    # This handles encoding, feature selection, zero-to-NaN, log transform, imputation, scaling, and optional PCA.
    X_train_prep, X_test_prep, preprocessing_objects = preprocess_features(
        X_train=X_train,
        X_test=X_test,
        selected_features=selected_features,
        log_features=log_features,
        zero_to_nan_features=zero_to_nan_features,
        scale=scale,
        use_pca=use_pca,
        n_components=n_components
    )

    # Create the requested regression model
    model = get_model(
        model_type=model_type,
        alpha=alpha
    )

    # Fit the model on the preprocessed training data
    model.fit(X_train_prep, y_train)

    # Evaluate the fitted model on train and test data
    metrics = train_test_metrics(
        model=model,
        X_train=X_train_prep,
        X_test=X_test_prep,
        y_train=y_train,
        y_test=y_test
    )

    # Return everything useful in one dictionary.
    return {
        "model": model,
        "preprocessing_objects": preprocessing_objects,
        "model_type": model_type,
        "alpha": alpha,
        "use_pca": use_pca,
        "n_components": n_components,
        "selected_features": selected_features,
        "train_rmse": metrics["train_rmse"],
        "test_rmse": metrics["test_rmse"],
        "train_r2": metrics["train_r2"],
        "test_r2": metrics["test_r2"]
    }


def cross_validate_regression_model(
    X,
    y,
    selected_features,
    log_features,
    zero_to_nan_features,
    model_type="linear",
    alpha=1.0,
    scale=True,
    use_pca=False,
    n_components=None,
    k=5,
    n_repeats=3,
    random_state=42
):
    """
    Evaluate a regression model using repeated K-fold cross-validation

    Parameters
    ----------
    X:
        Predictor DataFrame

    y:
        Target values

    selected_features:
        List of features to use

    log_features:
        List of features to log-transform

    zero_to_nan_features:
        List of features where zero should be treated as missing

    model_type:
        Regression model type: "linear", "ridge", or "lasso"

    alpha:
        Regularization strength for Ridge/Lasso

    scale:
        Whether to standardize features

    use_pca:
        Whether to apply PCA

    n_components:
        Number of PCA components if use_pca=True

    k:
        Number of folds

    n_repeats:
        Number of times to repeat K-fold cross-validation

    random_state:
        Random seed for reproducibility

    Returns
    -------
    dict:
        Cross-validation mean and standard deviation metrics
    """

    # Create the repeated K-fold splitter
    rkf = RepeatedKFold(
        n_splits=k,
        n_repeats=n_repeats,
        random_state=random_state
    )

    # Create empty lists to store validation metrics from each fold
    cv_rmse = []
    cv_r2 = []

    # Loop through each train/validation split
    for train_idx, val_idx in rkf.split(X):

        # Create the fold-specific training predictors
        X_subtrain = X.iloc[train_idx]

        # Create the fold-specific validation predictors
        X_val = X.iloc[val_idx]

        # Create the fold-specific training target
        y_subtrain = y.iloc[train_idx]

        # Create the fold-specific validation target.
        y_val = y.iloc[val_idx]

        # Train and evaluate the model on this fold
        fold_result = train_regression_model(
            X_train=X_subtrain,
            X_test=X_val,
            y_train=y_subtrain,
            y_test=y_val,
            selected_features=selected_features,
            log_features=log_features,
            zero_to_nan_features=zero_to_nan_features,
            model_type=model_type,
            alpha=alpha,
            scale=scale,
            use_pca=use_pca,
            n_components=n_components
        )

        # Store fold validation RMSE
        cv_rmse.append(fold_result["test_rmse"])

        # Store fold validation R-squared
        cv_r2.append(fold_result["test_r2"])

    # Return summary statistics across all folds
    return {
        "cv_rmse_mean": np.mean(cv_rmse),
        "cv_rmse_std": np.std(cv_rmse),
        "cv_r2_mean": np.mean(cv_r2),
        "cv_r2_std": np.std(cv_r2)
    }


def evaluate_model_workflow(
    X_train,
    X_test,
    y_train,
    y_test,
    X_cv,
    y_cv,
    selected_features,
    log_features,
    zero_to_nan_features,
    model_name,
    model_type="linear",
    alpha=1.0,
    scale=True,
    use_pca=False,
    n_components=None,
    setup_description=None
):
    """
    Train one final model and run cross-validation for comparison

    This function combines:
    1. Final train/test evaluation
    2. Repeated K-fold cross-validation
    3. A clean result dictionary for model comparison

    Parameters
    ----------
    X_train, X_test, y_train, y_test:
        Train/test split used for final model evaluation

    X_cv, y_cv:
        Data used for cross-validation
        Usually this should be the training data only

    selected_features:
        List of selected modeling features

    log_features:
        List of skewed features to log-transform

    zero_to_nan_features:
        List of features where zero should be treated as missing

    model_name:
        Human-readable model name

    model_type:
        "linear", "ridge", or "lasso"

    alpha:
        Regularization strength

    scale:
        Whether to standardize features

    use_pca:
        Whether to apply PCA

    n_components:
        Number of PCA components if use_pca=True

    setup_description:
        Short explanation of model setup

    Returns
    -------
    dict:
        Clean model comparison result
    """

    # Train the final model on the train/test split
    final_result = train_regression_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selected_features=selected_features,
        log_features=log_features,
        zero_to_nan_features=zero_to_nan_features,
        model_type=model_type,
        alpha=alpha,
        scale=scale,
        use_pca=use_pca,
        n_components=n_components
    )

    # Run repeated K-fold cross-validation
    cv_result = cross_validate_regression_model(
        X=X_cv,
        y=y_cv,
        selected_features=selected_features,
        log_features=log_features,
        zero_to_nan_features=zero_to_nan_features,
        model_type=model_type,
        alpha=alpha,
        scale=scale,
        use_pca=use_pca,
        n_components=n_components
    )

    # Return a clean comparison-ready dictionary
    return {
        "model": model_name,
        "setup": setup_description,
        "model_type": model_type,
        "alpha": alpha,
        "use_pca": use_pca,
        "n_components": n_components,
        "cv_rmse_mean": cv_result["cv_rmse_mean"],
        "cv_rmse_std": cv_result["cv_rmse_std"],
        "cv_r2_mean": cv_result["cv_r2_mean"],
        "cv_r2_std": cv_result["cv_r2_std"],
        "test_rmse": final_result["test_rmse"],
        "test_r2": final_result["test_r2"],
        "train_rmse": final_result["train_rmse"],
        "train_r2": final_result["train_r2"]
    }