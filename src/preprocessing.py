"""
preprocessing.py

Purpose:
Reusable preprocessing functions for the life expectancy regression project

This module handles:
1. Train/test feature preparation
2. Categorical encoding
3. Feature selection
4. Zero-to-missing replacement
5. Log transformations
6. KNN imputation
7. Feature scaling
8. Optional PCA transformation
"""

# imports
import numpy as np

import pandas as pd

from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

def clean_column_names(df):
    """
    Standardize column names in a pandas DataFrame

    This function:
    - strips leading/trailing whitespace
    - converts names to lowercase
    - replaces spaces with underscores
    - replaces slashes with underscores
    - replaces hyphens with underscores

    Parameters
    ----------
    df:
        Input pandas DataFrame

    Returns
    -------
    pandas.DataFrame:
        Copy of the DataFrame with cleaned column names
    """

    # Make a copy
    df = df.copy()

    # Clean all column names using pandas string methods
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # Return the cleaned DataFrame
    return df


def replace_zero_with_nan(X_train, X_test, zero_to_nan_features):
    """
    Replace zero values with NaN for selected features

    Some variables use 0 as a placeholder for missing or unavailable data
    Replacing those zeros with NaN allows the imputer to handle them properly

    Parameters
    ----------
    X_train:
        Training predictor DataFrame

    X_test:
        Test predictor DataFrame

    zero_to_nan_features:
        List of columns where zeros should be treated as missing values

    Returns
    -------
    tuple:
        Updated X_train and X_test DataFrames
    """

    # Copy the inputs
    X_train = X_train.copy()
    X_test = X_test.copy()

    # If no features are provided, return the data unchanged
    if zero_to_nan_features is None:
        return X_train, X_test

    # Loop through each column that should have zeros replaced
    for col in zero_to_nan_features:

        # Replace zeros with NaN in the training set if the column exists
        if col in X_train.columns:
            X_train[col] = X_train[col].replace(0, np.nan)

        # Replace zeros with NaN in the test set if the column exists
        if col in X_test.columns:
            X_test[col] = X_test[col].replace(0, np.nan)

    # Return the updated data
    return X_train, X_test


def apply_log_transform(X_train, X_test, log_features):
    """
    Apply log transformation to selected features

    Log transforms reduce skewness in heavily right-skewed variables
    This is useful for variables such as GDP, HIV/AIDS, and percentage expenditure

    Parameters
    ----------
    X_train:
        Training predictor DataFrame

    X_test:
        Test predictor DataFrame

    log_features:
        List of columns to log-transform

    Returns
    -------
    tuple:
        Updated X_train and X_test DataFrames
    """

    # Copy the inputs
    X_train = X_train.copy()
    X_test = X_test.copy()

    # If no log features are provided, return the data unchanged
    if log_features is None:
        return X_train, X_test

    # Loop through each column that should be transformed
    for col in log_features:

        # Apply log transform to the training column if it exists
        # np.log1p(x) means log(1 + x), which is safer when values can be zero
        if col in X_train.columns:
            X_train[col] = np.log1p(X_train[col])

        # Apply the same transform to the test column if it exists
        if col in X_test.columns:
            X_test[col] = np.log1p(X_test[col])

    # Return the transformed data.
    return X_train, X_test


def encode_categorical_features(X_train, X_test):
    """
    Encode categorical variables for modeling

    Currently this function dummy-encodes the status column
    Country is intentionally not dummy-encoded because it has many categories
    and can increase dimensionality significantly

    Parameters
    ----------
    X_train:
        Training predictor DataFrame

    X_test:
        Test predictor DataFrame

    Returns
    -------
    tuple:
        Encoded X_train and X_test DataFrames
    """

    # Copy the inputs
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Drop country if it exists
    # We keep country for SQL/dashboard summaries, but not for the regression model
    if "country" in X_train.columns:
        X_train = X_train.drop(columns=["country"])

    if "country" in X_test.columns:
        X_test = X_test.drop(columns=["country"])

    # Dummy encode status if it exists
    # drop_first=True prevents perfect multicollinearity between dummy columns
    if "status" in X_train.columns:
        X_train = pd.get_dummies(X_train, columns=["status"], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=["status"], drop_first=True)

    # Align training and test columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Return the encoded data
    return X_train, X_test


def select_features(X_train, X_test, selected_features):
    """
    Select model features from train and test data

    This keeps the selected numeric predictors and also keeps any status dummy columns

    Parameters
    ----------
    X_train:
        Training predictor DataFrame

    X_test:
        Test predictor DataFrame

    selected_features:
        List of selected feature names

    Returns
    -------
    tuple:
        Feature-selected X_train and X_test DataFrames
    """

    # Copy the inputs
    X_train = X_train.copy()
    X_test = X_test.copy()

    # If no feature list is provided, return all features
    if selected_features is None:
        return X_train, X_test

    # Keep selected features that actually exist in the data
    keep_cols = [col for col in selected_features if col in X_train.columns]

    # Also keep any dummy-encoded status columns
    keep_cols += [col for col in X_train.columns if col.startswith("status_")]

    # Select the same columns from train and test
    X_train = X_train[keep_cols]
    X_test = X_test[keep_cols]

    # Return the selected data
    return X_train, X_test


def preprocess_features(
    X_train,
    X_test,
    selected_features=None,
    log_features=None,
    zero_to_nan_features=None,
    imputer_k=5,
    scale=True,
    use_pca=False,
    n_components=None
):
    """
    Complete preprocessing pipeline for regression modeling

    This function applies:
    1. Categorical encoding
    2. Feature selection
    3. Zero-to-NaN replacement
    4. Log transformation
    5. KNN imputation
    6. Optional scaling
    7. Optional PCA

    Parameters
    ----------
    X_train:
        Training predictor DataFrame

    X_test:
        Test predictor DataFrame

    selected_features:
        List of features to use for modeling

    log_features:
        List of features to log-transform

    zero_to_nan_features:
        List of features where zero should be treated as missing

    imputer_k:
        Number of neighbors for KNN imputation

    scale:
        Whether to standardize features

    use_pca:
        Whether to apply PCA

    n_components:
        Number of PCA components to keep

    Returns
    -------
    tuple:
        X_train_final, X_test_final, preprocessing_objects
    """

    # Copy input
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Encode categorical variables such as status
    X_train, X_test = encode_categorical_features(X_train, X_test)

    # Keep only the selected modeling features
    X_train, X_test = select_features(X_train, X_test, selected_features)

    # Replace selected zero values with NaN before imputation
    X_train, X_test = replace_zero_with_nan(X_train, X_test, zero_to_nan_features)

    # Apply log transformations to selected skewed variables
    X_train, X_test = apply_log_transform(X_train, X_test, log_features)

    # Create the KNN imputer
    # weights="distance" gives closer neighbors more influence
    imputer = KNNImputer(n_neighbors=imputer_k, weights="distance")

    # Fit the imputer on the training set and transform the training set
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # Use the already-fitted imputer to transform the test set
    # This prevents data leakage
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Set default scaler to None
    scaler = None

    # If scale=True, standardize the features
    if scale:

        # Create the scaler
        scaler = StandardScaler()

        # Fit scaler only on training data, then transform training data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imputed),
            columns=X_train_imputed.columns,
            index=X_train_imputed.index
        )

        # Transform test data using the training scaler
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imputed),
            columns=X_test_imputed.columns,
            index=X_test_imputed.index
        )

    # If scale=False, keep the imputed values as the final non-PCA features
    else:
        X_train_scaled = X_train_imputed
        X_test_scaled = X_test_imputed

    # Set default PCA model to None
    pca_model = None

    # If PCA is requested, fit PCA on the training data and transform both sets
    if use_pca:

        # Create the PCA model
        pca_model = PCA(n_components=n_components)

        # Fit PCA only on training data
        X_train_final = pca_model.fit_transform(X_train_scaled)

        # Transform test data using the fitted PCA model
        X_test_final = pca_model.transform(X_test_scaled)

    # If PCA is not requested, final features are the scaled/imputed DataFrames
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # Store fitted preprocessing objects
    preprocessing_objects = {
        "imputer": imputer,
        "scaler": scaler,
        "pca_model": pca_model
    }

    # Return processed training data, processed test data, and fitted objects
    return X_train_final, X_test_final, preprocessing_objects