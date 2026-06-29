import numpy as np
import pandas as pd

from app.schemas import PredictionRequest


def request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """
    Convert a validated API request into a one-row pandas DataFrame.

    The API accepts a user-friendly status value:
    - "Developed"
    - "Developing"

    The trained Version 2 model expects the dummy variable:
    - status_Developing
    """

    input_data = request.model_dump()

    status = input_data.pop("status")

    input_data["status_Developing"] = 1 if status == "Developing" else 0

    return pd.DataFrame([input_data])


def apply_saved_preprocessing(input_df: pd.DataFrame, model_bundle: dict):
    """
    Apply the saved Version 2 preprocessing behavior to one prediction row.

    This uses the fitted preprocessing objects saved in the model bundle:
    - KNN imputer
    - StandardScaler
    - optional PCA model, if present

    It also applies the same log transforms and zero-to-NaN handling used
    during training.
    """

    input_df = input_df.copy()

    log_features = model_bundle["log_features"]
    zero_to_nan_features = model_bundle["zero_to_nan_features"]
    processed_feature_names = model_bundle["processed_feature_names"]
    preprocessing_objects = model_bundle["preprocessing_objects"]

    for col in zero_to_nan_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].replace(0, np.nan)

    for col in log_features:
        if col in input_df.columns:
            input_df[col] = np.log1p(input_df[col])

    for col in processed_feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[processed_feature_names]

    imputer = preprocessing_objects["imputer"]
    scaler = preprocessing_objects["scaler"]
    pca_model = preprocessing_objects["pca_model"]

    input_imputed = pd.DataFrame(
        imputer.transform(input_df),
        columns=processed_feature_names,
    )

    if scaler is not None:
        input_scaled = pd.DataFrame(
            scaler.transform(input_imputed),
            columns=processed_feature_names,
        )
    else:
        input_scaled = input_imputed

    if pca_model is not None:
        input_final = pca_model.transform(input_scaled)
    else:
        input_final = input_scaled

    return input_final


def make_prediction(request: PredictionRequest, model_bundle: dict) -> float:
    """
    Generate one life expectancy prediction from a validated API request.
    """

    input_df = request_to_dataframe(request)
    input_final = apply_saved_preprocessing(input_df, model_bundle)

    model = model_bundle["model"]
    prediction = model.predict(input_final)[0]

    return round(float(prediction), 2)
