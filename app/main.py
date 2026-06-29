from fastapi import FastAPI, HTTPException

from app.model_loader import load_model_and_metadata
from app.predict import make_prediction
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.config import load_config


config = load_config()

app = FastAPI(
    title=config["api"]["app_name"],
    description="FastAPI service for predicting life expectancy using the Version 3 deployed model.",
    version=config["api"]["model_version"],
)

try:
    model_bundle, model_metadata = load_model_and_metadata()
    MODEL_LOADED = True
except Exception as error:
    model_bundle = None
    model_metadata = {
        "model_name": config["api"]["model_name"],
        "model_version": config["api"]["model_version"],
        "model_type": "unknown",
    }
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(error)


@app.get("/")
def root():
    """
    Root endpoint for the API.
    """
    return {
        "message": "Life Expectancy Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint confirming whether the model artifact loaded.
    """
    return HealthResponse(
        status="ok" if MODEL_LOADED else "error",
        model_loaded=MODEL_LOADED,
        model_name=model_metadata["model_name"],
        model_version=model_metadata["model_version"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_life_expectancy(request: PredictionRequest):
    """
    Predict life expectancy from validated JSON inputs.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=500,
            detail=f"Model failed to load: {MODEL_LOAD_ERROR}",
        )

    prediction = make_prediction(request, model_bundle)

    return PredictionResponse(
        predicted_life_expectancy=prediction,
        model_name=model_metadata["model_name"],
        model_version=model_metadata["model_version"],
        model_type=model_metadata["model_type"],
    )
