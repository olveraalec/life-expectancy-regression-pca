from fastapi import FastAPI, HTTPException

from app.model_loader import load_model_and_metadata
from app.predict import make_prediction
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.config import load_config
from src.logger import setup_logger


config = load_config()
logger = setup_logger(__name__)

app = FastAPI(
    title=config["api"]["app_name"],
    description="FastAPI service for predicting life expectancy using the Version 3 deployed model.",
    version=config["api"]["model_version"],
)

try:
    model_bundle, model_metadata = load_model_and_metadata()
    MODEL_LOADED = True
    MODEL_LOAD_ERROR = None
    logger.info("Model bundle and metadata loaded successfully for API.")
except Exception as error:
    model_bundle = None
    model_metadata = {
        "model_name": config["api"]["model_name"],
        "model_version": config["api"]["model_version"],
        "model_type": "unknown",
    }
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(error)
    logger.error(f"Model failed to load for API: {MODEL_LOAD_ERROR}")


@app.get("/")
def root():
    """
    Root endpoint for the API.
    """
    logger.info("Root endpoint called.")

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
    logger.info("Health check endpoint called.")

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
    logger.info("Prediction endpoint called.")

    if not MODEL_LOADED:
        logger.error(f"Prediction failed because model is not loaded: {MODEL_LOAD_ERROR}")
        raise HTTPException(
            status_code=500,
            detail=f"Model failed to load: {MODEL_LOAD_ERROR}",
        )

    prediction = make_prediction(request, model_bundle)

    logger.info(f"Prediction completed: {prediction}")

    return PredictionResponse(
        predicted_life_expectancy=prediction,
        model_name=model_metadata["model_name"],
        model_version=model_metadata["model_version"],
        model_type=model_metadata["model_type"],
    )
