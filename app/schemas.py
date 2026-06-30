from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Input schema for a life expectancy prediction request.

    The API accepts raw user-friendly inputs.
    The prediction layer will convert status into the dummy variable
    expected by the trained model.
    """

    adult_mortality: float = Field(
        ...,
        ge=0,
        description="Adult mortality rate.",
        example=150,
    )
    income_composition_of_resources: float = Field(
        ...,
        ge=0,
        le=1,
        description="Income composition of resources index, usually between 0 and 1.",
        example=0.75,
    )
    schooling: float = Field(
        ...,
        ge=0,
        description="Average years of schooling.",
        example=13.2,
    )
    bmi: float = Field(
        ...,
        ge=0,
        description="Average BMI.",
        example=25.1,
    )
    gdp: float = Field(
        ...,
        ge=0,
        description="GDP value used by the model.",
        example=5000,
    )
    hiv_aids: float = Field(
        ...,
        ge=0,
        description="HIV/AIDS value used by the model.",
        example=0.2,
    )
    status: Literal["Developed", "Developing"] = Field(
        ...,
        description="Country development status.",
        example="Developing",
    )


class PredictionResponse(BaseModel):
    """
    Output schema for a prediction response.
    """

    predicted_life_expectancy: float
    model_name: str
    model_version: str
    model_type: str


class HealthResponse(BaseModel):
    """
    Output schema for the health check endpoint.
    """

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
