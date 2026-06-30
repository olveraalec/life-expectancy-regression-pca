from app.model_loader import load_model_bundle
from app.predict import make_prediction
from app.schemas import PredictionRequest


def test_make_prediction_returns_float():
    request = PredictionRequest(
        adult_mortality=150,
        income_composition_of_resources=0.75,
        schooling=13.2,
        bmi=25.1,
        gdp=5000,
        hiv_aids=0.2,
        status="Developing",
    )

    bundle = load_model_bundle()
    prediction = make_prediction(request, bundle)

    assert isinstance(prediction, float)
    assert 30 <= prediction <= 100
