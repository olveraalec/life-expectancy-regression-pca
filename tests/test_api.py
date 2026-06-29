from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Life Expectancy Prediction API"


def test_health_endpoint():
    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "Linear Regression"


def test_predict_endpoint():
    payload = {
        "adult_mortality": 150,
        "income_composition_of_resources": 0.75,
        "schooling": 13.2,
        "bmi": 25.1,
        "gdp": 5000,
        "hiv_aids": 0.2,
        "status": "Developing",
    }

    response = client.post("/predict", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert "predicted_life_expectancy" in data
    assert isinstance(data["predicted_life_expectancy"], float)
    assert data["model_name"] == "Linear Regression"
    assert data["model_version"] == "v3"
