"""Tests for the FastAPI prediction endpoint."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.api import app


@pytest.fixture
def client():
    return TestClient(app)


VALID_PAYLOAD = {
    "diameter": 12,
    "grade": "GR1",
    "CHEM1": 0.225, "CHEM2": 1.297, "CHEM3": 0.017, "CHEM4": 0.024,
    "CHEM5": 0.230, "CHEM6": 0.0026, "CHEM7": 0.0005, "CHEM8": 0.028,
    "CHEM9": 0.006, "CHEM10": 70.5,
    "TEMP1": 1000, "TEMP2": 965.0, "TEMP3": 70, "TEMP4": 158,
    "TEMP5": 581.0, "TEMP6": 549,
    "SPEED": 21, "PROCESS1": 2521, "PROCESS2": 2666, "PROCESS3": 1.32,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_has_available_models(self, client):
        data = client.get("/health").json()
        assert "available_models" in data
        assert isinstance(data["available_models"], list)


class TestModelInfoEndpoint:
    def test_model_info_valid(self, client):
        response = client.get("/model-info/12")
        assert response.status_code == 200
        data = response.json()
        assert data["diameter"] == 12
        assert "algorithm" in data

    def test_model_info_invalid_diameter(self, client):
        response = client.get("/model-info/15")
        assert response.status_code == 400


class TestPredictEndpoint:
    def test_predict_valid_input(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_fields(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "quality1_uts" in data
        assert "quality2_ys" in data
        assert "uts_ys_ratio" in data
        assert "passes_quality_gate" in data
        assert "model_used" in data

    def test_predict_plausible_values(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 400 < data["quality1_uts"] < 900
        assert 400 < data["quality2_ys"] < 900
        assert data["uts_ys_ratio"] > 0

    def test_predict_invalid_diameter(self, client):
        payload = {**VALID_PAYLOAD, "diameter": 15}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_invalid_grade(self, client):
        payload = {**VALID_PAYLOAD, "grade": "GR99"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_missing_field(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "CHEM1"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Pydantic validation error
