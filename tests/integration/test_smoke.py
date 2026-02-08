"""Integration smoke tests against the real Dockerized API.

These tests verify the complete system works end-to-end with real
Docker services (PostgreSQL, MinIO, MLflow, API). No mocks.

Prerequisites:
    make up   # start all Docker services

Run:
    make test-integration
"""

import httpx
import pytest

pytestmark = pytest.mark.integration


class TestHealthAndInfo:
    """Verify the API is healthy and reports model information."""

    def test_health_endpoint(self, api_client: httpx.Client):
        """GET /health returns 200 with a loaded model."""
        response = api_client.get("/health")

        assert (
            response.status_code == 200
        ), f"Health check failed: {response.status_code} {response.text}"

        data = response.json()
        assert data["status"] == "healthy"
        assert (
            data["model_loaded"] is True
        ), "No model loaded. Did you run 'make up' (which seeds MLflow)?"
        assert data["champion_loaded"] is True

    def test_model_info(self, api_client: httpx.Client):
        """GET /model/info returns champion model metadata."""
        response = api_client.get("/model/info")

        assert (
            response.status_code == 200
        ), f"Model info failed: {response.status_code} {response.text}"

        data = response.json()
        assert "champion" in data
        champion = data["champion"]
        assert champion["model_type"]
        assert champion["preprocessing_strategy"]
        assert champion["feature_names"]
        assert len(champion["feature_names"]) == 13
        assert champion["artifact_id"]


class TestPredictions:
    """Verify prediction endpoints work against the real model."""

    def test_single_prediction(self, api_client: httpx.Client, sample_valid_payload: dict):
        """POST /predict returns a valid prediction."""
        response = api_client.post("/predict", json=sample_valid_payload)

        assert (
            response.status_code == 200
        ), f"Prediction failed: {response.status_code} {response.text}"

        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
        assert data["prediction"] > 0, "House price should be positive"
        assert "prediction_formatted" in data
        assert "$" in data["prediction_formatted"]
        assert data["served_by"] in ("champion", "challenger")
        assert data["model_version"]

    def test_batch_prediction(self, api_client: httpx.Client, sample_valid_payload: dict):
        """POST /predict/batch with 3 items returns 3 predictions."""
        batch_payload = {"items": [sample_valid_payload] * 3}
        response = api_client.post("/predict/batch", json=batch_payload)

        assert (
            response.status_code == 200
        ), f"Batch prediction failed: {response.status_code} {response.text}"

        data = response.json()
        assert data["total_items"] == 3
        assert len(data["predictions"]) == 3

        for item in data["predictions"]:
            assert isinstance(item["prediction"], (int, float))
            assert item["prediction"] > 0
            assert "$" in item["prediction_formatted"]

    def test_invalid_input_error_format(self, api_client: httpx.Client):
        """POST /predict with invalid data returns 422 with ErrorResponse schema."""
        invalid_payload = {
            "CRIM": -1.0,  # Invalid: must be >= 0
            "ZN": 18.0,
            "INDUS": 2.31,
            "CHAS": 0,
            "NOX": 0.538,
            "RM": 6.575,
            "AGE": 65.2,
            "DIS": 4.09,
            "RAD": 1,
            "TAX": 296.0,
            "PTRATIO": 15.3,
            "B": 396.9,
            "LSTAT": 4.98,
        }

        response = api_client.post("/predict", json=invalid_payload)

        assert (
            response.status_code == 422
        ), f"Expected 422, got {response.status_code}: {response.text}"

        data = response.json()
        # Verify unified ErrorResponse schema
        assert "detail" in data, "ErrorResponse must have 'detail'"
        assert "code" in data, "ErrorResponse must have 'code'"
        assert data["code"] == "VALIDATION_ERROR"
        assert "errors" in data
        assert isinstance(data["errors"], list)
        assert len(data["errors"]) > 0
