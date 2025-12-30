"""Tests for the FastAPI application."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.schemas import HousingFeatures

# Sample valid input for predictions
VALID_FEATURES = {
    "CRIM": 0.00632,
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


@pytest.fixture
def client_with_model(trained_model_artifacts: dict):
    """Create test client with a loaded model."""
    from src.api import main

    # Inject model and scaler
    main.model = trained_model_artifacts["model"]
    main.scaler = trained_model_artifacts["scaler"]
    main.model_source = "test"

    with TestClient(main.app) as client:
        yield client

    # Cleanup
    main.model = None
    main.scaler = None
    main.model_source = None


@pytest.fixture
def client_without_model(monkeypatch):
    """Create test client without a loaded model."""
    from src.api import main

    # Patch the paths to non-existent files so lifespan doesn't load model
    monkeypatch.setattr(main, "MODEL_PATH", Path("/nonexistent/model.joblib"))
    monkeypatch.setattr(main, "SCALER_PATH", Path("/nonexistent/scaler.joblib"))
    monkeypatch.setattr(main, "MLFLOW_TRACKING_URI", None)

    # Reset global state
    main.model = None
    main.scaler = None
    main.model_source = None

    with TestClient(main.app) as client:
        yield client


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_api_info(self, client_with_model):
        """Test root endpoint returns API information."""
        response = client_with_model.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_with_model_loaded(self, client_with_model):
        """Test health endpoint when model is loaded."""
        response = client_with_model.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_source"] == "test"

    def test_health_without_model(self, client_without_model):
        """Test health endpoint when model is not loaded."""
        response = client_without_model.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert data["model_source"] is None


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_with_valid_input(self, client_with_model):
        """Test prediction with valid input."""
        response = client_with_model.post("/predict", json=VALID_FEATURES)

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert isinstance(data["prediction"], float)

    def test_predict_without_model(self, client_without_model):
        """Test prediction fails when model is not loaded."""
        response = client_without_model.post("/predict", json=VALID_FEATURES)

        assert response.status_code == 503
        data = response.json()
        assert "Model not loaded" in data["detail"]

    def test_predict_with_missing_features(self, client_with_model):
        """Test prediction fails with missing features."""
        incomplete_features = {"CRIM": 0.00632, "ZN": 18.0}

        response = client_with_model.post("/predict", json=incomplete_features)

        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_crim_negative(self, client_with_model):
        """Test validation rejects negative CRIM."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features["CRIM"] = -1.0

        response = client_with_model.post("/predict", json=invalid_features)

        assert response.status_code == 422

    def test_predict_with_invalid_chas_value(self, client_with_model):
        """Test validation rejects invalid CHAS value."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features["CHAS"] = 2  # Should be 0 or 1

        response = client_with_model.post("/predict", json=invalid_features)

        assert response.status_code == 422

    def test_predict_with_invalid_nox_range(self, client_with_model):
        """Test validation rejects NOX out of range."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features["NOX"] = 1.5  # Should be 0-1

        response = client_with_model.post("/predict", json=invalid_features)

        assert response.status_code == 422

    def test_predict_with_invalid_rm_range(self, client_with_model):
        """Test validation rejects RM out of range."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features["RM"] = 0.5  # Should be 1-15

        response = client_with_model.post("/predict", json=invalid_features)

        assert response.status_code == 422

    def test_predict_returns_reasonable_value(self, client_with_model):
        """Test prediction returns value in reasonable range."""
        response = client_with_model.post("/predict", json=VALID_FEATURES)

        assert response.status_code == 200
        prediction = response.json()["prediction"]
        # Housing prices should be between 5 and 50 (in $1000s)
        assert 0 < prediction < 100

    def test_predict_with_edge_values(self, client_with_model):
        """Test prediction with edge case values."""
        edge_features = {
            "CRIM": 0.0,
            "ZN": 0.0,
            "INDUS": 0.0,
            "CHAS": 0,
            "NOX": 0.0,
            "RM": 1.0,
            "AGE": 0.0,
            "DIS": 0.1,
            "RAD": 1,
            "TAX": 0.0,
            "PTRATIO": 0.0,
            "B": 0.0,
            "LSTAT": 0.0,
        }

        response = client_with_model.post("/predict", json=edge_features)

        assert response.status_code == 200


class TestSchemaValidation:
    """Tests for input schema validation."""

    def test_housing_features_valid(self):
        """Test HousingFeatures with valid data."""
        features = HousingFeatures(**VALID_FEATURES)

        assert features.CRIM == 0.00632
        assert features.CHAS == 0

    def test_housing_features_type_coercion(self):
        """Test HousingFeatures handles type coercion."""
        features_with_int = VALID_FEATURES.copy()
        features_with_int["ZN"] = 18  # int instead of float

        features = HousingFeatures(**features_with_int)

        assert features.ZN == 18.0

    def test_housing_features_validation_error(self):
        """Test HousingFeatures raises on invalid data."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features["CRIM"] = "not_a_number"

        with pytest.raises(Exception):
            HousingFeatures(**invalid_features)


class TestAPIDocumentation:
    """Tests for API documentation."""

    def test_openapi_schema_available(self, client_with_model):
        """Test OpenAPI schema is available."""
        response = client_with_model.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_swagger_ui_available(self, client_with_model):
        """Test Swagger UI is available."""
        response = client_with_model.get("/docs")

        assert response.status_code == 200
