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


class MockSettings:
    """Mock settings for testing."""

    def __init__(
        self,
        api_key=None,
        model_path=Path("models/housing_model.joblib"),
        scaler_path=Path("models/housing_model_scaler.joblib"),
        artifact_bundle_path=Path("/nonexistent/artifact_bundle"),
        mlflow_tracking_uri=None,
    ):
        self.api_title = "Test API"
        self.api_version = "1.0.0"
        self.api_key = api_key
        self.api_key_header = "X-API-Key"
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.artifact_bundle_path = artifact_bundle_path
        self.model_dir = Path("models")
        self.artifact_bundle_dir = "artifact_bundle"
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_model_name = "test-model"
        self.mlflow_model_alias = "champion"
        self.metrics_enabled = False

    @property
    def api_key_required(self):
        return self.api_key is not None and len(self.api_key) > 0


@pytest.fixture
def client_with_model(trained_model_artifacts: dict):
    """Create test client with a loaded model (legacy mode)."""
    from src.api import main

    # Reset artifact bundle and use legacy model
    main.artifact_bundle = None
    main.legacy_model = trained_model_artifacts["model"]
    main.legacy_scaler = trained_model_artifacts["scaler"]
    main.model_source = "test"

    with TestClient(main.app) as client:
        yield client

    # Cleanup
    main.artifact_bundle = None
    main.legacy_model = None
    main.legacy_scaler = None
    main.model_source = None


@pytest.fixture
def client_without_model(monkeypatch):
    """Create test client without a loaded model."""
    from src.api import main
    from src.config import settings as settings_module

    mock_settings = MockSettings(
        model_path=Path("/nonexistent/model.joblib"),
        scaler_path=Path("/nonexistent/scaler.joblib"),
        artifact_bundle_path=Path("/nonexistent/artifact_bundle"),
    )
    monkeypatch.setattr(main, "settings", mock_settings)
    monkeypatch.setattr(settings_module, "get_settings", lambda: mock_settings)

    # Reset global state
    main.artifact_bundle = None
    main.legacy_model = None
    main.legacy_scaler = None
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


class TestMetricsEndpoint:
    """Tests for the metrics endpoint."""

    def test_metrics_endpoint_returns_prometheus_format(self, client_with_model):
        """Test metrics endpoint returns Prometheus format."""
        response = client_with_model.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Check for some expected metric names
        content = response.text
        assert "http_requests_total" in content or "predictions_total" in content


class TestAPIKeySecurity:
    """Tests for API key authentication."""

    def test_predict_without_api_key_when_not_required(self, client_with_model):
        """Test prediction works without API key when not required."""
        response = client_with_model.post("/predict", json=VALID_FEATURES)

        assert response.status_code == 200

    def test_predict_with_api_key_when_required(self, trained_model_artifacts, monkeypatch):
        """Test prediction with valid API key when required."""
        from src.api import main
        from src.config import settings as settings_module

        mock_settings = MockSettings(
            api_key="test-secret-key",
            artifact_bundle_path=Path("/nonexistent/artifact_bundle"),
        )
        monkeypatch.setattr(main, "settings", mock_settings)
        monkeypatch.setattr(settings_module, "get_settings", lambda: mock_settings)

        # Inject model using legacy mode
        main.artifact_bundle = None
        main.legacy_model = trained_model_artifacts["model"]
        main.legacy_scaler = trained_model_artifacts["scaler"]
        main.model_source = "test"

        # Also patch security module
        from src.api import security

        monkeypatch.setattr(security, "settings", mock_settings)

        with TestClient(main.app) as client:
            # Request with valid API key
            response = client.post(
                "/predict",
                json=VALID_FEATURES,
                headers={"X-API-Key": "test-secret-key"},
            )
            assert response.status_code == 200

        # Cleanup
        main.artifact_bundle = None
        main.legacy_model = None
        main.legacy_scaler = None

    def test_predict_without_api_key_when_required(self, trained_model_artifacts, monkeypatch):
        """Test prediction fails without API key when required."""
        from src.api import main
        from src.config import settings as settings_module

        mock_settings = MockSettings(
            api_key="test-secret-key",
            artifact_bundle_path=Path("/nonexistent/artifact_bundle"),
        )
        monkeypatch.setattr(main, "settings", mock_settings)
        monkeypatch.setattr(settings_module, "get_settings", lambda: mock_settings)

        # Also patch security module
        from src.api import security

        monkeypatch.setattr(security, "settings", mock_settings)

        # Inject model using legacy mode
        main.artifact_bundle = None
        main.legacy_model = trained_model_artifacts["model"]
        main.legacy_scaler = trained_model_artifacts["scaler"]
        main.model_source = "test"

        with TestClient(main.app) as client:
            # Request without API key
            response = client.post("/predict", json=VALID_FEATURES)
            assert response.status_code == 401

        # Cleanup
        main.artifact_bundle = None
        main.legacy_model = None
        main.legacy_scaler = None

    def test_predict_with_invalid_api_key(self, trained_model_artifacts, monkeypatch):
        """Test prediction fails with invalid API key."""
        from src.api import main
        from src.config import settings as settings_module

        mock_settings = MockSettings(
            api_key="test-secret-key",
            artifact_bundle_path=Path("/nonexistent/artifact_bundle"),
        )
        monkeypatch.setattr(main, "settings", mock_settings)
        monkeypatch.setattr(settings_module, "get_settings", lambda: mock_settings)

        # Also patch security module
        from src.api import security

        monkeypatch.setattr(security, "settings", mock_settings)

        # Inject model using legacy mode
        main.artifact_bundle = None
        main.legacy_model = trained_model_artifacts["model"]
        main.legacy_scaler = trained_model_artifacts["scaler"]
        main.model_source = "test"

        with TestClient(main.app) as client:
            # Request with wrong API key
            response = client.post(
                "/predict",
                json=VALID_FEATURES,
                headers={"X-API-Key": "wrong-key"},
            )
            assert response.status_code == 403

        # Cleanup
        main.artifact_bundle = None
        main.legacy_model = None
        main.legacy_scaler = None
