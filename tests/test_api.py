"""Tests for API endpoints.

Justification: API is the interface with clients.
Must verify endpoints return correct responses and handle errors.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, check_feature_ranges
from src.api.schemas import HousingFeatures


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 OK."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root_returns_api_info(self, client):
        """Root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_with_valid_input_and_model(self, sample_features_dict, mock_artifact_bundle):
        """Predict returns 200 with prediction when model is loaded."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.post("/predict", json=sample_features_dict)

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "prediction_formatted" in data
        assert isinstance(data["prediction"], float)
        # Prediction should be a reasonable house price (in $1000s)
        assert 5.0 < data["prediction"] < 100.0

    def test_predict_with_invalid_input_returns_422(self, mock_artifact_bundle):
        """Predict returns 422 for invalid input types."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        invalid_data = {"CRIM": "not_a_number"}

        with TestClient(app) as client:
            response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422

    def test_predict_with_missing_fields_returns_422(self, mock_artifact_bundle):
        """Predict returns 422 when required fields are missing."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        incomplete_data = {"CRIM": 0.1, "ZN": 18.0}  # Missing 11 fields

        with TestClient(app) as client:
            response = client.post("/predict", json=incomplete_data)

        assert response.status_code == 422


class TestFeatureRangeChecking:
    """Tests for check_feature_ranges function."""

    def test_feature_in_range_no_warning(self, sample_features_dict, feature_stats):
        """No warning when feature is in range."""
        features = HousingFeatures(**sample_features_dict)

        warnings = check_feature_ranges(features, feature_stats)

        # CRIM=0.00632 is exactly at min, should not trigger warning
        crim_warnings = [w for w in warnings if "CRIM" in w]
        assert len(crim_warnings) == 0

    def test_feature_below_min_generates_warning(self, sample_features_dict, feature_stats):
        """Warning generated when feature below min."""
        # Use DIS which allows values > 0, set below the training min
        sample_features_dict["DIS"] = 0.5  # Below min of 1.1296
        features = HousingFeatures(**sample_features_dict)

        warnings = check_feature_ranges(features, feature_stats)

        assert any("DIS" in w and "below" in w for w in warnings)

    def test_feature_above_max_generates_warning(self, sample_features_dict, feature_stats):
        """Warning generated when feature above max."""
        sample_features_dict["CRIM"] = 100.0  # Above max of 88.9762
        features = HousingFeatures(**sample_features_dict)

        warnings = check_feature_ranges(features, feature_stats)

        assert any("CRIM" in w and "above" in w for w in warnings)

    def test_empty_stats_no_warnings(self, sample_features_dict):
        """No warnings with empty feature stats."""
        features = HousingFeatures(**sample_features_dict)

        warnings = check_feature_ranges(features, {})

        assert len(warnings) == 0
