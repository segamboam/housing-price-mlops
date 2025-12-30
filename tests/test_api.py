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

    def test_predict_with_valid_input(self, client, sample_features_dict):
        """Predict returns prediction for valid input."""
        response = client.post("/predict", json=sample_features_dict)

        # May return 503 if no model loaded, or 401 if API key required
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "prediction_formatted" in data
            assert isinstance(data["prediction"], float)
        else:
            # 503 = no model, 401 = API key required (both acceptable in test env)
            assert response.status_code in [401, 503]

    def test_predict_with_invalid_input_returns_error(self, client):
        """Predict returns error for invalid input."""
        invalid_data = {"CRIM": "not_a_number"}  # Wrong type

        response = client.post("/predict", json=invalid_data)

        # 422 = validation error, 401 = API key checked first
        assert response.status_code in [401, 422]


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
