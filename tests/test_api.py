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


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_success(self, sample_features_dict, mock_artifact_bundle):
        """Batch prediction returns all predictions."""
        import numpy as np
        from fastapi.testclient import TestClient

        from src.api.main import app

        mock_artifact_bundle.predict.return_value = np.array([25.5, 26.0])

        request_data = {"items": [sample_features_dict, sample_features_dict]}

        with TestClient(app) as client:
            response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["total_items"] == 2
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["index"] == 0
        assert data["predictions"][1]["index"] == 1
        assert "prediction" in data["predictions"][0]
        assert "prediction_formatted" in data["predictions"][0]

    def test_batch_predict_single_item(self, sample_features_dict, mock_artifact_bundle):
        """Batch prediction works with single item."""
        import numpy as np
        from fastapi.testclient import TestClient

        from src.api.main import app

        mock_artifact_bundle.predict.return_value = np.array([25.5])

        request_data = {"items": [sample_features_dict]}

        with TestClient(app) as client:
            response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["total_items"] == 1

    def test_batch_predict_empty_rejected(self, mock_artifact_bundle):
        """Batch rejects empty items list."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.post("/predict/batch", json={"items": []})

        assert response.status_code == 422

    def test_batch_predict_over_max_rejected(self, sample_features_dict, mock_artifact_bundle):
        """Batch rejects more than 100 items."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        items = [sample_features_dict] * 101

        with TestClient(app) as client:
            response = client.post("/predict/batch", json={"items": items})

        assert response.status_code == 422

    def test_batch_predict_invalid_item_rejects_all(
        self, sample_features_dict, mock_artifact_bundle
    ):
        """One invalid item rejects entire batch."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        items = [sample_features_dict, {"CRIM": "invalid"}]

        with TestClient(app) as client:
            response = client.post("/predict/batch", json={"items": items})

        assert response.status_code == 422

    def test_batch_predict_includes_warnings(
        self, sample_features_dict, feature_stats, mock_artifact_bundle
    ):
        """Batch predictions include per-item warnings."""
        import numpy as np
        from fastapi.testclient import TestClient

        from src.api.main import app

        mock_artifact_bundle.metadata.feature_stats = feature_stats
        mock_artifact_bundle.predict.return_value = np.array([25.5, 26.0])

        items = [sample_features_dict.copy(), sample_features_dict.copy()]
        items[1]["CRIM"] = 100.0  # Out of range

        with TestClient(app) as client:
            response = client.post("/predict/batch", json={"items": items})

        assert response.status_code == 200
        data = response.json()
        # First item should have no warnings
        assert len(data["predictions"][0]["warnings"]) == 0
        # Second item should have CRIM warning
        assert len(data["predictions"][1]["warnings"]) > 0
        assert any("CRIM" in w for w in data["predictions"][1]["warnings"])


class TestModelReloadEndpoint:
    """Tests for /model/reload endpoint."""

    def test_reload_success(self, mock_artifact_bundle, monkeypatch):
        """Reload succeeds and returns old vs new info."""
        from unittest.mock import MagicMock

        import numpy as np
        from fastapi.testclient import TestClient

        from src.api.main import app

        # Create a new mock for the reloaded bundle
        new_mock_metadata = MagicMock()
        new_mock_metadata.model_type = "gradient_boost"
        new_mock_metadata.artifact_id = "new-model-87654321"

        new_mock_bundle = MagicMock()
        new_mock_bundle.predict.return_value = np.array([30.0])
        new_mock_bundle.metadata = new_mock_metadata

        # Mock load_bundle_from_mlflow to return new bundle
        monkeypatch.setattr(
            "src.api.main.load_bundle_from_mlflow",
            lambda alias=None: (new_mock_bundle, "mlflow"),
        )

        with TestClient(app) as client:
            response = client.post("/model/reload")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "previous_model" in data
        assert "current_model" in data
        assert data["current_model"]["artifact_id"] == "new-mode"  # First 8 chars
        assert "reload_time_ms" in data

    def test_reload_with_alias(self, mock_artifact_bundle, monkeypatch):
        """Reload with specific alias parameter."""
        from unittest.mock import MagicMock

        import numpy as np
        from fastapi.testclient import TestClient

        from src.api.main import app

        new_mock_metadata = MagicMock()
        new_mock_metadata.model_type = "staging_model"
        new_mock_metadata.artifact_id = "staging-12345678"

        new_mock_bundle = MagicMock()
        new_mock_bundle.predict.return_value = np.array([28.0])
        new_mock_bundle.metadata = new_mock_metadata

        captured_alias = []

        def mock_load(alias=None):
            captured_alias.append(alias)
            return (new_mock_bundle, "mlflow")

        with TestClient(app) as client:
            # Apply mock after TestClient init (lifespan already ran)
            monkeypatch.setattr("src.api.main.load_bundle_from_mlflow", mock_load)
            response = client.post("/model/reload", json={"alias": "staging"})

        assert response.status_code == 200
        # The reload endpoint should have been called with "staging"
        assert "staging" in captured_alias

    def test_reload_failure_keeps_old_model(self, mock_artifact_bundle, monkeypatch):
        """Failed reload keeps existing model."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        # Mock load to fail
        monkeypatch.setattr(
            "src.api.main.load_bundle_from_mlflow",
            lambda alias=None: (None, None),
        )

        with TestClient(app) as client:
            response = client.post("/model/reload")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        # Model should be unchanged
        assert data["previous_model"] == data["current_model"]
