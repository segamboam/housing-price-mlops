"""Tests for API endpoints.

Justification: API is the interface with clients.
Must verify endpoints return correct responses and handle errors.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import HousingFeatures
from src.api.service import PredictionService
from src.artifacts.bundle import MLArtifactBundle
from src.config.settings import get_settings


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

    def test_health_includes_champion_challenger_status(self, mock_artifact_bundle):
        """Health endpoint shows champion and challenger load status."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "champion_loaded" in data
        assert "challenger_loaded" in data
        assert "traffic_split" in data
        assert data["champion_loaded"] is True
        assert data["challenger_loaded"] is True

    def test_health_shows_traffic_split(self, mock_artifact_bundle):
        """Health endpoint shows current effective traffic split."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.get("/health")

        data = response.json()
        split = data["traffic_split"]
        assert "champion" in split
        assert "challenger" in split
        assert split["champion"] + split["challenger"] == pytest.approx(1.0)


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root_returns_api_info(self, client):
        """Root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_root_includes_serving_strategy(self, client):
        """Root endpoint indicates champion/challenger serving strategy."""
        response = client.get("/")

        data = response.json()
        assert data["serving_strategy"] == "champion/challenger"


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

    def test_predict_returns_served_by(self, sample_features_dict, mock_artifact_bundle):
        """Predict response includes served_by field indicating which model was used."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.post("/predict", json=sample_features_dict)

        assert response.status_code == 200
        data = response.json()
        assert "served_by" in data
        assert data["served_by"] in ("champion", "challenger")

    def test_predict_with_invalid_input_returns_422(self, mock_artifact_bundle):
        """Predict returns 422 for invalid input types."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        invalid_data = {"CRIM": "not_a_number"}

        with TestClient(app) as client:
            response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422
        # Verify unified error format
        data = response.json()
        assert data["code"] == "VALIDATION_ERROR"
        assert "errors" in data
        assert len(data["errors"]) > 0

    def test_predict_with_missing_fields_returns_422(self, mock_artifact_bundle):
        """Predict returns 422 when required fields are missing."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        incomplete_data = {"CRIM": 0.1, "ZN": 18.0}  # Missing 11 fields

        with TestClient(app) as client:
            response = client.post("/predict", json=incomplete_data)

        assert response.status_code == 422
        data = response.json()
        assert data["code"] == "VALIDATION_ERROR"
        assert len(data["errors"]) >= 11  # At least 11 missing fields


class TestTrafficRouting:
    """Tests for champion/challenger traffic routing."""

    def test_all_traffic_to_champion_when_weight_is_1(
        self, sample_features_dict, mock_artifact_bundle
    ):
        """100% champion when weight is 1.0."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            # Set weight AFTER lifespan has run
            app.state.champion_weight = 1.0
            results = set()
            for _ in range(10):
                response = client.post("/predict", json=sample_features_dict)
                assert response.status_code == 200
                results.add(response.json()["served_by"])

        assert results == {"champion"}
        app.state.champion_weight = 0.5

    def test_all_traffic_to_challenger_when_weight_is_0(
        self, sample_features_dict, mock_artifact_bundle
    ):
        """100% challenger when weight is 0.0."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            # Set weight AFTER lifespan has run
            app.state.champion_weight = 0.0
            results = set()
            for _ in range(10):
                response = client.post("/predict", json=sample_features_dict)
                assert response.status_code == 200
                results.add(response.json()["served_by"])

        assert results == {"challenger"}
        app.state.champion_weight = 0.5

    def test_only_champion_when_challenger_is_none(
        self, sample_features_dict, mock_artifact_bundle
    ):
        """All traffic to champion when no challenger is loaded."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            # Remove challenger AFTER lifespan has run
            app.state.challenger_bundle = None
            results = set()
            for _ in range(10):
                response = client.post("/predict", json=sample_features_dict)
                assert response.status_code == 200
                results.add(response.json()["served_by"])

        assert results == {"champion"}

    def test_traffic_split_distributes_requests(self, sample_features_dict, mock_artifact_bundle):
        """With 50/50 weight and both models loaded, both should receive traffic."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        app.state.champion_weight = 0.5

        with TestClient(app) as client:
            results = {"champion": 0, "challenger": 0}
            for _ in range(50):
                response = client.post("/predict", json=sample_features_dict)
                assert response.status_code == 200
                served = response.json()["served_by"]
                results[served] += 1

        # Both should have received some traffic (probabilistic but very safe)
        assert results["champion"] > 0
        assert results["challenger"] > 0


class TestFeatureRangeChecking:
    """Tests for PredictionService._check_feature_ranges via the service layer."""

    def _make_service(self, mock_bundle: MLArtifactBundle) -> PredictionService:
        """Helper to create a PredictionService with the given bundle."""
        return PredictionService(
            bundle=mock_bundle, settings=get_settings(), model_alias="champion"
        )

    def test_feature_in_range_no_warning(
        self, sample_features_dict, feature_stats, mock_artifact_bundle
    ):
        """No warning when feature is in range."""
        mock_artifact_bundle.metadata.feature_stats = feature_stats
        service = self._make_service(mock_artifact_bundle)
        features = HousingFeatures(**sample_features_dict)

        warnings = service._check_feature_ranges(features)

        # CRIM=0.00632 is exactly at min, should not trigger warning
        crim_warnings = [w for w in warnings if "CRIM" in w]
        assert len(crim_warnings) == 0

    def test_feature_below_min_generates_warning(
        self, sample_features_dict, feature_stats, mock_artifact_bundle
    ):
        """Warning generated when feature below min."""
        mock_artifact_bundle.metadata.feature_stats = feature_stats
        service = self._make_service(mock_artifact_bundle)
        # Use DIS which allows values > 0, set below the training min
        sample_features_dict["DIS"] = 0.5  # Below min of 1.1296
        features = HousingFeatures(**sample_features_dict)

        warnings = service._check_feature_ranges(features)

        assert any("DIS" in w and "below" in w for w in warnings)

    def test_feature_above_max_generates_warning(
        self, sample_features_dict, feature_stats, mock_artifact_bundle
    ):
        """Warning generated when feature above max."""
        mock_artifact_bundle.metadata.feature_stats = feature_stats
        service = self._make_service(mock_artifact_bundle)
        sample_features_dict["CRIM"] = 100.0  # Above max of 88.9762
        features = HousingFeatures(**sample_features_dict)

        warnings = service._check_feature_ranges(features)

        assert any("CRIM" in w and "above" in w for w in warnings)

    def test_empty_stats_no_warnings(self, sample_features_dict, mock_artifact_bundle):
        """No warnings with empty feature stats."""
        mock_artifact_bundle.metadata.feature_stats = {}
        service = self._make_service(mock_artifact_bundle)
        features = HousingFeatures(**sample_features_dict)

        warnings = service._check_feature_ranges(features)

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
            # Force champion to use our configured mock
            app.state.champion_weight = 1.0
            response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["total_items"] == 2
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["index"] == 0
        assert data["predictions"][1]["index"] == 1
        assert "prediction" in data["predictions"][0]
        assert "prediction_formatted" in data["predictions"][0]
        app.state.champion_weight = 0.5

    def test_batch_predict_returns_served_by(self, sample_features_dict, mock_artifact_bundle):
        """Batch prediction response includes served_by field."""
        import numpy as np
        from fastapi.testclient import TestClient

        from src.api.main import app

        mock_artifact_bundle.predict.return_value = np.array([25.5])

        request_data = {"items": [sample_features_dict]}

        with TestClient(app) as client:
            response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "served_by" in data
        assert data["served_by"] in ("champion", "challenger")

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
            # Force champion to ensure we use the bundle with feature_stats
            app.state.champion_weight = 1.0
            response = client.post("/predict/batch", json={"items": items})

        assert response.status_code == 200
        data = response.json()
        # First item should have no warnings
        assert len(data["predictions"][0]["warnings"]) == 0
        # Second item should have CRIM warning
        assert len(data["predictions"][1]["warnings"]) > 0
        assert any("CRIM" in w for w in data["predictions"][1]["warnings"])
        app.state.champion_weight = 0.5


class TestModelReloadEndpoint:
    """Tests for /model/reload endpoint."""

    def test_reload_both_models(self, mock_artifact_bundle, monkeypatch):
        """Reload without alias reloads both champion and challenger."""
        from unittest.mock import MagicMock

        import numpy as np
        from fastapi.testclient import TestClient

        from src.api.main import app

        # Create new mock bundles for reload
        new_champion = MagicMock()
        new_champion.predict.return_value = np.array([30.0])
        new_champion.metadata = MagicMock()
        new_champion.metadata.model_type = "new_champion"
        new_champion.metadata.artifact_id = "newchamp-12345678"

        new_challenger = MagicMock()
        new_challenger.predict.return_value = np.array([31.0])
        new_challenger.metadata = MagicMock()
        new_challenger.metadata.model_type = "new_challenger"
        new_challenger.metadata.artifact_id = "newchall-87654321"

        def mock_load(alias=None):
            if alias == "champion":
                return (new_champion, "mlflow")
            elif alias == "challenger":
                return (new_challenger, "mlflow")
            return (new_champion, "mlflow")

        monkeypatch.setattr("src.api.main.load_bundle_from_mlflow", mock_load)

        with TestClient(app) as client:
            response = client.post("/model/reload")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["champion_model"] is not None
        assert data["challenger_model"] is not None
        assert "reload_time_ms" in data

    def test_reload_specific_alias(self, mock_artifact_bundle, monkeypatch):
        """Reload with specific alias only reloads that model."""
        from unittest.mock import MagicMock

        import numpy as np
        from fastapi.testclient import TestClient

        from src.api.main import app

        new_challenger = MagicMock()
        new_challenger.predict.return_value = np.array([28.0])
        new_challenger.metadata = MagicMock()
        new_challenger.metadata.model_type = "reloaded_challenger"
        new_challenger.metadata.artifact_id = "reloaded-12345678"

        captured_aliases = []

        def mock_load(alias=None):
            captured_aliases.append(alias)
            return (new_challenger, "mlflow")

        with TestClient(app) as client:
            monkeypatch.setattr("src.api.main.load_bundle_from_mlflow", mock_load)
            response = client.post("/model/reload", json={"alias": "challenger"})

        assert response.status_code == 200
        # Only challenger alias should have been loaded
        assert "challenger" in captured_aliases
        assert "champion" not in captured_aliases

    def test_reload_failure_keeps_old_models(self, mock_artifact_bundle, monkeypatch):
        """Failed reload keeps existing models."""
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


class TestTrafficConfigEndpoints:
    """Tests for /model/traffic GET and POST endpoints."""

    def test_get_traffic_config(self, mock_artifact_bundle):
        """GET /model/traffic returns current config."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.get("/model/traffic")

        assert response.status_code == 200
        data = response.json()
        assert "champion_weight" in data
        assert "challenger_weight" in data
        assert "champion_loaded" in data
        assert "challenger_loaded" in data
        assert "effective_split" in data
        assert data["champion_weight"] + data["challenger_weight"] == pytest.approx(1.0)

    def test_set_traffic_config(self, mock_artifact_bundle):
        """POST /model/traffic updates the split."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/model/traffic",
                json={"champion_weight": 0.7},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["champion_weight"] == pytest.approx(0.7)
        assert data["challenger_weight"] == pytest.approx(0.3)

        # Reset to 50/50
        app.state.champion_weight = 0.5

    def test_set_traffic_config_validates_range(self, mock_artifact_bundle):
        """POST /model/traffic rejects invalid weights."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/model/traffic",
                json={"champion_weight": 1.5},
            )

        assert response.status_code == 422

    def test_set_traffic_all_to_champion(self, mock_artifact_bundle):
        """POST /model/traffic can route 100% to champion."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/model/traffic",
                json={"champion_weight": 1.0},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["champion_weight"] == 1.0
        assert data["challenger_weight"] == 0.0

        # Reset
        app.state.champion_weight = 0.5


class TestUnifiedErrorResponses:
    """Tests for the unified error handling format."""

    def test_422_has_unified_format(self, mock_artifact_bundle):
        """Validation errors return ErrorResponse with code and errors list."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            response = client.post("/predict", json={"CRIM": "bad"})

        assert response.status_code == 422
        data = response.json()
        assert data["code"] == "VALIDATION_ERROR"
        assert data["detail"] == "Invalid input data"
        assert isinstance(data["errors"], list)
        assert len(data["errors"]) > 0
        # Each error should have field & message
        for err in data["errors"]:
            assert "message" in err

    def test_503_has_unified_format(self, mock_artifact_bundle, monkeypatch):
        """503 when no model loaded returns ErrorResponse format."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        # Mock loaders to return None (no model available)
        monkeypatch.setattr(
            "src.api.main.load_bundle_from_mlflow",
            lambda alias=None: (None, None),
        )
        monkeypatch.setattr(
            "src.api.main.load_artifact_bundle",
            lambda: (None, None),
        )

        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "CRIM": 0.1,
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
                },
            )

        assert response.status_code == 503
        data = response.json()
        assert data["code"] == "SERVICE_UNAVAILABLE"
        assert "detail" in data
