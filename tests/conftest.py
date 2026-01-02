"""Pytest fixtures for testing."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Import strategies to register them in factories
import src.data.preprocessing.strategies  # noqa: F401
import src.models.strategies  # noqa: F401


@pytest.fixture
def sample_features_dict() -> dict:
    """Sample housing features as dictionary."""
    return {
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
def sample_dataframe(sample_features_dict) -> pd.DataFrame:
    """Sample DataFrame with multiple rows for training."""
    # Create 20 samples with some variation
    np.random.seed(42)
    n_samples = 20

    data = {}
    for col, base_value in sample_features_dict.items():
        if col == "CHAS":
            data[col] = np.random.choice([0, 1], n_samples)
        else:
            # Add noise around base value
            noise = np.random.normal(0, abs(base_value) * 0.1 + 0.1, n_samples)
            data[col] = np.maximum(0, base_value + noise)

    return pd.DataFrame(data)


@pytest.fixture
def sample_target() -> np.ndarray:
    """Sample target values."""
    np.random.seed(42)
    return np.random.uniform(15, 35, 20)


@pytest.fixture
def feature_stats() -> dict:
    """Sample feature statistics for range checking."""
    return {
        "CRIM": {"min": 0.00632, "max": 88.9762},
        "ZN": {"min": 0.0, "max": 100.0},
        "INDUS": {"min": 0.46, "max": 27.74},
        "CHAS": {"min": 0, "max": 1},
        "NOX": {"min": 0.385, "max": 0.871},
        "RM": {"min": 3.561, "max": 8.78},
        "AGE": {"min": 2.9, "max": 100.0},
        "DIS": {"min": 1.1296, "max": 12.1265},
        "RAD": {"min": 1, "max": 24},
        "TAX": {"min": 187.0, "max": 711.0},
        "PTRATIO": {"min": 12.6, "max": 22.0},
        "B": {"min": 0.32, "max": 396.9},
        "LSTAT": {"min": 1.73, "max": 37.97},
    }


@pytest.fixture
def mock_artifact_bundle(monkeypatch):
    """Mock del artifact_bundle para tests de API sin modelo real.

    Also disables API key authentication for testing.
    Mocks the loading functions to prevent MLflow/external connections.
    """
    # Disable API key requirement by patching the settings object in security module
    monkeypatch.setattr("src.api.security.settings.api_key", None)

    # Mock the artifact bundle
    mock_metadata = MagicMock()
    mock_metadata.model_type = "mock_model"
    mock_metadata.artifact_id = "test-mock-12345678"
    mock_metadata.feature_stats = {}

    mock_bundle = MagicMock()
    mock_bundle.predict.return_value = np.array([25.5])
    mock_bundle.metadata = mock_metadata

    # Mock loading functions to prevent external connections during lifespan
    monkeypatch.setattr(
        "src.api.main.load_bundle_from_mlflow",
        lambda alias=None: (mock_bundle, "mlflow"),
    )
    monkeypatch.setattr(
        "src.api.main.load_artifact_bundle",
        lambda: (mock_bundle, "bundle"),
    )
    monkeypatch.setattr(
        "src.api.main.load_legacy_local_model",
        lambda: (None, None, None),
    )

    # Also set the global directly for tests that don't trigger lifespan
    monkeypatch.setattr("src.api.main.artifact_bundle", mock_bundle)
    monkeypatch.setattr("src.api.main.model_source", "mock")

    return mock_bundle
