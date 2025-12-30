"""Shared test fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_housing_data() -> pd.DataFrame:
    """Create a sample housing dataset for testing."""
    np.random.seed(42)
    n_samples = 50

    data = {
        "CRIM": np.random.uniform(0, 10, n_samples),
        "ZN": np.random.uniform(0, 100, n_samples),
        "INDUS": np.random.uniform(0, 30, n_samples),
        "CHAS": np.random.randint(0, 2, n_samples),
        "NOX": np.random.uniform(0.3, 0.9, n_samples),
        "RM": np.random.uniform(4, 9, n_samples),
        "AGE": np.random.uniform(0, 100, n_samples),
        "DIS": np.random.uniform(1, 12, n_samples),
        "RAD": np.random.randint(1, 25, n_samples),
        "TAX": np.random.uniform(100, 800, n_samples),
        "PTRATIO": np.random.uniform(12, 22, n_samples),
        "B": np.random.uniform(0, 400, n_samples),
        "LSTAT": np.random.uniform(1, 40, n_samples),
        "MEDV": np.random.uniform(5, 50, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_na(sample_housing_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample data with missing values."""
    df = sample_housing_data.copy()
    # Add some NaN values
    df.loc[0, "CRIM"] = np.nan
    df.loc[5, "ZN"] = np.nan
    df.loc[10, "RM"] = np.nan
    df.loc[15, "LSTAT"] = np.nan
    return df


@pytest.fixture
def temp_csv_file(sample_housing_data: pd.DataFrame) -> Path:
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_housing_data.to_csv(f.name, index=False)
        return Path(f.name)


@pytest.fixture
def temp_csv_with_na(sample_data_with_na: pd.DataFrame) -> Path:
    """Create a temporary CSV file with missing values."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data_with_na.to_csv(f.name, index=False)
        return Path(f.name)


@pytest.fixture
def trained_model_artifacts(sample_housing_data: pd.DataFrame, tmp_path: Path) -> dict:
    """Create trained model artifacts for testing."""
    from src.data.preprocessing import preprocess_pipeline
    from src.models.train import save_model, train_model

    data = preprocess_pipeline(sample_housing_data)
    model = train_model(data["X_train_scaled"], data["y_train"])
    paths = save_model(model, data["scaler"], tmp_path)

    return {
        "model": model,
        "scaler": data["scaler"],
        "model_path": paths["model_path"],
        "scaler_path": paths["scaler_path"],
        "data": data,
    }
