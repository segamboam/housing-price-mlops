"""Tests for data loader functions."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    get_data_summary,
    load_housing_data,
    validate_schema,
)


class TestConstants:
    """Tests for module constants."""

    def test_feature_columns_count(self):
        """Feature columns has expected count."""
        assert len(FEATURE_COLUMNS) == 13

    def test_target_column_is_medv(self):
        """Target column is MEDV."""
        assert TARGET_COLUMN == "MEDV"

    def test_expected_columns_includes_all(self):
        """Expected columns includes features + target."""
        assert len(EXPECTED_COLUMNS) == 14
        assert TARGET_COLUMN in EXPECTED_COLUMNS
        for col in FEATURE_COLUMNS:
            assert col in EXPECTED_COLUMNS

    def test_categorical_features(self):
        """CHAS is the only categorical feature."""
        assert CATEGORICAL_FEATURES == ["CHAS"]

    def test_numeric_features_excludes_categorical(self):
        """Numeric features excludes categorical."""
        assert "CHAS" not in NUMERIC_FEATURES
        assert len(NUMERIC_FEATURES) == len(FEATURE_COLUMNS) - len(CATEGORICAL_FEATURES)


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_valid_schema_passes(self):
        """Valid schema passes validation."""
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        validate_schema(df)  # Should not raise

    def test_missing_column_raises(self):
        """Missing column raises ValueError."""
        columns = [c for c in EXPECTED_COLUMNS if c != "CRIM"]
        df = pd.DataFrame(columns=columns)
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)

    def test_extra_column_raises(self):
        """Extra column raises ValueError."""
        columns = EXPECTED_COLUMNS + ["EXTRA"]
        df = pd.DataFrame(columns=columns)
        with pytest.raises(ValueError, match="Unexpected columns"):
            validate_schema(df)

    def test_multiple_missing_columns(self):
        """Multiple missing columns are reported."""
        columns = [c for c in EXPECTED_COLUMNS if c not in ["CRIM", "ZN", "MEDV"]]
        df = pd.DataFrame(columns=columns)
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)


class TestLoadHousingData:
    """Tests for load_housing_data function."""

    def test_file_not_found_raises(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_housing_data("/nonexistent/path/data.csv")

    def test_loads_valid_csv(self):
        """Valid CSV file is loaded correctly."""
        # Create a temporary CSV with valid schema
        data = {col: [1.0, 2.0, 3.0] for col in EXPECTED_COLUMNS}
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            result = load_housing_data(temp_path)
            assert len(result) == 3
            assert list(result.columns) == EXPECTED_COLUMNS
        finally:
            Path(temp_path).unlink()

    def test_loads_with_path_object(self):
        """Can load using Path object."""
        data = {col: [1.0] for col in EXPECTED_COLUMNS}
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)

        try:
            result = load_housing_data(temp_path)
            assert len(result) == 1
        finally:
            temp_path.unlink()

    def test_handles_na_values(self):
        """NA strings are converted to NaN."""
        data = {col: ["1.0", "NA", ""] for col in EXPECTED_COLUMNS}
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            result = load_housing_data(temp_path)
            # Check that NA values are converted to NaN
            assert result.isna().sum().sum() > 0
        finally:
            Path(temp_path).unlink()

    def test_invalid_schema_raises(self):
        """CSV with wrong schema raises ValueError."""
        data = {"wrong_column": [1.0, 2.0]}
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                load_housing_data(temp_path)
        finally:
            Path(temp_path).unlink()


class TestGetDataSummary:
    """Tests for get_data_summary function."""

    def test_returns_dict(self):
        """Summary returns a dictionary."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_data_summary(df)
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Summary contains all required keys."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_data_summary(df)
        assert "n_rows" in result
        assert "n_columns" in result
        assert "missing_values" in result
        assert "total_missing" in result

    def test_n_rows_correct(self):
        """Row count is correct."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = get_data_summary(df)
        assert result["n_rows"] == 5

    def test_n_columns_correct(self):
        """Column count is correct."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = get_data_summary(df)
        assert result["n_columns"] == 3

    def test_missing_values_per_column(self):
        """Missing values are reported per column."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})
        result = get_data_summary(df)
        assert result["missing_values"]["a"] == 1
        assert result["missing_values"]["b"] == 2

    def test_total_missing_correct(self):
        """Total missing count is correct."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})
        result = get_data_summary(df)
        assert result["total_missing"] == 3

    def test_no_missing_values(self):
        """Works with no missing values."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_data_summary(df)
        assert result["total_missing"] == 0
        assert all(v == 0 for v in result["missing_values"].values())

    def test_empty_dataframe(self):
        """Works with empty DataFrame."""
        df = pd.DataFrame()
        result = get_data_summary(df)
        assert result["n_rows"] == 0
        assert result["n_columns"] == 0
        assert result["total_missing"] == 0
