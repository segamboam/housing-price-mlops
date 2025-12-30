"""Tests for data loading and preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    EXPECTED_COLUMNS,
    get_data_summary,
    load_housing_data,
    validate_schema,
)
from src.data.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_train_test_split,
    impute_missing_values,
    preprocess_pipeline,
    scale_features,
    split_features_target,
)


class TestDataLoader:
    """Tests for data loading functions."""

    def test_load_housing_data_success(self, temp_csv_file: Path):
        """Test successful data loading."""
        df = load_housing_data(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_load_housing_data_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_housing_data("nonexistent.csv")

    def test_load_housing_data_with_na(self, temp_csv_with_na: Path):
        """Test loading data with NA values."""
        df = load_housing_data(temp_csv_with_na)

        assert df.isna().sum().sum() > 0

    def test_validate_schema_success(self, sample_housing_data: pd.DataFrame):
        """Test successful schema validation."""
        validate_schema(sample_housing_data)

    def test_validate_schema_missing_columns(self, sample_housing_data: pd.DataFrame):
        """Test error when columns are missing."""
        df = sample_housing_data.drop(columns=["MEDV"])

        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)

    def test_validate_schema_extra_columns(self, sample_housing_data: pd.DataFrame):
        """Test error when extra columns exist."""
        df = sample_housing_data.copy()
        df["EXTRA"] = 1

        with pytest.raises(ValueError, match="Unexpected columns"):
            validate_schema(df)

    def test_get_data_summary(self, sample_housing_data: pd.DataFrame):
        """Test data summary generation."""
        summary = get_data_summary(sample_housing_data)

        assert summary["n_rows"] == 50
        assert summary["n_columns"] == 14
        assert summary["total_missing"] == 0
        assert isinstance(summary["missing_values"], dict)


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_impute_missing_values_median(self, sample_data_with_na: pd.DataFrame):
        """Test median imputation of missing values."""
        df_imputed = impute_missing_values(sample_data_with_na, strategy="median")

        assert df_imputed.isna().sum().sum() == 0
        assert len(df_imputed) == len(sample_data_with_na)

    def test_impute_missing_values_mean(self, sample_data_with_na: pd.DataFrame):
        """Test mean imputation of missing values."""
        df_imputed = impute_missing_values(sample_data_with_na, strategy="mean")

        assert df_imputed.isna().sum().sum() == 0

    def test_impute_missing_values_invalid_strategy(self, sample_data_with_na: pd.DataFrame):
        """Test error with invalid imputation strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            impute_missing_values(sample_data_with_na, strategy="invalid")

    def test_split_features_target(self, sample_housing_data: pd.DataFrame):
        """Test feature-target splitting."""
        X, y = split_features_target(sample_housing_data)

        assert list(X.columns) == FEATURE_COLUMNS
        assert len(X) == len(sample_housing_data)
        assert len(y) == len(sample_housing_data)
        assert y.name == TARGET_COLUMN

    def test_create_train_test_split(self, sample_housing_data: pd.DataFrame):
        """Test train-test splitting."""
        X, y = split_features_target(sample_housing_data)
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) == 40
        assert len(X_test) == 10
        assert len(y_train) == 40
        assert len(y_test) == 10

    def test_create_train_test_split_reproducibility(self, sample_housing_data: pd.DataFrame):
        """Test that splitting is reproducible with same random state."""
        X, y = split_features_target(sample_housing_data)

        X_train1, _, _, _ = create_train_test_split(X, y, random_state=42)
        X_train2, _, _, _ = create_train_test_split(X, y, random_state=42)

        pd.testing.assert_frame_equal(X_train1, X_train2)

    def test_scale_features(self, sample_housing_data: pd.DataFrame):
        """Test feature scaling."""
        X, y = split_features_target(sample_housing_data)
        X_train, X_test, _, _ = create_train_test_split(X, y)

        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        # Check that training data is scaled (mean ~0, std ~1)
        assert np.abs(X_train_scaled.mean()) < 0.1
        assert np.abs(X_train_scaled.std() - 1) < 0.1

    def test_preprocess_pipeline(self, sample_housing_data: pd.DataFrame):
        """Test complete preprocessing pipeline."""
        result = preprocess_pipeline(sample_housing_data, test_size=0.2, random_state=42)

        assert "X_train_scaled" in result
        assert "X_test_scaled" in result
        assert "y_train" in result
        assert "y_test" in result
        assert "scaler" in result
        assert "feature_names" in result

        assert result["X_train_scaled"].shape[0] == 40
        assert result["X_test_scaled"].shape[0] == 10
        assert result["feature_names"] == FEATURE_COLUMNS

    def test_preprocess_pipeline_with_na(self, sample_data_with_na: pd.DataFrame):
        """Test preprocessing pipeline handles NA values."""
        result = preprocess_pipeline(sample_data_with_na)

        # Should not have any NaN after preprocessing
        assert not np.isnan(result["X_train_scaled"]).any()
        assert not np.isnan(result["X_test_scaled"]).any()
