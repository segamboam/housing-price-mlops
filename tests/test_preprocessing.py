"""Tests for preprocessing strategies.

Justification: Preprocessing is critical for model performance.
A bug here silently degrades predictions without obvious errors.
"""

import numpy as np
import pytest

from src.data.preprocessing.factory import PreprocessorFactory


class TestPreprocessorPipeline:
    """Tests for preprocessor fit/transform flow."""

    def test_fit_extracts_feature_names(self, sample_dataframe):
        """fit() extracts column names from DataFrame."""
        preprocessor = PreprocessorFactory.create("v1_median")

        preprocessor.fit(sample_dataframe)

        assert preprocessor.is_fitted
        assert preprocessor.feature_names == list(sample_dataframe.columns)

    def test_transform_without_fit_raises(self, sample_dataframe):
        """transform() without fit() raises RuntimeError."""
        preprocessor = PreprocessorFactory.create("v1_median")

        with pytest.raises(RuntimeError, match="Preprocessor must be fitted"):
            preprocessor.transform(sample_dataframe)

    def test_fit_transform_returns_scaled_array(self, sample_dataframe):
        """fit_transform() returns numpy array with expected shape."""
        preprocessor = PreprocessorFactory.create("v1_median")

        result = preprocessor.fit_transform(sample_dataframe)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_dataframe.shape
        # StandardScaler should center data around 0
        assert np.abs(result.mean()) < 1.0

    def test_all_strategies_produce_valid_output(self, sample_dataframe):
        """All registered strategies produce valid transformed data."""
        strategies = ["v1_median", "v2_knn", "v3_iterative"]

        for strategy in strategies:
            preprocessor = PreprocessorFactory.create(strategy)
            result = preprocessor.fit_transform(sample_dataframe)

            assert isinstance(result, np.ndarray), f"{strategy} failed"
            assert result.shape == sample_dataframe.shape, f"{strategy} shape mismatch"
            assert not np.isnan(result).any(), f"{strategy} produced NaN"
