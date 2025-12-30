"""Tests for Factory pattern implementations.

Justification: Factories are the entry point for creating models and preprocessors.
If factories fail, the entire pipeline breaks.
"""

import pytest

from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessingStrategy, PreprocessorFactory
from src.models.base import BaseModel
from src.models.factory import ModelFactory, ModelType


class TestPreprocessorFactory:
    """Tests for PreprocessorFactory."""

    def test_create_with_valid_string(self):
        """Factory creates preprocessor from valid string."""
        preprocessor = PreprocessorFactory.create("v1_median")

        assert preprocessor is not None
        assert isinstance(preprocessor, BasePreprocessor)
        assert "v1" in preprocessor.name.lower() or "median" in preprocessor.name.lower()

    def test_create_with_valid_enum(self):
        """Factory creates preprocessor from enum."""
        preprocessor = PreprocessorFactory.create(PreprocessingStrategy.V2_KNN)

        assert preprocessor is not None
        assert isinstance(preprocessor, BasePreprocessor)

    def test_create_with_invalid_string_raises(self):
        """Factory raises ValueError for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown preprocessing strategy"):
            PreprocessorFactory.create("invalid_strategy")

    def test_list_available_returns_all_strategies(self):
        """list_available returns registered strategies."""
        available = PreprocessorFactory.list_available()

        assert "v1_median" in available
        assert "v2_knn" in available
        assert "v3_iterative" in available


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_create_with_valid_string(self):
        """Factory creates model from valid string."""
        model = ModelFactory.create("random_forest")

        assert model is not None
        assert isinstance(model, BaseModel)

    def test_create_with_valid_enum(self):
        """Factory creates model from enum."""
        model = ModelFactory.create(ModelType.LINEAR)

        assert model is not None
        assert isinstance(model, BaseModel)

    def test_create_with_invalid_string_raises(self):
        """Factory raises ValueError for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create("invalid_model")

    def test_list_available_returns_all_models(self):
        """list_available returns registered models."""
        available = ModelFactory.list_available()

        assert "random_forest" in available
        assert "gradient_boost" in available
        assert "xgboost" in available
        assert "linear" in available
