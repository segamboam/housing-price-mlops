"""Tests for model strategies.

Justification: Models are the core ML component.
Must verify train/predict flow works correctly.
"""

import numpy as np
import pytest

from src.models.factory import ModelFactory


class TestModelTrainPredict:
    """Tests for model training and prediction flow."""

    def test_predict_without_train_raises(self):
        """predict() without train() raises RuntimeError."""
        model = ModelFactory.create("linear")
        X = np.random.randn(5, 13)

        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.predict(X)

    def test_train_sets_fitted_flag(self, sample_dataframe, sample_target):
        """train() sets internal fitted state."""
        model = ModelFactory.create("linear")
        X = sample_dataframe.values

        model.train(X, sample_target)

        assert model._is_fitted

    def test_predict_returns_array(self, sample_dataframe, sample_target):
        """predict() returns numpy array after training."""
        model = ModelFactory.create("linear")
        X = sample_dataframe.values

        model.train(X, sample_target)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_target)

    def test_all_models_train_and_predict(self, sample_dataframe, sample_target):
        """All registered models can train and predict."""
        model_types = ["random_forest", "gradient_boost", "xgboost", "linear"]
        X = sample_dataframe.values

        for model_type in model_types:
            model = ModelFactory.create(model_type)
            model.train(X, sample_target)
            predictions = model.predict(X)

            assert isinstance(predictions, np.ndarray), f"{model_type} failed"
            assert len(predictions) == len(sample_target), f"{model_type} wrong length"

    def test_feature_importance_available(self, sample_dataframe, sample_target):
        """Models provide feature importance after training."""
        model = ModelFactory.create("random_forest")
        X = sample_dataframe.values
        feature_names = list(sample_dataframe.columns)

        model.train(X, sample_target)
        importance = model.get_feature_importance(feature_names)

        assert importance is not None
        assert len(importance) == len(feature_names)
        assert all(v >= 0 for v in importance.values())
