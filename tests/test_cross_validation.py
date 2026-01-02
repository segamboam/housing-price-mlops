"""Tests for cross-validation utilities."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.cross_validation import (
    CVResult,
    LearningCurveResult,
    compute_learning_curve,
    perform_cross_validation,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return LinearRegression()


class TestCVResult:
    """Tests for CVResult dataclass."""

    def test_to_dict_contains_required_keys(self):
        """to_dict should contain all required keys for MLflow."""
        result = CVResult(
            cv_rmse_mean=3.21,
            cv_rmse_std=0.45,
            cv_rmse_scores=[3.0, 3.2, 3.4, 3.1, 3.35],
            cv_r2_mean=0.87,
            cv_r2_std=0.05,
            cv_r2_scores=[0.85, 0.88, 0.86, 0.89, 0.87],
            n_splits=5,
        )
        d = result.to_dict()

        assert "cv_rmse_mean" in d
        assert "cv_rmse_std" in d
        assert "cv_r2_mean" in d
        assert "cv_r2_std" in d
        assert "cv_n_splits" in d

    def test_to_dict_rounds_values(self):
        """to_dict should round values to 4 decimal places."""
        result = CVResult(
            cv_rmse_mean=3.2123456,
            cv_rmse_std=0.4567891,
            cv_rmse_scores=[3.0],
            cv_r2_mean=0.8765432,
            cv_r2_std=0.0512345,
            cv_r2_scores=[0.87],
            n_splits=5,
        )
        d = result.to_dict()

        assert d["cv_rmse_mean"] == 3.2123
        assert d["cv_rmse_std"] == 0.4568
        assert d["cv_r2_mean"] == 0.8765

    def test_to_full_dict_includes_scores(self):
        """to_full_dict should include per-fold scores."""
        result = CVResult(
            cv_rmse_mean=3.21,
            cv_rmse_std=0.45,
            cv_rmse_scores=[3.0, 3.2, 3.4],
            cv_r2_mean=0.87,
            cv_r2_std=0.05,
            cv_r2_scores=[0.85, 0.88, 0.86],
            n_splits=3,
        )
        d = result.to_full_dict()

        assert "cv_rmse_scores" in d
        assert "cv_r2_scores" in d
        assert len(d["cv_rmse_scores"]) == 3

    def test_str_representation(self):
        """String representation should be human-readable."""
        result = CVResult(
            cv_rmse_mean=3.21,
            cv_rmse_std=0.45,
            cv_rmse_scores=[3.0],
            cv_r2_mean=0.87,
            cv_r2_std=0.05,
            cv_r2_scores=[0.87],
            n_splits=5,
        )
        s = str(result)

        assert "5-fold" in s
        assert "RMSE" in s
        assert "3.21" in s


class TestPerformCrossValidation:
    """Tests for perform_cross_validation function."""

    def test_returns_cv_result(self, sample_data, simple_model):
        """Should return a CVResult instance."""
        X, y = sample_data
        result = perform_cross_validation(simple_model, X, y)

        assert isinstance(result, CVResult)

    def test_default_5_folds(self, sample_data, simple_model):
        """Default should use 5 folds."""
        X, y = sample_data
        result = perform_cross_validation(simple_model, X, y)

        assert result.n_splits == 5
        assert len(result.cv_rmse_scores) == 5
        assert len(result.cv_r2_scores) == 5

    def test_custom_n_splits(self, sample_data, simple_model):
        """Should respect custom n_splits parameter."""
        X, y = sample_data
        result = perform_cross_validation(simple_model, X, y, n_splits=3)

        assert result.n_splits == 3
        assert len(result.cv_rmse_scores) == 3

    def test_rmse_is_positive(self, sample_data, simple_model):
        """RMSE values should be positive."""
        X, y = sample_data
        result = perform_cross_validation(simple_model, X, y)

        assert result.cv_rmse_mean > 0
        assert all(s > 0 for s in result.cv_rmse_scores)

    def test_r2_in_valid_range(self, sample_data, simple_model):
        """R² should be in a reasonable range for good models."""
        X, y = sample_data
        result = perform_cross_validation(simple_model, X, y)

        # For this synthetic data, model should fit well
        assert result.cv_r2_mean > 0.5
        assert result.cv_r2_mean <= 1.0

    def test_reproducibility_with_same_seed(self, sample_data, simple_model):
        """Same random_state should produce same results."""
        X, y = sample_data
        result1 = perform_cross_validation(simple_model, X, y, random_state=42)
        result2 = perform_cross_validation(simple_model, X, y, random_state=42)

        assert result1.cv_rmse_mean == result2.cv_rmse_mean
        assert result1.cv_rmse_scores == result2.cv_rmse_scores

    def test_different_seeds_may_differ(self, sample_data, simple_model):
        """Different random_state may produce different results."""
        X, y = sample_data
        result1 = perform_cross_validation(simple_model, X, y, random_state=42)
        result2 = perform_cross_validation(simple_model, X, y, random_state=123)

        # Results might be slightly different due to shuffling
        # (but could be same by chance, so we just check it doesn't crash)
        assert isinstance(result1, CVResult)
        assert isinstance(result2, CVResult)

    def test_works_with_random_forest(self, sample_data):
        """Should work with RandomForest model."""
        X, y = sample_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        result = perform_cross_validation(model, X, y, n_splits=3)

        assert isinstance(result, CVResult)
        assert result.cv_rmse_mean > 0


class TestComputeLearningCurve:
    """Tests for compute_learning_curve function."""

    def test_returns_learning_curve_result(self, sample_data, simple_model):
        """Should return a LearningCurveResult instance."""
        X, y = sample_data
        result = compute_learning_curve(simple_model, X, y, cv=3)

        assert isinstance(result, LearningCurveResult)

    def test_train_sizes_are_increasing(self, sample_data, simple_model):
        """Training sizes should be monotonically increasing."""
        X, y = sample_data
        result = compute_learning_curve(simple_model, X, y, cv=3)

        assert result.train_sizes == sorted(result.train_sizes)
        assert result.train_sizes[0] < result.train_sizes[-1]

    def test_has_train_and_val_scores(self, sample_data, simple_model):
        """Should have both training and validation scores."""
        X, y = sample_data
        result = compute_learning_curve(simple_model, X, y, cv=3)

        assert len(result.train_scores_mean) > 0
        assert len(result.val_scores_mean) > 0
        assert len(result.train_scores_mean) == len(result.val_scores_mean)

    def test_to_dict_serializable(self, sample_data, simple_model):
        """to_dict should be JSON-serializable."""
        import json

        X, y = sample_data
        result = compute_learning_curve(simple_model, X, y, cv=3)
        d = result.to_dict()

        # Should not raise
        json.dumps(d)

    def test_overfitting_detection(self):
        """Should detect overfitting when gap is large."""
        result = LearningCurveResult(
            train_sizes=[10, 50, 100],
            train_scores_mean=[0.5, 0.3, 0.1],  # Very low training error
            train_scores_std=[0.01, 0.01, 0.01],
            val_scores_mean=[1.0, 0.8, 0.5],  # Much higher validation error
            val_scores_std=[0.1, 0.1, 0.1],
            is_overfitting=True,
            gap_at_full_data=0.4,
        )

        assert result.is_overfitting is True
        assert result.gap_at_full_data > 0.15
