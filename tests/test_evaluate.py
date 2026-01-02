"""Tests for evaluation metrics including business metrics."""

import numpy as np

from src.models.evaluate import (
    accuracy_within_tolerance,
    calculate_mape,
    calculate_metrics,
)


class TestMAPE:
    """Tests for Mean Absolute Percentage Error calculation."""

    def test_mape_perfect_predictions(self):
        """MAPE should be 0 for perfect predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        mape = calculate_mape(y_true, y_pred)
        assert mape == 0.0

    def test_mape_known_error(self):
        """MAPE should match expected value for known errors."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 220, 330])  # All 10% over
        mape = calculate_mape(y_true, y_pred)
        assert abs(mape - 10.0) < 0.01

    def test_mape_mixed_errors(self):
        """MAPE should average percentage errors correctly."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([90, 100, 110])  # -10%, 0%, +10%
        mape = calculate_mape(y_true, y_pred)
        # Average of |10|, |0|, |10| = 6.67%
        assert abs(mape - 6.67) < 0.1

    def test_mape_handles_zeros_in_true(self):
        """MAPE should skip zero values in y_true to avoid division by zero."""
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 220])
        mape = calculate_mape(y_true, y_pred)
        # Only calculates for non-zero: (10/100 + 20/200) / 2 = 10%
        assert isinstance(mape, float)
        assert mape > 0

    def test_mape_all_zeros(self):
        """MAPE should return 0 if all true values are zero."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([10, 20, 30])
        mape = calculate_mape(y_true, y_pred)
        assert mape == 0.0

    def test_mape_negative_values(self):
        """MAPE should work with negative values."""
        y_true = np.array([-100, -200, 300])
        y_pred = np.array([-90, -220, 330])
        mape = calculate_mape(y_true, y_pred)
        assert mape > 0


class TestAccuracyWithinTolerance:
    """Tests for accuracy within tolerance calculation."""

    def test_all_within_tolerance(self):
        """Should return 100% if all predictions are within tolerance."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([105, 195, 310])  # All within 10%
        acc = accuracy_within_tolerance(y_true, y_pred, tolerance=0.10)
        assert acc == 100.0

    def test_none_within_tolerance(self):
        """Should return 0% if no predictions are within tolerance."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([150, 300, 450])  # All 50% off
        acc = accuracy_within_tolerance(y_true, y_pred, tolerance=0.10)
        assert acc == 0.0

    def test_partial_within_tolerance(self):
        """Should return correct percentage for mixed results."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([95, 105, 150])  # 2/3 within 10%
        acc = accuracy_within_tolerance(y_true, y_pred, tolerance=0.10)
        assert abs(acc - 66.67) < 1

    def test_exact_tolerance_boundary(self):
        """Predictions exactly at tolerance should be included."""
        y_true = np.array([100])
        y_pred = np.array([110])  # Exactly 10%
        acc = accuracy_within_tolerance(y_true, y_pred, tolerance=0.10)
        assert acc == 100.0

    def test_custom_tolerance(self):
        """Should work with different tolerance values."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([95, 105, 115])  # 5%, 5%, 15% error

        # With 10% tolerance: 2/3 = 66.67%
        acc_10 = accuracy_within_tolerance(y_true, y_pred, tolerance=0.10)
        assert abs(acc_10 - 66.67) < 1

        # With 20% tolerance: 3/3 = 100%
        acc_20 = accuracy_within_tolerance(y_true, y_pred, tolerance=0.20)
        assert acc_20 == 100.0

    def test_default_tolerance_is_10_percent(self):
        """Default tolerance should be 10%."""
        y_true = np.array([100])
        y_pred = np.array([109])  # 9% error
        acc = accuracy_within_tolerance(y_true, y_pred)  # No tolerance specified
        assert acc == 100.0


class TestCalculateMetrics:
    """Tests for the combined metrics calculation function."""

    def test_includes_all_metrics(self):
        """Should include all expected metrics."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 195, 310, 390, 520])
        metrics = calculate_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert "accuracy_within_10pct" in metrics

    def test_metrics_are_rounded(self):
        """All metrics should be rounded to 4 decimal places."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([101, 199, 302])
        metrics = calculate_metrics(y_true, y_pred)

        for key, value in metrics.items():
            # Check that value has at most 4 decimal places
            assert round(value, 4) == value

    def test_perfect_predictions(self):
        """Perfect predictions should have RMSE=0, MAE=0, R2=1, MAPE=0, Acc=100."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["mape"] == 0.0
        assert metrics["accuracy_within_10pct"] == 100.0

    def test_metrics_consistency(self):
        """Metrics should be consistent with each other."""
        y_true = np.array([20, 25, 30, 35, 40])
        y_pred = np.array([22, 24, 32, 33, 42])
        metrics = calculate_metrics(y_true, y_pred)

        # RMSE should be >= MAE
        assert metrics["rmse"] >= metrics["mae"]

        # R2 should be between 0 and 1 for reasonable predictions
        assert 0 <= metrics["r2"] <= 1

        # MAPE should be positive
        assert metrics["mape"] >= 0

        # Accuracy should be between 0 and 100
        assert 0 <= metrics["accuracy_within_10pct"] <= 100
