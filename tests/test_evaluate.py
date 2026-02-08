"""Tests for evaluation metrics including business metrics."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.models.evaluate import (
    NumpyEncoder,
    accuracy_within_tolerance,
    calculate_mape,
    calculate_metrics,
    evaluate_model,
    generate_evaluation_report,
    print_metrics,
    save_report,
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

        for _key, value in metrics.items():
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


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_predictions_and_metrics(self):
        """Returns dictionary with predictions and metrics."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([100, 200, 300])

        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        y_test = np.array([100, 200, 300])

        result = evaluate_model(mock_model, X_test, y_test)

        assert "predictions" in result
        assert "metrics" in result

    def test_calls_model_predict(self):
        """Calls model.predict with X_test."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([100])

        X_test = np.array([[1, 2, 3]])
        y_test = np.array([100])

        evaluate_model(mock_model, X_test, y_test)

        mock_model.predict.assert_called_once()
        np.testing.assert_array_equal(mock_model.predict.call_args[0][0], X_test)

    def test_metrics_are_correct(self):
        """Metrics are calculated correctly."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([100, 200, 300])

        X_test = np.array([[1], [2], [3]])
        y_test = np.array([100, 200, 300])

        result = evaluate_model(mock_model, X_test, y_test)

        assert result["metrics"]["rmse"] == 0.0
        assert result["metrics"]["r2"] == 1.0


class TestGenerateEvaluationReport:
    """Tests for generate_evaluation_report function."""

    def test_includes_train_and_test_metrics(self):
        """Report includes train and test metrics."""
        train_metrics = {"r2": 0.95, "rmse": 2.0}
        test_metrics = {"r2": 0.90, "rmse": 3.0}

        report = generate_evaluation_report(train_metrics, test_metrics)

        assert report["train_metrics"] == train_metrics
        assert report["test_metrics"] == test_metrics

    def test_includes_feature_importance_when_provided(self):
        """Report includes feature importance when provided."""
        train_metrics = {"r2": 0.95}
        test_metrics = {"r2": 0.90}
        feature_importance = {"feature_a": 0.5, "feature_b": 0.3}

        report = generate_evaluation_report(
            train_metrics, test_metrics, feature_importance=feature_importance
        )

        assert report["feature_importance"] == feature_importance

    def test_excludes_feature_importance_when_none(self):
        """Report excludes feature importance when not provided."""
        train_metrics = {"r2": 0.95}
        test_metrics = {"r2": 0.90}

        report = generate_evaluation_report(train_metrics, test_metrics)

        assert "feature_importance" not in report

    def test_includes_model_params_when_provided(self):
        """Report includes model params when provided."""
        train_metrics = {"r2": 0.95}
        test_metrics = {"r2": 0.90}
        model_params = {"n_estimators": 100, "max_depth": 5}

        report = generate_evaluation_report(train_metrics, test_metrics, model_params=model_params)

        assert report["model_params"] == model_params

    def test_detects_overfitting(self):
        """Detects overfitting when R2 difference is large."""
        train_metrics = {"r2": 0.99}
        test_metrics = {"r2": 0.70}  # Big drop

        report = generate_evaluation_report(train_metrics, test_metrics)

        assert report["overfitting_warning"] is True

    def test_no_overfitting_warning_when_close(self):
        """No overfitting warning when R2 values are close."""
        train_metrics = {"r2": 0.92}
        test_metrics = {"r2": 0.90}  # Small drop

        report = generate_evaluation_report(train_metrics, test_metrics)

        assert report["overfitting_warning"] is False


class TestNumpyEncoder:
    """Tests for NumpyEncoder JSON encoder."""

    def test_encodes_numpy_int(self):
        """Encodes numpy integers."""
        data = {"value": np.int64(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert '"value": 42' in result

    def test_encodes_numpy_float(self):
        """Encodes numpy floats."""
        data = {"value": np.float64(3.14)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert "3.14" in result

    def test_encodes_numpy_array(self):
        """Encodes numpy arrays as lists."""
        data = {"values": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["values"] == [1, 2, 3]

    def test_encodes_mixed_types(self):
        """Encodes mixed numpy and Python types."""
        data = {
            "int_val": np.int32(10),
            "float_val": np.float32(2.5),
            "array_val": np.array([1, 2]),
            "python_int": 5,
            "python_str": "hello",
        }
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert parsed["int_val"] == 10
        assert abs(parsed["float_val"] - 2.5) < 0.01
        assert parsed["array_val"] == [1, 2]
        assert parsed["python_int"] == 5
        assert parsed["python_str"] == "hello"


class TestSaveReport:
    """Tests for save_report function."""

    def test_creates_json_file(self):
        """Creates a JSON file at specified path."""
        report = {"metrics": {"r2": 0.9}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.json"
            result = save_report(report, output_path)

            assert result.exists()
            assert result == output_path

    def test_creates_parent_directories(self):
        """Creates parent directories if they don't exist."""
        report = {"test": True}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "subdir" / "nested" / "report.json"
            result = save_report(report, output_path)

            assert result.exists()

    def test_writes_valid_json(self):
        """Writes valid JSON content."""
        report = {"metrics": {"r2": 0.85, "rmse": 3.5}, "model": "test"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.json"
            save_report(report, output_path)

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded == report

    def test_handles_numpy_types(self):
        """Handles numpy types in report."""
        report = {
            "value": np.float64(0.95),
            "array": np.array([1, 2, 3]),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.json"
            save_report(report, output_path)

            with open(output_path) as f:
                loaded = json.load(f)

            assert abs(loaded["value"] - 0.95) < 0.01
            assert loaded["array"] == [1, 2, 3]

    def test_accepts_string_path(self):
        """Accepts string path."""
        report = {"test": True}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/report.json"
            result = save_report(report, output_path)

            assert result.exists()


class TestPrintMetrics:
    """Tests for print_metrics function."""

    def test_prints_metrics(self, capsys):
        """Prints metrics to stdout."""
        metrics = {"r2": 0.95, "rmse": 3.5}

        print_metrics(metrics)

        captured = capsys.readouterr()
        assert "R2" in captured.out
        assert "0.95" in captured.out
        assert "RMSE" in captured.out
        assert "3.5" in captured.out

    def test_uses_custom_dataset_name(self, capsys):
        """Uses custom dataset name in header."""
        metrics = {"r2": 0.9}

        print_metrics(metrics, dataset_name="Validation")

        captured = capsys.readouterr()
        assert "Validation" in captured.out

    def test_default_dataset_name_is_test(self, capsys):
        """Default dataset name is 'Test'."""
        metrics = {"r2": 0.9}

        print_metrics(metrics)

        captured = capsys.readouterr()
        assert "Test" in captured.out
