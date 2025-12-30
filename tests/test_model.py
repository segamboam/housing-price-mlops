"""Tests for model training and evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data.preprocessing import FEATURE_COLUMNS, preprocess_pipeline
from src.models.evaluate import (
    calculate_metrics,
    evaluate_model,
    generate_evaluation_report,
    save_report,
)
from src.models.train import (
    DEFAULT_PARAMS,
    get_feature_importance,
    load_model,
    load_scaler,
    save_model,
    train_model,
)


class TestModelTraining:
    """Tests for model training functions."""

    def test_train_model_default_params(self, sample_housing_data: pd.DataFrame):
        """Test training with default parameters."""
        data = preprocess_pipeline(sample_housing_data)
        model = train_model(data["X_train_scaled"], data["y_train"])

        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == DEFAULT_PARAMS["n_estimators"]
        assert model.max_depth == DEFAULT_PARAMS["max_depth"]

    def test_train_model_custom_params(self, sample_housing_data: pd.DataFrame):
        """Test training with custom parameters."""
        data = preprocess_pipeline(sample_housing_data)
        custom_params = {"n_estimators": 50, "max_depth": 5}

        model = train_model(data["X_train_scaled"], data["y_train"], params=custom_params)

        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_train_model_can_predict(self, sample_housing_data: pd.DataFrame):
        """Test that trained model can make predictions."""
        data = preprocess_pipeline(sample_housing_data)
        model = train_model(data["X_train_scaled"], data["y_train"])

        predictions = model.predict(data["X_test_scaled"])

        assert len(predictions) == len(data["y_test"])
        assert all(isinstance(p, (int, float, np.floating)) for p in predictions)

    def test_save_model(self, sample_housing_data: pd.DataFrame, tmp_path: Path):
        """Test saving model and scaler."""
        data = preprocess_pipeline(sample_housing_data)
        model = train_model(data["X_train_scaled"], data["y_train"])

        paths = save_model(model, data["scaler"], tmp_path)

        assert paths["model_path"].exists()
        assert paths["scaler_path"].exists()
        assert paths["model_path"].suffix == ".joblib"
        assert paths["scaler_path"].suffix == ".joblib"

    def test_load_model(self, trained_model_artifacts: dict):
        """Test loading a saved model."""
        loaded_model = load_model(trained_model_artifacts["model_path"])

        assert isinstance(loaded_model, RandomForestRegressor)

    def test_load_scaler(self, trained_model_artifacts: dict):
        """Test loading a saved scaler."""
        loaded_scaler = load_scaler(trained_model_artifacts["scaler_path"])

        # Check scaler can transform data
        original_scaler = trained_model_artifacts["scaler"]
        assert loaded_scaler.mean_.shape == original_scaler.mean_.shape

    def test_model_persistence_predictions_match(self, trained_model_artifacts: dict):
        """Test that loaded model produces same predictions as original."""
        original_model = trained_model_artifacts["model"]
        loaded_model = load_model(trained_model_artifacts["model_path"])
        test_data = trained_model_artifacts["data"]["X_test_scaled"]

        original_preds = original_model.predict(test_data)
        loaded_preds = loaded_model.predict(test_data)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_get_feature_importance(self, sample_housing_data: pd.DataFrame):
        """Test feature importance extraction."""
        data = preprocess_pipeline(sample_housing_data)
        model = train_model(data["X_train_scaled"], data["y_train"])

        importance = get_feature_importance(model, FEATURE_COLUMNS)

        assert len(importance) == len(FEATURE_COLUMNS)
        assert all(isinstance(v, float) for v in importance.values())
        # Check sorted in descending order
        values = list(importance.values())
        assert values == sorted(values, reverse=True)


class TestModelEvaluation:
    """Tests for model evaluation functions."""

    def test_calculate_metrics(self):
        """Test metric calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = calculate_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > 0.9  # Should be high for close predictions

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = y_true.copy()

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_evaluate_model(self, sample_housing_data: pd.DataFrame):
        """Test model evaluation."""
        data = preprocess_pipeline(sample_housing_data)
        model = train_model(data["X_train_scaled"], data["y_train"])

        result = evaluate_model(model, data["X_test_scaled"], data["y_test"])

        assert "predictions" in result
        assert "metrics" in result
        assert len(result["predictions"]) == len(data["y_test"])

    def test_generate_evaluation_report(self):
        """Test evaluation report generation."""
        train_metrics = {"rmse": 1.0, "mae": 0.8, "r2": 0.95}
        test_metrics = {"rmse": 1.5, "mae": 1.0, "r2": 0.85}

        report = generate_evaluation_report(train_metrics, test_metrics)

        assert "train_metrics" in report
        assert "test_metrics" in report
        assert "overfitting_warning" in report

    def test_generate_evaluation_report_with_feature_importance(self):
        """Test report with feature importance."""
        train_metrics = {"rmse": 1.0, "mae": 0.8, "r2": 0.95}
        test_metrics = {"rmse": 1.5, "mae": 1.0, "r2": 0.85}
        feature_importance = {"RM": 0.5, "LSTAT": 0.3}

        report = generate_evaluation_report(
            train_metrics, test_metrics, feature_importance=feature_importance
        )

        assert "feature_importance" in report

    def test_generate_evaluation_report_overfitting_warning(self):
        """Test overfitting warning detection."""
        train_metrics = {"r2": 0.99}
        test_metrics = {"r2": 0.70}  # Large gap indicates overfitting

        report = generate_evaluation_report(train_metrics, test_metrics)

        assert report["overfitting_warning"] is True

    def test_generate_evaluation_report_no_overfitting(self):
        """Test no overfitting warning when gap is small."""
        train_metrics = {"r2": 0.90}
        test_metrics = {"r2": 0.85}

        report = generate_evaluation_report(train_metrics, test_metrics)

        assert report["overfitting_warning"] is False

    def test_save_report(self, tmp_path: Path):
        """Test saving evaluation report."""
        report = {
            "train_metrics": {"rmse": 1.0},
            "test_metrics": {"rmse": 1.5},
        }
        output_path = tmp_path / "report.json"

        saved_path = save_report(report, output_path)

        assert saved_path.exists()
        # Verify content
        import json

        with open(saved_path) as f:
            loaded = json.load(f)
        assert loaded == report
