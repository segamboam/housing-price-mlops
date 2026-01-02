"""End-to-end tests for the complete ML pipeline.

Justification: E2E tests verify the entire flow works together:
train → save bundle → load in API → predict.

These tests don't require external services (MLflow, PostgreSQL, MinIO)
because they use the bundle directly, simulating a production deployment.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.artifacts.bundle import MLArtifactBundle
from src.data.preprocessing.factory import PreprocessorFactory
from src.models.factory import ModelFactory


class TestEndToEndPipeline:
    """E2E tests for train → deploy → predict flow."""

    @pytest.fixture
    def training_data(self):
        """Create realistic training data."""
        np.random.seed(42)
        n_samples = 100

        # Simulate Boston Housing-like data
        data = {
            "CRIM": np.random.exponential(3, n_samples),
            "ZN": np.random.uniform(0, 100, n_samples),
            "INDUS": np.random.uniform(0, 28, n_samples),
            "CHAS": np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),
            "NOX": np.random.uniform(0.38, 0.87, n_samples),
            "RM": np.random.normal(6.3, 0.7, n_samples),
            "AGE": np.random.uniform(2, 100, n_samples),
            "DIS": np.random.uniform(1, 12, n_samples),
            "RAD": np.random.choice(range(1, 25), n_samples),
            "TAX": np.random.uniform(187, 711, n_samples),
            "PTRATIO": np.random.uniform(12, 22, n_samples),
            "B": np.random.uniform(0, 397, n_samples),
            "LSTAT": np.random.uniform(1, 38, n_samples),
        }
        X = pd.DataFrame(data)

        # Target with realistic relationship to features
        y = (
            20
            + 4 * X["RM"]
            - 0.5 * X["LSTAT"]
            - 0.1 * X["CRIM"]
            + np.random.normal(0, 2, n_samples)
        )

        return X, y.values

    @pytest.fixture
    def trained_bundle_path(self, training_data, tmp_path):
        """Train a model and save as bundle."""
        X, y = training_data

        # Split data
        train_size = 80
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train preprocessor
        preprocessor = PreprocessorFactory.create("v1_median")
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Train model
        model = ModelFactory.create("random_forest")
        model.train(X_train_transformed, y_train)

        # Evaluate
        train_preds = model.predict(X_train_transformed)
        test_preds = model.predict(X_test_transformed)

        train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_preds) ** 2))

        # Create and save bundle
        bundle = MLArtifactBundle.create(
            model=model,
            preprocessor=preprocessor,
            feature_names=list(X.columns),
            training_samples=train_size,
            test_samples=len(y_test),
            train_metrics={"rmse": float(train_rmse), "r2": 0.9},
            test_metrics={"rmse": float(test_rmse), "r2": 0.85},
        )

        bundle_path = tmp_path / "model_bundle"
        bundle.save(bundle_path)

        return bundle_path, X_test, y_test

    def test_e2e_train_save_load_predict(self, trained_bundle_path):
        """Complete E2E: train model, save bundle, load and predict."""
        bundle_path, X_test, y_test = trained_bundle_path

        # Load bundle (simulates what API does at startup)
        loaded_bundle = MLArtifactBundle.load(bundle_path)

        # Make predictions
        predictions = loaded_bundle.predict(X_test)

        # Verify predictions are reasonable
        assert len(predictions) == len(y_test)
        assert all(pred > 0 for pred in predictions)  # House prices are positive
        assert all(pred < 100 for pred in predictions)  # Reasonable upper bound

        # Verify model has acceptable performance
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        assert rmse < 10, f"RMSE {rmse} is too high for this synthetic data"

    def test_e2e_api_with_trained_bundle(self, trained_bundle_path, monkeypatch):
        """E2E: train model, save bundle, serve via API, make prediction."""
        bundle_path, X_test, _ = trained_bundle_path

        # Load the bundle we trained
        loaded_bundle = MLArtifactBundle.load(bundle_path)

        # Mock the API to use our bundle instead of loading from MLflow
        monkeypatch.setattr("src.api.security.settings.api_key", None)
        monkeypatch.setattr(
            "src.api.main.load_bundle_from_mlflow",
            lambda alias=None: (loaded_bundle, "e2e_test"),
        )
        monkeypatch.setattr(
            "src.api.main.load_artifact_bundle",
            lambda: (loaded_bundle, "e2e_test"),
        )

        # Import app after patching
        from src.api.main import app

        # Create test client and make prediction
        with TestClient(app) as client:
            # Health check
            health_response = client.get("/health")
            assert health_response.status_code == 200
            assert health_response.json()["model_loaded"] is True

            # Single prediction
            sample = X_test.iloc[0].to_dict()
            predict_response = client.post("/predict", json=sample)

            assert predict_response.status_code == 200
            data = predict_response.json()
            assert "prediction" in data
            assert "prediction_formatted" in data
            assert isinstance(data["prediction"], float)
            assert data["prediction"] > 0

            # Batch prediction
            batch_samples = X_test.head(5).to_dict(orient="records")
            batch_response = client.post("/predict/batch", json={"items": batch_samples})

            assert batch_response.status_code == 200
            batch_data = batch_response.json()
            assert batch_data["total_items"] == 5
            assert len(batch_data["predictions"]) == 5

    def test_e2e_different_models_produce_different_results(self, training_data, tmp_path):
        """Verify different model types produce different predictions."""
        X, y = training_data
        X_train, y_train = X[:80], y[:80]
        X_test = X[80:]

        predictions_by_model = {}

        for model_type in ["linear", "random_forest", "gradient_boost"]:
            # Train
            preprocessor = PreprocessorFactory.create("v1_median")
            X_transformed = preprocessor.fit_transform(X_train)

            model = ModelFactory.create(model_type)
            model.train(X_transformed, y_train)

            # Create bundle
            bundle = MLArtifactBundle.create(
                model=model,
                preprocessor=preprocessor,
                feature_names=list(X.columns),
                training_samples=80,
                test_samples=20,
                train_metrics={"rmse": 1.0, "r2": 0.9},
                test_metrics={"rmse": 1.5, "r2": 0.85},
            )

            # Save and reload
            bundle_path = tmp_path / f"bundle_{model_type}"
            bundle.save(bundle_path)
            loaded = MLArtifactBundle.load(bundle_path)

            # Predict
            predictions_by_model[model_type] = loaded.predict(X_test)

        # Verify models produce different predictions
        linear_preds = predictions_by_model["linear"]
        rf_preds = predictions_by_model["random_forest"]
        gb_preds = predictions_by_model["gradient_boost"]

        # They shouldn't be identical
        assert not np.allclose(linear_preds, rf_preds, rtol=0.01)
        assert not np.allclose(rf_preds, gb_preds, rtol=0.01)

    def test_e2e_preprocessing_strategies_are_consistent(self, training_data, tmp_path):
        """Verify preprocessing is applied consistently in bundle."""
        X, y = training_data
        X_train, y_train = X[:80], y[:80]
        X_test = X[80:]

        # Train with specific preprocessor
        preprocessor = PreprocessorFactory.create("v1_median")
        X_transformed = preprocessor.fit_transform(X_train)

        model = ModelFactory.create("linear")
        model.train(X_transformed, y_train)

        bundle = MLArtifactBundle.create(
            model=model,
            preprocessor=preprocessor,
            feature_names=list(X.columns),
            training_samples=80,
            test_samples=20,
            train_metrics={"rmse": 1.0, "r2": 0.9},
            test_metrics={"rmse": 1.5, "r2": 0.85},
        )

        # Predictions before save
        preds_before = bundle.predict(X_test)

        # Save, reload, predict again
        bundle_path = tmp_path / "bundle_preproc"
        bundle.save(bundle_path)
        loaded = MLArtifactBundle.load(bundle_path)
        preds_after = loaded.predict(X_test)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(preds_before, preds_after)

        # Verify preprocessor metadata is preserved
        assert loaded.metadata.preprocessing_strategy == "v1_median"


class TestEndToEndEdgeCases:
    """E2E tests for edge cases and error handling."""

    def test_e2e_prediction_with_extreme_values(self, sample_dataframe, sample_target, tmp_path):
        """Model handles extreme input values gracefully."""
        # Train a simple bundle
        preprocessor = PreprocessorFactory.create("v1_median")
        X_transformed = preprocessor.fit_transform(sample_dataframe)

        model = ModelFactory.create("linear")
        model.train(X_transformed, sample_target)

        bundle = MLArtifactBundle.create(
            model=model,
            preprocessor=preprocessor,
            feature_names=list(sample_dataframe.columns),
            training_samples=len(sample_target),
            test_samples=5,
            train_metrics={"rmse": 1.0, "r2": 0.9},
            test_metrics={"rmse": 1.5, "r2": 0.85},
        )

        bundle_path = tmp_path / "bundle_extreme"
        bundle.save(bundle_path)
        loaded = MLArtifactBundle.load(bundle_path)

        # Create extreme input
        extreme_input = sample_dataframe.iloc[[0]].copy()
        extreme_input["CRIM"] = 1000  # Way above training range
        extreme_input["RM"] = 1  # Way below training range

        # Should not crash, but prediction may be unreliable
        prediction = loaded.predict(extreme_input)
        assert len(prediction) == 1
        assert np.isfinite(prediction[0])  # Should not be NaN or Inf

    def test_e2e_bundle_metadata_integrity(self, sample_dataframe, sample_target, tmp_path):
        """Bundle preserves all metadata after save/load."""
        preprocessor = PreprocessorFactory.create("v2_knn")
        X_transformed = preprocessor.fit_transform(sample_dataframe)

        model = ModelFactory.create("gradient_boost")
        model.train(X_transformed, sample_target)

        original_metrics = {
            "rmse": 2.5,
            "mae": 1.8,
            "r2": 0.85,
            "mape": 8.5,
            "accuracy_within_10pct": 92.0,
        }

        bundle = MLArtifactBundle.create(
            model=model,
            preprocessor=preprocessor,
            feature_names=list(sample_dataframe.columns),
            training_samples=len(sample_target),
            test_samples=5,
            train_metrics=original_metrics,
            test_metrics={"rmse": 3.0, "mae": 2.1, "r2": 0.80},
        )

        bundle_path = tmp_path / "bundle_metadata"
        bundle.save(bundle_path)
        loaded = MLArtifactBundle.load(bundle_path)

        # Verify metadata
        assert loaded.metadata.model_type == "gradient_boost"
        assert loaded.metadata.preprocessing_strategy == "v2_knn"
        assert loaded.metadata.training_samples == len(sample_target)
        assert loaded.metadata.feature_names == list(sample_dataframe.columns)
        assert loaded.metadata.train_metrics["rmse"] == 2.5
        assert loaded.metadata.train_metrics["accuracy_within_10pct"] == 92.0
