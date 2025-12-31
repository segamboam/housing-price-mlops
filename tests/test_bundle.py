"""Tests for MLArtifactBundle serialization.

Justification: Bundle is the contract between training and serving.
If save/load breaks, the entire deployment pipeline fails.
"""

import numpy as np
import pytest

from src.artifacts.bundle import MLArtifactBundle
from src.data.preprocessing.factory import PreprocessorFactory
from src.models.factory import ModelFactory


@pytest.fixture
def trained_bundle(sample_dataframe, sample_target):
    """Create a trained bundle for testing."""
    # Train preprocessor
    preprocessor = PreprocessorFactory.create("v1_median")
    X_transformed = preprocessor.fit_transform(sample_dataframe)

    # Train model
    model = ModelFactory.create("linear")
    model.train(X_transformed, sample_target)

    # Create bundle
    bundle = MLArtifactBundle.create(
        model=model,
        preprocessor=preprocessor,
        feature_names=list(sample_dataframe.columns),
        training_samples=len(sample_target),
        test_samples=5,
        train_metrics={"rmse": 2.5, "mae": 1.8, "r2": 0.85},
        test_metrics={"rmse": 3.0, "mae": 2.1, "r2": 0.80},
    )
    return bundle


class TestArtifactBundle:
    """Tests for MLArtifactBundle save/load."""

    def test_save_creates_files(self, trained_bundle, tmp_path):
        """save() creates all required files."""
        bundle_dir = tmp_path / "bundle"

        trained_bundle.save(bundle_dir)

        assert (bundle_dir / "metadata.json").exists()
        assert (bundle_dir / "model.joblib").exists()
        assert (bundle_dir / "preprocessor.joblib").exists()

    def test_load_restores_bundle(self, trained_bundle, tmp_path):
        """load() restores bundle with all components."""
        bundle_dir = tmp_path / "bundle"
        trained_bundle.save(bundle_dir)

        loaded = MLArtifactBundle.load(bundle_dir)

        assert loaded.metadata.model_type == trained_bundle.metadata.model_type
        assert loaded.metadata.training_samples == trained_bundle.metadata.training_samples

    def test_roundtrip_predictions_match(self, trained_bundle, sample_dataframe, tmp_path):
        """Predictions match after save/load roundtrip."""
        bundle_dir = tmp_path / "bundle"

        # Get predictions before save
        original_preds = trained_bundle.predict(sample_dataframe)

        # Save and reload
        trained_bundle.save(bundle_dir)
        loaded = MLArtifactBundle.load(bundle_dir)

        # Get predictions after load
        loaded_preds = loaded.predict(sample_dataframe)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_load_nonexistent_raises(self, tmp_path):
        """load() raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            MLArtifactBundle.load(tmp_path / "nonexistent")

    def test_predict_handles_dataframe(self, trained_bundle, sample_dataframe):
        """predict() works with DataFrame input."""
        predictions = trained_bundle.predict(sample_dataframe)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_dataframe)
