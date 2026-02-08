"""Model sanity tests — validates the seed artifact bundle in CI.

These tests load the pre-trained model from seed/artifact_bundle/ and verify
that it can produce valid predictions. This acts as a regression gate: if
someone changes preprocessing, features, or model code in a way that breaks
the shipped model, CI will catch it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SEED_BUNDLE_DIR = Path(__file__).resolve().parent.parent / "seed" / "artifact_bundle"

# Known input from the first row of the Boston Housing dataset
KNOWN_INPUT = {
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.9,
    "LSTAT": 4.98,
}


@pytest.fixture(scope="module")
def bundle():
    """Load the seed artifact bundle once for all tests in this module."""
    from src.artifacts.bundle import MLArtifactBundle

    return MLArtifactBundle.load(SEED_BUNDLE_DIR)


class TestSeedBundleIntegrity:
    """Verify the seed model bundle loads and has valid metadata."""

    def test_bundle_loads_successfully(self, bundle):
        """The seed bundle directory must contain all required files."""
        assert bundle is not None

    def test_metadata_has_expected_features(self, bundle):
        """Metadata must list exactly 13 housing features."""
        assert len(bundle.metadata.feature_names) == 13
        assert "RM" in bundle.metadata.feature_names
        assert "LSTAT" in bundle.metadata.feature_names

    def test_metadata_has_test_metrics(self, bundle):
        """Metadata must include test set metrics."""
        metrics = bundle.metadata.test_metrics
        assert "r2" in metrics
        assert "rmse" in metrics

    def test_test_r2_above_threshold(self, bundle):
        """The seed model must have a test R2 score above 0.80."""
        r2 = bundle.metadata.test_metrics["r2"]
        assert r2 > 0.80, f"Test R2 ({r2:.4f}) is below minimum threshold 0.80"

    def test_test_rmse_below_threshold(self, bundle):
        """The seed model must have a test RMSE below 5.0 ($5k)."""
        rmse = bundle.metadata.test_metrics["rmse"]
        assert rmse < 5.0, f"Test RMSE ({rmse:.4f}) exceeds maximum threshold 5.0"


class TestSeedBundlePredictions:
    """Verify the seed model produces valid predictions."""

    def test_single_prediction_is_finite_positive(self, bundle):
        """A prediction on known input must be a finite positive number."""
        df = pd.DataFrame([KNOWN_INPUT])
        prediction = bundle.predict(df)

        assert len(prediction) == 1
        assert np.isfinite(prediction[0]), f"Prediction is not finite: {prediction[0]}"
        assert prediction[0] > 0, f"House price must be positive, got: {prediction[0]}"

    def test_prediction_in_reasonable_range(self, bundle):
        """Prediction for typical Boston suburb should be between $5k and $50k."""
        df = pd.DataFrame([KNOWN_INPUT])
        prediction = bundle.predict(df)[0]

        assert 5.0 < prediction < 50.0, (
            f"Prediction {prediction:.2f} outside reasonable range [5, 50] "
            f"(values are in $1000s)"
        )

    def test_batch_predictions_shape(self, bundle):
        """Batch of 5 identical inputs must return 5 predictions."""
        df = pd.DataFrame([KNOWN_INPUT] * 5)
        predictions = bundle.predict(df)

        assert predictions.shape == (5,)
        assert all(np.isfinite(predictions))

    def test_predictions_deterministic(self, bundle):
        """Two calls with the same input must return identical results."""
        df = pd.DataFrame([KNOWN_INPUT])
        pred_a = bundle.predict(df)[0]
        pred_b = bundle.predict(df)[0]

        assert pred_a == pred_b, f"Non-deterministic: {pred_a} != {pred_b}"
