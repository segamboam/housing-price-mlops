"""Prediction service layer.

Encapsulates business logic for predictions, separating it from
HTTP endpoint concerns. This makes the logic testable independently
and keeps endpoints thin.
"""

import time
from dataclasses import dataclass, field

import pandas as pd

from src.api.middleware import (
    record_out_of_range,
    record_prediction,
    record_prediction_value,
)
from src.api.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    HousingFeatures,
    ModelInfoResponse,
    PredictionResponse,
)
from src.artifacts.bundle import MLArtifactBundle
from src.config.settings import Settings
from src.data.loader import FEATURE_COLUMNS


@dataclass
class PredictionResult:
    """Internal result from a single prediction."""

    value: float
    formatted: str
    model_version: str
    model_type: str | None
    served_by: str
    warnings: list[str] = field(default_factory=list)


class PredictionService:
    """Service for housing price predictions.

    Encapsulates prediction logic, feature validation, and price formatting.
    Receives its dependencies (bundle, settings) via constructor injection.

    Args:
        bundle: The loaded ML artifact bundle (model + preprocessor).
        settings: Application settings.
        model_alias: Which alias this bundle corresponds to (champion/challenger).
    """

    def __init__(
        self,
        bundle: MLArtifactBundle,
        settings: Settings,
        model_alias: str = "champion",
    ) -> None:
        self._bundle = bundle
        self._settings = settings
        self._model_alias = model_alias

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: HousingFeatures) -> PredictionResponse:
        """Run a single prediction and return the API response.

        Args:
            features: Validated input features.

        Returns:
            PredictionResponse ready to be returned by the endpoint.
        """
        start_time = time.perf_counter()

        warnings = self._check_feature_ranges(features)

        X = pd.DataFrame([self._features_to_dict(features)])
        prediction = self._bundle.predict(X)[0]
        model_version = self._bundle.metadata.artifact_id[:8]

        duration = time.perf_counter() - start_time
        record_prediction(duration, model_alias=self._model_alias)
        record_prediction_value(float(prediction), model_alias=self._model_alias)

        prediction_value = round(float(prediction), 2)
        return PredictionResponse(
            prediction=prediction_value,
            prediction_formatted=self._format_price(prediction_value),
            model_version=model_version,
            model_type=self._bundle.metadata.model_type,
            served_by=self._model_alias,
            warnings=warnings,
        )

    def predict_batch(self, items: list[HousingFeatures]) -> BatchPredictionResponse:
        """Run batch predictions and return the API response.

        Args:
            items: List of validated input features.

        Returns:
            BatchPredictionResponse ready to be returned by the endpoint.
        """
        start_time = time.perf_counter()

        # Collect per-item warnings
        all_warnings = [self._check_feature_ranges(f) for f in items]

        # Build DataFrame for efficient batch prediction
        X = pd.DataFrame([self._features_to_dict(f) for f in items])

        predictions = self._bundle.predict(X)
        model_version = self._bundle.metadata.artifact_id[:8]
        model_type = self._bundle.metadata.model_type

        # Build response items
        prediction_items = []
        for i, (pred, warnings) in enumerate(zip(predictions, all_warnings)):
            pred_value = round(float(pred), 2)
            prediction_items.append(
                BatchPredictionItem(
                    index=i,
                    prediction=pred_value,
                    prediction_formatted=self._format_price(pred_value),
                    warnings=warnings,
                )
            )
            record_prediction_value(float(pred), model_alias=self._model_alias)

        duration = time.perf_counter() - start_time
        record_prediction(duration, model_alias=self._model_alias)

        return BatchPredictionResponse(
            predictions=prediction_items,
            model_version=model_version,
            model_type=model_type,
            served_by=self._model_alias,
            total_items=len(items),
            processing_time_ms=duration * 1000,
        )

    def get_model_info(self, model_source: str | None) -> ModelInfoResponse:
        """Build model info response from bundle metadata.

        Args:
            model_source: Source identifier for the loaded model.

        Returns:
            ModelInfoResponse ready to be returned by the endpoint.
        """
        metadata = self._bundle.metadata
        return ModelInfoResponse(
            model_type=metadata.model_type,
            preprocessing_strategy=metadata.preprocessing_strategy,
            preprocessing_version=metadata.preprocessing_version,
            feature_names=metadata.feature_names,
            training_samples=metadata.training_samples,
            test_samples=metadata.test_samples if metadata.test_samples else None,
            train_metrics=metadata.train_metrics,
            test_metrics=metadata.test_metrics,
            feature_importance=(
                metadata.feature_importance if metadata.feature_importance else None
            ),
            artifact_id=metadata.artifact_id,
            mlflow_run_id=metadata.mlflow_run_id,
            mlflow_experiment=metadata.mlflow_experiment_name,
            created_at=metadata.created_at.isoformat() if metadata.created_at else None,
            prediction_source=model_source,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_feature_ranges(self, features: HousingFeatures) -> list[str]:
        """Check if input features are within training data ranges.

        Args:
            features: Input features from the request.

        Returns:
            List of warning messages for out-of-range features.
        """
        feature_stats = self._bundle.metadata.feature_stats
        if not feature_stats:
            return []

        warnings: list[str] = []
        for col in FEATURE_COLUMNS:
            value = getattr(features, col)
            stats = feature_stats.get(col)
            if stats is None:
                continue

            min_val = stats.get("min")
            max_val = stats.get("max")

            if min_val is not None and value < min_val:
                warnings.append(f"{col} ({value}) is below training min ({min_val:.2f})")
                record_out_of_range(col)
            elif max_val is not None and value > max_val:
                warnings.append(f"{col} ({value}) is above training max ({max_val:.2f})")
                record_out_of_range(col)

        return warnings

    def _format_price(self, value: float) -> str:
        """Format prediction value as currency string.

        Args:
            value: Raw prediction value (in model units, e.g., $1000s).

        Returns:
            Formatted price string (e.g., "$24,500").
        """
        display_value = value * self._settings.price_multiplier
        return f"{self._settings.currency_symbol}{display_value:,.0f}"

    @staticmethod
    def _features_to_dict(features: HousingFeatures) -> dict[str, float]:
        """Convert HousingFeatures to dict with correct column order."""
        return {col: getattr(features, col) for col in FEATURE_COLUMNS}
