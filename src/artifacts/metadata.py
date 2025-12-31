"""Artifact metadata schema for ML bundles."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ArtifactMetadata(BaseModel):
    """Metadata for an ML artifact bundle.

    Contains all information needed to reconstruct and understand
    a trained model and its preprocessing pipeline.
    """

    # Identifiers
    artifact_id: str = Field(description="Unique artifact identifier (UUID)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when artifact was created",
    )

    # Model information
    model_type: str = Field(
        description="Model strategy type (e.g., 'random_forest', 'gradient_boost')"
    )
    model_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters used for training",
    )

    # Preprocessing information
    preprocessing_strategy: str = Field(
        description="Preprocessing strategy (e.g., 'v1_median', 'v2_knn')"
    )
    preprocessing_version: str = Field(
        description="Version of the preprocessing strategy",
    )

    # Feature information
    feature_names: list[str] = Field(
        description="Ordered list of feature names as used during training"
    )
    target_name: str = Field(
        default="MEDV",
        description="Name of the target variable",
    )

    # Training information
    training_samples: int = Field(
        description="Number of samples used for training",
    )
    test_samples: int = Field(
        default=0,
        description="Number of samples used for testing",
    )
    random_state: int = Field(
        default=42,
        description="Random seed used for reproducibility",
    )

    # Feature statistics for monitoring
    feature_stats: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Statistics per feature (min, max) from training data for drift detection",
    )

    # Metrics
    train_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metrics on training set (rmse, mae, r2)",
    )
    test_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metrics on test set (rmse, mae, r2)",
    )
    feature_importance: dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance scores",
    )

    # MLflow tracking
    mlflow_run_id: str | None = Field(
        default=None,
        description="MLflow run ID if tracked",
    )
    mlflow_experiment_name: str | None = Field(
        default=None,
        description="MLflow experiment name if tracked",
    )

    # Versioning
    artifact_version: str = Field(
        default="1.0.0",
        description="Semantic version of the artifact format",
    )
    framework_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Versions of key frameworks (scikit-learn, python, etc.)",
    )

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }
