"""MLflow helper functions to reduce code duplication."""

import mlflow
from mlflow import MlflowClient

from src.config.settings import get_settings


def initialize_mlflow(
    tracking_uri: str | None = None,
    configure_s3: bool = True,
) -> MlflowClient:
    """Initialize MLflow with consistent configuration.

    Args:
        tracking_uri: Optional custom tracking URI. If None, uses settings.
        configure_s3: Whether to configure S3/MinIO credentials.

    Returns:
        Configured MlflowClient instance.
    """
    settings = get_settings()

    if configure_s3:
        settings.configure_mlflow_s3()

    effective_uri = tracking_uri or str(settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(effective_uri)

    return MlflowClient()


def tag_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    model_type: str,
    preprocessing: str,
    test_r2: float,
    test_rmse: float,
    source: str = "unknown",
) -> None:
    """Add standard description and tags to a model version.

    Args:
        client: MLflow client instance.
        model_name: Name of the registered model.
        version: Version number to tag.
        model_type: Type of model (e.g., 'gradient_boost').
        preprocessing: Preprocessing strategy used.
        test_r2: Test R² score.
        test_rmse: Test RMSE score.
        source: Source of the model (e.g., 'cli', 'experiment_runner').
    """
    version_description = (
        f"{model_type} with {preprocessing} preprocessing. "
        f"Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}"
    )

    client.update_model_version(
        name=model_name,
        version=version,
        description=version_description,
    )

    client.set_model_version_tag(model_name, version, "model_type", model_type)
    client.set_model_version_tag(model_name, version, "preprocessing", preprocessing)
    client.set_model_version_tag(model_name, version, "test_r2", f"{test_r2:.4f}")
    client.set_model_version_tag(model_name, version, "source", source)
