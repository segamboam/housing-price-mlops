"""Centralized application settings using pydantic-settings."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Settings
    api_title: str = "Housing Price Prediction API"
    api_version: str = "1.0.0"
    api_key: str | None = None
    api_key_header: str = "X-API-Key"
    api_port: int = 8000

    # Model Settings
    model_dir: Path = Path("models")
    model_name: str = "housing_model"
    artifact_bundle_dir: str = "artifact_bundle"

    # MLflow Settings
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_name: str = "housing-price-model"
    mlflow_model_alias: str = "production"
    mlflow_port: int = 5000
    mlflow_bucket_name: str = "mlflow-artifacts"

    # PostgreSQL Settings (MLflow backend store)
    postgres_user: str = "mlflow"
    postgres_password: str = "mlflow123"
    postgres_db: str = "mlflow"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # MinIO/S3 Settings (artifact storage)
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin123"
    minio_api_port: int = 9000
    minio_console_port: int = 9001
    mlflow_s3_endpoint_url: str = "http://localhost:9000"

    # Prometheus Settings
    metrics_enabled: bool = True

    # ML Defaults
    default_model_type: str = "gradient_boost"
    default_preprocessing: str = "v2_knn"
    default_test_size: float = 0.2
    default_random_state: int = 42
    default_cv_splits: int = 5

    # ML Evaluation Thresholds
    # Threshold for R² difference between train/test to flag overfitting
    overfitting_r2_threshold: float = 0.1
    # RMSE gap threshold for learning curve overfitting detection
    overfitting_rmse_threshold: float = 0.15
    # Default tolerance for accuracy_within_tolerance metric (10%)
    accuracy_tolerance: float = 0.10

    # API Limits
    # Maximum items allowed in batch prediction requests
    batch_max_items: int = 100

    # Prediction Display
    # Currency symbol for formatted predictions
    currency_symbol: str = "$"
    # Multiplier to convert model output to display units (e.g., 1000 for $1000s)
    price_multiplier: float = 1000.0

    # MLflow Experiment
    mlflow_experiment_name: str = "housing-price-prediction"

    # Data
    data_path: Path = Path("data/HousingData.csv")

    @property
    def model_path(self) -> Path:
        """Path to the model file."""
        return self.model_dir / f"{self.model_name}.joblib"

    @property
    def scaler_path(self) -> Path:
        """Path to the scaler file."""
        return self.model_dir / f"{self.model_name}_scaler.joblib"

    @property
    def artifact_bundle_path(self) -> Path:
        """Path to the artifact bundle directory."""
        return self.model_dir / self.artifact_bundle_dir

    @property
    def api_key_required(self) -> bool:
        """Check if API key authentication is required."""
        return self.api_key is not None and len(self.api_key) > 0

    @property
    def postgres_uri(self) -> str:
        """PostgreSQL connection URI for MLflow backend store."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def s3_artifact_root(self) -> str:
        """S3 artifact root for MLflow."""
        return f"s3://{self.mlflow_bucket_name}"

    def configure_mlflow_s3(self) -> None:
        """Configure environment variables for S3/MinIO access.

        MLflow uses boto3 which reads credentials from environment variables.
        Call this method before any MLflow operations that involve artifacts.
        """
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = self.minio_root_user
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.minio_root_password


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
