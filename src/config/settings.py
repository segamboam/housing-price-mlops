"""Centralized application settings using pydantic-settings."""

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

    # Model Settings
    model_dir: Path = Path("models")
    model_name: str = "housing_model"

    # MLflow Settings
    mlflow_tracking_uri: str | None = None
    mlflow_model_name: str = "housing-price-model"
    mlflow_model_alias: str = "champion"

    # Prometheus Settings
    metrics_enabled: bool = True

    @property
    def model_path(self) -> Path:
        """Path to the model file."""
        return self.model_dir / f"{self.model_name}.joblib"

    @property
    def scaler_path(self) -> Path:
        """Path to the scaler file."""
        return self.model_dir / f"{self.model_name}_scaler.joblib"

    @property
    def api_key_required(self) -> bool:
        """Check if API key authentication is required."""
        return self.api_key is not None and len(self.api_key) > 0


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
