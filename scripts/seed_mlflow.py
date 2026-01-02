#!/usr/bin/env python
"""Seed script to initialize MLflow with a pre-trained model.

This script uploads the pre-trained artifact bundle from seed/ to MLflow
(PostgreSQL + MinIO) and registers it as the production model.

Usage:
    # With Docker infrastructure running:
    uv run python scripts/seed_mlflow.py

    # Or via Makefile:
    make seed
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from mlflow import MlflowClient

from src.config.settings import get_settings


def seed_mlflow() -> None:
    """Seed MLflow with the pre-trained model from seed/artifact_bundle/."""
    settings = get_settings()

    # Configure S3/MinIO access
    settings.configure_mlflow_s3()

    # Set tracking URI
    tracking_uri = settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Seeding MLflow at: {tracking_uri}")
    print(f"Artifact store: {settings.s3_artifact_root}")

    # Check connection
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        print(f"Connected! Found {len(experiments)} existing experiments")
    except Exception as e:
        print(f"ERROR: Cannot connect to MLflow: {e}")
        print("Make sure Docker infrastructure is running: make infra-up")
        sys.exit(1)

    # Check if seed bundle exists
    seed_path = Path("seed/artifact_bundle")
    if not seed_path.exists():
        print(f"ERROR: Seed bundle not found at {seed_path}")
        sys.exit(1)

    # Load metadata from seed bundle
    metadata_path = seed_path / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"\nSeed model info:")
    print(f"  Model type: {metadata.get('model_type', 'unknown')}")
    print(f"  Preprocessing: {metadata.get('preprocessing_strategy', 'unknown')}")
    print(f"  Test R2: {metadata.get('test_metrics', {}).get('r2', 'N/A'):.4f}")

    # Create/get experiment
    experiment_name = "housing-price-prediction"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        print(f"\nCreated experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"\nUsing existing experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)

    # Create a run and log the pre-trained model
    print("\nUploading seed model to MLflow...")

    with mlflow.start_run(run_name="seed_model") as run:
        # Log parameters from metadata
        mlflow.log_params(
            {
                "model_type": metadata.get("model_type", "unknown"),
                "preprocessing_strategy": metadata.get("preprocessing_strategy", "unknown"),
                "source": "seed",
                "training_samples": metadata.get("training_samples", 0),
                "test_samples": metadata.get("test_samples", 0),
            }
        )

        # Log metrics from metadata
        train_metrics = metadata.get("train_metrics", {})
        test_metrics = metadata.get("test_metrics", {})
        mlflow.log_metrics(
            {
                "train_rmse": train_metrics.get("rmse", 0),
                "train_mae": train_metrics.get("mae", 0),
                "train_r2": train_metrics.get("r2", 0),
                "test_rmse": test_metrics.get("rmse", 0),
                "test_mae": test_metrics.get("mae", 0),
                "test_r2": test_metrics.get("r2", 0),
            }
        )

        # Log the artifact bundle
        mlflow.log_artifacts(str(seed_path), artifact_path="artifact_bundle")

        # Load and log the sklearn model
        import joblib

        model_path = seed_path / "model.joblib"
        model = joblib.load(model_path)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name,
        )

        run_id = run.info.run_id
        print(f"  Run ID: {run_id[:8]}...")

    # Set production alias
    print("\nSetting production alias...")
    try:
        # Get the latest version
        versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
        if versions:
            latest_version = max(versions, key=lambda v: int(v.version))
            client.set_registered_model_alias(
                name=settings.mlflow_model_name,
                alias="production",
                version=latest_version.version,
            )
            print(f"  Set 'production' alias to version {latest_version.version}")
    except Exception as e:
        print(f"  Warning: Could not set alias: {e}")

    print("\n" + "=" * 60)
    print("SEED COMPLETE!")
    print("=" * 60)
    print(f"MLflow UI:      {tracking_uri}")
    print(f"MinIO Console:  http://localhost:{settings.minio_console_port}")
    print(f"  User: {settings.minio_root_user}")
    print(f"  Pass: {settings.minio_root_password}")
    print("=" * 60)


if __name__ == "__main__":
    seed_mlflow()
