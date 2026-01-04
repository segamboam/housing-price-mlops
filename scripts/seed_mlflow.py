#!/usr/bin/env python
"""Seed script to initialize MLflow with a pre-trained model.

This script uploads the pre-trained artifact bundle from seed/ to MLflow
(PostgreSQL + MinIO) and registers it as the production model.

It also ensures DVC is initialized for data versioning.

Usage:
    # With Docker infrastructure running:
    uv run python scripts/seed_mlflow.py

    # Or via Makefile:
    make seed
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from src.config.settings import get_settings
from src.utils.mlflow_helpers import initialize_mlflow, tag_model_version


def ensure_dvc_initialized() -> None:
    """Ensure DVC is initialized for the ML pipeline.
    
    This runs the DVC initialization which:
    1. Initializes DVC if needed
    2. Configures MinIO remote
    3. Verifies raw data exists in Git
    """
    from scripts.init_dvc import init_dvc
    init_dvc()
    print()


def seed_mlflow() -> None:
    """Seed MLflow with the pre-trained model from seed/artifact_bundle/."""
    # Ensure DVC is initialized first
    ensure_dvc_initialized()
    
    settings = get_settings()

    # Initialize MLflow
    print(f"Seeding MLflow at: {settings.mlflow_tracking_uri}")
    print(f"Artifact store: {settings.s3_artifact_root}")

    # Check connection
    try:
        client = initialize_mlflow()
        experiments = client.search_experiments()
        print(f"Connected! Found {len(experiments)} existing experiments")
    except Exception as e:
        print(f"ERROR: Cannot connect to MLflow: {e}")
        print("Make sure Docker infrastructure is running: make infra-up")
        sys.exit(1)

    # Check if model already exists with production alias
    try:
        model_version = client.get_model_version_by_alias(
            name=settings.mlflow_model_name, alias="production"
        )
        print(f"\nModel already seeded! Production alias points to version {model_version.version}")
        print("Skipping seed (use --force to re-seed)")
        sys.exit(0)
    except mlflow.exceptions.MlflowException:
        # No production alias exists, proceed with seeding
        pass

    # Check if seed bundle exists
    seed_path = Path("seed/artifact_bundle")
    if not seed_path.exists():
        print(f"ERROR: Seed bundle not found at {seed_path}")
        sys.exit(1)

    # Load metadata from seed bundle
    metadata_path = seed_path / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    print("\nSeed model info:")
    print(f"  Model type: {metadata.get('model_type', 'unknown')}")
    print(f"  Preprocessing: {metadata.get('preprocessing_strategy', 'unknown')}")
    print(f"  Test R2: {metadata.get('test_metrics', {}).get('r2', 'N/A'):.4f}")

    # Create/get experiment
    experiment_name = settings.mlflow_experiment_name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name)
        print(f"\nCreated experiment: {experiment_name}")
    else:
        print(f"\nUsing existing experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)

    # Create a run and log the pre-trained model
    print("\nUploading seed model to MLflow...")

    # Extract model info for descriptive naming
    model_type = metadata.get("model_type", "unknown")
    preprocessing = metadata.get("preprocessing_strategy", "unknown")
    run_name = f"seed_{model_type}_{preprocessing}"

    with mlflow.start_run(run_name=run_name) as run:
        # Set run tags for filtering and organization (MLflow 3.x)
        mlflow.set_tags(
            {
                "model_type": model_type,
                "preprocessing": preprocessing,
                "source": "seed",
                "environment": "demo",
            }
        )

        # Log parameters from metadata
        mlflow.log_params(
            {
                "model_type": model_type,
                "preprocessing_strategy": preprocessing,
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

        # Use descriptive artifact_path for LoggedModel name in MLflow 3.x
        artifact_path = f"sklearn_{model_type}"
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=settings.mlflow_model_name,
        )

        run_id = run.info.run_id
        print(f"  Run ID: {run_id[:8]}...")

    # Set production alias and add model version metadata
    print("\nSetting production alias and metadata...")
    try:
        # Get the latest version
        versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
        if versions:
            latest_version = max(versions, key=lambda v: int(v.version))
            version_num = latest_version.version

            # Add description and tags
            test_r2 = test_metrics.get("r2", 0)
            test_rmse = test_metrics.get("rmse", 0)
            tag_model_version(
                client=client,
                model_name=settings.mlflow_model_name,
                version=version_num,
                model_type=model_type,
                preprocessing=preprocessing,
                test_r2=test_r2,
                test_rmse=test_rmse,
                source="seed",
            )

            # Set production alias
            client.set_registered_model_alias(
                name=settings.mlflow_model_name,
                alias="production",
                version=version_num,
            )
            print(f"  Set 'production' alias to version {version_num}")
            print(f"  Added description and tags to version {version_num}")
    except Exception as e:
        print(f"  Warning: Could not set alias/metadata: {e}")

    print("\n" + "=" * 60)
    print("SEED COMPLETE!")
    print("=" * 60)
    print(f"MLflow UI:      {settings.mlflow_tracking_uri}")
    print(f"MinIO Console:  http://localhost:{settings.minio_console_port}")
    print(f"  User: {settings.minio_root_user}")
    print(f"  Pass: {settings.minio_root_password}")
    print("=" * 60)


if __name__ == "__main__":
    seed_mlflow()
