"""Core training logic shared by CLI and experiment runner.

This module uses the preprocessing cache system which:
1. Checks if preprocessed data exists locally
2. If not, tries to pull from S3 (MinIO)
3. If not in remote, creates and pushes to remote

This ensures preprocessed data is computed only once and shared across runs.
"""

import tempfile
from dataclasses import dataclass
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import src.data.preprocessing.strategies  # noqa: F401
import src.models.strategies  # noqa: F401
from src.artifacts.bundle import MLArtifactBundle
from src.config.settings import get_settings
from src.data.cache import get_cached_data
from src.data.loader import FEATURE_COLUMNS, TARGET_COLUMN
from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessorFactory
from src.models.base import BaseModel
from src.models.cross_validation import CVResult, perform_cross_validation
from src.models.evaluate import evaluate_model
from src.models.factory import ModelFactory
from src.utils import compute_dataset_hash
from src.utils.mlflow_helpers import initialize_mlflow


@dataclass
class TrainingResult:
    """Result from training a model."""

    run_id: str
    model: BaseModel
    preprocessor: BasePreprocessor
    bundle: MLArtifactBundle
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    cv_result: CVResult | None
    feature_importance: dict[str, float]
    training_samples: int
    test_samples: int


def train_model(
    model_type: str,
    preprocessing: str,
    data_path: str = "data/HousingData.csv",  # Kept for compatibility but not used
    test_size: float = 0.2,
    random_state: int = 42,
    hyperparameters: dict | None = None,
    enable_cv: bool = True,
    cv_splits: int = 5,
    run_name: str | None = None,
    source_tag: str = "training",
) -> TrainingResult:
    """Train a model with the given configuration.

    This is the core training function used by both the CLI and experiment runner.
    It uses the preprocessing cache system to avoid redundant computation:
    - If cache exists locally, uses it
    - If not, tries to pull from S3 (MinIO)
    - If not in remote, computes and pushes to remote

    Args:
        model_type: Type of model to train (e.g., 'gradient_boost', 'random_forest').
        preprocessing: Preprocessing strategy (e.g., 'v2_knn', 'v3_iterative').
        data_path: Path to the training data CSV (kept for compatibility).
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.
        hyperparameters: Optional model hyperparameters.
        enable_cv: Whether to perform cross-validation.
        cv_splits: Number of CV folds.
        run_name: Optional custom run name for MLflow.
        source_tag: Tag to identify the source (e.g., 'cli', 'experiment_runner').

    Returns:
        TrainingResult with all training artifacts and metrics.
    """
    hyperparameters = hyperparameters or {}

    # Configure MLflow
    settings = get_settings()
    initialize_mlflow()
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # Get preprocessed data from cache (or create if needed)
    X_train, X_test, y_train, y_test, metadata, preprocessor_pipeline = get_cached_data(
        preprocessing_version=preprocessing,
        test_size=test_size,
        random_state=random_state,
    )

    # Reconstruct preprocessor object with the cached pipeline
    preprocessor = PreprocessorFactory.create(preprocessing)
    preprocessor.pipeline = preprocessor_pipeline
    preprocessor.feature_names = FEATURE_COLUMNS

    # Calculate feature stats from preprocessed data
    feature_stats = {
        col: {
            "min": float(X_train[:, i].min()),
            "max": float(X_train[:, i].max()),
        }
        for i, col in enumerate(FEATURE_COLUMNS)
    }

    # Build run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_type}_{preprocessing}_{timestamp}"

    # Train with MLflow tracking
    with mlflow.start_run(run_name=run_name) as run:
        # Set run tags
        mlflow.set_tags(
            {
                "model_type": model_type,
                "preprocessing": preprocessing,
                "preprocessing_version": metadata.get("preprocessing_version", preprocessing),
                "source": source_tag,
                "environment": "development",
                "cache_used": "true",
            }
        )

        # Log parameters
        mlflow.log_params(
            {
                "model_type": model_type,
                "preprocessing_strategy": preprocessing,
                "test_size": test_size,
                "random_state": random_state,
                "n_train_samples": len(y_train),
                "n_test_samples": len(y_test),
                "enable_cv": enable_cv,
            }
        )

        # Log hyperparameters
        if hyperparameters:
            mlflow.log_params({f"hp_{k}": v for k, v in hyperparameters.items()})

        # Create and train model
        model = ModelFactory.create(model_type)
        train_params = {"random_state": random_state, **hyperparameters}
        model.train(X_train, y_train, **train_params)

        # Log model params
        mlflow.log_params({f"model_{k}": v for k, v in model.params.items()})

        # Evaluate
        train_result = evaluate_model(model.model, X_train, y_train)
        test_result = evaluate_model(model.model, X_test, y_test)

        # Log metrics
        for prefix, metrics in [
            ("train", train_result["metrics"]),
            ("test", test_result["metrics"]),
        ]:
            for name, value in metrics.items():
                mlflow.log_metric(f"{prefix}_{name}", value)

        # Cross-validation
        cv_result: CVResult | None = None
        if enable_cv:
            cv_result = perform_cross_validation(
                model.model,
                X_train,
                y_train,
                n_splits=cv_splits,
                random_state=random_state,
            )
            mlflow.log_metrics(cv_result.to_dict())

        # Get feature importance
        feature_importance = model.get_feature_importance(FEATURE_COLUMNS) or {}

        # Create artifact bundle
        bundle = MLArtifactBundle.create(
            model=model,
            preprocessor=preprocessor,
            feature_names=FEATURE_COLUMNS,
            training_samples=len(y_train),
            test_samples=len(y_test),
            train_metrics=train_result["metrics"],
            test_metrics=test_result["metrics"],
            cv_metrics=cv_result.to_dict() if cv_result else {},
            feature_importance=feature_importance,
            feature_stats=feature_stats,
            mlflow_run_id=run.info.run_id,
            mlflow_experiment_name=settings.mlflow_experiment_name,
            random_state=random_state,
        )

        # Save bundle to temp directory and log to MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = bundle.save(f"{tmp_dir}/artifact_bundle")
            mlflow.log_artifacts(str(bundle_path), artifact_path="artifact_bundle")

        # Log model with signature
        signature = infer_signature(
            X_train,
            model.model.predict(X_train[:1]),
        )

        model_artifact_path = f"sklearn_{model_type}"
        mlflow.sklearn.log_model(
            model.model,
            artifact_path=model_artifact_path,
            signature=signature,
        )

        return TrainingResult(
            run_id=run.info.run_id,
            model=model,
            preprocessor=preprocessor,
            bundle=bundle,
            train_metrics=train_result["metrics"],
            test_metrics=test_result["metrics"],
            cv_result=cv_result,
            feature_importance=feature_importance,
            training_samples=len(y_train),
            test_samples=len(y_test),
        )
