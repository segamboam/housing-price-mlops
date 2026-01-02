"""Single experiment training function."""

import time
from datetime import datetime

import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

# Import strategies to register them with factories
import src.data.preprocessing.strategies  # noqa: F401
import src.models.strategies  # noqa: F401
from src.config.settings import get_settings
from src.data.loader import FEATURE_COLUMNS, TARGET_COLUMN, load_housing_data
from src.data.preprocessing.factory import PreprocessorFactory
from src.experiments.runner import ExperimentConfig, ExperimentResult
from src.models.cross_validation import perform_cross_validation
from src.models.evaluate import evaluate_model
from src.models.factory import ModelFactory
from src.utils import compute_dataset_hash


def train_single_experiment(
    config: ExperimentConfig,
    data_path: str = "data/HousingData.csv",
) -> ExperimentResult:
    """Train a single experiment and return results.

    Args:
        config: Experiment configuration.
        data_path: Path to the dataset.

    Returns:
        ExperimentResult with metrics and run info.
    """
    start_time = time.time()

    # Configure MLflow
    settings = get_settings()
    settings.configure_mlflow_s3()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # Load data
    df = load_housing_data(data_path)
    dataset_hash = compute_dataset_hash(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    # Preprocess
    preprocessor = PreprocessorFactory.create(config.preprocessing)
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Build run name with timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.model_type}_{config.preprocessing}_{timestamp}"

    # Train with MLflow tracking
    with mlflow.start_run(run_name=run_name) as run:
        # Set run tags for filtering and organization (MLflow 3.x)
        mlflow.set_tags(
            {
                "model_type": config.model_type,
                "preprocessing": config.preprocessing,
                "source": "experiment_runner",
                "environment": "development",
            }
        )

        # Log parameters
        mlflow.log_params(
            {
                "model_type": config.model_type,
                "preprocessing": config.preprocessing,
                "test_size": config.test_size,
                "random_state": config.random_state,
                "dataset_hash": dataset_hash,
                "enable_cv": config.enable_cv,
            }
        )

        # Log hyperparameters
        if config.hyperparameters:
            mlflow.log_params(
                {f"hp_{k}": v for k, v in config.hyperparameters.items()}
            )

        # Create and train model
        model = ModelFactory.create(config.model_type)
        model.train(
            X_train_transformed,
            y_train.values,
            random_state=config.random_state,
            **config.hyperparameters,
        )

        # Log model params
        mlflow.log_params({f"model_{k}": v for k, v in model.params.items()})

        # Evaluate
        train_result = evaluate_model(model.model, X_train_transformed, y_train.values)
        test_result = evaluate_model(model.model, X_test_transformed, y_test.values)

        # Log metrics
        for prefix, metrics in [("train", train_result["metrics"]), ("test", test_result["metrics"])]:
            for name, value in metrics.items():
                mlflow.log_metric(f"{prefix}_{name}", value)

        # Cross-validation
        cv_metrics = None
        if config.enable_cv:
            cv_result = perform_cross_validation(
                model.model,
                X_train_transformed,
                y_train.values,
                n_splits=config.cv_splits,
                random_state=config.random_state,
            )
            cv_metrics = cv_result.to_dict()
            mlflow.log_metrics(cv_metrics)

        # Always log model artifact (for later registration if needed)
        model_artifact_path = f"sklearn_{config.model_type}"
        registered_model_name = settings.mlflow_model_name if config.register_model else None
        model_info = mlflow.sklearn.log_model(
            model.model,
            artifact_path=model_artifact_path,
            registered_model_name=registered_model_name,
        )

        # Add description and tags to the model version if registered
        if config.register_model and model_info.registered_model_version:
            client = MlflowClient()
            version = model_info.registered_model_version
            version_description = (
                f"{config.model_type} with {config.preprocessing} preprocessing. "
                f"Test R²: {test_result['metrics']['r2']:.4f}, "
                f"RMSE: {test_result['metrics']['rmse']:.4f}"
            )
            client.update_model_version(
                name=settings.mlflow_model_name,
                version=version,
                description=version_description,
            )
            client.set_model_version_tag(
                settings.mlflow_model_name, version, "model_type", config.model_type
            )
            client.set_model_version_tag(
                settings.mlflow_model_name, version, "preprocessing", config.preprocessing
            )
            client.set_model_version_tag(
                settings.mlflow_model_name,
                version,
                "test_r2",
                f"{test_result['metrics']['r2']:.4f}",
            )

        training_time = time.time() - start_time

        return ExperimentResult(
            run_id=run.info.run_id,
            model_type=config.model_type,
            preprocessing=config.preprocessing,
            hyperparameters=config.hyperparameters,
            train_metrics=train_result["metrics"],
            test_metrics=test_result["metrics"],
            cv_metrics=cv_metrics,
            training_time=training_time,
        )
