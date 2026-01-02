"""CLI command to train a model."""

import hashlib
from pathlib import Path

import pandas as pd
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
import src.data.preprocessing.strategies  # noqa: F401
import src.models.strategies  # noqa: F401
from mlflow.models import infer_signature
from src.artifacts.bundle import MLArtifactBundle
from src.cli.utils import (
    config_panel,
    console,
    create_feature_importance_table,
    create_metrics_table,
    error_panel,
    select_option,
    success_panel,
)
from src.config.settings import get_settings
from src.data.loader import get_data_summary, load_housing_data
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from src.data.preprocessing.factory import PreprocessorFactory
from src.models.evaluate import evaluate_model, generate_evaluation_report, save_report
from src.models.factory import ModelFactory


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the dataset for versioning."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:12]


def train(
    model_type: str = typer.Option(
        "gradient_boost",
        "--model-type",
        "-m",
        help=f"Model type: {', '.join(ModelFactory.list_available())}",
    ),
    preprocessing: str = typer.Option(
        "v2_knn",
        "--preprocessing",
        "-p",
        help=f"Preprocessing strategy: {', '.join(PreprocessorFactory.list_available())}",
    ),
    experiment_name: str = typer.Option(
        "housing-price-prediction",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    data_path: Path = typer.Option(
        Path("data/HousingData.csv"),
        "--data",
        "-d",
        help="Path to training data CSV",
    ),
    output_dir: Path = typer.Option(
        Path("models"),
        "--output",
        "-o",
        help="Directory to save model artifacts",
    ),
    test_size: float = typer.Option(
        0.2,
        "--test-size",
        help="Proportion of data for testing",
    ),
    random_state: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    register: bool = typer.Option(
        True,
        "--register/--no-register",
        help="Register model in MLflow Registry",
    ),
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI or http://localhost:5000)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactive mode: select model and preprocessing from menu",
    ),
) -> None:
    """Train a housing price prediction model."""
    # Interactive mode: prompt user for selections
    if interactive:
        console.print("\n[bold]🚀 Interactive Training Mode[/bold]")

        model_type = select_option(
            "Select model type:",
            ModelFactory.list_available(),
            default=model_type,
        )

        preprocessing = select_option(
            "Select preprocessing strategy:",
            PreprocessorFactory.list_available(),
            default=preprocessing,
        )

        console.print()

    # Show configuration
    config = {
        "Model": model_type,
        "Preprocessing": preprocessing,
        "Experiment": experiment_name,
        "Data": str(data_path),
        "Test Size": f"{test_size:.0%}",
        "Random Seed": str(random_state),
        "Register": "Yes" if register else "No",
    }
    console.print(config_panel(config, title="Training Configuration"))
    console.print()

    # Validate inputs
    if model_type not in ModelFactory.list_available():
        console.print(error_panel(f"Unknown model type: {model_type}"))
        raise typer.Exit(1)

    if preprocessing not in PreprocessorFactory.list_available():
        console.print(error_panel(f"Unknown preprocessing strategy: {preprocessing}"))
        raise typer.Exit(1)

    if not data_path.exists():
        console.print(error_panel(f"Data file not found: {data_path}"))
        raise typer.Exit(1)

    # Configure MLflow with S3/MinIO credentials
    settings = get_settings()
    settings.configure_mlflow_s3()

    # Use provided tracking URI or fall back to settings
    effective_tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(effective_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load data
        task = progress.add_task("Loading data...", total=None)
        df = load_housing_data(str(data_path))
        summary = get_data_summary(df)
        dataset_hash = compute_dataset_hash(df)
        progress.update(task, description=f"[green]Loaded {summary['n_rows']} rows")

        # Prepare features
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        # Calculate feature stats for monitoring
        feature_stats = {
            col: {"min": float(X[col].min()), "max": float(X[col].max())} for col in FEATURE_COLUMNS
        }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Preprocess
        progress.update(task, description="Preprocessing...")
        preprocessor = PreprocessorFactory.create(preprocessing)
        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        progress.update(task, description=f"[green]Preprocessed with {preprocessing}")

        # Train
        run_name = f"{model_type}_{preprocessing}"

        with mlflow.start_run(run_name=run_name) as run:
            progress.update(task, description=f"Training {model_type}...")

            # Log parameters
            mlflow.log_params(
                {
                    "model_type": model_type,
                    "preprocessing_strategy": preprocessing,
                    "test_size": test_size,
                    "random_state": random_state,
                    "n_samples": summary["n_rows"],
                    "dataset_hash": dataset_hash,
                }
            )

            # Create and train model
            model = ModelFactory.create(model_type)
            model.train(X_train_transformed, y_train.values, random_state=random_state)

            # Log model params
            mlflow.log_params({f"model_{k}": v for k, v in model.params.items()})

            progress.update(task, description="[green]Training complete")

            # Evaluate
            progress.update(task, description="Evaluating...")
            train_result = evaluate_model(model.model, X_train_transformed, y_train.values)
            test_result = evaluate_model(model.model, X_test_transformed, y_test.values)

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_rmse": train_result["metrics"]["rmse"],
                    "train_mae": train_result["metrics"]["mae"],
                    "train_r2": train_result["metrics"]["r2"],
                    "test_rmse": test_result["metrics"]["rmse"],
                    "test_mae": test_result["metrics"]["mae"],
                    "test_r2": test_result["metrics"]["r2"],
                }
            )

            progress.update(task, description="[green]Evaluation complete")

            # Get feature importance
            feature_importance = model.get_feature_importance(FEATURE_COLUMNS) or {}

            # Save artifacts
            progress.update(task, description="Saving artifacts...")

            bundle = MLArtifactBundle.create(
                model=model,
                preprocessor=preprocessor,
                feature_names=FEATURE_COLUMNS,
                training_samples=len(y_train),
                test_samples=len(y_test),
                train_metrics=train_result["metrics"],
                test_metrics=test_result["metrics"],
                feature_importance=feature_importance,
                feature_stats=feature_stats,
                mlflow_run_id=run.info.run_id,
                mlflow_experiment_name=experiment_name,
                random_state=random_state,
            )

            bundle_path = output_dir / "artifact_bundle"
            bundle.save(bundle_path)

            # Save evaluation report
            report = generate_evaluation_report(
                train_metrics=train_result["metrics"],
                test_metrics=test_result["metrics"],
                feature_importance=feature_importance,
                model_params=model.params,
            )
            save_report(report, output_dir / "evaluation_report.json")

            # Log to MLflow
            mlflow.log_artifacts(str(bundle_path), artifact_path="artifact_bundle")

            # Log model with signature
            signature = infer_signature(
                X_train_transformed,
                model.model.predict(X_train_transformed[:1]),
            )

            mlflow.sklearn.log_model(
                model.model,
                artifact_path="model",
                signature=signature,
                registered_model_name="housing-price-model" if register else None,
            )

            progress.update(task, description="[green]Artifacts saved")

    # Show results
    console.print()
    console.print(create_metrics_table(train_result["metrics"], test_result["metrics"]))
    console.print()

    if feature_importance:
        console.print(create_feature_importance_table(feature_importance))
        console.print()

    # Success message
    result_info = f"""[bold]Run ID:[/bold] {run.info.run_id[:8]}...
[bold]Artifact ID:[/bold] {bundle.metadata.artifact_id[:8]}...
[bold]Bundle saved:[/bold] {bundle_path}
[bold]MLflow UI:[/bold] {effective_tracking_uri}"""

    console.print(success_panel(result_info, title="Training Complete"))
