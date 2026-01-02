"""CLI command to train a model."""

from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import typer
from mlflow import MlflowClient
from mlflow.models import infer_signature
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.model_selection import train_test_split

# Import strategies to register them with factories
import src.data.preprocessing.strategies  # noqa: F401
import src.models.strategies  # noqa: F401
from src.artifacts.bundle import MLArtifactBundle
from src.cli.utils import (
    config_panel,
    confirm_action,
    console,
    create_cv_results_table,
    create_feature_importance_table,
    create_metrics_table,
    error_panel,
    select_hyperparameters,
    select_option,
    success_panel,
)
from src.config.settings import get_settings
from src.data.loader import FEATURE_COLUMNS, TARGET_COLUMN, get_data_summary, load_housing_data
from src.data.preprocessing.factory import PreprocessorFactory
from src.models.cross_validation import CVResult, perform_cross_validation
from src.models.evaluate import evaluate_model, generate_evaluation_report, save_report
from src.models.factory import ModelFactory
from src.utils import compute_dataset_hash

# Get settings for defaults
_settings = get_settings()


def train(
    model_type: str = typer.Option(
        _settings.default_model_type,
        "--model-type",
        "-m",
        help=f"Model type: {', '.join(ModelFactory.list_available())}",
    ),
    preprocessing: str = typer.Option(
        _settings.default_preprocessing,
        "--preprocessing",
        "-p",
        help=f"Preprocessing strategy: {', '.join(PreprocessorFactory.list_available())}",
    ),
    experiment_name: str = typer.Option(
        _settings.mlflow_experiment_name,
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    data_path: Path = typer.Option(
        _settings.data_path,
        "--data",
        "-d",
        help="Path to training data CSV",
    ),
    output_dir: Path = typer.Option(
        _settings.model_dir,
        "--output",
        "-o",
        help="Directory to save model artifacts",
    ),
    test_size: float = typer.Option(
        _settings.default_test_size,
        "--test-size",
        help="Proportion of data for testing",
    ),
    random_state: int = typer.Option(
        _settings.default_random_state,
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
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactive mode: select model and preprocessing from menu",
    ),
    enable_cv: bool = typer.Option(
        False,
        "--cv/--no-cv",
        help="Enable cross-validation",
    ),
    cv_splits: int = typer.Option(
        _settings.default_cv_splits,
        "--cv-splits",
        help="Number of cross-validation folds",
    ),
) -> None:
    """Train a housing price prediction model."""
    # Interactive mode: prompt user for selections
    custom_hyperparams: dict = {}
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

        # Ask if user wants to configure hyperparameters
        if confirm_action("Configure hyperparameters?", default=False):
            custom_hyperparams = select_hyperparameters(model_type)

        # Ask if user wants to enable cross-validation
        if not enable_cv:
            enable_cv = confirm_action("Enable cross-validation?", default=True)

        console.print()

    # Show configuration
    config = {
        "Model": model_type,
        "Preprocessing": preprocessing,
        "Experiment": experiment_name,
        "Data": str(data_path),
        "Test Size": f"{test_size:.0%}",
        "Random Seed": str(random_state),
        "Cross-Validation": f"{cv_splits}-fold" if enable_cv else "Disabled",
        "Register": "Yes" if register else "No",
    }
    if custom_hyperparams:
        config["Custom Params"] = ", ".join(f"{k}={v}" for k, v in custom_hyperparams.items())
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

        # Train - use descriptive run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_type}_{preprocessing}_{timestamp}"

        with mlflow.start_run(run_name=run_name) as run:
            progress.update(task, description=f"Training {model_type}...")

            # Set run tags for filtering and organization (MLflow 3.x)
            mlflow.set_tags(
                {
                    "model_type": model_type,
                    "preprocessing": preprocessing,
                    "source": "cli",
                    "environment": "development",
                }
            )

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
            train_params = {"random_state": random_state, **custom_hyperparams}
            model.train(X_train_transformed, y_train.values, **train_params)

            # Log model params
            mlflow.log_params({f"model_{k}": v for k, v in model.params.items()})

            progress.update(task, description="[green]Training complete")

            # Evaluate
            progress.update(task, description="Evaluating...")
            train_result = evaluate_model(model.model, X_train_transformed, y_train.values)
            test_result = evaluate_model(model.model, X_test_transformed, y_test.values)

            # Log metrics (now includes mape and accuracy_within_10pct)
            mlflow.log_metrics(
                {
                    "train_rmse": train_result["metrics"]["rmse"],
                    "train_mae": train_result["metrics"]["mae"],
                    "train_r2": train_result["metrics"]["r2"],
                    "train_mape": train_result["metrics"]["mape"],
                    "train_accuracy_within_10pct": train_result["metrics"]["accuracy_within_10pct"],
                    "test_rmse": test_result["metrics"]["rmse"],
                    "test_mae": test_result["metrics"]["mae"],
                    "test_r2": test_result["metrics"]["r2"],
                    "test_mape": test_result["metrics"]["mape"],
                    "test_accuracy_within_10pct": test_result["metrics"]["accuracy_within_10pct"],
                }
            )

            progress.update(task, description="[green]Evaluation complete")

            # Cross-validation (optional)
            cv_result: CVResult | None = None
            if enable_cv:
                progress.update(task, description=f"Running {cv_splits}-fold cross-validation...")
                cv_result = perform_cross_validation(
                    model.model,
                    X_train_transformed,
                    y_train.values,
                    n_splits=cv_splits,
                    random_state=random_state,
                )
                # Log CV metrics to MLflow
                mlflow.log_metrics(cv_result.to_dict())
                progress.update(
                    task,
                    description=f"[green]CV complete: RMSE {cv_result.cv_rmse_mean:.4f} ± {cv_result.cv_rmse_std:.4f}",
                )

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
                cv_metrics=cv_result.to_dict() if cv_result else {},
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

            # Log model with descriptive artifact_path for LoggedModel name in MLflow 3.x
            model_artifact_path = f"sklearn_{model_type}"
            mlflow.sklearn.log_model(
                model.model,
                artifact_path=model_artifact_path,
                signature=signature,
            )

            progress.update(task, description="[green]Artifacts saved")

            # Store run info for later use
            run_id = run.info.run_id

    # Show results
    console.print()
    console.print(create_metrics_table(train_result["metrics"], test_result["metrics"]))
    console.print()

    if cv_result:
        console.print(create_cv_results_table(cv_result.to_dict()))
        console.print()

    if feature_importance:
        console.print(create_feature_importance_table(feature_importance))
        console.print()

    # In interactive mode, ask user if they want to register the model
    should_register = register
    if interactive:
        console.print()
        should_register = confirm_action(
            "Register this model in MLflow Registry?", default=False
        )

    # Register model if requested
    registered_version = None
    if should_register:
        model_uri = f"runs:/{run_id}/sklearn_{model_type}"
        model_name = settings.mlflow_model_name

        # Register the model (no alias - use 'meli promote' to assign production)
        result = mlflow.register_model(model_uri, model_name)
        registered_version = result.version

        # Add description and tags to the model version (MLflow 3.x)
        client = MlflowClient()
        version_description = (
            f"{model_type} with {preprocessing} preprocessing. "
            f"Test R²: {test_result['metrics']['r2']:.4f}, "
            f"RMSE: {test_result['metrics']['rmse']:.4f}"
        )
        client.update_model_version(
            name=model_name,
            version=registered_version,
            description=version_description,
        )

        # Add tags to model version for filtering
        client.set_model_version_tag(model_name, registered_version, "model_type", model_type)
        client.set_model_version_tag(model_name, registered_version, "preprocessing", preprocessing)
        client.set_model_version_tag(model_name, registered_version, "test_r2", f"{test_result['metrics']['r2']:.4f}")

        console.print(
            f"[green]Model registered as {model_name} v{registered_version}[/green]\n"
            f"[dim]Use 'uv run meli promote --version {registered_version} --alias production' "
            f"to promote[/dim]"
        )

    # Success message
    result_info = f"""[bold]Run ID:[/bold] {run_id[:8]}...
[bold]Artifact ID:[/bold] {bundle.metadata.artifact_id[:8]}...
[bold]Bundle saved:[/bold] {bundle_path}
[bold]MLflow UI:[/bold] {effective_tracking_uri}"""

    if registered_version:
        result_info += f"\n[bold]Registered:[/bold] v{registered_version}"

    console.print(success_panel(result_info, title="Training Complete"))
