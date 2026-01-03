"""CLI command to train a model."""

import mlflow
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

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
from src.data.preprocessing.factory import PreprocessorFactory
from src.models.evaluate import generate_evaluation_report, save_report
from src.models.factory import ModelFactory
from src.training.core import train_model
from src.utils.mlflow_helpers import initialize_mlflow, tag_model_version

# Get settings for defaults
_settings = get_settings()


def train() -> None:
    """Train a housing price prediction model interactively."""
    console.print("\n[bold]🚀 Interactive Training Mode[/bold]")

    # Interactive selections
    model_type = select_option(
        "Select model type:",
        ModelFactory.list_available(),
        default=_settings.default_model_type,
    )

    preprocessing = select_option(
        "Select preprocessing strategy:",
        PreprocessorFactory.list_available(),
        default=_settings.default_preprocessing,
    )

    # Ask if user wants to configure hyperparameters
    custom_hyperparams: dict = {}
    if confirm_action("Configure hyperparameters?", default=False):
        custom_hyperparams = select_hyperparameters(model_type)

    # Ask if user wants to enable cross-validation
    enable_cv = confirm_action("Enable cross-validation?", default=True)

    console.print()

    # Use settings for all other values
    data_path = _settings.data_path
    output_dir = _settings.model_dir
    test_size = _settings.default_test_size
    random_state = _settings.default_random_state
    cv_splits = _settings.default_cv_splits

    # Show configuration
    config = {
        "Model": model_type,
        "Preprocessing": preprocessing,
        "Experiment": _settings.mlflow_experiment_name,
        "Data": str(data_path),
        "Test Size": f"{test_size:.0%}",
        "Random Seed": str(random_state),
        "Cross-Validation": f"{cv_splits}-fold" if enable_cv else "Disabled",
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

    # Train with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training model...", total=None)

        # Use shared training logic
        result = train_model(
            model_type=model_type,
            preprocessing=preprocessing,
            data_path=str(data_path),
            test_size=test_size,
            random_state=random_state,
            hyperparameters=custom_hyperparams,
            enable_cv=enable_cv,
            cv_splits=cv_splits,
            source_tag="cli",
        )

        progress.update(task, description="[green]Training complete")

        # Save artifacts locally
        progress.update(task, description="Saving local artifacts...")
        bundle_path = output_dir / "artifact_bundle"
        result.bundle.save(bundle_path)

        # Save evaluation report
        report = generate_evaluation_report(
            train_metrics=result.train_metrics,
            test_metrics=result.test_metrics,
            feature_importance=result.feature_importance,
            model_params=result.model.params,
        )
        save_report(report, output_dir / "evaluation_report.json")
        progress.update(task, description="[green]Artifacts saved")

    # Show results
    console.print()
    console.print(create_metrics_table(result.train_metrics, result.test_metrics))
    console.print()

    if result.cv_result:
        console.print(create_cv_results_table(result.cv_result.to_dict()))
        console.print()

    if result.feature_importance:
        console.print(create_feature_importance_table(result.feature_importance))
        console.print()

    # Ask user if they want to register the model
    console.print()
    should_register = confirm_action("Register this model in MLflow Registry?", default=True)

    # Register model if requested
    settings = get_settings()
    registered_version = None
    if should_register:
        model_uri = f"runs:/{result.run_id}/sklearn_{model_type}"
        model_name = settings.mlflow_model_name

        # Register the model
        reg_result = mlflow.register_model(model_uri, model_name)
        registered_version = reg_result.version

        # Add description and tags to the model version
        client = initialize_mlflow()
        tag_model_version(
            client=client,
            model_name=model_name,
            version=registered_version,
            model_type=model_type,
            preprocessing=preprocessing,
            test_r2=result.test_metrics["r2"],
            test_rmse=result.test_metrics["rmse"],
            source="cli",
        )

        console.print(
            f"[green]Model registered as {model_name} v{registered_version}[/green]\n"
            f"[dim]Use 'make promote VERSION={registered_version}' to promote[/dim]"
        )

    # Success message
    result_info = f"""[bold]Run ID:[/bold] {result.run_id[:8]}...
[bold]Artifact ID:[/bold] {result.bundle.metadata.artifact_id[:8]}...
[bold]Bundle saved:[/bold] {bundle_path}
[bold]MLflow UI:[/bold] {settings.mlflow_tracking_uri}"""

    if registered_version:
        result_info += f"\n[bold]Registered:[/bold] v{registered_version}"

    console.print(success_panel(result_info, title="Training Complete"))
