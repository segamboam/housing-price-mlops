"""CLI command to register an existing run as a model version."""

import mlflow
import typer

from src.cli.utils import console, error_panel, success_panel
from src.config.settings import get_settings
from src.utils.mlflow_helpers import initialize_mlflow, tag_model_version

settings = get_settings()


def register(
    run_id: str = typer.Argument(
        ...,
        help="MLflow run ID to register (can be short prefix)",
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        "-m",
        help="Name for the registered model (default: from settings)",
    ),
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default: from settings)",
    ),
) -> None:
    """Register an existing MLflow run as a model version.

    This allows you to register models from experiment runs that were
    trained without --register flag.

    Example:
        uv run python -m src.cli.main register abc123def
        uv run python -m src.cli.main register abc123def --model-name my-model
    """
    # Configure MLflow
    client = initialize_mlflow(tracking_uri=tracking_uri)
    uri = tracking_uri or str(settings.mlflow_tracking_uri)

    console.print(f"[dim]Tracking URI: {uri}[/dim]\n")

    # Find the run
    try:
        # Search for run by ID prefix
        runs = client.search_runs(
            experiment_ids=[],  # Search all experiments
            filter_string=f"run_id LIKE '{run_id}%'",
            max_results=10,
        )

        if not runs:
            # Try exact match
            try:
                run = client.get_run(run_id)
                runs = [run]
            except mlflow.exceptions.MlflowException:
                console.print(error_panel(f"No run found with ID starting with '{run_id}'"))
                raise typer.Exit(1)

        if len(runs) > 1:
            console.print(f"[yellow]Multiple runs found matching '{run_id}':[/yellow]\n")
            for r in runs:
                model_type = r.data.tags.get("model_type", "unknown")
                preprocessing = r.data.tags.get("preprocessing", "unknown")
                console.print(f"  - {r.info.run_id[:8]}... ({model_type}, {preprocessing})")
            console.print("\n[yellow]Please provide a more specific run ID[/yellow]")
            raise typer.Exit(1)

        run = runs[0]
        full_run_id = run.info.run_id

        # Get run info
        model_type = run.data.tags.get("model_type", "unknown")
        preprocessing = run.data.tags.get("preprocessing", "unknown")
        test_r2 = run.data.metrics.get("test_r2", 0)
        test_rmse = run.data.metrics.get("test_rmse", 0)

        console.print(f"[bold]Found run:[/bold] {full_run_id[:8]}...")
        console.print(f"  Model type: {model_type}")
        console.print(f"  Preprocessing: {preprocessing}")
        console.print(f"  Test R²: {test_r2:.4f}")
        console.print(f"  Test RMSE: {test_rmse:.4f}")
        console.print()

        # Check if run has artifact_bundle
        artifacts = client.list_artifacts(full_run_id)
        artifact_names = [a.path for a in artifacts]

        if "artifact_bundle" not in artifact_names:
            console.print(
                error_panel(
                    "This run doesn't have an artifact_bundle.\n"
                    "Only runs from 'make train' can be registered.\n"
                    "Experiment runs don't include the preprocessor bundle."
                )
            )
            raise typer.Exit(1)

        # Build model artifact path (MLflow 3.x convention)
        model_artifact_path = f"sklearn_{model_type}"

        # Register the model
        effective_model_name = model_name or settings.mlflow_model_name
        model_uri = f"runs:/{full_run_id}/{model_artifact_path}"

        console.print("[bold]Registering model...[/bold]")
        console.print(f"  URI: {model_uri}")
        console.print(f"  Model name: {effective_model_name}")
        console.print()

        result = mlflow.register_model(model_uri, effective_model_name)
        version = result.version

        # Add description and tags
        tag_model_version(
            client=client,
            model_name=effective_model_name,
            version=version,
            model_type=model_type,
            preprocessing=preprocessing,
            test_r2=test_r2,
            test_rmse=test_rmse,
            source="cli_register",
        )

        console.print(
            success_panel(
                f"Registered as {effective_model_name} v{version}\n\n"
                f"To promote to production:\n"
                f"  make promote VERSION={version}",
                title="Registration Complete",
            )
        )

    except mlflow.exceptions.MlflowException as e:
        console.print(error_panel(f"MLflow error: {e}"))
        raise typer.Exit(1)
