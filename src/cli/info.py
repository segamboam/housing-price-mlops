"""CLI command to show model information."""

import json
from pathlib import Path

import typer

from src.cli.utils import (
    console,
    create_feature_importance_table,
    create_metrics_table,
    error_panel,
    info_panel,
)
from src.config.settings import get_settings

settings = get_settings()


def info(
    bundle_path: Path = typer.Option(
        None,
        "--bundle",
        "-b",
        help="Path to artifact bundle (default: models/artifact_bundle)",
    ),
) -> None:
    """Show information about the current model."""
    if bundle_path is None:
        bundle_path = settings.artifact_bundle_path

    metadata_path = bundle_path / "metadata.json"

    if not metadata_path.exists():
        console.print(error_panel(f"No model found at {bundle_path}"))
        raise typer.Exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Model info panel
    model_info = f"""[bold]Model Type:[/bold] {metadata.get('model_type', 'unknown')}
[bold]Preprocessing:[/bold] {metadata.get('preprocessing_strategy', 'unknown')} v{metadata.get('preprocessing_version', '?')}
[bold]Artifact ID:[/bold] {metadata.get('artifact_id', 'unknown')[:8]}...
[bold]Created:[/bold] {metadata.get('created_at', 'unknown')[:19]}
[bold]Training Samples:[/bold] {metadata.get('training_samples', 0)}
[bold]Test Samples:[/bold] {metadata.get('test_samples', 0)}
[bold]MLflow Run:[/bold] {metadata.get('mlflow_run_id', 'N/A')[:8] if metadata.get('mlflow_run_id') else 'N/A'}..."""

    console.print(info_panel(model_info, title="Model Information"))
    console.print()

    # Metrics table
    train_metrics = metadata.get("train_metrics", {})
    test_metrics = metadata.get("test_metrics", {})
    if train_metrics or test_metrics:
        console.print(create_metrics_table(train_metrics, test_metrics))
        console.print()

    # Feature importance table
    feature_importance = metadata.get("feature_importance", {})
    if feature_importance:
        console.print(create_feature_importance_table(feature_importance))
        console.print()

    # Feature stats summary
    feature_stats = metadata.get("feature_stats", {})
    if feature_stats:
        console.print(f"[dim]Feature statistics available for {len(feature_stats)} features[/dim]")
