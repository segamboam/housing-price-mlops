"""CLI command to promote a model to production."""

from datetime import datetime

import typer

import mlflow
from mlflow import MlflowClient
from src.cli.utils import console, create_versions_table, error_panel, success_panel
from src.config.settings import get_settings

settings = get_settings()


def get_model_versions(client: MlflowClient, model_name: str) -> list[dict]:
    """Get all versions of a model with their aliases."""
    try:
        model = client.get_registered_model(model_name)
        aliases = model.aliases or {}

        versions = client.search_model_versions(f"name='{model_name}'")
        result = []

        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            version_aliases = [alias for alias, ver in aliases.items() if ver == v.version]
            created = datetime.fromtimestamp(v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M")
            result.append(
                {
                    "version": v.version,
                    "aliases": version_aliases,
                    "run_id": v.run_id,
                    "created": created,
                }
            )

        return result
    except mlflow.exceptions.MlflowException:
        return []


def promote(
    version: str = typer.Option(
        None,
        "--version",
        "-v",
        help="Model version to promote",
    ),
    alias: str = typer.Option(
        "production",
        "--alias",
        "-a",
        help="Alias to assign (default: production)",
    ),
    model_name: str = typer.Option(
        "housing-price-model",
        "--model-name",
        "-m",
        help="Name of the registered model",
    ),
    tracking_uri: str = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default: from settings)",
    ),
    list_versions: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all model versions",
    ),
) -> None:
    """Promote a model version to production (or other alias)."""
    uri = tracking_uri or str(settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    console.print(f"[dim]Tracking URI: {uri}[/dim]\n")

    # Get current versions
    versions = get_model_versions(client, model_name)

    if not versions:
        console.print(error_panel(f"No versions found for model '{model_name}'"))
        raise typer.Exit(1)

    # List mode
    if list_versions or version is None:
        console.print(f"[bold]Model:[/bold] {model_name}\n")
        console.print(create_versions_table(versions))

        if version is None and not list_versions:
            console.print("\n[yellow]Tip: Use --version to promote a specific version[/yellow]")
        return

    # Promote mode
    try:
        # Verify version exists
        client.get_model_version(model_name, version)

        # Find current production model
        current_production = None
        for v in versions:
            if alias in v.get("aliases", []):
                current_production = v
                break

        # Show what will change
        console.print(f"[bold]Promoting version {version} to '{alias}'[/bold]\n")

        if current_production:
            console.print(
                f"[yellow]Current {alias}:[/yellow] v{current_production['version']} "
                f"(run: {current_production['run_id'][:8]})"
            )
        console.print(f"[green]New {alias}:[/green] v{version}\n")

        # Perform promotion
        client.set_registered_model_alias(model_name, alias, version)

        console.print(
            success_panel(
                f"Model '{model_name}' version {version} promoted to '{alias}'",
                title="Promotion Complete",
            )
        )

        # Show updated versions
        console.print()
        updated_versions = get_model_versions(client, model_name)
        console.print(create_versions_table(updated_versions))

    except mlflow.exceptions.MlflowException as e:
        console.print(error_panel(f"Failed to promote model: {e}"))
        raise typer.Exit(1)
