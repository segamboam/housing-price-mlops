"""CLI command to list MLflow runs."""

from datetime import datetime

import typer

import mlflow
from src.cli.utils import console, create_runs_table, error_panel
from src.config.settings import get_settings

settings = get_settings()


def runs(
    experiment: str = typer.Option(
        "housing-price-prediction",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Maximum number of runs to show",
    ),
    tracking_uri: str = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default: from settings)",
    ),
) -> None:
    """List recent MLflow experiment runs."""
    uri = tracking_uri or str(settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(uri)

    try:
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            console.print(error_panel(f"Experiment '{experiment}' not found"))
            raise typer.Exit(1)

        runs_data = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )

        if runs_data.empty:
            console.print(error_panel(f"No runs found in experiment '{experiment}'"))
            raise typer.Exit(1)

        # Convert to list of dicts for our table
        runs_list = []
        for _, row in runs_data.iterrows():
            start_time = row.get("start_time")
            if isinstance(start_time, datetime):
                date_str = start_time.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = str(start_time)[:16] if start_time else "unknown"

            runs_list.append(
                {
                    "run_id": row.get("run_id", ""),
                    "model_type": row.get("params.model_type", "unknown"),
                    "preprocessing": row.get("params.preprocessing_strategy", "unknown"),
                    "rmse": row.get("metrics.test_rmse", 0) or 0,
                    "r2": row.get("metrics.test_r2", 0) or 0,
                    "date": date_str,
                }
            )

        console.print(f"\n[bold]Experiment:[/bold] {experiment}")
        console.print(f"[dim]Tracking URI: {uri}[/dim]\n")
        console.print(create_runs_table(runs_list))
        console.print(f"\n[dim]Showing {len(runs_list)} of {len(runs_data)} runs[/dim]")

    except Exception as e:
        console.print(error_panel(f"Failed to fetch runs: {e}"))
        raise typer.Exit(1)
