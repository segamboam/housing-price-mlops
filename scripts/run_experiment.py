#!/usr/bin/env python3
"""Run ML experiments from YAML configuration.

All results are logged to MLflow. Use MLflow UI to compare runs and visualize metrics.

Usage:
    python scripts/run_experiment.py --config src/experiments/config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.config.settings import get_settings
from src.experiments.runner import ExperimentRunner

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Run ML experiments from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run experiments from config
    python scripts/run_experiment.py --config src/experiments/config.yaml

    # Quiet mode (less output)
    python scripts/run_experiment.py --config src/experiments/config.yaml --quiet

After running, compare results in MLflow UI:
    make infra-up  # Start MLflow
    # Open http://localhost:5000
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/experiments/config.yaml"),
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not args.config.exists():
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    # Run experiments
    console.print(f"\n[bold]Loading config: {args.config}[/bold]")
    runner = ExperimentRunner(args.config)

    console.print(f"[dim]Experiment: {runner.experiment_name}[/dim]")

    configs = runner.generate_experiment_configs()
    console.print(f"[dim]Total combinations: {len(configs)}[/dim]")

    results = runner.run_all(verbose=not args.quiet)

    # Print summary table
    runner.print_summary_table()

    # Show MLflow info
    settings = get_settings()
    console.print("\n[bold]Results logged to MLflow[/bold]")
    console.print(f"  MLflow UI: {settings.mlflow_tracking_uri}")
    console.print(f"  Runs: {len(results)}")

    best = runner.get_best_model()
    console.print(f"\n[bold green]Best Model:[/bold green] {best.get_display_name()}")
    console.print(f"  Run ID: {best.run_id}")
    console.print(f"  Test RMSE: {best.test_metrics.get('rmse', 0):.4f}")
    console.print(f"  Test R²: {best.test_metrics.get('r2', 0):.4f}")

    console.print("\n[dim]Compare runs in MLflow UI: Select runs → Compare[/dim]")
    console.print("[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
