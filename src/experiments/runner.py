"""Experiment runner for grid search experiments."""

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    model_type: str
    preprocessing: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    test_size: float = 0.2
    random_state: int = 42
    enable_cv: bool = True
    cv_splits: int = 5
    register_model: bool = False


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    run_id: str
    model_type: str
    preprocessing: str
    hyperparameters: dict[str, Any]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    cv_metrics: dict[str, float] | None = None
    training_time: float = 0.0

    def get_display_name(self) -> str:
        """Get a display name for this result."""
        name = f"{self.model_type}_{self.preprocessing}"
        if self.hyperparameters:
            key_params = []
            if "n_estimators" in self.hyperparameters:
                key_params.append(f"n={self.hyperparameters['n_estimators']}")
            if "max_depth" in self.hyperparameters:
                key_params.append(f"d={self.hyperparameters['max_depth']}")
            if "learning_rate" in self.hyperparameters:
                key_params.append(f"lr={self.hyperparameters['learning_rate']}")
            if key_params:
                name += f" ({', '.join(key_params)})"
        return name


class ExperimentRunner:
    """Runs experiments from YAML configuration."""

    def __init__(self, config_path: str | Path):
        """Initialize the runner with a config file.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config: DictConfig = OmegaConf.load(self.config_path)
        self.results: list[ExperimentResult] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = self.config.experiment.name

    def generate_experiment_configs(self) -> list[ExperimentConfig]:
        """Generate all experiment configurations from grid.

        Returns:
            List of ExperimentConfig for each combination.
        """
        configs = []
        grid = self.config.grid
        settings = self.config.settings

        models = OmegaConf.to_container(grid.models, resolve=True)
        preprocessors = OmegaConf.to_container(grid.preprocessors, resolve=True)
        hyperparameters = OmegaConf.to_container(grid.get("hyperparameters", {}), resolve=True)

        for model, preproc in itertools.product(models, preprocessors):
            model_hparams = hyperparameters.get(model, {})

            if model_hparams:
                # Generate all combinations of hyperparameters
                hparam_keys = list(model_hparams.keys())
                hparam_values = [v if isinstance(v, list) else [v] for v in model_hparams.values()]

                for combo in itertools.product(*hparam_values):
                    hparams = dict(zip(hparam_keys, combo))
                    configs.append(
                        ExperimentConfig(
                            model_type=model,
                            preprocessing=preproc,
                            hyperparameters=hparams,
                            test_size=settings.test_size,
                            random_state=settings.random_state,
                            enable_cv=settings.enable_cv,
                            cv_splits=settings.cv_splits,
                            register_model=settings.register_models,
                        )
                    )
            else:
                # No hyperparameters specified, use defaults
                configs.append(
                    ExperimentConfig(
                        model_type=model,
                        preprocessing=preproc,
                        test_size=settings.test_size,
                        random_state=settings.random_state,
                        enable_cv=settings.enable_cv,
                        cv_splits=settings.cv_splits,
                        register_model=settings.register_models,
                    )
                )

        return configs

    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment and return results.

        Args:
            config: Configuration for this experiment.

        Returns:
            ExperimentResult with metrics and run info.
        """
        from src.experiments.train_experiment import train_single_experiment

        return train_single_experiment(config, data_path=self.config.settings.data_path)

    def run_all(self, verbose: bool = True) -> list[ExperimentResult]:
        """Run all experiments in the grid.

        Args:
            verbose: Whether to show progress.

        Returns:
            List of ExperimentResult for all experiments.
        """
        configs = self.generate_experiment_configs()
        total = len(configs)

        if verbose:
            console.print(f"\n[bold]Running {total} experiments...[/bold]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Experiments", total=total)

            for i, config in enumerate(configs, 1):
                desc = f"[{i}/{total}] {config.model_type} + {config.preprocessing}"
                if config.hyperparameters:
                    params_str = ", ".join(f"{k}={v}" for k, v in config.hyperparameters.items())
                    desc += f" ({params_str})"
                progress.update(task, description=desc)

                result = self.run_single_experiment(config)
                self.results.append(result)
                progress.advance(task)

        if verbose:
            console.print(f"\n[green]Completed {total} experiments[/green]")

        return self.results

    def get_best_model(
        self, metric: str = "test_rmse", higher_is_better: bool | None = None
    ) -> ExperimentResult:
        """Get the best model based on a metric.

        Args:
            metric: Metric name to optimize.
            higher_is_better: If None, auto-detect based on metric name.

        Returns:
            Best ExperimentResult.
        """
        if not self.results:
            raise ValueError("No results available. Run experiments first.")

        # Auto-detect if not specified
        if higher_is_better is None:
            higher_is_better = metric in ["r2", "accuracy_within_10pct", "cv_r2_mean"]

        def get_metric_value(result: ExperimentResult) -> float:
            if metric.startswith("cv_"):
                return result.cv_metrics.get(metric, float("-inf" if higher_is_better else "inf"))
            elif metric.startswith("test_"):
                key = metric.replace("test_", "")
                return result.test_metrics.get(key, float("-inf" if higher_is_better else "inf"))
            elif metric.startswith("train_"):
                key = metric.replace("train_", "")
                return result.train_metrics.get(key, float("-inf" if higher_is_better else "inf"))
            return result.test_metrics.get(metric, float("-inf" if higher_is_better else "inf"))

        return sorted(self.results, key=get_metric_value, reverse=higher_is_better)[0]

    def print_summary_table(self) -> None:
        """Print a summary table of all results."""
        if not self.results:
            console.print("[yellow]No results to display[/yellow]")
            return

        table = Table(title="Experiment Results")
        table.add_column("Model", style="cyan")
        table.add_column("Preprocessing", style="magenta")
        table.add_column("Test RMSE", justify="right")
        table.add_column("Test R²", justify="right")
        table.add_column("MAPE", justify="right")
        table.add_column("Acc@10%", justify="right")
        table.add_column("CV RMSE", justify="right")

        # Find best model
        best = self.get_best_model("test_rmse", higher_is_better=False)

        for result in sorted(self.results, key=lambda r: r.test_metrics.get("rmse", 999)):
            is_best = result.run_id == best.run_id
            style = "bold green" if is_best else ""

            cv_rmse = ""
            if result.cv_metrics:
                mean = result.cv_metrics.get("cv_rmse_mean", 0)
                std = result.cv_metrics.get("cv_rmse_std", 0)
                cv_rmse = f"{mean:.4f} ± {std:.4f}"

            model_name = result.model_type
            if result.hyperparameters:
                params = []
                if "n_estimators" in result.hyperparameters:
                    params.append(f"n={result.hyperparameters['n_estimators']}")
                if "max_depth" in result.hyperparameters:
                    params.append(f"d={result.hyperparameters['max_depth']}")
                if params:
                    model_name += f"\n({', '.join(params)})"

            table.add_row(
                model_name,
                result.preprocessing,
                f"{result.test_metrics.get('rmse', 0):.4f}",
                f"{result.test_metrics.get('r2', 0):.4f}",
                f"{result.test_metrics.get('mape', 0):.2f}%",
                f"{result.test_metrics.get('accuracy_within_10pct', 0):.1f}%",
                cv_rmse,
                style=style,
            )

        console.print(table)

        # Print best model info
        console.print(f"\n[bold green]Best Model:[/bold green] {best.get_display_name()}")
        console.print(f"  Test RMSE: {best.test_metrics.get('rmse', 0):.4f}")
        console.print(f"  Test R²: {best.test_metrics.get('r2', 0):.4f}")
