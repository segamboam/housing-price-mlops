"""Rich UI utilities for CLI commands."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


def confirm_action(prompt: str, default: bool = False) -> bool:
    """Ask user for yes/no confirmation.

    Args:
        prompt: The question to ask
        default: Default value if user just presses Enter

    Returns:
        True if user confirmed, False otherwise
    """
    return Confirm.ask(f"[bold cyan]{prompt}[/bold cyan]", default=default)


def select_option(prompt: str, options: list[str], default: str | None = None) -> str:
    """Display an interactive menu and return the selected option.

    Args:
        prompt: The question to ask
        options: List of available options
        default: Default option (will be highlighted)

    Returns:
        The selected option string
    """
    console.print(f"\n[bold cyan]{prompt}[/bold cyan]")

    for i, option in enumerate(options, 1):
        if option == default:
            console.print(f"  [bold green]{i}.[/bold green] {option} [dim](default)[/dim]")
        else:
            console.print(f"  [dim]{i}.[/dim] {option}")

    while True:
        choice = Prompt.ask(
            "\nSelect option",
            default=str(options.index(default) + 1) if default else "1",
        )

        # Handle numeric input
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected = options[idx]
                console.print(f"[green]✓[/green] Selected: [bold]{selected}[/bold]")
                return selected

        # Handle text input (exact match)
        if choice in options:
            console.print(f"[green]✓[/green] Selected: [bold]{choice}[/bold]")
            return choice

        console.print(f"[red]Invalid choice. Enter 1-{len(options)} or option name.[/red]")


def create_metrics_table(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> Table:
    """Create a Rich table comparing train and test metrics."""
    table = Table(title="Model Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Train", justify="right")
    table.add_column("Test", justify="right")

    for metric in ["rmse", "mae", "r2"]:
        train_val = train_metrics.get(metric, 0)
        test_val = test_metrics.get(metric, 0)
        table.add_row(
            metric.upper(),
            f"{train_val:.4f}",
            f"{test_val:.4f}",
        )

    return table


def create_feature_importance_table(
    feature_importance: dict[str, float],
    top_n: int = 5,
) -> Table:
    """Create a Rich table with top feature importances."""
    table = Table(title="Top Feature Importance", show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Feature")
    table.add_column("Importance", justify="right")

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
        table.add_row(str(i), feature, f"{importance:.4f}")

    return table


def create_runs_table(runs: list[dict]) -> Table:
    """Create a Rich table with MLflow runs."""
    table = Table(title="Recent Runs", show_header=True, header_style="bold cyan")
    table.add_column("Run ID", style="dim")
    table.add_column("Model")
    table.add_column("Preprocessing")
    table.add_column("RMSE", justify="right")
    table.add_column("R2", justify="right")
    table.add_column("Date")

    best_r2 = max((r.get("r2", 0) for r in runs), default=0)

    for run in runs:
        run_id = run.get("run_id", "")[:8]
        model = run.get("model_type", "unknown")
        preproc = run.get("preprocessing", "unknown")
        rmse = run.get("rmse", 0)
        r2 = run.get("r2", 0)
        date = run.get("date", "")

        # Highlight best model
        style = "bold green" if r2 == best_r2 else None
        suffix = " *" if r2 == best_r2 else ""

        table.add_row(
            run_id,
            model,
            preproc,
            f"{rmse:.4f}",
            f"{r2:.4f}{suffix}",
            date,
            style=style,
        )

    return table


def create_versions_table(versions: list[dict]) -> Table:
    """Create a Rich table with model versions."""
    table = Table(title="Model Versions", show_header=True, header_style="bold cyan")
    table.add_column("Version", style="dim")
    table.add_column("Aliases")
    table.add_column("Run ID")
    table.add_column("Created")

    for v in versions:
        aliases = ", ".join(v.get("aliases", [])) or "-"
        style = "bold green" if "production" in v.get("aliases", []) else None
        table.add_row(
            str(v.get("version", "")),
            aliases,
            v.get("run_id", "")[:8],
            v.get("created", ""),
            style=style,
        )

    return table


def success_panel(message: str, title: str = "Success") -> Panel:
    """Create a green success panel."""
    return Panel(message, title=title, border_style="green")


def error_panel(message: str, title: str = "Error") -> Panel:
    """Create a red error panel."""
    return Panel(message, title=title, border_style="red")


def info_panel(message: str, title: str = "Info") -> Panel:
    """Create a blue info panel."""
    return Panel(message, title=title, border_style="blue")


def config_panel(config: dict[str, str], title: str = "Configuration") -> Panel:
    """Create a panel showing configuration."""
    lines = [f"[bold]{k}:[/bold] {v}" for k, v in config.items()]
    return Panel("\n".join(lines), title=title, border_style="cyan")


def prompt_integer(
    prompt: str,
    default: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Prompt user for an integer value with validation.

    Args:
        prompt: The question to ask
        default: Default value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        The entered integer value
    """
    while True:
        value = Prompt.ask(f"[cyan]{prompt}[/cyan]", default=str(default))
        try:
            int_val = int(value)
            if min_val is not None and int_val < min_val:
                console.print(f"[red]Value must be >= {min_val}[/red]")
                continue
            if max_val is not None and int_val > max_val:
                console.print(f"[red]Value must be <= {max_val}[/red]")
                continue
            return int_val
        except ValueError:
            console.print("[red]Please enter a valid integer[/red]")


def prompt_float(
    prompt: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Prompt user for a float value with validation.

    Args:
        prompt: The question to ask
        default: Default value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        The entered float value
    """
    while True:
        value = Prompt.ask(f"[cyan]{prompt}[/cyan]", default=str(default))
        try:
            float_val = float(value)
            if min_val is not None and float_val < min_val:
                console.print(f"[red]Value must be >= {min_val}[/red]")
                continue
            if max_val is not None and float_val > max_val:
                console.print(f"[red]Value must be <= {max_val}[/red]")
                continue
            return float_val
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


# Hyperparameter configurations for interactive mode
HYPERPARAMETER_CONFIGS: dict[str, dict] = {
    "random_forest": {
        "n_estimators": {"default": 100, "min": 10, "max": 500, "type": "int", "desc": "Number of trees"},
        "max_depth": {"default": 15, "min": 1, "max": 50, "type": "int", "desc": "Maximum tree depth"},
        "min_samples_split": {"default": 2, "min": 2, "max": 20, "type": "int", "desc": "Min samples to split"},
        "min_samples_leaf": {"default": 1, "min": 1, "max": 10, "type": "int", "desc": "Min samples per leaf"},
    },
    "gradient_boost": {
        "n_estimators": {"default": 100, "min": 10, "max": 500, "type": "int", "desc": "Number of boosting stages"},
        "learning_rate": {"default": 0.1, "min": 0.01, "max": 1.0, "type": "float", "desc": "Learning rate"},
        "max_depth": {"default": 3, "min": 1, "max": 20, "type": "int", "desc": "Maximum tree depth"},
        "min_samples_split": {"default": 2, "min": 2, "max": 20, "type": "int", "desc": "Min samples to split"},
    },
    "xgboost": {
        "n_estimators": {"default": 100, "min": 10, "max": 500, "type": "int", "desc": "Number of boosting rounds"},
        "learning_rate": {"default": 0.1, "min": 0.01, "max": 1.0, "type": "float", "desc": "Learning rate (eta)"},
        "max_depth": {"default": 6, "min": 1, "max": 20, "type": "int", "desc": "Maximum tree depth"},
        "subsample": {"default": 1.0, "min": 0.5, "max": 1.0, "type": "float", "desc": "Subsample ratio"},
    },
    "linear": {
        # Linear regression has no hyperparameters to tune
    },
}


def select_hyperparameters(model_type: str) -> dict:
    """Interactive selection of hyperparameters for a model.

    Args:
        model_type: The type of model (random_forest, gradient_boost, etc.)

    Returns:
        Dictionary of hyperparameter name -> value
    """
    config = HYPERPARAMETER_CONFIGS.get(model_type, {})

    if not config:
        console.print(f"[dim]No hyperparameters to configure for {model_type}[/dim]")
        return {}

    console.print(f"\n[bold]Configure hyperparameters for {model_type}:[/bold]")
    console.print("[dim]Press Enter to use default values[/dim]\n")

    hyperparams = {}
    for param_name, param_config in config.items():
        desc = param_config["desc"]
        default = param_config["default"]
        min_val = param_config["min"]
        max_val = param_config["max"]

        prompt = f"  {param_name} ({desc}) [{min_val}-{max_val}]"

        if param_config["type"] == "int":
            hyperparams[param_name] = prompt_integer(prompt, default, min_val, max_val)
        else:
            hyperparams[param_name] = prompt_float(prompt, default, min_val, max_val)

    console.print()
    return hyperparams


def create_cv_results_table(cv_metrics: dict) -> Table:
    """Create a Rich table showing cross-validation results.

    Args:
        cv_metrics: Dictionary with cv_rmse_mean, cv_rmse_std, cv_r2_mean, cv_r2_std

    Returns:
        Rich Table with CV results
    """
    table = Table(title="Cross-Validation Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")

    rmse_mean = cv_metrics.get("cv_rmse_mean", 0)
    rmse_std = cv_metrics.get("cv_rmse_std", 0)
    r2_mean = cv_metrics.get("cv_r2_mean", 0)
    r2_std = cv_metrics.get("cv_r2_std", 0)

    table.add_row("RMSE", f"{rmse_mean:.4f}", f"± {rmse_std:.4f}")
    table.add_row("R²", f"{r2_mean:.4f}", f"± {r2_std:.4f}")

    return table
