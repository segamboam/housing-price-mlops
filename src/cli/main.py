"""Main CLI application for Housing Price Prediction."""

import typer

from src.cli import info, promote, runs, train

app = typer.Typer(
    name="housing",
    help="Housing Price Prediction - CLI Tools",
    add_completion=False,
    no_args_is_help=True,
)

# Register commands
app.command(name="train", help="Train a new model")(train.train)
app.command(name="info", help="Show current model information")(info.info)
app.command(name="runs", help="List MLflow experiment runs")(runs.runs)
app.command(name="promote", help="Promote a model version to champion")(promote.promote)


if __name__ == "__main__":
    app()
