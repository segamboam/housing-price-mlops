import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.settings import get_settings

_settings = get_settings()


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        MAPE as percentage (0-100). Returns 0 if all true values are zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Avoid division by zero - only calculate for non-zero true values
    mask = y_true != 0
    if not mask.any():
        return 0.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def accuracy_within_tolerance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float | None = None,
) -> float:
    """Calculate percentage of predictions within a tolerance threshold.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        tolerance: Allowed relative deviation as fraction. Defaults to settings.accuracy_tolerance.

    Returns:
        Percentage of predictions within tolerance (0-100).
    """
    if tolerance is None:
        tolerance = _settings.accuracy_tolerance

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Use small epsilon to avoid division by zero
    relative_error = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)
    within_tolerance = relative_error <= tolerance

    return float(np.mean(within_tolerance) * 100)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics including business metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with RMSE, MAE, R2, MAPE, and accuracy within 10% metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    acc_10 = accuracy_within_tolerance(y_true, y_pred)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
        "accuracy_within_10pct": round(acc_10, 4),
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a trained model on test data.

    Args:
        model: Trained model with predict method.
        X_test: Test features.
        y_test: True test target values.

    Returns:
        Dictionary with predictions and metrics.
    """
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    return {
        "predictions": y_pred,
        "metrics": metrics,
    }


def generate_evaluation_report(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    feature_importance: dict[str, float] | None = None,
    model_params: dict | None = None,
) -> dict:
    """Generate a comprehensive evaluation report.

    Args:
        train_metrics: Metrics calculated on training data.
        test_metrics: Metrics calculated on test data.
        feature_importance: Optional feature importance scores.
        model_params: Optional model hyperparameters.

    Returns:
        Dictionary containing the full evaluation report.
    """
    report = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    if feature_importance:
        report["feature_importance"] = feature_importance

    if model_params:
        report["model_params"] = model_params

    # Check for potential overfitting using configured threshold
    r2_diff = train_metrics.get("r2", 0) - test_metrics.get("r2", 0)
    report["overfitting_warning"] = r2_diff > _settings.overfitting_r2_threshold

    return report


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_report(report: dict, output_path: str | Path) -> Path:
    """Save evaluation report to JSON file.

    Args:
        report: Evaluation report dictionary.
        output_path: Path to save the report.

    Returns:
        Path to the saved report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    return output_path


def print_metrics(metrics: dict[str, float], dataset_name: str = "Test") -> None:
    """Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics.
        dataset_name: Name of the dataset (for display).
    """
    print(f"\n{dataset_name} Metrics:")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value}")
