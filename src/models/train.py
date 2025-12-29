from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None
) -> RandomForestRegressor:
    """Train a RandomForest regression model.

    Args:
        X_train: Training features (scaled).
        y_train: Training target values.
        params: Model hyperparameters. Uses defaults if None.

    Returns:
        Trained RandomForestRegressor model.
    """
    model_params = DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)

    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)

    return model


def save_model(
    model: RandomForestRegressor,
    scaler,
    output_dir: str | Path,
    model_name: str = "housing_model"
) -> dict[str, Path]:
    """Save model and scaler to disk.

    Args:
        model: Trained model.
        scaler: Fitted scaler.
        output_dir: Directory to save artifacts.
        model_name: Base name for saved files.

    Returns:
        Dictionary with paths to saved artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.joblib"
    scaler_path = output_dir / f"{model_name}_scaler.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
    }


def load_model(model_path: str | Path) -> RandomForestRegressor:
    """Load a trained model from disk.

    Args:
        model_path: Path to the saved model.

    Returns:
        Loaded RandomForestRegressor model.
    """
    return joblib.load(model_path)


def load_scaler(scaler_path: str | Path):
    """Load a fitted scaler from disk.

    Args:
        scaler_path: Path to the saved scaler.

    Returns:
        Loaded scaler.
    """
    return joblib.load(scaler_path)


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str]
) -> dict[str, float]:
    """Get feature importance from trained model.

    Args:
        model: Trained RandomForestRegressor.
        feature_names: List of feature names.

    Returns:
        Dictionary mapping feature names to importance scores.
    """
    importance = model.feature_importances_
    return dict(sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    ))
