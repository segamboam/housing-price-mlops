"""Cross-validation utilities for model evaluation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, cross_val_score, learning_curve

from src.config.settings import get_settings

_settings = get_settings()


@dataclass
class CVResult:
    """Results from cross-validation.

    Attributes:
        cv_rmse_mean: Mean RMSE across all folds.
        cv_rmse_std: Standard deviation of RMSE across folds.
        cv_rmse_scores: RMSE score for each fold.
        cv_r2_mean: Mean R² across all folds.
        cv_r2_std: Standard deviation of R² across folds.
        cv_r2_scores: R² score for each fold.
        n_splits: Number of CV folds used.
    """

    cv_rmse_mean: float
    cv_rmse_std: float
    cv_rmse_scores: list[float]
    cv_r2_mean: float
    cv_r2_std: float
    cv_r2_scores: list[float]
    n_splits: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MLflow logging.

        Returns:
            Dictionary with CV metrics suitable for MLflow.
        """
        return {
            "cv_rmse_mean": round(self.cv_rmse_mean, 4),
            "cv_rmse_std": round(self.cv_rmse_std, 4),
            "cv_r2_mean": round(self.cv_r2_mean, 4),
            "cv_r2_std": round(self.cv_r2_std, 4),
            "cv_n_splits": self.n_splits,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Convert to dictionary including all scores per fold.

        Returns:
            Dictionary with all CV data including per-fold scores.
        """
        return {
            **self.to_dict(),
            "cv_rmse_scores": [round(s, 4) for s in self.cv_rmse_scores],
            "cv_r2_scores": [round(s, 4) for s in self.cv_r2_scores],
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"CV Results ({self.n_splits}-fold):\n"
            f"  RMSE: {self.cv_rmse_mean:.4f} ± {self.cv_rmse_std:.4f}\n"
            f"  R²:   {self.cv_r2_mean:.4f} ± {self.cv_r2_std:.4f}"
        )


@dataclass
class LearningCurveResult:
    """Results from learning curve analysis.

    Attributes:
        train_sizes: Number of training samples for each point.
        train_scores_mean: Mean training score for each training size.
        train_scores_std: Std of training scores.
        val_scores_mean: Mean validation score for each training size.
        val_scores_std: Std of validation scores.
        is_overfitting: Whether the model shows signs of overfitting.
        gap_at_full_data: Gap between train and validation at full data.
    """

    train_sizes: list[int]
    train_scores_mean: list[float]
    train_scores_std: list[float]
    val_scores_mean: list[float]
    val_scores_std: list[float]
    is_overfitting: bool = False
    gap_at_full_data: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "train_sizes": self.train_sizes,
            "train_scores_mean": [round(s, 4) for s in self.train_scores_mean],
            "train_scores_std": [round(s, 4) for s in self.train_scores_std],
            "val_scores_mean": [round(s, 4) for s in self.val_scores_mean],
            "val_scores_std": [round(s, 4) for s in self.val_scores_std],
            "is_overfitting": self.is_overfitting,
            "gap_at_full_data": round(self.gap_at_full_data, 4),
        }


def perform_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> CVResult:
    """Perform k-fold cross-validation on a model.

    Args:
        model: Sklearn-compatible model (fitted or unfitted).
               If using a BaseModel strategy, pass model.model.
        X: Feature array (already preprocessed).
        y: Target array.
        n_splits: Number of CV folds.
        shuffle: Whether to shuffle data before splitting.
        random_state: Random seed for reproducibility.

    Returns:
        CVResult with cross-validation metrics.

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=100, random_state=42)
        >>> cv_result = perform_cross_validation(model, X_train, y_train)
        >>> print(cv_result)
        CV Results (5-fold):
          RMSE: 3.21 ± 0.45
          R²:   0.87 ± 0.05
    """
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # RMSE scores (sklearn returns negative MSE, we convert to positive RMSE)
    mse_scores = cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = np.sqrt(-mse_scores)

    # R² scores
    r2_scores = cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring="r2",
        n_jobs=-1,
    )

    return CVResult(
        cv_rmse_mean=float(rmse_scores.mean()),
        cv_rmse_std=float(rmse_scores.std()),
        cv_rmse_scores=rmse_scores.tolist(),
        cv_r2_mean=float(r2_scores.mean()),
        cv_r2_std=float(r2_scores.std()),
        cv_r2_scores=r2_scores.tolist(),
        n_splits=n_splits,
    )


def compute_learning_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray | None = None,
    cv: int = 5,
    random_state: int | None = None,
    overfitting_threshold: float | None = None,
) -> LearningCurveResult:
    """Compute learning curves to detect overfitting.

    Learning curves plot training and validation scores as a function
    of training set size. A large gap between curves indicates overfitting.

    Args:
        model: Sklearn-compatible model.
        X: Feature array.
        y: Target array.
        train_sizes: Array of training set sizes to evaluate.
                    Defaults to 10 points from 10% to 100%.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducibility. Defaults to settings.default_random_state.
        overfitting_threshold: RMSE gap threshold to flag overfitting.
                              Defaults to settings.overfitting_rmse_threshold.

    Returns:
        LearningCurveResult with training and validation scores.
    """
    if random_state is None:
        random_state = _settings.default_random_state
    if overfitting_threshold is None:
        overfitting_threshold = _settings.overfitting_rmse_threshold
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )

    # Convert negative RMSE to positive
    train_scores = -train_scores
    val_scores = -val_scores

    # Calculate means and stds
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Check for overfitting: gap between train and val at full data
    gap_at_full_data = float(val_mean[-1] - train_mean[-1])
    is_overfitting = gap_at_full_data > overfitting_threshold

    return LearningCurveResult(
        train_sizes=train_sizes_abs.tolist(),
        train_scores_mean=train_mean.tolist(),
        train_scores_std=train_std.tolist(),
        val_scores_mean=val_mean.tolist(),
        val_scores_std=val_std.tolist(),
        is_overfitting=is_overfitting,
        gap_at_full_data=gap_at_full_data,
    )
