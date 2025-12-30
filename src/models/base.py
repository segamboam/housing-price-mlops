"""Base class for ML model strategies."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all ML model strategies.

    Implements the Strategy pattern to allow interchangeable model algorithms
    while maintaining a consistent interface for training, prediction, and
    feature importance extraction.
    """

    def __init__(self):
        self._model: Any = None
        self._is_fitted: bool = False
        self._params: dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the model strategy."""
        ...

    @property
    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Default hyperparameters for the model."""
        ...

    @abstractmethod
    def _create_model(self, **params: Any) -> Any:
        """Create the underlying sklearn-compatible model instance.

        Args:
            **params: Hyperparameters for model initialization.

        Returns:
            An sklearn-compatible model instance.
        """
        ...

    def train(self, X: np.ndarray, y: np.ndarray, **params: Any) -> "BaseModel":
        """Train the model on data.

        Args:
            X: Training features (already preprocessed/scaled).
            y: Training target values.
            **params: Additional hyperparameters to override defaults.

        Returns:
            Self for method chaining.
        """
        self._params = {**self.default_params, **params}
        self._model = self._create_model(**self._params)
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on data.

        Args:
            X: Features to predict (already preprocessed/scaled).

        Returns:
            Array of predictions.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        return self._model.predict(X)

    @abstractmethod
    def get_feature_importance(
        self, feature_names: list[str]
    ) -> dict[str, float] | None:
        """Get feature importance scores if supported by the model.

        Args:
            feature_names: List of feature names in order.

        Returns:
            Dictionary mapping feature names to importance scores,
            sorted by importance descending. None if not supported.
        """
        ...

    @property
    def model(self) -> Any:
        """Access the underlying sklearn-compatible model."""
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        """Set the underlying model (used when loading from disk)."""
        self._model = value
        self._is_fitted = value is not None

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been trained."""
        return self._is_fitted

    @property
    def params(self) -> dict[str, Any]:
        """Get the parameters used for training."""
        return self._params.copy()

    @params.setter
    def params(self, value: dict[str, Any]) -> None:
        """Set parameters (used when loading from disk)."""
        self._params = value
