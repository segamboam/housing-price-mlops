"""Base class for preprocessing strategies."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class BasePreprocessor(ABC):
    """Abstract base class for preprocessing strategies.

    Implements the Strategy pattern for interchangeable preprocessing pipelines.
    Each strategy encapsulates imputation and scaling logic in an sklearn Pipeline.
    """

    def __init__(self):
        self._pipeline: Pipeline | None = None
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the preprocessing strategy."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string for tracking changes."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the strategy."""
        ...

    @abstractmethod
    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline for this strategy.

        Returns:
            A configured sklearn Pipeline.
        """
        ...

    def fit(self, X: pd.DataFrame | np.ndarray) -> "BasePreprocessor":
        """Fit the preprocessor on training data.

        Args:
            X: Training features (DataFrame or array).

        Returns:
            Self for method chaining.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        else:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data using the fitted pipeline.

        Args:
            X: Features to transform.

        Returns:
            Transformed feature array.

        Raises:
            RuntimeError: If preprocessor has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return self._pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Features to fit and transform.

        Returns:
            Transformed feature array.
        """
        self.fit(X)
        return self.transform(X)

    @property
    def pipeline(self) -> Pipeline | None:
        """Access the underlying sklearn pipeline."""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: Pipeline) -> None:
        """Set the pipeline (used when loading from disk)."""
        self._pipeline = value
        self._is_fitted = value is not None

    @property
    def feature_names(self) -> list[str]:
        """Get feature names in order."""
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value: list[str]) -> None:
        """Set feature names (used when loading from disk)."""
        self._feature_names = value

    @property
    def is_fitted(self) -> bool:
        """Check if the preprocessor has been fitted."""
        return self._is_fitted

    def get_params(self) -> dict[str, Any]:
        """Get preprocessor parameters for logging/metadata.

        Returns:
            Dictionary with strategy metadata.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
        }
