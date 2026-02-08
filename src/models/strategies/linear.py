"""Linear regression strategy."""

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from src.models.base import BaseModel
from src.models.factory import ModelFactory, ModelType


@ModelFactory.register(ModelType.LINEAR)
class LinearRegressionStrategy(BaseModel):
    """Linear Regression strategy using Ridge regularization.

    Simple baseline model. Uses Ridge (L2 regularization) to prevent
    overfitting. Feature importance is derived from coefficient magnitudes.
    """

    @property
    def name(self) -> str:
        return "linear"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "alpha": 1.0,
            "random_state": 42,
        }

    def _create_model(self, **params: Any) -> Ridge:
        return Ridge(**params)

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float] | None:
        if self._model is None:
            return None

        # Use absolute coefficient values as importance
        importance = np.abs(self._model.coef_)
        # Normalize to sum to 1
        importance = importance / importance.sum()

        return dict(
            sorted(
                zip(feature_names, importance, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
        )
