"""Random Forest regression strategy."""

from typing import Any

from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseModel
from src.models.factory import ModelFactory, ModelType


@ModelFactory.register(ModelType.RANDOM_FOREST)
class RandomForestStrategy(BaseModel):
    """Random Forest Regressor strategy.

    A robust ensemble method that combines multiple decision trees.
    Good balance between accuracy and interpretability, handles
    non-linear relationships well.
    """

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": 150,
            "max_depth": 15,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "random_state": 42,
            "n_jobs": -1,
        }

    def _create_model(self, **params: Any) -> RandomForestRegressor:
        return RandomForestRegressor(**params)

    def get_feature_importance(
        self, feature_names: list[str]
    ) -> dict[str, float] | None:
        if self._model is None:
            return None

        importance = self._model.feature_importances_
        return dict(
            sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )
