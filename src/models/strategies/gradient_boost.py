"""Gradient Boosting regression strategy."""

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor

from src.models.base import BaseModel
from src.models.factory import ModelFactory, ModelType


@ModelFactory.register(ModelType.GRADIENT_BOOST)
class GradientBoostStrategy(BaseModel):
    """Gradient Boosting Regressor strategy.

    Sequential ensemble method that builds trees to correct errors
    of previous trees. Often achieves better accuracy than Random Forest
    but can be more prone to overfitting.
    """

    @property
    def name(self) -> str:
        return "gradient_boost"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "subsample": 0.8,
            "random_state": 42,
        }

    def _create_model(self, **params: Any) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(**params)

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
