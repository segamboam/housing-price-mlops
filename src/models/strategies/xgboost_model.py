"""XGBoost regression strategy."""

from typing import Any

from src.models.base import BaseModel
from src.models.factory import ModelFactory, ModelType

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None


@ModelFactory.register(ModelType.XGBOOST)
class XGBoostStrategy(BaseModel):
    """XGBoost Regressor strategy.

    Optimized gradient boosting implementation with regularization.
    Often achieves state-of-the-art results on tabular data.

    Note: Requires xgboost package to be installed.
    """

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

    def _create_model(self, **params: Any) -> "XGBRegressor":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
        return XGBRegressor(**params)

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float] | None:
        if self._model is None:
            return None

        importance = self._model.feature_importances_
        return dict(
            sorted(
                zip(feature_names, importance, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
        )
