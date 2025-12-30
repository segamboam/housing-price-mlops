"""Model strategies module.

Importing this module registers all available model strategies with the ModelFactory.
"""

from src.models.strategies.gradient_boost import GradientBoostStrategy
from src.models.strategies.linear import LinearRegressionStrategy
from src.models.strategies.random_forest import RandomForestStrategy
from src.models.strategies.xgboost_model import XGBoostStrategy

__all__ = [
    "RandomForestStrategy",
    "GradientBoostStrategy",
    "XGBoostStrategy",
    "LinearRegressionStrategy",
]
