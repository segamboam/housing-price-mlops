"""V3 Iterative preprocessing strategy - advanced imputation."""

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessingStrategy, PreprocessorFactory


@PreprocessorFactory.register(PreprocessingStrategy.V3_ITERATIVE)
class V3IterativePreprocessor(BasePreprocessor):
    """Iterative imputation preprocessing strategy.

    Pipeline:
        1. IterativeImputer (MICE-like multivariate imputation)
        2. RobustScaler (uses median and IQR, robust to outliers)

    Most sophisticated approach - models each feature with missing values
    as a function of other features in a round-robin fashion.
    """

    def __init__(self, max_iter: int = 10, random_state: int = 42):
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "v3_iterative"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return f"IterativeImputer (max_iter={self.max_iter}) + RobustScaler"

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "imputer",
                    IterativeImputer(
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
                ("scaler", RobustScaler()),
            ]
        )
