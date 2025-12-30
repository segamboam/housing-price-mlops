"""V1 Median preprocessing strategy - baseline approach."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessingStrategy, PreprocessorFactory


@PreprocessorFactory.register(PreprocessingStrategy.V1_MEDIAN)
class V1MedianPreprocessor(BasePreprocessor):
    """Baseline preprocessing strategy using median imputation.

    Pipeline:
        1. SimpleImputer with median strategy (robust to outliers)
        2. StandardScaler (zero mean, unit variance)

    This is the original preprocessing approach, simple and effective.
    """

    @property
    def name(self) -> str:
        return "v1_median"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Median imputation + StandardScaler (baseline)"

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
