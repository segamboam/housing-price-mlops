"""V2 KNN preprocessing strategy - neighbor-based imputation."""

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessingStrategy, PreprocessorFactory


@PreprocessorFactory.register(PreprocessingStrategy.V2_KNN)
class V2KNNPreprocessor(BasePreprocessor):
    """KNN-based preprocessing strategy.

    Pipeline:
        1. KNNImputer (uses k nearest neighbors to impute missing values)
        2. StandardScaler (zero mean, unit variance)

    More sophisticated than median imputation as it considers the
    relationship between features when imputing missing values.
    """

    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors

    @property
    def name(self) -> str:
        return "v2_knn"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return f"KNN imputation (k={self.n_neighbors}) + StandardScaler"

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", KNNImputer(n_neighbors=self.n_neighbors)),
                ("scaler", StandardScaler()),
            ]
        )
