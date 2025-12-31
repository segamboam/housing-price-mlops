"""V4 Robust Column preprocessing strategy - ColumnTransformer with RobustScaler."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessingStrategy, PreprocessorFactory


# Feature definitions for Boston Housing dataset
NUMERIC_FEATURES = [
    "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]
CATEGORICAL_FEATURES = ["CHAS"]


@PreprocessorFactory.register(PreprocessingStrategy.V4_ROBUST_COL)
class V4RobustColPreprocessor(BasePreprocessor):
    """Advanced preprocessing with ColumnTransformer and RobustScaler.

    Pipeline:
        - Numeric features: SimpleImputer(median) + RobustScaler
        - Categorical features: SimpleImputer(most_frequent)

    RobustScaler is more robust to outliers than StandardScaler,
    using median and IQR instead of mean and std.
    """

    @property
    def name(self) -> str:
        return "v4_robust_col"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "ColumnTransformer + RobustScaler (outlier-resistant)"

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]), NUMERIC_FEATURES),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]), CATEGORICAL_FEATURES),
            ],
            remainder="passthrough",
        )

        return Pipeline([
            ("preprocessor", preprocessor),
        ])
