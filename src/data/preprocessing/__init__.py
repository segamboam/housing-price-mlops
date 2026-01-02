"""Preprocessing module with strategy pattern support."""

# Feature and target column definitions
from src.data.loader import FEATURE_COLUMNS, TARGET_COLUMN

# Strategy pattern exports
from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessingStrategy, PreprocessorFactory

__all__ = [
    # Strategy pattern
    "BasePreprocessor",
    "PreprocessingStrategy",
    "PreprocessorFactory",
    # Column definitions
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
]
