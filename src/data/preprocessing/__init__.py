"""Preprocessing module with strategy pattern support.

This module provides both:
1. Strategy pattern classes for flexible preprocessing pipelines
2. Legacy functions for backward compatibility
"""

# Strategy pattern exports
# Legacy exports (backward compatibility)
from src.data.preprocessing._legacy import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_train_test_split,
    impute_missing_values,
    preprocess_pipeline,
    scale_features,
    split_features_target,
)
from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessingStrategy, PreprocessorFactory

__all__ = [
    # Strategy pattern
    "BasePreprocessor",
    "PreprocessingStrategy",
    "PreprocessorFactory",
    # Legacy constants
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    # Legacy functions
    "impute_missing_values",
    "split_features_target",
    "create_train_test_split",
    "scale_features",
    "preprocess_pipeline",
]
