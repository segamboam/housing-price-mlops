"""Preprocessing strategies module.

Importing this module registers all available preprocessing strategies
with the PreprocessorFactory.
"""

from src.data.preprocessing.strategies.v1_median import V1MedianPreprocessor
from src.data.preprocessing.strategies.v2_knn import V2KNNPreprocessor
from src.data.preprocessing.strategies.v3_iterative import V3IterativePreprocessor
from src.data.preprocessing.strategies.v4_robust_col import V4RobustColPreprocessor

__all__ = [
    "V1MedianPreprocessor",
    "V2KNNPreprocessor",
    "V3IterativePreprocessor",
    "V4RobustColPreprocessor",
]
