"""Hashing utilities for data versioning."""

import hashlib

import pandas as pd


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the dataset for versioning.

    Args:
        df: DataFrame to hash.

    Returns:
        12-character hexadecimal hash string.
    """
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:12]
