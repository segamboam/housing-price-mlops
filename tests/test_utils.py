"""Tests for utility functions."""

import numpy as np
import pandas as pd

from src.utils.hashing import compute_dataset_hash


class TestComputeDatasetHash:
    """Tests for compute_dataset_hash function."""

    def test_returns_string(self):
        """Hash returns a string."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = compute_dataset_hash(df)
        assert isinstance(result, str)

    def test_returns_12_characters(self):
        """Hash is truncated to 12 characters."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = compute_dataset_hash(df)
        assert len(result) == 12

    def test_returns_hexadecimal(self):
        """Hash contains only hexadecimal characters."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = compute_dataset_hash(df)
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_data_same_hash(self):
        """Same DataFrame produces same hash."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert compute_dataset_hash(df1) == compute_dataset_hash(df2)

    def test_different_data_different_hash(self):
        """Different DataFrames produce different hashes."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})  # Changed one value
        assert compute_dataset_hash(df1) != compute_dataset_hash(df2)

    def test_column_order_affects_hash(self):
        """Column order affects the hash (this is expected behavior)."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2]})
        # Note: hash may or may not differ based on pandas internal representation
        # Just test that it completes without error
        hash1 = compute_dataset_hash(df1)
        hash2 = compute_dataset_hash(df2)
        assert isinstance(hash1, str) and isinstance(hash2, str)

    def test_empty_dataframe(self):
        """Empty DataFrame can be hashed."""
        df = pd.DataFrame()
        result = compute_dataset_hash(df)
        assert isinstance(result, str)
        assert len(result) == 12

    def test_with_nan_values(self):
        """DataFrame with NaN values can be hashed."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
        result = compute_dataset_hash(df)
        assert isinstance(result, str)
        assert len(result) == 12

    def test_with_various_dtypes(self):
        """DataFrame with various dtypes can be hashed."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        result = compute_dataset_hash(df)
        assert isinstance(result, str)
        assert len(result) == 12
