"""Tests for preprocessing cache manager."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
from botocore.exceptions import ClientError

from src.data.cache import ProcessedDataCache, get_cached_data


class TestProcessedDataCacheInit:
    """Tests for ProcessedDataCache initialization."""

    def test_init_sets_version(self):
        """Initialization sets preprocessing version."""
        cache = ProcessedDataCache("v1_median")
        assert cache.preprocessing_version == "v1_median"

    def test_init_sets_cache_path(self):
        """Initialization sets correct cache path."""
        cache = ProcessedDataCache("v1_median")
        assert cache.cache_path == Path("data/processed/v1_median")

    def test_init_s3_client_is_none(self):
        """S3 client is not initialized until accessed."""
        cache = ProcessedDataCache("v1_median")
        assert cache._s3_client is None


class TestS3Key:
    """Tests for S3 key generation."""

    def test_s3_key_format(self):
        """S3 key has correct format."""
        cache = ProcessedDataCache("v1_median")
        key = cache._s3_key("train_X.parquet")
        assert key == "processed-cache/v1_median/train_X.parquet"

    def test_s3_key_with_different_version(self):
        """S3 key uses correct version."""
        cache = ProcessedDataCache("v2_knn")
        key = cache._s3_key("metadata.json")
        assert key == "processed-cache/v2_knn/metadata.json"


class TestExistsLocally:
    """Tests for exists_locally method."""

    def test_returns_false_when_no_files(self):
        """Returns False when cache directory doesn't exist."""
        cache = ProcessedDataCache("nonexistent_version_xyz")
        assert cache.exists_locally() is False

    def test_returns_true_when_all_files_exist(self, tmp_path):
        """Returns True when all required files exist."""
        cache = ProcessedDataCache("test_version")
        cache.cache_path = tmp_path

        # Create all required files
        for filename in ProcessedDataCache.REQUIRED_FILES:
            (tmp_path / filename).touch()

        assert cache.exists_locally() is True

    def test_returns_false_when_missing_file(self, tmp_path):
        """Returns False when any required file is missing."""
        cache = ProcessedDataCache("test_version")
        cache.cache_path = tmp_path

        # Create all but one file
        for filename in ProcessedDataCache.REQUIRED_FILES[:-1]:
            (tmp_path / filename).touch()

        assert cache.exists_locally() is False


class TestExistsInS3:
    """Tests for exists_in_s3 method."""

    def test_returns_true_when_metadata_exists(self):
        """Returns True when metadata.json exists in S3."""
        cache = ProcessedDataCache("v1_median")
        mock_client = MagicMock()
        cache._s3_client = mock_client

        assert cache.exists_in_s3() is True
        mock_client.head_object.assert_called_once()

    def test_returns_false_on_client_error(self):
        """Returns False when S3 returns error."""
        cache = ProcessedDataCache("v1_median")
        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        cache._s3_client = mock_client

        assert cache.exists_in_s3() is False


class TestTryPullFromRemote:
    """Tests for try_pull_from_remote method."""

    def test_returns_false_when_not_in_s3(self):
        """Returns False when data doesn't exist in S3."""
        cache = ProcessedDataCache("v1_median")
        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        cache._s3_client = mock_client

        assert cache.try_pull_from_remote() is False

    def test_downloads_all_files_when_exists(self, tmp_path):
        """Downloads all required files when data exists in S3."""
        cache = ProcessedDataCache("v1_median")
        cache.cache_path = tmp_path

        mock_client = MagicMock()
        cache._s3_client = mock_client

        result = cache.try_pull_from_remote()

        assert result is True
        assert mock_client.download_file.call_count == len(ProcessedDataCache.REQUIRED_FILES)

    def test_returns_false_on_download_error(self, tmp_path):
        """Returns False when download fails."""
        cache = ProcessedDataCache("v1_median")
        cache.cache_path = tmp_path

        mock_client = MagicMock()
        mock_client.download_file.side_effect = ClientError({"Error": {"Code": "500"}}, "GetObject")
        cache._s3_client = mock_client

        assert cache.try_pull_from_remote() is False


class TestPushToRemote:
    """Tests for push_to_remote method."""

    def test_returns_false_when_no_local_cache(self):
        """Returns False when local cache doesn't exist."""
        cache = ProcessedDataCache("nonexistent_xyz")
        assert cache.push_to_remote() is False

    def test_uploads_all_files(self, tmp_path):
        """Uploads all required files when local cache exists."""
        cache = ProcessedDataCache("test_version")
        cache.cache_path = tmp_path

        # Create all required files
        for filename in ProcessedDataCache.REQUIRED_FILES:
            (tmp_path / filename).touch()

        mock_client = MagicMock()
        cache._s3_client = mock_client

        result = cache.push_to_remote()

        assert result is True
        assert mock_client.upload_file.call_count == len(ProcessedDataCache.REQUIRED_FILES)

    def test_returns_false_on_upload_error(self, tmp_path):
        """Returns False when upload fails."""
        cache = ProcessedDataCache("test_version")
        cache.cache_path = tmp_path

        # Create all required files
        for filename in ProcessedDataCache.REQUIRED_FILES:
            (tmp_path / filename).touch()

        mock_client = MagicMock()
        mock_client.upload_file.side_effect = ClientError({"Error": {"Code": "500"}}, "PutObject")
        cache._s3_client = mock_client

        assert cache.push_to_remote() is False


class TestLoad:
    """Tests for load method."""

    def test_loads_all_data(self, tmp_path):
        """Loads all data components correctly."""
        cache = ProcessedDataCache("test_version")
        cache.cache_path = tmp_path

        # Create test data
        X_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X_test = pd.DataFrame({"a": [5.0], "b": [6.0]})
        y_train = pd.DataFrame({"target": [10.0, 20.0]})
        y_test = pd.DataFrame({"target": [30.0]})
        metadata = {"version": "test", "samples": 2}

        # Save test data
        X_train.to_parquet(tmp_path / "train_X.parquet")
        X_test.to_parquet(tmp_path / "test_X.parquet")
        y_train.to_parquet(tmp_path / "train_y.parquet")
        y_test.to_parquet(tmp_path / "test_y.parquet")

        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create a mock preprocessor
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit([[1, 2], [3, 4]])
        joblib.dump(scaler, tmp_path / "preprocessor.joblib")

        # Load and verify
        result = cache.load()

        assert len(result) == 6
        X_tr, X_te, y_tr, y_te, meta, preprocessor = result

        assert X_tr.shape == (2, 2)
        assert X_te.shape == (1, 2)
        assert len(y_tr) == 2
        assert len(y_te) == 1
        assert meta["version"] == "test"
        assert preprocessor is not None


class TestGetOrCreate:
    """Tests for get_or_create method."""

    def test_uses_local_cache_when_exists(self, tmp_path):
        """Uses local cache when available."""
        cache = ProcessedDataCache("test_version")
        cache.cache_path = tmp_path

        # Create minimal valid cache
        X_train = pd.DataFrame({"a": [1.0]})
        X_test = pd.DataFrame({"a": [2.0]})
        y_train = pd.DataFrame({"target": [10.0]})
        y_test = pd.DataFrame({"target": [20.0]})

        X_train.to_parquet(tmp_path / "train_X.parquet")
        X_test.to_parquet(tmp_path / "test_X.parquet")
        y_train.to_parquet(tmp_path / "train_y.parquet")
        y_test.to_parquet(tmp_path / "test_y.parquet")

        with open(tmp_path / "metadata.json", "w") as f:
            json.dump({"test": True}, f)

        from sklearn.preprocessing import StandardScaler

        joblib.dump(StandardScaler().fit([[1]]), tmp_path / "preprocessor.joblib")

        # Mock S3 client to verify it's not called
        mock_client = MagicMock()
        cache._s3_client = mock_client

        result = cache.get_or_create()

        assert len(result) == 6
        # S3 should not be called when local exists
        mock_client.head_object.assert_not_called()


class TestGetCachedDataFunction:
    """Tests for get_cached_data convenience function."""

    def test_creates_cache_instance(self):
        """Creates ProcessedDataCache with correct version."""
        with patch.object(ProcessedDataCache, "get_or_create") as mock_get:
            mock_get.return_value = (None, None, None, None, {}, None)

            get_cached_data("v1_median")

            mock_get.assert_called_once()

    def test_passes_parameters(self):
        """Passes test_size and random_state to cache."""
        with patch.object(ProcessedDataCache, "get_or_create") as mock_get:
            mock_get.return_value = (None, None, None, None, {}, None)

            get_cached_data("v1_median", test_size=0.3, random_state=123)

            mock_get.assert_called_once_with(test_size=0.3, random_state=123)
