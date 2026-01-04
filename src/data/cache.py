"""Preprocessing cache manager with S3 (MinIO) integration.

This module provides transparent caching for preprocessed data.
When training or running experiments:
1. Check if cache exists locally
2. If not, try to pull from S3 (MinIO)
3. If not in remote, compute and push to remote

This ensures:
- Preprocessed data is computed only once per strategy
- All data is stored in S3 (MinIO) for persistence across sessions
- Both `make train` and `make experiment` use the same cache
"""

import json
import os
import sys
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ProcessedDataCache:
    """Manages preprocessed data cache with S3 (MinIO) synchronization."""

    CACHE_DIR = Path("data/processed")
    RAW_DATA_PATH = Path("data/HousingData.csv")
    S3_BUCKET = "data-cache"
    S3_PREFIX = "processed-cache"

    REQUIRED_FILES = [
        "train_X.parquet",
        "train_y.parquet",
        "test_X.parquet",
        "test_y.parquet",
        "preprocessor.joblib",
        "metadata.json",
    ]

    def __init__(self, preprocessing_version: str):
        """Initialize cache for a specific preprocessing version.

        Args:
            preprocessing_version: The preprocessing strategy name (e.g., 'v1_median').
        """
        self.preprocessing_version = preprocessing_version
        self.cache_path = self.CACHE_DIR / preprocessing_version
        self._s3_client = None

    @property
    def s3_client(self):
        """Lazy-load S3 client configured for MinIO."""
        if self._s3_client is None:
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
            )
        return self._s3_client

    def _s3_key(self, filename: str) -> str:
        """Get S3 key for a file."""
        return f"{self.S3_PREFIX}/{self.preprocessing_version}/{filename}"

    def exists_locally(self) -> bool:
        """Check if processed data exists locally."""
        return all((self.cache_path / f).exists() for f in self.REQUIRED_FILES)

    def exists_in_s3(self) -> bool:
        """Check if processed data exists in S3 (MinIO)."""
        try:
            # Just check if metadata.json exists as a proxy
            self.s3_client.head_object(Bucket=self.S3_BUCKET, Key=self._s3_key("metadata.json"))
            return True
        except ClientError:
            return False

    def try_pull_from_remote(self) -> bool:
        """Try to pull cached data from S3 (MinIO).

        Returns:
            True if pull was successful and cache now exists locally.
        """
        try:
            if not self.exists_in_s3():
                return False

            print(f"  Pulling {self.preprocessing_version} from MinIO...")
            self.cache_path.mkdir(parents=True, exist_ok=True)

            for filename in self.REQUIRED_FILES:
                local_path = self.cache_path / filename
                self.s3_client.download_file(self.S3_BUCKET, self._s3_key(filename), str(local_path))

            print(f"  ✓ Pulled from MinIO: {self.preprocessing_version}")
            return True

        except ClientError as e:
            print(f"  ⚠ Could not pull from MinIO: {e}")
            return False

    def push_to_remote(self) -> bool:
        """Push cached data to S3 (MinIO).

        Returns:
            True if push was successful.
        """
        if not self.exists_locally():
            return False

        print(f"  Pushing {self.preprocessing_version} to MinIO...")

        try:
            for filename in self.REQUIRED_FILES:
                local_path = self.cache_path / filename
                self.s3_client.upload_file(str(local_path), self.S3_BUCKET, self._s3_key(filename))

            print(f"  ✓ Pushed to MinIO: {self.preprocessing_version}")
            return True

        except ClientError as e:
            print(f"  ⚠ Could not push to MinIO: {e}")
            return False

    def create_cache(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """Create preprocessed data cache.

        Args:
            test_size: Fraction of data for test split.
            random_state: Random seed for reproducibility.

        Returns:
            Metadata dictionary from preprocessing.
        """
        from src.data.preprocess import preprocess_and_save

        print(f"  Creating cache for {self.preprocessing_version}...")

        metadata = preprocess_and_save(
            input_path=str(self.RAW_DATA_PATH),
            output_dir=str(self.cache_path),
            preprocessing_version=self.preprocessing_version,
            test_size=test_size,
            random_state=random_state,
        )

        # Push to remote
        self.push_to_remote()

        return metadata

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, object]:
        """Load preprocessed data from cache.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata, preprocessor_pipeline)
        """
        X_train = pd.read_parquet(self.cache_path / "train_X.parquet").values
        X_test = pd.read_parquet(self.cache_path / "test_X.parquet").values
        y_train = pd.read_parquet(self.cache_path / "train_y.parquet").values.ravel()
        y_test = pd.read_parquet(self.cache_path / "test_y.parquet").values.ravel()

        with open(self.cache_path / "metadata.json") as f:
            metadata = json.load(f)

        preprocessor_pipeline = joblib.load(self.cache_path / "preprocessor.joblib")

        return X_train, X_test, y_train, y_test, metadata, preprocessor_pipeline

    def get_or_create(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, object]:
        """Get cached data, creating if necessary.

        This is the main entry point. It:
        1. Checks local cache
        2. If not found, tries to pull from S3 (MinIO)
        3. If not in remote, creates the cache and pushes

        Args:
            test_size: Fraction of data for test split (only used if creating).
            random_state: Random seed (only used if creating).

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata, preprocessor_pipeline)
        """
        print(f"\n[Cache] Preprocessing: {self.preprocessing_version}")

        # 1. Check local cache
        if self.exists_locally():
            print(f"  ✓ Using local cache")
            return self.load()

        # 2. Try to pull from remote
        if self.try_pull_from_remote():
            return self.load()

        # 3. Create cache
        print(f"  Cache not found, creating...")
        self.create_cache(test_size=test_size, random_state=random_state)
        return self.load()


def get_cached_data(
    preprocessing_version: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, object]:
    """Convenience function to get cached preprocessed data.

    Args:
        preprocessing_version: Preprocessing strategy (e.g., 'v1_median').
        test_size: Fraction of data for test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata, preprocessor_pipeline)
    """
    cache = ProcessedDataCache(preprocessing_version)
    return cache.get_or_create(test_size=test_size, random_state=random_state)

