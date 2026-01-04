"""Preprocessing cache manager with DVC integration.

This module provides transparent caching for preprocessed data.
When training or running experiments:
1. Check if cache exists locally
2. If not, try to pull from DVC remote (MinIO)
3. If not in remote, compute and push to remote

This ensures:
- Preprocessed data is computed only once per strategy
- All data is versioned and stored in S3 (MinIO)
- Both `make train` and `make experiment` use the same cache
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ProcessedDataCache:
    """Manages preprocessed data cache with DVC synchronization."""

    CACHE_DIR = Path("data/processed")
    RAW_DATA_PATH = Path("data/HousingData.csv")

    def __init__(self, preprocessing_version: str):
        """Initialize cache for a specific preprocessing version.

        Args:
            preprocessing_version: The preprocessing strategy name (e.g., 'v1_median').
        """
        self.preprocessing_version = preprocessing_version
        self.cache_path = self.CACHE_DIR / preprocessing_version

    def exists_locally(self) -> bool:
        """Check if processed data exists locally."""
        required_files = [
            "train_X.parquet",
            "train_y.parquet",
            "test_X.parquet",
            "test_y.parquet",
            "preprocessor.joblib",
            "metadata.json",
        ]
        return all((self.cache_path / f).exists() for f in required_files)

    def _run_dvc_command(self, args: list[str], check: bool = False) -> subprocess.CompletedProcess:
        """Run a DVC command."""
        cmd = ["uv", "run", "dvc"] + args
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    def try_pull_from_remote(self) -> bool:
        """Try to pull cached data from DVC remote.

        This uses dvc pull with the cache path. It works when:
        - The data was previously pushed with `dvc push`
        - DVC is tracking the files (via dvc.yaml outs or dvc add)

        Returns:
            True if pull was successful and cache now exists locally.
        """
        # First check if there's a dvc.lock that might have our data
        print(f"  Checking DVC remote for {self.preprocessing_version}...")
        
        # Try to pull the specific path
        result = self._run_dvc_command(["pull", str(self.cache_path), "-f"], check=False)

        if result.returncode == 0 and self.exists_locally():
            print(f"  ✓ Pulled from remote: {self.preprocessing_version}")
            return True

        # Also try general pull in case dvc.lock has the reference
        result = self._run_dvc_command(["pull"], check=False)
        if self.exists_locally():
            print(f"  ✓ Pulled from remote: {self.preprocessing_version}")
            return True

        return False

    def push_to_remote(self) -> bool:
        """Push cached data to DVC remote.

        Returns:
            True if push was successful.
        """
        if not self.exists_locally():
            return False

        print(f"  Pushing {self.preprocessing_version} to MinIO...")
        result = self._run_dvc_command(["push", str(self.cache_path)], check=False)

        if result.returncode == 0:
            print(f"  ✓ Pushed to remote: {self.preprocessing_version}")
            return True

        # Non-critical failure - data is still available locally
        print(f"  ⚠ Could not push to remote (MinIO may be down)")
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
        2. If not found, tries to pull from DVC remote
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

