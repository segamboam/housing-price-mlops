"""Tests for preprocessing script functions."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.loader import FEATURE_COLUMNS, TARGET_COLUMN


class TestPreprocessAndSave:
    """Tests for preprocess_and_save function."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing."""
        # Create sample data with all expected columns
        np.random.seed(42)
        n_samples = 50

        data = {}
        for col in FEATURE_COLUMNS:
            if col == "CHAS":
                data[col] = np.random.choice([0, 1], n_samples)
            else:
                data[col] = np.random.uniform(0.1, 100, n_samples)

        data[TARGET_COLUMN] = np.random.uniform(10, 50, n_samples)

        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_creates_output_directory(self, sample_csv, tmp_path):
        """Creates output directory if it doesn't exist."""
        from src.data.preprocess import preprocess_and_save

        output_dir = tmp_path / "output" / "nested"

        preprocess_and_save(
            input_path=str(sample_csv),
            output_dir=str(output_dir),
            preprocessing_version="v1_median",
        )

        assert output_dir.exists()

    def test_creates_all_required_files(self, sample_csv, tmp_path):
        """Creates all required output files."""
        from src.data.preprocess import preprocess_and_save

        output_dir = tmp_path / "output"

        preprocess_and_save(
            input_path=str(sample_csv),
            output_dir=str(output_dir),
            preprocessing_version="v1_median",
        )

        assert (output_dir / "train_X.parquet").exists()
        assert (output_dir / "train_y.parquet").exists()
        assert (output_dir / "test_X.parquet").exists()
        assert (output_dir / "test_y.parquet").exists()
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "preprocessor.joblib").exists()

    def test_returns_metadata(self, sample_csv, tmp_path):
        """Returns metadata dictionary."""
        from src.data.preprocess import preprocess_and_save

        output_dir = tmp_path / "output"

        result = preprocess_and_save(
            input_path=str(sample_csv),
            output_dir=str(output_dir),
            preprocessing_version="v1_median",
        )

        assert isinstance(result, dict)
        assert "preprocessing_version" in result
        assert result["preprocessing_version"] == "v1_median"

    def test_metadata_contains_required_fields(self, sample_csv, tmp_path):
        """Metadata contains all required fields."""
        from src.data.preprocess import preprocess_and_save

        output_dir = tmp_path / "output"

        result = preprocess_and_save(
            input_path=str(sample_csv),
            output_dir=str(output_dir),
            preprocessing_version="v1_median",
        )

        assert "preprocessing_version" in result
        assert "preprocessing_name" in result
        assert "created_at" in result
        assert "train_samples" in result
        assert "test_samples" in result
        assert "feature_columns" in result
        assert "feature_stats" in result

    def test_respects_test_size(self, sample_csv, tmp_path):
        """Respects test_size parameter."""
        from src.data.preprocess import preprocess_and_save

        output_dir = tmp_path / "output"

        result = preprocess_and_save(
            input_path=str(sample_csv),
            output_dir=str(output_dir),
            preprocessing_version="v1_median",
            test_size=0.3,
        )

        total = result["train_samples"] + result["test_samples"]
        test_ratio = result["test_samples"] / total
        assert abs(test_ratio - 0.3) < 0.1  # Allow some tolerance

    def test_different_preprocessing_versions(self, sample_csv, tmp_path):
        """Works with different preprocessing versions."""
        from src.data.preprocess import preprocess_and_save

        for version in ["v1_median", "v2_knn"]:
            output_dir = tmp_path / f"output_{version}"

            result = preprocess_and_save(
                input_path=str(sample_csv),
                output_dir=str(output_dir),
                preprocessing_version=version,
            )

            assert result["preprocessing_version"] == version


class TestLoadProcessedData:
    """Tests for load_processed_data function."""

    @pytest.fixture
    def processed_dir(self, tmp_path):
        """Create a directory with processed data."""
        # Create sample processed data
        X_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X_test = pd.DataFrame({"a": [5.0], "b": [6.0]})
        y_train = pd.DataFrame({"MEDV": [10.0, 20.0]})
        y_test = pd.DataFrame({"MEDV": [30.0]})

        X_train.to_parquet(tmp_path / "train_X.parquet")
        X_test.to_parquet(tmp_path / "test_X.parquet")
        y_train.to_parquet(tmp_path / "train_y.parquet")
        y_test.to_parquet(tmp_path / "test_y.parquet")

        metadata = {"version": "test", "feature_columns": ["a", "b"]}
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return tmp_path

    def test_loads_all_components(self, processed_dir):
        """Loads all data components."""
        from src.data.preprocess import load_processed_data

        result = load_processed_data(str(processed_dir))

        assert len(result) == 5
        X_train, X_test, y_train, y_test, metadata = result

    def test_returns_correct_shapes(self, processed_dir):
        """Returns data with correct shapes."""
        from src.data.preprocess import load_processed_data

        X_train, X_test, y_train, y_test, metadata = load_processed_data(str(processed_dir))

        assert len(X_train) == 2
        assert len(X_test) == 1
        assert len(y_train) == 2
        assert len(y_test) == 1

    def test_raises_on_missing_directory(self):
        """Raises FileNotFoundError for missing directory."""
        from src.data.preprocess import load_processed_data

        with pytest.raises(FileNotFoundError, match="Processed data not found"):
            load_processed_data("/nonexistent/path")

    def test_loads_metadata(self, processed_dir):
        """Loads metadata correctly."""
        from src.data.preprocess import load_processed_data

        _, _, _, _, metadata = load_processed_data(str(processed_dir))

        assert metadata["version"] == "test"
        assert metadata["feature_columns"] == ["a", "b"]
