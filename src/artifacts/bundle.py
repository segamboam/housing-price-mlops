"""ML Artifact Bundle for packaging model and preprocessor together."""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.artifacts.metadata import ArtifactMetadata
from src.data.preprocessing.base import BasePreprocessor
from src.data.preprocessing.factory import PreprocessorFactory
from src.models.base import BaseModel
from src.models.factory import ModelFactory


class MLArtifactBundle:
    """Manages saving and loading of ML artifacts as a single bundle.

    The bundle contains:
        - metadata.json: Full metadata about the artifact
        - model.joblib: Serialized sklearn model
        - preprocessor.joblib: Serialized sklearn Pipeline

    This ensures consistency between training and inference by packaging
    the exact model and preprocessor used together.
    """

    METADATA_FILE = "metadata.json"
    MODEL_FILE = "model.joblib"
    PREPROCESSOR_FILE = "preprocessor.joblib"

    def __init__(
        self,
        model: BaseModel,
        preprocessor: BasePreprocessor,
        metadata: ArtifactMetadata,
    ):
        """Initialize the bundle with components.

        Args:
            model: Trained model strategy.
            preprocessor: Fitted preprocessor strategy.
            metadata: Artifact metadata.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata

    @classmethod
    def create(
        cls,
        model: BaseModel,
        preprocessor: BasePreprocessor,
        feature_names: list[str],
        training_samples: int,
        test_samples: int = 0,
        model_params: dict[str, Any] | None = None,
        train_metrics: dict[str, float] | None = None,
        test_metrics: dict[str, float] | None = None,
        feature_importance: dict[str, float] | None = None,
        feature_stats: dict[str, dict[str, float]] | None = None,
        mlflow_run_id: str | None = None,
        mlflow_experiment_name: str | None = None,
        random_state: int = 42,
    ) -> "MLArtifactBundle":
        """Create a new artifact bundle from components.

        Args:
            model: Trained model strategy.
            preprocessor: Fitted preprocessor strategy.
            feature_names: Ordered list of feature names.
            training_samples: Number of training samples.
            test_samples: Number of test samples.
            model_params: Hyperparameters used for training.
            train_metrics: Metrics on training set.
            test_metrics: Metrics on test set.
            feature_importance: Feature importance scores.
            feature_stats: Statistics per feature (min, max) for monitoring.
            mlflow_run_id: MLflow run ID if tracked.
            mlflow_experiment_name: MLflow experiment name.
            random_state: Random seed used.

        Returns:
            A new MLArtifactBundle instance.
        """
        metadata = ArtifactMetadata(
            artifact_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            model_type=model.name,
            model_params=model_params or model.params,
            preprocessing_strategy=preprocessor.name,
            preprocessing_version=preprocessor.version,
            feature_names=feature_names,
            training_samples=training_samples,
            test_samples=test_samples,
            random_state=random_state,
            feature_stats=feature_stats or {},
            train_metrics=train_metrics or {},
            test_metrics=test_metrics or {},
            feature_importance=feature_importance or {},
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_name=mlflow_experiment_name,
            framework_versions={
                "scikit-learn": sklearn.__version__,
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
        )
        return cls(model, preprocessor, metadata)

    def save(self, output_dir: Path | str) -> Path:
        """Save the artifact bundle to a directory.

        Args:
            output_dir: Directory to save the bundle.

        Returns:
            Path to the saved bundle directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = output_dir / self.METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.model_dump(mode="json"), f, indent=2, default=str)

        # Save model (the underlying sklearn model)
        model_path = output_dir / self.MODEL_FILE
        joblib.dump(self.model.model, model_path)

        # Save preprocessor pipeline
        preprocessor_path = output_dir / self.PREPROCESSOR_FILE
        joblib.dump(self.preprocessor.pipeline, preprocessor_path)

        return output_dir

    @classmethod
    def load(cls, bundle_dir: Path | str) -> "MLArtifactBundle":
        """Load an artifact bundle from a directory.

        Uses metadata to auto-detect the correct model and preprocessor
        strategy types.

        Args:
            bundle_dir: Directory containing the bundle.

        Returns:
            Loaded MLArtifactBundle instance.

        Raises:
            FileNotFoundError: If bundle directory or files don't exist.
            ValueError: If metadata references unknown strategies.
        """
        bundle_dir = Path(bundle_dir)

        if not bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

        # Load metadata
        metadata_path = bundle_dir / cls.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path) as f:
            metadata_dict = json.load(f)

        # Parse datetime if it's a string
        if isinstance(metadata_dict.get("created_at"), str):
            metadata_dict["created_at"] = datetime.fromisoformat(metadata_dict["created_at"])

        metadata = ArtifactMetadata(**metadata_dict)

        # Import strategies to ensure they're registered
        import src.data.preprocessing.strategies  # noqa: F401
        import src.models.strategies  # noqa: F401

        # Reconstruct model strategy wrapper
        model_strategy = ModelFactory.create(metadata.model_type)
        model_path = bundle_dir / cls.MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_strategy.model = joblib.load(model_path)
        model_strategy.params = metadata.model_params

        # Reconstruct preprocessor strategy wrapper
        preprocessor_strategy = PreprocessorFactory.create(metadata.preprocessing_strategy)
        preprocessor_path = bundle_dir / cls.PREPROCESSOR_FILE
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        preprocessor_strategy.pipeline = joblib.load(preprocessor_path)
        preprocessor_strategy.feature_names = metadata.feature_names

        return cls(model_strategy, preprocessor_strategy, metadata)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Preprocess and predict in one step.

        This is the main inference method that ensures consistency
        with the training pipeline.

        Args:
            X: Features to predict. If DataFrame, columns should match
               feature_names. If array, columns should be in same order.

        Returns:
            Array of predictions.
        """
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict(X_transformed)

    def __repr__(self) -> str:
        return (
            f"MLArtifactBundle("
            f"model={self.metadata.model_type}, "
            f"preprocessing={self.metadata.preprocessing_strategy}, "
            f"id={self.metadata.artifact_id[:8]}...)"
        )
