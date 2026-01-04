"""Single experiment training function.

Uses the preprocessing cache system - each preprocessing version is computed
only once and stored in MinIO via DVC. Subsequent experiments with the same
preprocessing will use the cached data.
"""

import time

from src.experiments.runner import ExperimentConfig, ExperimentResult
from src.training.core import train_model


def train_single_experiment(
    config: ExperimentConfig,
    data_path: str = "data/HousingData.csv",  # Kept for compatibility
) -> ExperimentResult:
    """Train a single experiment and return results.

    Uses the preprocessing cache system:
    - First experiment with a preprocessing version computes and caches data
    - Subsequent experiments reuse the cached preprocessed data
    - All cached data is stored in MinIO via DVC

    Args:
        config: Experiment configuration.
        data_path: Path to the dataset (kept for compatibility, not used directly).

    Returns:
        ExperimentResult with metrics and run info.
    """
    start_time = time.time()

    # Use shared training logic (which uses preprocessing cache)
    result = train_model(
        model_type=config.model_type,
        preprocessing=config.preprocessing,
        data_path=data_path,
        test_size=config.test_size,
        random_state=config.random_state,
        hyperparameters=config.hyperparameters,
        enable_cv=config.enable_cv,
        cv_splits=config.cv_splits,
        source_tag="experiment_runner",
    )

    training_time = time.time() - start_time

    return ExperimentResult(
        run_id=result.run_id,
        model_type=config.model_type,
        preprocessing=config.preprocessing,
        hyperparameters=config.hyperparameters,
        train_metrics=result.train_metrics,
        test_metrics=result.test_metrics,
        cv_metrics=result.cv_result.to_dict() if result.cv_result else None,
        training_time=training_time,
    )
