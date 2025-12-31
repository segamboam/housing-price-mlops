#!/usr/bin/env python3
"""Main training script for the Housing Price Prediction model.

Supports multiple model strategies and preprocessing pipelines via the Strategy pattern.
Outputs a unified MLArtifactBundle for consistent training/inference.
"""

import argparse
import hashlib
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

import src.data.preprocessing.strategies  # noqa: F401
import src.models.strategies  # noqa: F401
from src.artifacts.bundle import MLArtifactBundle
from src.data.loader import get_data_summary, load_housing_data
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from src.data.preprocessing.factory import PreprocessorFactory
from src.models.evaluate import (
    evaluate_model,
    generate_evaluation_report,
    print_metrics,
    save_report,
)
from src.models.factory import ModelFactory


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a housing price prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/HousingData.csv",
        help="Path to the housing dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Model strategy selection
    parser.add_argument(
        "--model-type",
        type=str,
        choices=ModelFactory.list_available()
        or ["random_forest", "gradient_boost", "xgboost", "linear"],
        default="random_forest",
        help="Model strategy to use",
    )

    # Preprocessing strategy selection
    parser.add_argument(
        "--preprocessing-strategy",
        type=str,
        choices=PreprocessorFactory.list_available() or ["v1_median", "v2_knn", "v3_iterative"],
        default="v1_median",
        help="Preprocessing strategy to use",
    )

    # Model hyperparameters (optional overrides)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of estimators (for ensemble models)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of trees",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (for boosting models)",
    )

    # MLflow configuration
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="housing-price-prediction",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register model in MLflow Model Registry",
    )
    return parser.parse_args()


def build_model_params(args: argparse.Namespace) -> dict:
    """Build model hyperparameters from CLI arguments.

    Only includes parameters that were explicitly provided.
    """
    params = {}
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.random_state is not None:
        params["random_state"] = args.random_state
    return params


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the dataset for versioning."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:12]


def create_input_example(X: pd.DataFrame) -> dict:
    """Create a sample input for model signature."""
    return X.iloc[0].to_dict()


def save_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: dict[str, float],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate and save evaluation plots.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        feature_importance: Feature importance scores.
        output_dir: Directory to save plots.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plots = {}

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted Values")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = plots_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    plots["actual_vs_predicted"] = path

    # 2. Residuals plot
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none")
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residual Plot")
    ax.grid(True, alpha=0.3)
    path = plots_dir / "residuals.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    plots["residuals"] = path

    # 3. Feature Importance (if available)
    if feature_importance:
        fig, ax = plt.subplots(figsize=(10, 6))
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
        ax.barh(features, importances, color=colors)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.grid(True, alpha=0.3, axis="x")
        path = plots_dir / "feature_importance.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        plots["feature_importance"] = path

    # 4. Residuals histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="r", linestyle="--")
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Residuals")
    ax.grid(True, alpha=0.3)
    path = plots_dir / "residuals_histogram.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    plots["residuals_histogram"] = path

    return plots


def main() -> None:
    """Run the complete training pipeline."""
    args = parse_args()

    print("=" * 60)
    print("Housing Price Prediction - Training Pipeline")
    print("=" * 60)
    print(f"\nModel type: {args.model_type}")
    print(f"Preprocessing: {args.preprocessing_strategy}")

    # Configure MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    print(f"\nMLflow tracking URI: {args.mlflow_tracking_uri}")
    print(f"MLflow experiment: {args.experiment_name}")

    # Load data
    print(f"\n[1/6] Loading data from {args.data_path}...")
    df = load_housing_data(args.data_path)
    summary = get_data_summary(df)
    dataset_hash = compute_dataset_hash(df)
    print(f"  Loaded {summary['n_rows']} rows, {summary['n_columns']} columns")
    print(f"  Total missing values: {summary['total_missing']}")
    print(f"  Dataset hash: {dataset_hash}")

    # Split features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Calculate feature statistics for model monitoring (before any preprocessing)
    feature_stats = {
        col: {"min": float(X[col].min()), "max": float(X[col].max())} for col in FEATURE_COLUMNS
    }
    print(f"  Feature stats calculated for {len(feature_stats)} features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print("\n[2/6] Data split:")
    print(f"  Train size: {len(y_train)}")
    print(f"  Test size: {len(y_test)}")

    # Create and fit preprocessor
    print(f"\n[3/6] Preprocessing with {args.preprocessing_strategy}...")
    preprocessor = PreprocessorFactory.create(args.preprocessing_strategy)
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"  Strategy: {preprocessor.description}")

    # Create and train model first to get params for run name
    print(f"\n[4/6] Training {args.model_type} model...")
    model = ModelFactory.create(args.model_type)
    extra_params = build_model_params(args)
    model.train(X_train_transformed, y_train.values, **extra_params)

    # Build descriptive run name with key hyperparameters
    key_param_names = ["n_estimators", "max_depth", "learning_rate"]
    key_params = {k: v for k, v in model.params.items() if k in key_param_names}
    param_str = "_".join(f"{k}={v}" for k, v in key_params.items()) if key_params else ""
    run_name = f"{args.model_type}_{args.preprocessing_strategy}"
    if param_str:
        run_name = f"{run_name}_{param_str}"

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\nMLflow run ID: {run.info.run_id}")

        # Log configuration parameters
        mlflow.log_params(
            {
                "data_path": args.data_path,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_samples": summary["n_rows"],
                "n_features": summary["n_columns"] - 1,
                "missing_values": summary["total_missing"],
                "model_type": args.model_type,
                "preprocessing_strategy": args.preprocessing_strategy,
                "preprocessing_version": preprocessor.version,
                "dataset_hash": dataset_hash,
            }
        )

        # Log dataset info as tags for easier filtering
        mlflow.set_tags(
            {
                "dataset.name": Path(args.data_path).stem,
                "dataset.hash": dataset_hash,
                "dataset.samples": str(summary["n_rows"]),
            }
        )

        # Log model hyperparameters (model already trained before run started)
        mlflow.log_params({f"model_{k}": v for k, v in model.params.items()})
        print(f"  Model params: {model.params}")

        # Evaluate
        print("\n[5/6] Evaluating model...")
        train_result = evaluate_model(model.model, X_train_transformed, y_train.values)
        test_result = evaluate_model(model.model, X_test_transformed, y_test.values)
        print_metrics(train_result["metrics"], "Train")
        print_metrics(test_result["metrics"], "Test")

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "train_rmse": train_result["metrics"]["rmse"],
                "train_mae": train_result["metrics"]["mae"],
                "train_r2": train_result["metrics"]["r2"],
                "test_rmse": test_result["metrics"]["rmse"],
                "test_mae": test_result["metrics"]["mae"],
                "test_r2": test_result["metrics"]["r2"],
            }
        )

        # Feature importance
        feature_importance = model.get_feature_importance(FEATURE_COLUMNS)
        if feature_importance:
            print("\nTop 5 Feature Importance:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                print(f"  {i + 1}. {feature}: {importance:.4f}")

            # Log feature importance as metrics
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)
        else:
            feature_importance = {}

        # Generate and log evaluation plots
        output_dir = Path(args.output_dir)
        print("\n  Generating evaluation plots...")
        plots = save_evaluation_plots(
            y_test.values,
            test_result["predictions"],
            feature_importance,
            output_dir,
        )
        for name, plot_path in plots.items():
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
        print(f"  Logged {len(plots)} plots to MLflow")

        # Log detailed data summary as artifact
        data_summary = {
            "n_samples": summary["n_rows"],
            "n_features": len(FEATURE_COLUMNS),
            "missing_values_total": summary["total_missing"],
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "feature_stats": {
                col: {
                    "mean": float(X[col].mean()),
                    "std": float(X[col].std()),
                    "min": float(X[col].min()),
                    "max": float(X[col].max()),
                    "missing": int(X[col].isna().sum()),
                }
                for col in FEATURE_COLUMNS
            },
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
            },
        }
        mlflow.log_dict(data_summary, "data_summary.json")

        # Add markdown description for the run (visible in MLflow UI)
        top_features = (
            ", ".join(list(feature_importance.keys())[:3]) if feature_importance else "N/A"
        )
        run_description = f"""## Experiment: {args.experiment_name}

### Model: {model.name}
### Preprocessing: {preprocessor.name} (v{preprocessor.version})

**Configuration:**
- Dataset: {Path(args.data_path).stem} ({summary["n_rows"]} samples)
- Train/Test split: {int((1 - args.test_size) * 100)}% / {int(args.test_size * 100)}%
- Random state: {args.random_state}

**Results:**
- Test R²: {test_result["metrics"]["r2"]:.4f}
- Test RMSE: {test_result["metrics"]["rmse"]:.4f}
- Test MAE: {test_result["metrics"]["mae"]:.4f}

**Top Features:** {top_features}

**Hyperparameters:**
{chr(10).join(f"- {k}: {v}" for k, v in model.params.items())}
"""
        mlflow.set_tag("mlflow.note.content", run_description)

        # Add additional descriptive tags
        mlflow.set_tags(
            {
                "author": os.getenv("USER", "unknown"),
                "purpose": "training",
                "environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        # Create artifact bundle
        print(f"\n[6/6] Saving artifact bundle to {args.output_dir}...")
        bundle = MLArtifactBundle.create(
            model=model,
            preprocessor=preprocessor,
            feature_names=FEATURE_COLUMNS,
            training_samples=len(y_train),
            test_samples=len(y_test),
            train_metrics=train_result["metrics"],
            test_metrics=test_result["metrics"],
            feature_importance=feature_importance,
            feature_stats=feature_stats,
            mlflow_run_id=run.info.run_id,
            mlflow_experiment_name=args.experiment_name,
            random_state=args.random_state,
        )

        bundle_path = output_dir / "artifact_bundle"
        bundle.save(bundle_path)
        print(f"  Bundle saved: {bundle_path}")
        print(f"  Artifact ID: {bundle.metadata.artifact_id}")

        # Also save legacy format for backward compatibility
        from src.models.train import save_model as save_model_legacy

        legacy_paths = save_model_legacy(model.model, preprocessor.pipeline, output_dir)
        print(f"  Legacy model saved: {legacy_paths['model_path']}")
        print(f"  Legacy scaler saved: {legacy_paths['scaler_path']}")

        # Save evaluation report
        report = generate_evaluation_report(
            train_metrics=train_result["metrics"],
            test_metrics=test_result["metrics"],
            feature_importance=feature_importance,
            model_params=model.params,
        )
        report_path = save_report(report, output_dir / "evaluation_report.json")
        print(f"  Report saved: {report_path}")

        # Log artifacts to MLflow
        mlflow.log_artifacts(str(bundle_path), artifact_path="artifact_bundle")
        mlflow.log_artifact(str(legacy_paths["scaler_path"]))
        mlflow.log_artifact(str(report_path))

        # Create model signature with input example
        input_example = create_input_example(X_train)
        signature = infer_signature(
            X_train_transformed,
            model.model.predict(X_train_transformed[:1]),
        )

        # Log model to MLflow with signature and input example
        model_info = mlflow.sklearn.log_model(
            model.model,
            artifact_path="model",
            signature=signature,
            input_example=np.array([list(input_example.values())]),
            registered_model_name="housing-price-model" if args.register_model else None,
        )
        print(f"  Model logged to MLflow: {model_info.model_uri}")

        if args.register_model:
            print("  Model registered as 'housing-price-model'")
            # Set alias for the new version if requested
            client = mlflow.MlflowClient()
            # Use search_model_versions instead of deprecated get_latest_versions
            versions = client.search_model_versions("name='housing-price-model'")
            if versions:
                # Get the latest version (highest version number)
                latest_version = max(v.version for v in versions)
                # Mark as challenger (ready for testing)
                try:
                    client.set_registered_model_alias(
                        "housing-price-model", "challenger", latest_version
                    )
                    print(f"  Model version {latest_version} aliased as 'challenger'")
                    print("  To promote: python scripts/promote_model.py --version", latest_version)
                except Exception as e:
                    print(f"  Warning: Could not set alias: {e}")

        if report["overfitting_warning"]:
            mlflow.set_tag("overfitting_warning", "true")
            print("\nWarning: Potential overfitting detected (R2 diff > 0.1)")

        # Set useful tags
        mlflow.set_tags(
            {
                "model_type": args.model_type,
                "preprocessing_strategy": args.preprocessing_strategy,
                "framework": "scikit-learn",
            }
        )

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"View results in MLflow UI: mlflow ui --backend-store-uri {args.mlflow_tracking_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
