#!/usr/bin/env python3
"""Main training script for the Housing Price Prediction model."""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn

from src.data.loader import get_data_summary, load_housing_data
from src.data.preprocessing import preprocess_pipeline
from src.models.evaluate import (
    evaluate_model,
    generate_evaluation_report,
    print_metrics,
    save_report,
)
from src.models.train import (
    DEFAULT_PARAMS,
    get_feature_importance,
    save_model,
    train_model,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a housing price prediction model")
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
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in RandomForest (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum depth of trees (default: 10)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="housing-price-prediction",
        help="MLflow experiment name (default: housing-price-prediction)",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register model in MLflow Model Registry",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete training pipeline."""
    args = parse_args()

    print("=" * 50)
    print("Housing Price Prediction - Training Pipeline")
    print("=" * 50)

    # Configure MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    print(f"\nMLflow tracking URI: {args.mlflow_tracking_uri}")
    print(f"MLflow experiment: {args.experiment_name}")

    # Load data
    print(f"\n[1/5] Loading data from {args.data_path}...")
    df = load_housing_data(args.data_path)
    summary = get_data_summary(df)
    print(f"  Loaded {summary['n_rows']} rows, {summary['n_columns']} columns")
    print(f"  Total missing values: {summary['total_missing']}")

    # Preprocess
    print("\n[2/5] Preprocessing data...")
    data = preprocess_pipeline(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"  Train size: {len(data['y_train'])}")
    print(f"  Test size: {len(data['y_test'])}")

    # Start MLflow run
    with mlflow.start_run() as run:
        print(f"\nMLflow run ID: {run.info.run_id}")

        # Log dataset parameters
        mlflow.log_params(
            {
                "data_path": args.data_path,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_samples": summary["n_rows"],
                "n_features": summary["n_columns"] - 1,  # Exclude target
                "missing_values": summary["total_missing"],
            }
        )

        # Train model
        print("\n[3/5] Training RandomForest model...")
        model_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
        }
        model = train_model(
            data["X_train_scaled"],
            data["y_train"],
            params=model_params,
        )
        print(f"  Model trained with params: {model_params}")

        # Log model hyperparameters
        full_params = {**DEFAULT_PARAMS, **model_params}
        mlflow.log_params({f"model_{k}": v for k, v in full_params.items()})

        # Evaluate
        print("\n[4/5] Evaluating model...")
        train_result = evaluate_model(model, data["X_train_scaled"], data["y_train"])
        test_result = evaluate_model(model, data["X_test_scaled"], data["y_test"])
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
        feature_importance = get_feature_importance(model, data["feature_names"])
        print("\nTop 5 Feature Importance:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            print(f"  {i + 1}. {feature}: {importance:.4f}")

        # Log feature importance as metrics
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)

        # Save artifacts locally
        print(f"\n[5/5] Saving model to {args.output_dir}...")
        paths = save_model(model, data["scaler"], args.output_dir)
        print(f"  Model saved: {paths['model_path']}")
        print(f"  Scaler saved: {paths['scaler_path']}")

        # Save evaluation report
        report = generate_evaluation_report(
            train_metrics=train_result["metrics"],
            test_metrics=test_result["metrics"],
            feature_importance=feature_importance,
            model_params=full_params,
        )
        report_path = save_report(report, Path(args.output_dir) / "evaluation_report.json")
        print(f"  Report saved: {report_path}")

        # Log artifacts to MLflow
        mlflow.log_artifact(str(paths["scaler_path"]))
        mlflow.log_artifact(str(report_path))

        # Log model to MLflow with signature
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="housing-price-model" if args.register_model else None,
        )
        print("  Model logged to MLflow")

        if args.register_model:
            print("  Model registered as 'housing-price-model'")

        if report["overfitting_warning"]:
            mlflow.set_tag("overfitting_warning", "true")
            print("\nWarning: Potential overfitting detected (R2 diff > 0.1)")

        # Set useful tags
        mlflow.set_tags(
            {
                "model_type": "RandomForestRegressor",
                "framework": "scikit-learn",
            }
        )

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"View results in MLflow UI: mlflow ui --backend-store-uri {args.mlflow_tracking_uri}")
    print("=" * 50)


if __name__ == "__main__":
    main()
