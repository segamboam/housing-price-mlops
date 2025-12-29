#!/usr/bin/env python3
"""Main training script for the Housing Price Prediction model."""

import argparse
from pathlib import Path

from src.data.loader import load_housing_data, get_data_summary
from src.data.preprocessing import preprocess_pipeline
from src.models.train import (
    train_model,
    save_model,
    get_feature_importance,
    DEFAULT_PARAMS,
)
from src.models.evaluate import (
    evaluate_model,
    calculate_metrics,
    generate_evaluation_report,
    save_report,
    print_metrics,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a housing price prediction model"
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
    return parser.parse_args()


def main() -> None:
    """Run the complete training pipeline."""
    args = parse_args()

    print("=" * 50)
    print("Housing Price Prediction - Training Pipeline")
    print("=" * 50)

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

    # Evaluate
    print("\n[4/5] Evaluating model...")
    train_result = evaluate_model(
        model, data["X_train_scaled"], data["y_train"]
    )
    test_result = evaluate_model(
        model, data["X_test_scaled"], data["y_test"]
    )
    print_metrics(train_result["metrics"], "Train")
    print_metrics(test_result["metrics"], "Test")

    # Feature importance
    feature_importance = get_feature_importance(model, data["feature_names"])
    print("\nTop 5 Feature Importance:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")

    # Save artifacts
    print(f"\n[5/5] Saving model to {args.output_dir}...")
    paths = save_model(model, data["scaler"], args.output_dir)
    print(f"  Model saved: {paths['model_path']}")
    print(f"  Scaler saved: {paths['scaler_path']}")

    # Save evaluation report
    report = generate_evaluation_report(
        train_metrics=train_result["metrics"],
        test_metrics=test_result["metrics"],
        feature_importance=feature_importance,
        model_params={**DEFAULT_PARAMS, **model_params},
    )
    report_path = save_report(report, Path(args.output_dir) / "evaluation_report.json")
    print(f"  Report saved: {report_path}")

    if report["overfitting_warning"]:
        print("\n⚠️  Warning: Potential overfitting detected (R2 diff > 0.1)")

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
