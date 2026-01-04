#!/usr/bin/env python
"""Standalone preprocessing script.

This script preprocesses raw data and saves it to disk for reuse.
Each preprocessing version creates a separate dataset that can be
cached and reused across multiple training runs.

Usage:
    uv run python -m src.data.preprocess --version v1_median

    # Or with custom paths:
    uv run python -m src.data.preprocess \
        --version v2_knn \
        --input data/HousingData.csv \
        --output data/processed/v2_knn

The output directory will contain:
    - train_X.parquet: Preprocessed training features
    - train_y.parquet: Training targets
    - test_X.parquet: Preprocessed test features  
    - test_y.parquet: Test targets
    - metadata.json: Preprocessing metadata and stats
    - preprocessor.joblib: Fitted preprocessor for inference
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after path setup
import src.data.preprocessing.strategies  # noqa: F401 - Register strategies
from src.data.loader import FEATURE_COLUMNS, TARGET_COLUMN, load_housing_data
from src.data.preprocessing.factory import PreprocessorFactory


def preprocess_and_save(
    input_path: str,
    output_dir: str,
    preprocessing_version: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Preprocess data and save to disk.
    
    Args:
        input_path: Path to raw CSV data.
        output_dir: Directory to save processed data.
        preprocessing_version: Preprocessing strategy (e.g., 'v1_median').
        test_size: Fraction for test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary with preprocessing metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING: {preprocessing_version}")
    print(f"{'='*60}")
    
    # Load raw data
    print(f"\n[1/5] Loading data from {input_path}...")
    df = load_housing_data(input_path)
    print(f"  Loaded {len(df)} samples")
    
    # Split features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # Train/test split (BEFORE preprocessing to avoid data leakage)
    print(f"\n[2/5] Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Create and fit preprocessor
    print(f"\n[3/5] Fitting preprocessor '{preprocessing_version}'...")
    preprocessor = PreprocessorFactory.create(preprocessing_version)
    X_train_processed = preprocessor.fit_transform(X_train)
    print(f"  Preprocessor: {preprocessor.description}")
    
    # Transform test data
    print(f"\n[4/5] Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Save processed data
    print(f"\n[5/5] Saving to {output_path}...")
    
    # Save as parquet for efficiency
    train_X_df = pd.DataFrame(X_train_processed, columns=FEATURE_COLUMNS)
    test_X_df = pd.DataFrame(X_test_processed, columns=FEATURE_COLUMNS)
    train_y_df = pd.DataFrame(y_train.values, columns=[TARGET_COLUMN])
    test_y_df = pd.DataFrame(y_test.values, columns=[TARGET_COLUMN])
    
    train_X_df.to_parquet(output_path / "train_X.parquet", index=False)
    train_y_df.to_parquet(output_path / "train_y.parquet", index=False)
    test_X_df.to_parquet(output_path / "test_X.parquet", index=False)
    test_y_df.to_parquet(output_path / "test_y.parquet", index=False)
    print(f"  Saved: train_X.parquet, train_y.parquet, test_X.parquet, test_y.parquet")
    
    # Save fitted preprocessor for inference
    joblib.dump(preprocessor.pipeline, output_path / "preprocessor.joblib")
    print(f"  Saved: preprocessor.joblib")
    
    # Calculate feature stats for monitoring
    feature_stats = {
        col: {
            "min": float(train_X_df[col].min()),
            "max": float(train_X_df[col].max()),
            "mean": float(train_X_df[col].mean()),
            "std": float(train_X_df[col].std()),
        }
        for col in FEATURE_COLUMNS
    }
    
    # Create metadata
    metadata = {
        "preprocessing_version": preprocessing_version,
        "preprocessing_name": preprocessor.name,
        "preprocessing_description": preprocessor.description,
        "created_at": datetime.now(UTC).isoformat(),
        "input_file": str(input_path),
        "test_size": test_size,
        "random_state": random_state,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "feature_stats": feature_stats,
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: metadata.json")
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {output_path}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"{'='*60}\n")
    
    return metadata


def load_processed_data(processed_dir: str) -> tuple:
    """Load preprocessed data from disk.
    
    Args:
        processed_dir: Directory containing processed data.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    path = Path(processed_dir)
    
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    
    X_train = pd.read_parquet(path / "train_X.parquet")
    y_train = pd.read_parquet(path / "train_y.parquet")
    X_test = pd.read_parquet(path / "test_X.parquet")
    y_test = pd.read_parquet(path / "test_y.parquet")
    
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    
    return X_train, X_test, y_train, y_test, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preprocess with v1_median strategy
    uv run python -m src.data.preprocess --version v1_median
    
    # Preprocess with v2_knn and custom output
    uv run python -m src.data.preprocess --version v2_knn --output data/processed/v2_knn
    
    # List available preprocessing strategies
    uv run python -m src.data.preprocess --list
        """
    )
    
    parser.add_argument(
        "--version", "-v",
        type=str,
        help="Preprocessing version (e.g., v1_median, v2_knn)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/HousingData.csv",
        help="Input CSV file (default: data/HousingData.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory (default: data/processed/{version})"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available preprocessing strategies"
    )
    
    args = parser.parse_args()
    
    # List strategies
    if args.list:
        print("\nAvailable preprocessing strategies:")
        for strategy in PreprocessorFactory.list_available():
            preprocessor = PreprocessorFactory.create(strategy)
            print(f"  {strategy}: {preprocessor.description}")
        return
    
    # Validate version
    if not args.version:
        parser.error("--version is required (use --list to see available)")
    
    # Set default output
    output_dir = args.output or f"data/processed/{args.version}"
    
    # Run preprocessing
    preprocess_and_save(
        input_path=args.input,
        output_dir=output_dir,
        preprocessing_version=args.version,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

