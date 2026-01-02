from pathlib import Path

import pandas as pd

# Feature columns used for model training
FEATURE_COLUMNS = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]

# Target column for prediction
TARGET_COLUMN = "MEDV"

# All expected columns in the dataset (features + target)
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_housing_data(file_path: str | Path) -> pd.DataFrame:
    """Load the Boston Housing dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with the housing data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the schema doesn't match expected columns.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, na_values=["NA", ""])

    validate_schema(df)

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has the expected columns.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If columns don't match expected schema.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if extra_cols:
        raise ValueError(f"Unexpected columns: {extra_cols}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get a summary of the dataset.

    Args:
        df: DataFrame to summarize.

    Returns:
        Dictionary with summary statistics.
    """
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "total_missing": df.isna().sum().sum(),
    }
