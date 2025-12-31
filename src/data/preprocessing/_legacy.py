import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "MEDV"
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


def impute_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing values in the DataFrame.

    Args:
        df: DataFrame with potential missing values.
        strategy: Imputation strategy ('median' or 'mean').

    Returns:
        DataFrame with imputed values.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isna().any():
            if strategy == "median":
                fill_value = df[col].median()
            elif strategy == "mean":
                fill_value = df[col].mean()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            df[col] = df[col].fillna(fill_value)

    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    X = df[FEATURE_COLUMNS]
    y = df[target_col]
    return X, y


def create_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Args:
        X: Features DataFrame.
        y: Target Series.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using StandardScaler.

    Fits the scaler on training data only to prevent data leakage.

    Args:
        X_train: Training features.
        X_test: Test features.

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    impute_strategy: str = "median",
) -> dict:
    """Run the complete preprocessing pipeline.

    Args:
        df: Raw DataFrame.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.
        impute_strategy: Strategy for missing value imputation.

    Returns:
        Dictionary containing:
            - X_train_scaled: Scaled training features
            - X_test_scaled: Scaled test features
            - y_train: Training target
            - y_test: Test target
            - scaler: Fitted StandardScaler
            - feature_names: List of feature column names
    """
    df_clean = impute_missing_values(df, strategy=impute_strategy)

    X, y = split_features_target(df_clean)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
    }
