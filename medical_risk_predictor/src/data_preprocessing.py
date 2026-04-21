from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PreprocessingArtifacts:
    standard_scaler: StandardScaler
    minmax_scaler: MinMaxScaler
    feature_columns: list[str]


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    return pd.read_csv(path)


def replace_zero_with_nan(df: pd.DataFrame, medical_columns: Iterable[str]) -> pd.DataFrame:
    """Replace medically implausible 0 values with NaN in selected columns."""
    out = df.copy()
    for col in medical_columns:
        if col in out.columns:
            out[col] = out[col].replace(0, np.nan)
    return out


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill missing values with median (default) or mean."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if out[col].isna().any():
            fill_value = out[col].median() if strategy == "median" else out[col].mean()
            out[col] = out[col].fillna(fill_value)

    return out


def iqr_outlier_clip(df: pd.DataFrame, exclude_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Clip outliers using the IQR rule to preserve records while reducing extreme influence."""
    out = df.copy()
    exclude_columns = set(exclude_columns or [])

    for col in out.select_dtypes(include=[np.number]).columns:
        if col in exclude_columns:
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out[col] = out[col].clip(lower=lower, upper=upper)

    return out


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessingArtifacts]:
    """Apply StandardScaler then MinMaxScaler and return scaled dataframes."""
    std_scaler = StandardScaler()
    mm_scaler = MinMaxScaler()

    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    X_train_scaled = mm_scaler.fit_transform(X_train_std)
    X_test_scaled = mm_scaler.transform(X_test_std)

    train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    artifacts = PreprocessingArtifacts(
        standard_scaler=std_scaler,
        minmax_scaler=mm_scaler,
        feature_columns=list(X_train.columns),
    )
    return train_df, test_df, artifacts


def split_and_balance(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test split and apply SMOTE to training data."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
    y_train_resampled = pd.Series(y_train_resampled, name=target_column)

    return X_train_resampled, X_test, y_train_resampled, y_test


def preprocess_data(
    path: str,
    target_column: str,
    medical_zero_invalid_cols: Iterable[str],
    missing_strategy: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, PreprocessingArtifacts]:
    """End-to-end preprocessing pipeline returning train/test sets and fitted scalers."""
    df = load_dataset(path)
    df = replace_zero_with_nan(df, medical_zero_invalid_cols)
    df = handle_missing_values(df, strategy=missing_strategy)
    df = iqr_outlier_clip(df, exclude_columns=[target_column])

    X_train, X_test, y_train, y_test = split_and_balance(df, target_column=target_column)
    X_train_scaled, X_test_scaled, artifacts = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, artifacts
