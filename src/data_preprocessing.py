from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PreprocessingArtifacts:
    standard_scaler: StandardScaler | None
    minmax_scaler: MinMaxScaler | None
    feature_columns: list[str]
    scaled_columns: list[str]
    passthrough_columns: list[str]


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


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    passthrough_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessingArtifacts]:
    """Apply StandardScaler then MinMaxScaler and return scaled dataframes."""
    passthrough = [c for c in (passthrough_columns or []) if c in X_train.columns]
    scaled_cols = [c for c in X_train.columns if c not in passthrough]

    train_df = X_train.copy()
    test_df = X_test.copy()

    std_scaler: StandardScaler | None = None
    mm_scaler: MinMaxScaler | None = None

    if scaled_cols:
        std_scaler = StandardScaler()
        mm_scaler = MinMaxScaler()

        X_train_std = std_scaler.fit_transform(X_train[scaled_cols])
        X_test_std = std_scaler.transform(X_test[scaled_cols])

        X_train_scaled = mm_scaler.fit_transform(X_train_std)
        X_test_scaled = mm_scaler.transform(X_test_std)

        train_scaled_df = pd.DataFrame(X_train_scaled, columns=scaled_cols, index=train_df.index)
        test_scaled_df = pd.DataFrame(X_test_scaled, columns=scaled_cols, index=test_df.index)

        train_df = train_df.astype({col: float for col in scaled_cols})
        test_df = test_df.astype({col: float for col in scaled_cols})

        train_df[scaled_cols] = train_scaled_df
        test_df[scaled_cols] = test_scaled_df

    artifacts = PreprocessingArtifacts(
        standard_scaler=std_scaler,
        minmax_scaler=mm_scaler,
        feature_columns=list(X_train.columns),
        scaled_columns=scaled_cols,
        passthrough_columns=passthrough,
    )
    return train_df, test_df, artifacts


def split_and_balance(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    categorical_columns: Iterable[str] | None = None,
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

    categorical_columns = [c for c in (categorical_columns or []) if c in X.columns]
    categorical_idx = [X.columns.get_loc(c) for c in categorical_columns]

    if categorical_idx:
        smote = SMOTENC(categorical_features=categorical_idx, random_state=random_state)
    else:
        smote = SMOTE(random_state=random_state)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
    for col in categorical_columns:
        fill_value = None
        if col in X_train.columns:
            series = pd.to_numeric(X_train[col], errors="coerce")
            if not series.dropna().empty:
                fill_value = series.mode().iloc[0]
        if fill_value is None:
            fill_value = 0
        X_train_resampled[col] = (
            pd.to_numeric(X_train_resampled[col], errors="coerce")
            .fillna(fill_value)
            .round()
            .astype(int)
        )

    y_train_resampled = pd.Series(y_train_resampled, name=target_column)

    return X_train_resampled, X_test, y_train_resampled, y_test


def preprocess_data(
    path: str,
    target_column: str,
    medical_zero_invalid_cols: Iterable[str],
    missing_strategy: str = "median",
    categorical_passthrough_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, PreprocessingArtifacts]:
    """End-to-end preprocessing pipeline returning train/test sets and fitted scalers."""
    df = load_dataset(path)
    df = replace_zero_with_nan(df, medical_zero_invalid_cols)
    df = handle_missing_values(df, strategy=missing_strategy)
    df = iqr_outlier_clip(df, exclude_columns=[target_column])

    categorical_passthrough_cols = [c for c in (categorical_passthrough_cols or []) if c in df.columns]
    X_train, X_test, y_train, y_test = split_and_balance(
        df,
        target_column=target_column,
        categorical_columns=categorical_passthrough_cols,
    )

    X_train_scaled, X_test_scaled, artifacts = scale_features(
        X_train,
        X_test,
        passthrough_columns=categorical_passthrough_cols,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, artifacts
