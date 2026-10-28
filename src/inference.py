"""Shared inference-time preprocessing used by the dashboard and the REST API.

Raw patient values go through the same steps as the training data: invalid zeros -> NaN,
median imputation, StandardScaler -> MinMaxScaler on numeric columns, then feature engineering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_preprocessing import handle_missing_values, replace_zero_with_nan
from feature_engineering import build_feature_matrix


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fit_reference_transform(
    dataset: pd.DataFrame,
    target_col: str,
    zero_invalid_cols: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    raw = dataset.drop(columns=[target_col], errors="ignore")
    raw = coerce_numeric(raw)
    raw = replace_zero_with_nan(raw, zero_invalid_cols)
    raw = handle_missing_values(raw)

    raw_columns = list(raw.columns)
    passthrough_cols = [c for c in categorical_columns if c in raw_columns]
    scaled_cols = [c for c in raw_columns if c not in passthrough_cols]

    std: StandardScaler | None = None
    mm: MinMaxScaler | None = None
    scaled = raw.copy()
    if scaled_cols := [c for c in raw.columns if c not in passthrough_cols]:
        scaled[scaled_cols] = scaled[scaled_cols].astype(float)

    if scaled_cols:
        std = StandardScaler()
        mm = MinMaxScaler()
        scaled_std = std.fit_transform(raw[scaled_cols])
        scaled_vals = mm.fit_transform(scaled_std)
        scaled.loc[:, scaled_cols] = scaled_vals

    engineered = build_feature_matrix(scaled, categorical_columns=passthrough_cols)

    return {
        "raw_columns": raw_columns,
        "raw_medians": raw.median(numeric_only=True).to_dict(),
        "std_scaler": std,
        "mm_scaler": mm,
        "scaled_columns": scaled_cols,
        "categorical_columns": passthrough_cols,
        "engineered_reference": engineered,
        "raw_reference": raw,
    }


def transform_raw_to_model_features(
    raw_input: pd.DataFrame,
    ref_transform: dict[str, Any],
    feature_order: list[str],
    zero_invalid_cols: list[str],
) -> pd.DataFrame:
    row = raw_input.reindex(columns=ref_transform["raw_columns"], fill_value=np.nan)
    row = coerce_numeric(row)
    row = replace_zero_with_nan(row, zero_invalid_cols)

    medians = ref_transform["raw_medians"]
    for col, value in medians.items():
        if col in row.columns:
            row[col] = row[col].fillna(value)

    row = handle_missing_values(row)

    scaled_cols = ref_transform["scaled_columns"]
    row_scaled = row.copy()
    if scaled_cols:
        row_scaled[scaled_cols] = row_scaled[scaled_cols].astype(float)
    if scaled_cols:
        row_std = ref_transform["std_scaler"].transform(row[scaled_cols])
        row_scaled_vals = ref_transform["mm_scaler"].transform(row_std)
        row_scaled.loc[:, scaled_cols] = row_scaled_vals

    engineered = build_feature_matrix(row_scaled, categorical_columns=ref_transform["categorical_columns"], drop_first=False)
    return engineered.reindex(columns=feature_order, fill_value=0.0)


def transform_dataset_to_model_features(
    dataset: pd.DataFrame,
    target_col: str,
    ref_transform: dict[str, Any],
    feature_order: list[str],
    zero_invalid_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    raw = dataset.drop(columns=[target_col], errors="ignore")
    raw = coerce_numeric(raw)
    raw = replace_zero_with_nan(raw, zero_invalid_cols)
    raw = handle_missing_values(raw)

    medians = ref_transform["raw_medians"]
    for col, value in medians.items():
        if col in raw.columns:
            raw[col] = raw[col].fillna(value)

    raw = raw.reindex(columns=ref_transform["raw_columns"], fill_value=0.0)
    scaled_cols = ref_transform["scaled_columns"]
    raw_scaled = raw.copy()
    if scaled_cols:
        raw_scaled[scaled_cols] = raw_scaled[scaled_cols].astype(float)
    if scaled_cols:
        raw_std = ref_transform["std_scaler"].transform(raw[scaled_cols])
        raw_scaled_vals = ref_transform["mm_scaler"].transform(raw_std)
        raw_scaled.loc[:, scaled_cols] = raw_scaled_vals

    engineered = build_feature_matrix(raw_scaled, categorical_columns=ref_transform["categorical_columns"], drop_first=False)
    X_model = engineered.reindex(columns=feature_order, fill_value=0.0)
    y = dataset[target_col] if target_col in dataset.columns else pd.Series(dtype=float)
    return X_model, y
