from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


HEART_CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def _create_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BMI" not in out.columns:
        return out

    bins = [0, 18.5, 25, 30, np.inf]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]
    out["BMI_Category"] = pd.cut(out["BMI"], bins=bins, labels=labels, include_lowest=True)
    out["BMI_Category"] = out["BMI_Category"].astype(str)
    return out


def _create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Age" not in out.columns:
        return out

    bins = [0, 35, 55, np.inf]
    labels = ["Young", "Middle", "Senior"]
    out["Age_Group"] = pd.cut(out["Age"], bins=bins, labels=labels, include_lowest=True)
    out["Age_Group"] = out["Age_Group"].astype(str)
    return out


def _create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Glucose" in out.columns and "Insulin" in out.columns:
        out["Glucose_Insulin_Ratio"] = out["Glucose"] / (out["Insulin"] + 1e-5)

    if "Age" in out.columns and "BMI" in out.columns:
        out["Age_BMI_Interaction"] = out["Age"] * out["BMI"]

    if "BloodPressure" in out.columns and "BMI" in out.columns:
        out["BP_BMI_Interaction"] = out["BloodPressure"] * out["BMI"]

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Apply deterministic feature construction and one-hot encoding.

    Use drop_first=False at inference time: with a single patient row, drop_first would drop
    that patient's own category, zeroing every one-hot column. Reindexing to the model's
    feature list afterwards removes the baseline columns the model never saw.
    """
    enriched = _create_bmi_category(df)
    enriched = _create_age_group(enriched)
    enriched = _create_interaction_features(enriched)

    categorical_columns = [c for c in (categorical_columns or []) if c in enriched.columns]
    for col in categorical_columns:
        # Force categorical treatment to avoid ordinal assumptions on coded values.
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce").round().astype("Int64").astype(str)

    encoded = pd.get_dummies(
        enriched,
        columns=[c for c in ["BMI_Category", "Age_Group"] if c in enriched.columns] + categorical_columns,
        drop_first=drop_first,
    )
    return encoded


def drop_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop one feature from each pair with absolute correlation above threshold."""
    corr_matrix = df.corr(numeric_only=True)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col].abs() > threshold)]
    reduced_df = df.drop(columns=to_drop, errors="ignore")
    return reduced_df, corr_matrix


def select_k_best_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, SelectKBest]:
    """Select top-k features by ANOVA F-statistics."""
    non_constant_cols = X.columns[X.nunique(dropna=False) > 1]
    X = X[non_constant_cols]

    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected_array = selector.fit_transform(X, y)

    selected_columns = X.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected_array, columns=selected_columns, index=X.index)

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "f_score": selector.scores_,
            "selected": selector.get_support(),
        }
    ).sort_values(by="f_score", ascending=False)

    return X_selected, importance, selector


def engineer_features(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.9,
    k_best: int = 10,
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SelectKBest, list[str], list[str]]:
    """Build derived features, remove strong multicollinearity, and select best predictors."""
    encoded = build_feature_matrix(X, categorical_columns=categorical_columns)

    reduced, corr_matrix = drop_highly_correlated_features(encoded, threshold=corr_threshold)
    dropped_columns = [c for c in encoded.columns if c not in reduced.columns]
    selected, importance_df, selector = select_k_best_features(reduced, y, k=k_best)
    reduced_columns = reduced.columns.tolist()

    return selected, importance_df, corr_matrix, selector, dropped_columns, reduced_columns


def transform_features_for_inference(
    X: pd.DataFrame,
    dropped_columns: list[str],
    reduced_columns: list[str],
    selector: SelectKBest,
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply train-time feature construction and selection on new data."""
    encoded = build_feature_matrix(X, categorical_columns=categorical_columns)
    reduced = encoded.drop(columns=dropped_columns, errors="ignore")
    selector_input_columns = list(getattr(selector, "feature_names_in_", reduced_columns))
    reduced = reduced.reindex(columns=selector_input_columns, fill_value=0)
    selected = selector.transform(reduced)
    selected_columns = [col for col, keep in zip(reduced.columns, selector.get_support()) if keep]
    return pd.DataFrame(selected, columns=selected_columns, index=X.index)
