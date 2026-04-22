from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from anomaly_detection import detect_anomalies
from data_preprocessing import handle_missing_values, replace_zero_with_nan
from explainability import compute_shap_values
from feature_engineering import build_feature_matrix


st.set_page_config(page_title="Medical Disease Risk Dashboard", layout="wide")
st.title("Medical Disease Risk Prediction System")
st.caption("Diabetes and Heart Disease risk prediction with explainability, anomaly detection, and PDF reporting")

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "saved_models"


DISEASE_CFG = {
    "Diabetes": {
        "model": MODEL_DIR / "best_diabetes.joblib",
        "meta": MODEL_DIR / "best_diabetes_meta.joblib",
        "data": DATA_DIR / "diabetes_processed.csv",
        "target": "Outcome",
        "zero_invalid": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    },
    "Heart Disease": {
        "model": MODEL_DIR / "best_heart.joblib",
        "meta": MODEL_DIR / "best_heart_meta.joblib",
        "data": DATA_DIR / "heart_processed.csv",
        "target": "target",
        "zero_invalid": [],
        "category_labels": {
            "sex": {0: "Female", 1: "Male"},
            "cp": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"},
            "fbs": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"},
            "restecg": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"},
            "exang": {0: "No", 1: "Yes"},
            "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
            "ca": {0: "0 Vessels", 1: "1 Vessel", 2: "2 Vessels", 3: "3 Vessels", 4: "4 Vessels"},
            "thal": {0: "Unknown", 1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"},
        },
    },
}


def _risk_color(prob: float) -> str:
    if prob < 0.33:
        return "#2E8B57"
    if prob < 0.66:
        return "#F0AD4E"
    return "#D9534F"


def _risk_label(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"


def _gauge(prob: float) -> go.Figure:
    color = _risk_color(prob)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Predicted Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 33], "color": "rgba(46,139,87,0.2)"},
                    {"range": [33, 66], "color": "rgba(240,173,78,0.2)"},
                    {"range": [66, 100], "color": "rgba(217,83,79,0.2)"},
                ],
            },
        )
    )
    fig.update_layout(height=320)
    return fig


def _load_disease_assets(disease: str):
    cfg = DISEASE_CFG[disease]
    if not cfg["model"].exists() or not cfg["meta"].exists():
        return None, None, None

    model = joblib.load(cfg["model"])
    meta = joblib.load(cfg["meta"])

    data = None
    if cfg["data"].exists():
        data = pd.read_csv(cfg["data"])

    return model, meta, data


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _fit_reference_transform(
    dataset: pd.DataFrame,
    target_col: str,
    zero_invalid_cols: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    raw = dataset.drop(columns=[target_col], errors="ignore")
    raw = _coerce_numeric(raw)
    raw = replace_zero_with_nan(raw, zero_invalid_cols)
    raw = handle_missing_values(raw)

    raw_columns = list(raw.columns)
    passthrough_cols = [c for c in categorical_columns if c in raw_columns]
    scaled_cols = [c for c in raw_columns if c not in passthrough_cols]

    std: StandardScaler | None = None
    mm: MinMaxScaler | None = None
    scaled = raw.copy()

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


def _transform_raw_to_model_features(
    raw_input: pd.DataFrame,
    ref_transform: dict[str, Any],
    feature_order: list[str],
    zero_invalid_cols: list[str],
) -> pd.DataFrame:
    row = raw_input.reindex(columns=ref_transform["raw_columns"], fill_value=np.nan)
    row = _coerce_numeric(row)
    row = replace_zero_with_nan(row, zero_invalid_cols)

    medians = ref_transform["raw_medians"]
    for col, value in medians.items():
        if col in row.columns:
            row[col] = row[col].fillna(value)

    row = handle_missing_values(row)

    row_scaled = row.copy()
    scaled_cols = ref_transform["scaled_columns"]
    if scaled_cols:
        row_std = ref_transform["std_scaler"].transform(row[scaled_cols])
        row_scaled_vals = ref_transform["mm_scaler"].transform(row_std)
        row_scaled.loc[:, scaled_cols] = row_scaled_vals

    engineered = build_feature_matrix(row_scaled, categorical_columns=ref_transform["categorical_columns"])
    return engineered.reindex(columns=feature_order, fill_value=0.0)


def _transform_dataset_to_model_features(
    dataset: pd.DataFrame,
    target_col: str,
    ref_transform: dict[str, Any],
    feature_order: list[str],
    zero_invalid_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    raw = dataset.drop(columns=[target_col], errors="ignore")
    raw = _coerce_numeric(raw)
    raw = replace_zero_with_nan(raw, zero_invalid_cols)
    raw = handle_missing_values(raw)

    medians = ref_transform["raw_medians"]
    for col, value in medians.items():
        if col in raw.columns:
            raw[col] = raw[col].fillna(value)

    raw = raw.reindex(columns=ref_transform["raw_columns"], fill_value=0.0)
    raw_scaled = raw.copy()
    scaled_cols = ref_transform["scaled_columns"]
    if scaled_cols:
        raw_std = ref_transform["std_scaler"].transform(raw[scaled_cols])
        raw_scaled_vals = ref_transform["mm_scaler"].transform(raw_std)
        raw_scaled.loc[:, scaled_cols] = raw_scaled_vals

    engineered = build_feature_matrix(raw_scaled, categorical_columns=ref_transform["categorical_columns"])
    X_model = engineered.reindex(columns=feature_order, fill_value=0.0)
    y = dataset[target_col] if target_col in dataset.columns else pd.Series(dtype=float)
    return X_model, y


def _build_patient_input(
    raw_features: list[str],
    data: pd.DataFrame | None,
    category_labels: dict[str, dict[int, str]],
) -> pd.DataFrame:
    st.sidebar.header("Patient Input")
    values = {}

    for feature in raw_features:
        if data is None or feature not in data.columns:
            values[feature] = st.sidebar.number_input(feature, value=0.0)
            continue

        series = pd.to_numeric(data[feature], errors="coerce").dropna()
        if series.empty:
            values[feature] = st.sidebar.number_input(feature, value=0.0)
            continue

        if feature in category_labels:
            unique_vals = sorted(series.unique().tolist())
            options = [int(v) for v in unique_vals] if unique_vals else sorted(category_labels[feature].keys())
            default_guess = int(round(float(series.median())))
            default_idx = min(range(len(options)), key=lambda i: abs(options[i] - default_guess)) if options else 0
            values[feature] = st.sidebar.selectbox(
                feature,
                options=options,
                index=default_idx,
                format_func=lambda v, f=feature: f"{category_labels[f].get(int(v), f'Code {int(v)}')} ({int(v)})",
            )
            continue

        is_integer_like = bool(np.allclose(series.to_numpy(), np.round(series.to_numpy())))
        unique_vals = sorted(series.unique().tolist())

        if is_integer_like and len(unique_vals) <= 8:
            options = [int(v) for v in unique_vals]
            default_guess = int(round(float(series.median())))
            default_idx = min(range(len(options)), key=lambda i: abs(options[i] - default_guess))
            values[feature] = st.sidebar.selectbox(feature, options=options, index=default_idx)
        else:
            min_v = float(series.min())
            max_v = float(series.max())
            if feature.lower() == "age":
                max_v = max(max_v, 100.0)
            if max_v <= min_v:
                max_v = min_v + 1.0
            default_v = float(series.median())
            step = 1.0 if is_integer_like else max((max_v - min_v) / 200.0, 0.01)
            values[feature] = st.sidebar.number_input(
                feature,
                min_value=min_v,
                max_value=max_v,
                value=default_v,
                step=step,
            )

    return pd.DataFrame([values], columns=raw_features)


def _list_comparison_models() -> dict[str, object]:
    models = {}
    for p in sorted(MODEL_DIR.glob("*.joblib")):
        if p.name.endswith("_meta.joblib"):
            continue
        if p.name.startswith("best_"):
            continue
        try:
            obj = joblib.load(p)
            if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
                models[p.stem] = obj
        except Exception:
            continue
    return models


disease = st.sidebar.selectbox("Disease", list(DISEASE_CFG.keys()))
model, meta, dataset = _load_disease_assets(disease)

if model is None or meta is None:
    st.error("Best model artifacts not found. Run scripts/train_and_evaluate.py first.")
    st.stop()

feature_order = list(meta.get("features", []))
if not feature_order:
    st.error("Model metadata missing feature list.")
    st.stop()

if dataset is None:
    st.error("Processed dataset is required to build patient-friendly input ranges.")
    st.stop()

target_col = DISEASE_CFG[disease]["target"]
zero_invalid_cols = DISEASE_CFG[disease]["zero_invalid"]
category_labels = DISEASE_CFG[disease].get("category_labels", {})
meta_raw_columns = list(meta.get("raw_columns", []))
if not meta_raw_columns:
    meta_raw_columns = [c for c in dataset.columns if c != target_col]
meta_categorical = [c for c in meta.get("categorical_columns", []) if c in meta_raw_columns]

ref_transform = _fit_reference_transform(dataset, target_col, zero_invalid_cols, categorical_columns=meta_categorical)

patient_raw_df = _build_patient_input(ref_transform["raw_columns"], ref_transform["raw_reference"], category_labels)
patient_df = _transform_raw_to_model_features(patient_raw_df, ref_transform, feature_order, zero_invalid_cols)
risk_prob = float(model.predict_proba(patient_df)[0][1])

X_model_all, y_all = _transform_dataset_to_model_features(
    dataset,
    target_col,
    ref_transform,
    feature_order,
    zero_invalid_cols,
)

pred_tab, shap_tab, comp_tab, anomaly_tab = st.tabs([
    "Prediction",
    "SHAP Explanation",
    "Model Comparison",
    "Anomaly Detection",
])

with pred_tab:
    st.subheader(f"{disease} Risk Prediction")
    st.plotly_chart(_gauge(risk_prob), use_container_width=True)
    label = _risk_label(risk_prob)
    if label == "Low":
        st.markdown("### Risk Level: :green[Low]")
    elif label == "Medium":
        st.markdown("### Risk Level: :orange[Medium]")
    else:
        st.markdown("### Risk Level: :red[High]")

with shap_tab:
    st.subheader("Patient-Level Feature Contributions")
    if X_model_all.empty:
        st.info("Processed dataset not available for SHAP background context.")
    else:
        bg = X_model_all.sample(min(200, len(X_model_all)), random_state=42)

        _, shap_values = compute_shap_values(model, bg, patient_df)

        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]

        contrib = pd.DataFrame(
            {
                "feature": feature_order,
                "impact": values[0],
                "abs_impact": np.abs(values[0]),
            }
        ).sort_values("abs_impact", ascending=False)

        st.dataframe(contrib[["feature", "impact"]], use_container_width=True)

        fig = go.Figure(
            [
                go.Bar(
                    x=contrib["feature"],
                    y=contrib["impact"],
                    marker_color=["#D9534F" if v > 0 else "#2E8B57" for v in contrib["impact"]],
                )
            ]
        )
        fig.update_layout(title="SHAP Contribution by Feature", xaxis_title="Feature", yaxis_title="Impact")
        st.plotly_chart(fig, use_container_width=True)

with comp_tab:
    st.subheader("Model Comparison")
    if X_model_all.empty or y_all.empty:
        st.info("Processed dataset not found for model comparison.")
    else:
        models = _list_comparison_models()
        if not models:
            st.info("No generic model files found for comparison.")
        else:
            rows = []
            for name, m in models.items():
                try:
                    preds = m.predict(X_model_all)
                    probs = m.predict_proba(X_model_all)[:, 1]
                    rows.append(
                        {
                            "model": name,
                            "accuracy": float((preds == y_all).mean()),
                            "avg_probability": float(np.mean(probs)),
                        }
                    )
                except Exception:
                    continue

            if rows:
                out = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
                st.dataframe(out, use_container_width=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(name="accuracy", x=out["model"], y=out["accuracy"]))
                fig.add_trace(go.Bar(name="avg_probability", x=out["model"], y=out["avg_probability"]))
                fig.update_layout(barmode="group", title="Model Comparison")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No comparable models could be evaluated on the selected disease dataset.")

with anomaly_tab:
    st.subheader("Anomaly Detection")
    if X_model_all.empty:
        st.info("Processed dataset not found for anomaly detection.")
    else:
        anomalies, scores, reasons, _ = detect_anomalies(X_model_all, verbose=False)
        st.metric("Anomalous Patient Rows", len(anomalies))

        if anomalies:
            result = pd.DataFrame(
                {
                    "index": anomalies,
                    "anomaly_score": scores.loc[anomalies].values,
                    "most_unusual_feature": reasons.loc[anomalies, "most_unusual_feature"].values,
                    "feature_zscore": reasons.loc[anomalies, "feature_zscore"].values,
                }
            )
            st.dataframe(result, use_container_width=True)
        else:
            st.success("No anomalies detected with current settings.")
