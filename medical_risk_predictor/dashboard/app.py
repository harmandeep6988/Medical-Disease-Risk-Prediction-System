from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from anomaly_detection import detect_anomalies
from explainability import compute_shap_values


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
    },
    "Heart Disease": {
        "model": MODEL_DIR / "best_heart.joblib",
        "meta": MODEL_DIR / "best_heart_meta.joblib",
        "data": DATA_DIR / "heart_processed.csv",
        "target": "target",
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


def _build_patient_input(features: list[str], data: pd.DataFrame | None) -> pd.DataFrame:
    st.sidebar.header("Patient Input")
    values = {}

    medians = {}
    if data is not None:
        numeric = data.select_dtypes(include=[np.number])
        medians = numeric.median().to_dict()

    for feature in features:
        default_val = float(medians.get(feature, 0.5))
        default_val = min(max(default_val, 0.0), 1.0)
        values[feature] = st.sidebar.slider(feature, min_value=0.0, max_value=1.0, value=default_val, step=0.01)

    return pd.DataFrame([values], columns=features)


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

patient_df = _build_patient_input(feature_order, dataset)
risk_prob = float(model.predict_proba(patient_df)[0][1])

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
    if dataset is None:
        st.info("Processed dataset not found for SHAP background context.")
    else:
        target_col = DISEASE_CFG[disease]["target"]
        bg = dataset.drop(columns=[target_col], errors="ignore")
        bg = bg.reindex(columns=feature_order, fill_value=0)
        bg = bg.sample(min(200, len(bg)), random_state=42)

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
    if dataset is None:
        st.info("Processed dataset not found for model comparison.")
    else:
        target_col = DISEASE_CFG[disease]["target"]
        if target_col not in dataset.columns:
            st.warning(f"Missing target column {target_col} in processed dataset.")
        else:
            models = _list_comparison_models()
            if not models:
                st.info("No generic model files found for comparison.")
            else:
                X = dataset.drop(columns=[target_col], errors="ignore")
                y = dataset[target_col]
                X = X.reindex(columns=feature_order, fill_value=0)

                rows = []
                for name, m in models.items():
                    try:
                        preds = m.predict(X)
                        probs = m.predict_proba(X)[:, 1]
                        rows.append(
                            {
                                "model": name,
                                "accuracy": float((preds == y).mean()),
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
    if dataset is None:
        st.info("Processed dataset not found for anomaly detection.")
    else:
        target_col = DISEASE_CFG[disease]["target"]
        work = dataset.drop(columns=[target_col], errors="ignore")
        work = work.reindex(columns=feature_order, fill_value=0)

        anomalies, scores, reasons, _ = detect_anomalies(work, verbose=False)
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
