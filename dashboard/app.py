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
from report_generator import generate_patient_report


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
            "cp": {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"},
            "fbs": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"},
            "restecg": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"},
            "exang": {0: "No", 1: "Yes"},
            "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
            "ca": {0: "0 Vessels", 1: "1 Vessel", 2: "2 Vessels", 3: "3 Vessels", 4: "4 Vessels"},
            "thal": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"},
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


def _assessment_risk_band(prob: float) -> str:
    if prob < 0.4:
        return "Low"
    if prob <= 0.7:
        return "Moderate"
    return "High"


def _show_assessment_help() -> None:
    """Display help expander explaining results in plain language."""
    with st.expander("📚 How to Read Your Results", expanded=False):
        st.markdown("""
        ### Understanding Your Results

        **What is the Risk Score?**
        
        This is a percentage that estimates how likely you are to develop this disease based on the information you provided.
        - A score of **20%** means out of 100 people with your profile, roughly 20 might develop this disease
        - A score of **60%** means the prediction is less certain but leans toward higher risk
        
        **Risk Levels Explained:**
        
        🟢 **Low Risk** (below 40%): Your characteristics suggest lower risk right now. Keep up healthy habits!
        
        🟡 **Moderate Risk** (40-70%): Your factors suggest moderate risk. It's a good time to talk with your doctor about prevention.
        
        🔴 **High Risk** (above 70%): Your characteristics suggest higher risk. Please schedule a check-up with your doctor to discuss.
        
        **What are SHAP Factors?**
        
        These are the top factors that influenced the prediction, listed in order of importance:
        - **Increases risk**: Factors that pushed the prediction toward higher risk
        - **Decreases risk**: Factors that pushed the prediction toward lower risk
        
        For example, if you see "Glucose Level increases risk" with a high impact, that means high glucose was an important reason for the prediction.
        
        **Important Notes:**
        - This tool is educational and uses medical data patterns
        - It is NOT a diagnosis—only a doctor can diagnose
        - Your actual risk depends on many factors not captured here
        - Always consult with a healthcare professional for medical decisions
        """)


def _assessment_risk_color(prob: float) -> str:
    if prob < 0.4:
        return "#2E8B57"
    if prob <= 0.7:
        return "#D18B00"
    return "#B22222"


def _assessment_gauge(prob: float, disease: str) -> go.Figure:
    color = _assessment_risk_color(prob)
    band = _assessment_risk_band(prob)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": f"{disease} Self-Assessment Risk · {band}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "rgba(46,139,87,0.18)"},
                    {"range": [40, 70], "color": "rgba(209,139,0,0.18)"},
                    {"range": [70, 100], "color": "rgba(178,34,34,0.18)"},
                ],
                "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": prob * 100},
            },
        )
    )
    fig.update_layout(height=380, margin={"l": 20, "r": 20, "t": 70, "b": 20})
    return fig


def _assessment_card(prob: float, disease: str) -> str:
    if prob < 0.4:
        title = "Low Risk"
        message = "maintain healthy habits"
        bg = "#E8F5E9"
        border = "#2E8B57"
    elif prob <= 0.7:
        title = "Moderate Risk"
        message = "consult your doctor"
        bg = "#FFF4D6"
        border = "#D18B00"
    else:
        title = "High Risk"
        message = "seek medical attention soon"
        bg = "#FDECEC"
        border = "#B22222"

    return (
        f"<div style='padding: 1rem 1.1rem; border-radius: 16px; background: {bg}; border: 2px solid {border};'>"
        f"<div style='font-size: 1.1rem; font-weight: 700; color: {border};'>{disease}: {title}</div>"
        f"<div style='margin-top: 0.35rem; font-size: 0.98rem; color: #24303f;'>"
        f"{title} — {message}."
        f"</div></div>"
    )


def _feature_label(feature: str, disease: str) -> str:
    diabetes_labels = {
        "Pregnancies": "pregnancy count",
        "Glucose": "glucose level",
        "BloodPressure": "blood pressure",
        "SkinThickness": "skin fold thickness",
        "Insulin": "insulin level",
        "BMI": "BMI",
        "DiabetesPedigreeFunction": "family history of diabetes",
        "Age": "age",
        "BMI_Category": "BMI category",
        "Age_Group": "age group",
    }
    heart_labels = {
        "age": "age",
        "sex": "biological sex",
        "cp": "chest pain type",
        "trestbps": "resting blood pressure",
        "chol": "cholesterol level",
        "fbs": "fasting blood sugar",
        "restecg": "resting ECG result",
        "thalach": "maximum heart rate",
        "exang": "exercise chest pain",
        "oldpeak": "ST depression during exercise",
        "slope": "exercise ST segment slope",
        "ca": "major vessels visible by fluoroscopy",
        "thal": "thalassemia type",
    }

    if disease == "Diabetes":
        if feature.startswith("BMI_Category_"):
            return f"BMI category: {feature.split('_', 2)[-1].replace('_', ' ')}"
        if feature.startswith("Age_Group_"):
            return f"age group: {feature.split('_', 2)[-1].replace('_', ' ')}"
        return diabetes_labels.get(feature, feature.replace("_", " ").lower())

    if feature == "sex_1":
        return "biological sex: male"
    if feature.startswith("cp_"):
        cp_map = {"1": "Typical angina", "2": "Atypical angina", "3": "Non-anginal pain", "4": "Asymptomatic"}
        code = feature.split("_", 1)[1]
        return f"chest pain type: {cp_map.get(code, code)}"
    if feature.startswith("fbs_"):
        return "fasting blood sugar above 120 mg/dL"
    if feature.startswith("restecg_"):
        ecg_map = {"1": "ST-T wave abnormality", "2": "Left ventricular hypertrophy"}
        code = feature.split("_", 1)[1]
        return f"resting ECG: {ecg_map.get(code, code)}"
    if feature.startswith("exang_"):
        return "exercise-induced chest pain: yes"
    if feature.startswith("slope_"):
        slope_map = {"1": "Upsloping", "2": "Flat", "3": "Downsloping"}
        code = feature.split("_", 1)[1]
        return f"exercise ST slope: {slope_map.get(code, code)}"
    if feature.startswith("ca_"):
        code = feature.split("_", 1)[1]
        return f"major vessels visible: {code}"
    if feature.startswith("thal_"):
        thal_map = {"6": "Fixed defect", "7": "Reversible defect"}
        code = feature.split("_", 1)[1]
        return f"thalassemia type: {thal_map.get(code, code)}"
    return heart_labels.get(feature, feature.replace("_", " ").lower())


def _plain_shap_sentence(feature: str, impact: float, disease: str) -> str:
    label = _feature_label(feature, disease)
    if impact >= 0:
        return f"Your {label} is pushing the risk higher."
    return f"Your {label} is pulling the risk lower."


def _load_assessment_context(disease: str) -> dict[str, Any] | None:
    model, meta, dataset = _load_disease_assets(disease)
    if model is None or meta is None or dataset is None:
        return None

    feature_order = list(meta.get("features", []))
    if not feature_order:
        return None

    target_col = DISEASE_CFG[disease]["target"]
    zero_invalid_cols = DISEASE_CFG[disease]["zero_invalid"]
    meta_raw_columns = list(meta.get("raw_columns", []))
    if not meta_raw_columns:
        meta_raw_columns = [c for c in dataset.columns if c != target_col]
    meta_categorical = [c for c in meta.get("categorical_columns", []) if c in meta_raw_columns]
    ref_transform = _fit_reference_transform(dataset, target_col, zero_invalid_cols, categorical_columns=meta_categorical)

    X_model_all, _ = _transform_dataset_to_model_features(
        dataset,
        target_col,
        ref_transform,
        feature_order,
        zero_invalid_cols,
    )

    return {
        "model": model,
        "meta": meta,
        "dataset": dataset,
        "feature_order": feature_order,
        "ref_transform": ref_transform,
        "X_model_all": X_model_all,
        "target_col": target_col,
        "zero_invalid_cols": zero_invalid_cols,
    }


def _build_assessment_result(disease: str, raw_patient: dict[str, Any]) -> dict[str, Any] | None:
    context = _load_assessment_context(disease)
    if context is None:
        return None

    patient_raw_df = pd.DataFrame([raw_patient])
    patient_df = _transform_raw_to_model_features(
        patient_raw_df,
        context["ref_transform"],
        context["feature_order"],
        context["zero_invalid_cols"],
    )

    risk_prob = float(context["model"].predict_proba(patient_df)[0][1])
    background = context["X_model_all"].sample(min(200, len(context["X_model_all"])), random_state=42) if not context["X_model_all"].empty else patient_df
    explainer, shap_values = compute_shap_values(context["model"], background, patient_df)

    shap_array = shap_values.values
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1] if shap_array.shape[2] > 1 else shap_array[:, :, 0]

    row_impact = shap_array[0]
    ranked = np.argsort(np.abs(row_impact))[::-1][:3]
    top_factors = [_plain_shap_sentence(context["feature_order"][idx], float(row_impact[idx]), disease) for idx in ranked]

    report_path = generate_patient_report(
        patient_df.iloc[0].to_dict(),
        context["model"],
        explainer,
        context["feature_order"],
        filename_prefix=f"{disease.lower().replace(' ', '_')}_self_assessment",
    )

    return {
        "probability": risk_prob,
        "band": _assessment_risk_band(risk_prob),
        "top_factors": top_factors,
        "report_path": report_path,
    }


def _render_assessment_result(disease: str, result: dict[str, Any]) -> None:
    st.plotly_chart(_assessment_gauge(float(result["probability"]), disease), use_container_width=True)

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown(_assessment_card(float(result["probability"]), disease), unsafe_allow_html=True)
    with right:
        st.metric("Risk Band", result["band"])
        st.metric("Estimated Risk", f"{float(result['probability']) * 100:.1f}%")

    st.subheader("Top 3 SHAP factors")
    for factor in result["top_factors"]:
        st.write(factor)

    report_path = Path(result["report_path"])
    if report_path.exists():
        st.download_button(
            "Download My Report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="application/pdf",
        )
    else:
        st.info("The PDF report could not be found. Please submit the assessment again.")


def _render_diabetes_self_assessment() -> None:
    st.subheader("Diabetes")
    st.info("These questions use everyday language. If you know your recent clinic numbers, enter those values; if not, use your best estimate.")
    with st.expander("Help guide: what each input means"):
        st.markdown(
            """
            | Input | What it means | Normal / Typical |
            | --- | --- | --- |
            | Age | your age in years | varies by person |
            | Biological sex | used only to decide whether the pregnancy question applies | n/a |
            | Pregnancies | how many times you have been pregnant | 0 if never pregnant |
            | Fasting blood sugar | glucose after fasting for several hours | 70-99 mg/dL |
            | BMI | body mass index from height and weight | 18.5-24.9 |
            | Blood pressure | diastolic pressure at rest (bottom number) | 60-80 mm Hg |
            | Skin fold thickness | fat layer under the skin measured in clinic | 10-50 mm (varies) |
            | Insulin | fasting insulin level from a blood test | 2-25 uU/mL |
            | Family history of diabetes | simple proxy for genetic risk | no family history |
            """
        )

    # Show validation warnings container
    warnings_container = st.container()

    with st.form("diabetes_self_assessment_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input(
                "How old are you?",
                min_value=1,
                max_value=100,
                value=40,
                step=1,
                help="Age in years. No single normal range.",
            )
            st.caption("Normal: age varies (no single normal range).")
            sex = st.selectbox(
                "What is your biological sex?",
                ["Female", "Male"],
                help="Used only to decide whether the pregnancy question applies.",
            )
            pregnancies = 0
            if sex == "Female":
                pregnancies = st.number_input(
                    "How many times have you been pregnant?",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1,
                    help="Use 0 if you have never been pregnant. Normal: 0 if never pregnant.",
                )
                st.caption("Normal: 0 if never pregnant.")
            glucose = st.number_input(
                "What was your last fasting blood sugar reading? (mg/dL)",
                min_value=70,
                max_value=200,
                value=100,
                step=1,
                help="Blood sugar after not eating for several hours. Normal: 70-99 mg/dL.",
            )
            st.caption("Normal fasting: 70-99 mg/dL.")
            bmi = st.number_input(
                "What is your BMI?",
                min_value=10.0,
                max_value=70.0,
                value=25.0,
                step=0.1,
                help="BMI is a weight-to-height ratio used to estimate body fat. Normal: 18.5-24.9.",
            )
            st.caption("Normal: 18.5-24.9.")
            with st.expander("BMI helper: calculate BMI from weight and height"):
                bmi_helper_cols = st.columns(2)
                with bmi_helper_cols[0]:
                    helper_weight = st.number_input("Weight (kg)", min_value=20.0, max_value=350.0, value=70.0, step=0.5)
                with bmi_helper_cols[1]:
                    helper_height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
                helper_bmi = helper_weight / (helper_height ** 2) if helper_height > 0 else 0.0
                st.caption(f"Estimated BMI: {helper_bmi:.1f}")
                st.caption("Formula: weight in kilograms divided by height in meters squared.")

        with right:
            blood_pressure = st.number_input(
                "What is your blood pressure? (mm Hg)",
                min_value=40,
                max_value=140,
                value=80,
                step=1,
                help="Diastolic pressure inside your arteries at rest. Normal: 60-80 mm Hg.",
            )
            st.caption("Normal diastolic: 60-80 mm Hg.")
            skin_thickness = st.number_input(
                "What is your skin fold thickness? (mm) — ask your doctor",
                min_value=0,
                max_value=100,
                value=20,
                step=1,
                help="Fat layer under the skin, usually measured at a clinic. Typical: 10-50 mm (varies).",
            )
            st.caption("Typical: 10-50 mm (varies).")
            insulin = st.number_input(
                "What is your insulin level? (mu U/ml)",
                min_value=0,
                max_value=900,
                value=80,
                step=1,
                help="Insulin helps move sugar from blood into cells. Typical fasting: 2-25 uU/mL.",
            )
            st.caption("Typical fasting: 2-25 uU/mL.")
            diabetes_history = st.selectbox(
                "Do you have a family history of diabetes?",
                ["No", "Somewhat", "Yes"],
                help="This maps to a simple risk score used by the model. No normal range.",
            )
            st.caption("Normal: No family history.")

        submitted = st.form_submit_button("Check My Diabetes Risk")

    # Show validation warnings outside the form
    if submitted:
        validation_issues = []
        if glucose == 0:
            validation_issues.append("• Blood glucose of 0 mg/dL seems medically unlikely. Please double-check your value.")
        if bmi == 0:
            validation_issues.append("• BMI of 0 seems medically unlikely. Please double-check your value.")

        if validation_issues:
            with warnings_container:
                st.warning(
                    "⚠️ **Data Alert**\n\n" + "\n".join(validation_issues) +
                    "\n\nWe still processed your submission, but please review these values."
                )

        raw_patient = {
            "Pregnancies": int(pregnancies),
            "Glucose": float(glucose),
            "BloodPressure": float(blood_pressure),
            "SkinThickness": float(skin_thickness),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": {"No": 0.1, "Somewhat": 0.4, "Yes": 0.8}[diabetes_history],
            "Age": float(age),
        }
        result = _build_assessment_result("Diabetes", raw_patient)
        if result is None:
            st.error("Diabetes model artifacts are not available.")
        else:
            st.session_state.setdefault("self_assessment_results", {})["Diabetes"] = result

    result = st.session_state.get("self_assessment_results", {}).get("Diabetes")
    if result:
        _render_assessment_result("Diabetes", result)


def _render_heart_self_assessment() -> None:
    st.subheader("Heart Disease")
    st.info("These questions ask about common heart-health measurements. If you do not know a value, use your latest clinic result or ask your clinician.")
    with st.expander("Help guide: what each input means"):
        st.markdown(
            """
            | Input | What it means | Normal / Typical |
            | --- | --- | --- |
            | Age | your age in years | varies by person |
            | Biological sex | male or female, coded by the model | n/a |
            | Chest pain type | whether you have chest pain and what type | no chest pain |
            | Resting blood pressure | systolic pressure at rest (top number) | 90-120 mm Hg |
            | Cholesterol | total cholesterol in the blood | under 200 mg/dL |
            | Fasting blood sugar above 120 | whether fasting glucose is over 120 mg/dL | fasting under 100 mg/dL |
            | Resting ECG | electrical activity of the heart at rest | Normal |
            | Max heart rate achieved | highest heart rate during exercise testing | about 220 minus age |
            | Exercise chest pain | chest pain triggered by exercise | No |
            | ST depression (oldpeak) | ECG change during exercise | 0-1.0 |
            | ST segment slope | shape of the ECG line after exercise | upsloping is most common |
            | Major vessels (fluoroscopy) | number of vessels seen on imaging | 0 vessels |
            | Thalassemia type | imaging marker related to blood flow patterns | Normal |
            """
        )

    # Show validation warnings container
    warnings_container = st.container()

    with st.form("heart_self_assessment_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input(
                "How old are you?",
                min_value=18,
                max_value=100,
                value=50,
                step=1,
                help="Age in years. No single normal range.",
            )
            st.caption("Normal: age varies (no single normal range).")
            sex_label = st.selectbox(
                "What is your biological sex?",
                ["Female", "Male"],
                help="The model uses this as a coded input.",
            )
            chest_pain_label = st.selectbox(
                "Do you experience chest pain? If yes, what type?",
                ["No pain", "Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"],
                help="Chest pain is discomfort or pressure in the chest. Normal: no chest pain.",
            )
            st.caption("Normal: no chest pain.")
            trestbps = st.number_input(
                "What is your resting blood pressure? (mm Hg)",
                min_value=80,
                max_value=200,
                value=120,
                step=1,
                help="Resting systolic blood pressure. Normal: 90-120 mm Hg.",
            )
            st.caption("Normal systolic: 90-120 mm Hg.")
            chol = st.number_input(
                "What is your cholesterol level? (mg/dL)",
                min_value=100,
                max_value=600,
                value=200,
                step=1,
                help="Cholesterol is a fat-like substance in your blood. Normal: under 200 mg/dL.",
            )
            st.caption("Normal total cholesterol: under 200 mg/dL.")
            fbs_label = st.selectbox(
                "Is your fasting blood sugar above 120 mg/dL?",
                ["No", "Yes"],
                help="This checks whether your blood sugar is above the normal fasting threshold. Normal fasting: under 100 mg/dL.",
            )
            st.caption("Normal fasting: under 100 mg/dL.")
            restecg_label = st.selectbox(
                "What was your resting ECG result?",
                ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
                help="An ECG records the heart's electrical activity. Normal: Normal.",
            )
            st.caption("Normal: Normal.")
            thalach = st.number_input(
                "What is your maximum heart rate achieved?",
                min_value=60,
                max_value=220,
                value=150,
                step=1,
                help="Fastest heart rate during exercise testing. Typical max about 220 minus age.",
            )
            st.caption("Typical max: about 220 minus age.")
            exang_label = st.selectbox(
                "Do you experience chest pain during exercise?",
                ["No", "Yes"],
                help="Exercise-induced chest pain can be a sign of reduced blood flow to the heart. Normal: No.",
            )
            st.caption("Normal: No.")

        with right:
            oldpeak = st.number_input(
                "ST depression during exercise (oldpeak)",
                min_value=0.0,
                max_value=6.0,
                value=1.0,
                step=0.1,
                help="Ask your cardiologist for this value from your ECG report. Normal: 0-1.0.",
            )
            st.caption("Normal: 0-1.0.")
            st.info("Oldpeak is the amount the ECG pattern changes during exercise. Normal is around 0-1.0; higher values can suggest more strain.")
            slope_label = st.selectbox(
                "Slope of peak exercise ST segment",
                ["Upsloping", "Flat", "Downsloping"],
                help="This describes the shape of the ECG line after exercise. Normal: upsloping is most common.",
            )
            st.caption("Normal: upsloping is most common.")
            ca = st.selectbox(
                "Number of major vessels coloured by fluoroscopy",
                [0, 1, 2, 3],
                help="Fluoroscopy is an imaging test that can show blood vessels around the heart. Normal: 0 vessels.",
            )
            st.caption("Normal: 0 vessels.")
            thal_label = st.selectbox(
                "Thalassemia type",
                ["Normal", "Fixed defect", "Reversible defect"],
                help="Heart-imaging marker that can suggest how blood flows through the heart muscle. Normal: Normal.",
            )
            st.caption("Normal: Normal.")

        submitted = st.form_submit_button("Check My Heart Disease Risk")

    # Show validation warnings outside the form
    if submitted:
        validation_issues = []
        if chol == 0:
            validation_issues.append("• Cholesterol of 0 mg/dL seems medically unlikely. Please double-check your value.")

        if validation_issues:
            with warnings_container:
                st.warning(
                    "⚠️ **Data Alert**\n\n" + "\n".join(validation_issues) +
                    "\n\nWe still processed your submission, but please review these values."
                )

        raw_patient = {
            "age": float(age),
            "sex": 0 if sex_label == "Female" else 1,
            "cp": {"No pain": 0, "Typical angina": 1, "Atypical angina": 2, "Non-anginal": 3, "Asymptomatic": 4}[chest_pain_label],
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": 1 if fbs_label == "Yes" else 0,
            "restecg": {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}[restecg_label],
            "thalach": float(thalach),
            "exang": 1 if exang_label == "Yes" else 0,
            "oldpeak": float(oldpeak),
            "slope": {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope_label],
            "ca": int(ca),
            "thal": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal_label],
        }
        result = _build_assessment_result("Heart Disease", raw_patient)
        if result is None:
            st.error("Heart disease model artifacts are not available.")
        else:
            st.session_state.setdefault("self_assessment_results", {})["Heart Disease"] = result

    result = st.session_state.get("self_assessment_results", {}).get("Heart Disease")
    if result:
        _render_assessment_result("Heart Disease", result)


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

pred_tab, shap_tab, comp_tab, anomaly_tab, self_tab = st.tabs([
    "Prediction",
    "SHAP Explanation",
    "Model Comparison",
    "Anomaly Detection",
    "Patient Self-Assessment",
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

with self_tab:
    st.subheader("Patient Self-Assessment")
    _show_assessment_help()
    diabetes_tab, heart_tab = st.tabs(["Diabetes", "Heart Disease"])

    with diabetes_tab:
        _render_diabetes_self_assessment()

    with heart_tab:
        _render_heart_self_assessment()

    st.markdown("---")
    st.caption(
        "This tool is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional."
    )
