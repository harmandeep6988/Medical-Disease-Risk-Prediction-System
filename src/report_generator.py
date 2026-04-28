from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fpdf import FPDF


REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "generated"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _risk_band(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"


def _next_steps(risk_band: str) -> list[str]:
    if risk_band == "Low":
        return [
            "Maintain current lifestyle and annual health screening.",
            "Keep balanced diet and regular physical activity.",
        ]
    if risk_band == "Medium":
        return [
            "Schedule clinician consultation within 4 weeks.",
            "Increase physical activity to at least 150 min/week.",
            "Track glucose and blood pressure trends weekly.",
        ]
    return [
        "Consult physician as soon as possible for full assessment.",
        "Begin supervised nutrition and activity intervention plan.",
        "Monitor key vitals frequently and review medications.",
    ]


def _estimate_savings(probability: float, baseline_cost: float = 5000.0) -> float:
    # Simple heuristic for communication purposes.
    return baseline_cost * min(max(probability, 0.0), 1.0) * 0.4


def generate_patient_report(
    patient_data: dict[str, Any],
    model: Any,
    explainer: Any,
    feature_order: list[str],
    filename_prefix: str = "patient_risk_report",
) -> Path:
    patient_df = pd.DataFrame([patient_data]).reindex(columns=feature_order, fill_value=0)

    pred_class = int(model.predict(patient_df)[0])
    risk_probability = float(model.predict_proba(patient_df)[0][1])

    shap_values = explainer(patient_df)
    values = shap_values.values[0]
    feature_names = shap_values.feature_names

    top_idx = np.argsort(np.abs(values))[::-1][:3]
    top_features = [(feature_names[i], float(values[i])) for i in top_idx]

    risk_band = _risk_band(risk_probability)
    est_savings = _estimate_savings(risk_probability)

    pdf = FPDF()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_top_margin(15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    line_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Medical Disease Risk Report", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Patient Profile", ln=True)
    pdf.set_font("Arial", "", 11)
    for key, val in patient_data.items():
        safe_key = str(key).replace("_", " ")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(line_w, 7, f"- {safe_key}: {val}")

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Risk Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Predicted Class: {pred_class}", ln=True)
    pdf.cell(0, 7, f"Risk Probability: {risk_probability * 100:.2f}% ({risk_band})", ln=True)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Top Contributing Risk Factors", ln=True)
    pdf.set_font("Arial", "", 11)
    for feature, contribution in top_features:
        safe_feature = str(feature).replace("_", " ")
        direction = "increases" if contribution >= 0 else "reduces"
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(line_w, 7, f"- {safe_feature}: {direction} risk (SHAP={contribution:.4f})")

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Recommended Next Steps", ln=True)
    pdf.set_font("Arial", "", 11)
    for step in _next_steps(risk_band):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(line_w, 7, f"- {step}")

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Cost-Benefit Insight", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(line_w, 7, f"Early intervention saves approx ${est_savings:,.0f} in avoidable long-term care costs.")

    out_path = REPORT_DIR / f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(str(out_path))
    return out_path
