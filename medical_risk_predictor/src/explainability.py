from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


SHAP_DIR = Path(__file__).resolve().parents[1] / "reports" / "generated"
SHAP_DIR.mkdir(parents=True, exist_ok=True)


def _get_explainer(model: Any, background_data: pd.DataFrame) -> shap.Explainer:
    if hasattr(model, "estimators_") or model.__class__.__name__.startswith(("XGB", "RandomForest", "GradientBoosting")):
        return shap.TreeExplainer(model)
    return shap.Explainer(model, background_data)


def compute_shap_values(model: Any, X_background: pd.DataFrame, X_samples: pd.DataFrame):
    explainer = _get_explainer(model, X_background)
    shap_values = explainer(X_samples)
    return explainer, shap_values


def _select_binary_class_explanation(shap_values, class_index: int = 1):
    """Reduce multi-output explanations to a single class for binary-risk plots."""
    values = shap_values.values
    if values.ndim == 3:
        cls = class_index if values.shape[2] > class_index else 0
        return shap.Explanation(
            values=values[:, :, cls],
            base_values=shap_values.base_values[:, cls] if np.array(shap_values.base_values).ndim > 1 else shap_values.base_values,
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    return shap_values


def generate_shap_summary_plot(shap_values, X_samples: pd.DataFrame) -> Path:
    shap_values = _select_binary_class_explanation(shap_values)
    shap.plots.beeswarm(shap_values, show=False)
    path = SHAP_DIR / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def generate_shap_waterfall_plot(shap_values, patient_index: int = 0) -> Path:
    shap_values = _select_binary_class_explanation(shap_values)
    shap.plots.waterfall(shap_values[patient_index], show=False)
    path = SHAP_DIR / "shap_waterfall_patient.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def generate_shap_bar_plot(shap_values) -> Path:
    shap_values = _select_binary_class_explanation(shap_values)
    shap.plots.bar(shap_values, show=False)
    path = SHAP_DIR / "shap_global_bar.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def build_human_readable_explanation(
    shap_values,
    patient_features: pd.Series,
    top_n: int = 3,
) -> str:
    shap_values = _select_binary_class_explanation(shap_values)
    contributions = shap_values.values
    if contributions.ndim > 1:
        contributions = contributions[0]

    ranked_idx = np.argsort(np.abs(contributions))[::-1][:top_n]
    lines = []

    for idx in ranked_idx:
        feature = shap_values.feature_names[idx]
        val = float(contributions[idx])
        direction = "up" if val >= 0 else "down"
        pct = abs(val) * 100
        lines.append(f"{feature} ({direction} risk by {pct:.1f}%)")

    patient_snapshot = ", ".join([f"{k}={v}" for k, v in patient_features.to_dict().items()][:5])
    return f"Top risk factors for this patient: {', '.join(lines)}. Snapshot: {patient_snapshot}."


def explain_model_predictions(
    model: Any,
    X_background: pd.DataFrame,
    X_samples: pd.DataFrame,
    patient_index: int = 0,
) -> tuple[str, dict[str, Path], Any, Any]:
    explainer, shap_values = compute_shap_values(model, X_background, X_samples)
    files = {
        "beeswarm": generate_shap_summary_plot(shap_values, X_samples),
        "waterfall": generate_shap_waterfall_plot(shap_values, patient_index=patient_index),
        "bar": generate_shap_bar_plot(shap_values),
    }
    explanation = build_human_readable_explanation(
        shap_values[patient_index],
        X_samples.iloc[patient_index],
    )
    return explanation, files, explainer, shap_values
