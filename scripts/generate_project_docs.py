from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from anomaly_detection import detect_anomalies  # noqa: E402
from data_preprocessing import preprocess_data  # noqa: E402
from explainability import explain_model_predictions  # noqa: E402
from feature_engineering import HEART_CATEGORICAL_COLUMNS, build_feature_matrix  # noqa: E402
from model_evaluation import evaluate_and_visualize  # noqa: E402
from model_training import get_model_grids  # noqa: E402
from report_generator import generate_patient_report  # noqa: E402


REPORTS_DIR = ROOT / "reports" / "generated"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DiseaseConfig:
    key: str
    display_name: str
    dataset_path: Path
    target_column: str
    best_model_path: Path
    meta_path: Path


DISEASES: list[DiseaseConfig] = [
    DiseaseConfig(
        key="diabetes",
        display_name="Diabetes",
        dataset_path=ROOT / "data" / "processed" / "diabetes_processed.csv",
        target_column="Outcome",
        best_model_path=ROOT / "models" / "saved_models" / "best_diabetes.joblib",
        meta_path=ROOT / "models" / "saved_models" / "best_diabetes_meta.joblib",
    ),
    DiseaseConfig(
        key="heart",
        display_name="Heart Disease",
        dataset_path=ROOT / "data" / "processed" / "heart_processed.csv",
        target_column="target",
        best_model_path=ROOT / "models" / "saved_models" / "best_heart.joblib",
        meta_path=ROOT / "models" / "saved_models" / "best_heart_meta.joblib",
    ),
]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_rename(src: Path, dst: Path) -> Path:
    if not src.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    return dst


def _package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        from importlib import metadata

        for p in packages:
            try:
                versions[p] = metadata.version(p)
            except Exception:
                versions[p] = "(not found)"
    except Exception:
        for p in packages:
            versions[p] = "(unknown)"
    return versions


def _invalid_zero_cols(source_columns: list[str]) -> list[str]:
    candidates = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "chol", "trestbps", "thalach"]
    return [c for c in candidates if c in source_columns]


def _compute_binary_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


def _class_balance(y: pd.Series) -> dict[str, Any]:
    counts = y.value_counts(dropna=False).to_dict()
    total = float(len(y)) if len(y) else 0.0
    pos_rate = float((y == 1).mean()) if len(y) else 0.0
    return {"counts": {str(k): int(v) for k, v in counts.items()}, "positive_rate": pos_rate, "n": int(total)}


def evaluate_saved_best_model(cfg: DiseaseConfig) -> dict[str, Any]:
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {cfg.dataset_path}")
    if not cfg.best_model_path.exists() or not cfg.meta_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts for {cfg.display_name}. Run scripts/run_all.py first."
        )

    df = pd.read_csv(cfg.dataset_path)
    source_columns = list(df.columns)

    categorical_cols = [c for c in HEART_CATEGORICAL_COLUMNS if c in source_columns]
    invalid_zero_cols = _invalid_zero_cols(source_columns)

    X_train, X_test, y_train, y_test, prep_artifacts = preprocess_data(
        path=str(cfg.dataset_path),
        target_column=cfg.target_column,
        medical_zero_invalid_cols=invalid_zero_cols,
        categorical_passthrough_cols=categorical_cols,
    )

    meta: dict[str, Any] = joblib.load(cfg.meta_path)
    feature_order: list[str] = list(meta.get("features", []))

    X_train_encoded = build_feature_matrix(X_train, categorical_columns=categorical_cols)
    X_test_encoded = build_feature_matrix(X_test, categorical_columns=categorical_cols)

    X_train_model = X_train_encoded.reindex(columns=feature_order, fill_value=0.0)
    X_test_model = X_test_encoded.reindex(columns=feature_order, fill_value=0.0)

    model = joblib.load(cfg.best_model_path)

    y_pred = model.predict(X_test_model)
    y_proba = model.predict_proba(X_test_model)[:, 1]

    metrics = _compute_binary_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Generate evaluation plots (ROC, confusion matrix, comparison chart) using existing helper,
    # then rename to avoid overwriting between diseases.
    _, _, eval_artifacts = evaluate_and_visualize({"best_model": model}, X_test_model, y_test)
    eval_files = {
        k: _safe_rename(Path(v), REPORTS_DIR / f"{Path(v).stem}_{cfg.key}{Path(v).suffix}")
        for k, v in eval_artifacts.items()
    }

    # Generate SHAP plots + a short explanation, then rename files.
    explanation, shap_files, explainer, _ = explain_model_predictions(
        model,
        X_train_model.sample(min(200, len(X_train_model)), random_state=42),
        X_test_model.head(50),
    )
    shap_files_renamed = {
        k: _safe_rename(Path(v), REPORTS_DIR / f"{Path(v).stem}_{cfg.key}{Path(v).suffix}")
        for k, v in shap_files.items()
    }

    # Anomaly detection (on model-feature space for consistent scaling/encoding).
    _, anomaly_scores, anomaly_reasons, anomaly_plot = detect_anomalies(
        X_train_model,
        contamination=0.05,
        random_state=42,
        verbose=False,
    )
    anomaly_plot_renamed = _safe_rename(
        Path(anomaly_plot), REPORTS_DIR / f"{Path(anomaly_plot).stem}_{cfg.key}{Path(anomaly_plot).suffix}"
    )

    # Generate one sample PDF report from the first test record.
    patient_data = X_test_model.iloc[0].to_dict()
    pdf_path = generate_patient_report(
        patient_data=patient_data,
        model=model,
        explainer=explainer,
        feature_order=feature_order,
        filename_prefix=f"{cfg.key}_patient_report",
    )

    return {
        "disease": cfg.display_name,
        "key": cfg.key,
        "dataset": {
            "path": str(cfg.dataset_path.relative_to(ROOT)),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "target": cfg.target_column,
            "class_balance_test": _class_balance(y_test),
            "class_balance_train_after_smote": _class_balance(y_train),
        },
        "preprocessing": {
            "test_size": 0.2,
            "random_state": 42,
            "missing_strategy": "median",
            "zero_invalid_columns": invalid_zero_cols,
            "outlier_clip": "IQR 1.5x",
            "scaling": "StandardScaler -> MinMaxScaler (categoricals passthrough)",
            "categorical_passthrough_columns": categorical_cols,
            "balanced_with": "SMOTENC" if categorical_cols else "SMOTE",
            "scaled_columns": getattr(prep_artifacts, "scaled_columns", []),
        },
        "model": {
            "artifact": str(cfg.best_model_path.relative_to(ROOT)),
            "meta": str(cfg.meta_path.relative_to(ROOT)),
            "type": model.__class__.__name__,
            "n_features": int(len(feature_order)),
            "params": {
                # Keep this to a manageable subset in markdown; full params are still saved in JSON.
                k: v
                for k, v in model.get_params().items()
                if k
                in {
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "min_samples_split",
                    "min_samples_leaf",
                    "C",
                    "penalty",
                    "solver",
                    "random_state",
                }
            },
        },
        "metrics": metrics,
        "confusion_matrix": cm,
        "explainability": {
            "explanation_text": explanation,
            "shap_files": {k: str(Path(v).relative_to(ROOT)) for k, v in shap_files_renamed.items()},
        },
        "evaluation_artifacts": {k: str(Path(v).relative_to(ROOT)) for k, v in eval_files.items()},
        "anomaly_detection": {
            "plot": str(anomaly_plot_renamed.relative_to(ROOT)),
            "anomaly_rate_train": float((anomaly_scores < 0).mean()) if len(anomaly_scores) else 0.0,
            "top_reason_examples": anomaly_reasons.head(5).to_dict(orient="records"),
        },
        "pdf_report": str(Path(pdf_path).relative_to(ROOT)),
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_float(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def write_project_run_report(results: list[dict[str, Any]], env: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Project Run Report — Medical Disease Risk Prediction System")
    lines.append("")
    lines.append(f"Generated: {env['generated_at']}")
    lines.append("")
    lines.append("## What Was Run")
    lines.append("- Data prep: `scripts/prepare_datasets.py` (OpenML/URL fallback)")
    lines.append("- Training + evaluation: `scripts/train_and_evaluate.py` (grid search + ROC/AUC ranking)")
    lines.append("- This report: `scripts/generate_project_docs.py` (evaluates saved best models + generates artifacts)")
    lines.append("")

    lines.append("## Environment")
    lines.append(f"- OS: `{env['os']}`")
    lines.append(f"- Python: `{env['python']}`")
    lines.append("- Key packages:")
    for k, v in env["packages"].items():
        lines.append(f"  - `{k}=={v}`")
    lines.append("")

    lines.append("## Results (Best Saved Models)")
    lines.append("| Disease | AUC | Accuracy | F1 | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['disease']} | {_fmt_float(m['auc'])} | {_fmt_pct(m['accuracy'])} | {_fmt_float(m['f1'])} | {_fmt_float(m['precision'])} | {_fmt_float(m['recall'])} |"
        )
    lines.append("")

    lines.append("## Parameters (Pipeline)")
    lines.append("- Split: stratified `test_size=0.2`, `random_state=42`")
    lines.append("- Missing values: median imputation (numeric)")
    lines.append("- Medical 0 handling: selected columns -> `NaN` then imputed")
    lines.append("- Outliers: IQR clipping (1.5×)")
    lines.append("- Scaling: StandardScaler -> MinMaxScaler (categoricals passthrough)")
    lines.append("- Imbalance: SMOTE or SMOTENC depending on categorical columns")
    lines.append("- Feature engineering: BMI/age bucketing + interactions + one-hot encoding")
    lines.append("- Training: `GridSearchCV` with 5-fold CV, scoring = `roc_auc`, `n_jobs=-1`")
    lines.append("")

    for r in results:
        lines.append(f"## {r['disease']}")
        lines.append("- Dataset:")
        lines.append(f"  - Path: `{r['dataset']['path']}`")
        lines.append(f"  - Shape: `{r['dataset']['shape'][0]} rows × {r['dataset']['shape'][1]} cols`")
        lines.append(f"  - Target: `{r['dataset']['target']}`")
        lines.append(f"  - Test positive rate: `{_fmt_pct(r['dataset']['class_balance_test']['positive_rate'])}`")
        lines.append("- Preprocessing:")
        lines.append(
            f"  - Zero-invalid columns: `{', '.join(r['preprocessing']['zero_invalid_columns']) if r['preprocessing']['zero_invalid_columns'] else '(none)'}`"
        )
        lines.append(
            f"  - Categorical passthrough: `{', '.join(r['preprocessing']['categorical_passthrough_columns']) if r['preprocessing']['categorical_passthrough_columns'] else '(none)'}`"
        )
        lines.append(f"  - Balancing: `{r['preprocessing']['balanced_with']}`")
        lines.append("- Best saved model:")
        lines.append(f"  - Type: `{r['model']['type']}`")
        lines.append(f"  - Artifact: `{r['model']['artifact']}`")
        lines.append(f"  - Features used: `{r['model']['n_features']}`")
        lines.append("  - Key hyperparameters:")
        if r["model"]["params"]:
            for k, v in r["model"]["params"].items():
                lines.append(f"    - `{k}`: `{v}`")
        else:
            lines.append("    - (no concise subset available)")
        lines.append("- Explainability:")
        lines.append(f"  - Example explanation: {r['explainability']['explanation_text']}")
        lines.append("  - SHAP artifacts:")
        for k, v in r["explainability"]["shap_files"].items():
            lines.append(f"    - `{k}`: `{v}`")
        lines.append("- Evaluation artifacts:")
        for k, v in r["evaluation_artifacts"].items():
            lines.append(f"  - `{k}`: `{v}`")
        lines.append(f"- Anomaly detection plot: `{r['anomaly_detection']['plot']}`")
        lines.append(f"- Sample PDF report: `{r['pdf_report']}`")
        lines.append("")

    lines.append("## Notes / Gaps")
    lines.append("- This repo currently runs **Diabetes** + **Heart Disease**. Liver disease + FastAPI endpoints are not present in the codebase yet.")
    lines.append("- Plot filenames in the training script are generic (risk of overwrite). This script renames per-disease artifacts to preserve both.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_workflow_doc(out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Project Workflow — Medical Disease Risk Prediction System")
    lines.append("")
    lines.append("This document describes the end-to-end workflow, which modules implement each step, and what to upgrade next.")
    lines.append("")

    lines.append("## End-to-End Flow")
    lines.append("1. **Dataset acquisition** — `scripts/prepare_datasets.py`")
    lines.append("   - Fetches Diabetes and Heart datasets (OpenML first; URL fallback for heart).")
    lines.append("   - Writes to `data/raw/` and `data/processed/`.")
    lines.append("2. **Preprocessing** — `src/data_preprocessing.py`")
    lines.append("   - Zero-invalid columns -> NaN, median imputation, IQR outlier clipping.")
    lines.append("   - Stratified split; SMOTE/SMOTENC for imbalance.")
    lines.append("   - Scaling: StandardScaler then MinMaxScaler (categoricals passthrough).")
    lines.append("3. **Feature engineering** — `src/feature_engineering.py`")
    lines.append("   - Derived features (BMI categories, age groups, interactions).")
    lines.append("   - One-hot encoding + correlation pruning + SelectKBest.")
    lines.append("4. **Model training** — `src/model_training.py`")
    lines.append("   - GridSearchCV over LogisticRegression, RandomForest, GradientBoosting, XGBoost.")
    lines.append("   - CV scoring uses ROC-AUC.")
    lines.append("5. **Evaluation + artifacts** — `src/model_evaluation.py`")
    lines.append("   - Accuracy/Precision/Recall/F1/AUC + ROC + confusion matrix + comparison chart.")
    lines.append("6. **Explainability** — `src/explainability.py`")
    lines.append("   - SHAP plots (beeswarm, waterfall, global bar) and human-readable explanation.")
    lines.append("7. **Anomaly detection** — `src/anomaly_detection.py`")
    lines.append("   - IsolationForest + PCA scatter for outlier profiles.")
    lines.append("8. **UI dashboard** — `dashboard/app.py`")
    lines.append("   - Streamlit app for inference, SHAP, anomaly view, and PDF report generation.")
    lines.append("9. **PDF clinical report** — `src/report_generator.py`")
    lines.append("   - Creates a patient-facing PDF with risk band, top SHAP factors, and next steps.")
    lines.append("")

    lines.append("## Tech Used (What + How)")
    lines.append("- **Pandas/NumPy**: dataset IO + vectorized feature work")
    lines.append("- **Scikit-learn**: preprocessing, SMOTE/SMOTENC integration, GridSearchCV, evaluation metrics")
    lines.append("- **XGBoost / GradientBoosting / RandomForest / LogisticRegression**: candidate classifiers")
    lines.append("- **SHAP**: local + global feature attribution plots")
    lines.append("- **IsolationForest**: unsupervised outlier detection for reliability")
    lines.append("- **Matplotlib/Seaborn**: static evaluation plots")
    lines.append("- **Streamlit**: interactive dashboard for inference + report download")
    lines.append("- **Docker/Docker Compose**: containerized dashboard/training/pipeline/notebook workflows")
    lines.append("- **GitHub Actions**: CI test run on Python 3.12 (`.github/workflows/test.yml`)")
    lines.append("")

    lines.append("## Upgrade Checklist (Recommended)")
    lines.append("- **Add Liver Disease module**: dataset, preprocessing config, model training/eval, Streamlit tab")
    lines.append("- **Add FastAPI service** (currently missing): `api/` package, Pydantic schemas, `/predict/{disease}` endpoints")
    lines.append("- **Prevent artifact overwrites**: prefix plot/model filenames per disease in training scripts")
    lines.append("- **Calibration & thresholds**: reliability curves + threshold selection for clinical use")
    lines.append("- **Data validation**: schema checks, value ranges, and missingness alerts")
    lines.append("- **Model registry/versioning**: track dataset hash + params + metrics per model build")
    lines.append("- **Monitoring**: drift detection + periodic retraining triggers")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()

    env = {
        "generated_at": _now_stamp(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.replace("\n", " "),
        "packages": _package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "shap",
                "matplotlib",
                "seaborn",
                "imbalanced-learn",
                "streamlit",
                "fpdf2",
            ]
        ),
        "param_grids": {
            name: grid for name, (_, grid) in get_model_grids().items()
        },
    }

    results: list[dict[str, Any]] = []
    for cfg in DISEASES:
        results.append(evaluate_saved_best_model(cfg))

    run_report_path = ROOT / "PROJECT_RUN_REPORT.md"
    workflow_path = ROOT / "PROJECT_WORKFLOW.md"

    write_project_run_report(results, env, run_report_path)
    write_workflow_doc(workflow_path)

    out_json = {
        "env": env,
        "results": results,
        "runtime_seconds": time.time() - start,
    }
    (REPORTS_DIR / "project_run_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    print(f"Wrote: {run_report_path}")
    print(f"Wrote: {workflow_path}")
    print(f"Wrote: {REPORTS_DIR / 'project_run_summary.json'}")


if __name__ == "__main__":
    main()
