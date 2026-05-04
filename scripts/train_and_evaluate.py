from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from data_preprocessing import preprocess_data
from explainability import explain_model_predictions
from feature_engineering import HEART_CATEGORICAL_COLUMNS, engineer_features, transform_features_for_inference
from model_evaluation import evaluate_and_visualize
from model_training import train_models


def _rename_artifacts(model_prefix: str, files: Mapping[str, Path]) -> dict[str, Path]:
    """Rename generated artifact files to include disease prefix to avoid overwrites."""
    out: dict[str, Path] = {}
    for key, src in files.items():
        src = Path(src)
        dst = src.with_name(f"{src.stem}_{model_prefix}{src.suffix}")
        if not src.exists():
            out[key] = dst
            continue
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        out[key] = dst
    return out


def run_pipeline(dataset_path: Path, target_column: str, model_prefix: str) -> None:
    print(f"Running pipeline for: {dataset_path.name}")

    source_columns = list(pd.read_csv(dataset_path, nrows=2).columns)
    categorical_cols = [c for c in HEART_CATEGORICAL_COLUMNS if c in source_columns]

    invalid_zero_cols = [
        c
        for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "chol", "trestbps", "thalach"]
        if c in source_columns
    ]

    X_train, X_test, y_train, y_test, _ = preprocess_data(
        path=str(dataset_path),
        target_column=target_column,
        medical_zero_invalid_cols=invalid_zero_cols,
        categorical_passthrough_cols=categorical_cols,
    )

    X_train_eng, importance, _, selector, dropped_columns, reduced_columns = engineer_features(
        X_train,
        y_train,
        k_best=min(10, X_train.shape[1]),
        categorical_columns=categorical_cols,
    )
    X_test_eng = transform_features_for_inference(
        X_test,
        dropped_columns,
        reduced_columns,
        selector,
        categorical_columns=categorical_cols,
    )

    models, cv_summary = train_models(X_train_eng, y_train)
    eval_summary, best_model_name, artifacts = evaluate_and_visualize(models, X_test_eng, y_test)
    artifacts = _rename_artifacts(model_prefix, artifacts)

    best_model = models[best_model_name]
    explanation, shap_files, explainer, _ = explain_model_predictions(
        best_model,
        X_train_eng.sample(min(200, len(X_train_eng)), random_state=42),
        X_test_eng.head(50),
    )
    shap_files = _rename_artifacts(model_prefix, shap_files)

    # Save disease-specific best model and metadata.
    out_dir = ROOT / "models" / "saved_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_dir / f"best_{model_prefix}.joblib")
    joblib.dump(
        {
            "features": X_train_eng.columns.tolist(),
            "target": target_column,
            "raw_columns": [c for c in source_columns if c != target_column],
            "categorical_columns": categorical_cols,
            "invalid_zero_columns": invalid_zero_cols,
        },
        out_dir / f"best_{model_prefix}_meta.joblib",
    )

    print("CV Summary:")
    print(cv_summary)
    print("Evaluation Summary:")
    print(eval_summary)
    print(f"Best model: {best_model_name}")
    print(f"Explainability text: {explanation}")
    print(f"Evaluation artifact files: {artifacts}")
    print(f"SHAP artifact files: {shap_files}")


def main() -> None:
    diabetes_path = ROOT / "data" / "raw" / "diabetes_merged.csv"
    heart_path = ROOT / "data" / "raw" / "heart_merged.csv"

    if not diabetes_path.exists() or not heart_path.exists():
        raise FileNotFoundError("Merged datasets missing. Run scripts/fetch_and_merge_datasets.py first.")

    run_pipeline(diabetes_path, target_column="Outcome", model_prefix="diabetes")
    run_pipeline(heart_path, target_column="target", model_prefix="heart")


if __name__ == "__main__":
    main()
