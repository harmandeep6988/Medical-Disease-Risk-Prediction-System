from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from data_preprocessing import preprocess_data
from feature_engineering import build_feature_matrix

REPORTS_DIR = ROOT / "reports" / "generated"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _select_validation_param(model: Any) -> tuple[str | None, list[Any]]:
    if hasattr(model, "n_estimators"):
        return "n_estimators", [50, 100, 200, 300]
    if hasattr(model, "C"):
        return "C", [0.01, 0.1, 1.0, 10.0]
    if hasattr(model, "max_depth"):
        return "max_depth", [2, 3, 5, 7]
    return None, []


def _plot_overfitting_curves(
    disease: str,
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    param_name: str | None,
    param_range: list[Any],
    vtrain_scores: np.ndarray | None,
    vtest_scores: np.ndarray | None,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_sizes, train_scores, marker="o", label="Train")
    axes[0].plot(train_sizes, test_scores, marker="o", label="Test")
    axes[0].set_title("Learning Curve (Accuracy)")
    axes[0].set_xlabel("Training Set Size")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="best")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    if param_name and vtrain_scores is not None and vtest_scores is not None:
        axes[1].plot(param_range, vtrain_scores, marker="o", label="Train")
        axes[1].plot(param_range, vtest_scores, marker="o", label="Test")
        axes[1].set_title("Validation Curve (Accuracy)")
        axes[1].set_xlabel(param_name)
        axes[1].set_ylabel("Accuracy")
        axes[1].legend(loc="best")
        axes[1].grid(True, linestyle="--", alpha=0.4)
    else:
        axes[1].text(0.5, 0.5, "Validation curve not available", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    out_path = REPORTS_DIR / f"overfitting_analysis_{disease}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def analyze_overfitting_from_features(
    disease: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> dict[str, Any]:
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    train_auc = float(roc_auc_score(y_train, train_proba))
    test_auc = float(roc_auc_score(y_test, test_proba))

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    cv_auc_mean = float(np.mean(cv_scores))
    cv_auc_std = float(np.std(cv_scores))

    gap = train_auc - test_auc
    overfitting = gap > 0.05

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=random_state,
    )

    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    param_name, param_range = _select_validation_param(model)
    vtrain_scores_mean = None
    vtest_scores_mean = None

    if param_name:
        vtrain_scores, vtest_scores = validation_curve(
            model,
            X_train,
            y_train,
            param_name=param_name,
            param_range=param_range,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        vtrain_scores_mean = vtrain_scores.mean(axis=1)
        vtest_scores_mean = vtest_scores.mean(axis=1)

    plot_path = _plot_overfitting_curves(
        disease,
        train_sizes,
        train_scores_mean,
        test_scores_mean,
        param_name,
        param_range,
        vtrain_scores_mean,
        vtest_scores_mean,
    )

    status_text = "⚠️ Overfitting Detected" if overfitting else "✓ No Overfitting"
    print(f"Overfitting Status: {status_text}")
    print(f"Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
    print(f"AUC Gap: {gap:.4f}")
    print(f"Saved plot: {plot_path}")

    return {
        "disease": disease,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "gap": gap,
        "overfitting": overfitting,
        "status": status_text,
        "plot": str(plot_path),
    }


def _prepare_features_for_analysis(
    dataset_path: Path,
    target_column: str,
    meta: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    source_columns = list(pd.read_csv(dataset_path, nrows=2).columns)

    categorical_cols = meta.get("categorical_columns", [])
    invalid_zero_cols = meta.get("invalid_zero_columns", [])

    X_train, X_test, y_train, y_test, _ = preprocess_data(
        path=str(dataset_path),
        target_column=target_column,
        medical_zero_invalid_cols=invalid_zero_cols,
        categorical_passthrough_cols=[c for c in categorical_cols if c in source_columns],
    )

    X_train_eng = build_feature_matrix(X_train, categorical_columns=categorical_cols)
    X_test_eng = build_feature_matrix(X_test, categorical_columns=categorical_cols)

    target_features = meta.get("features", [])
    X_train_eng = X_train_eng.reindex(columns=target_features, fill_value=0)
    X_test_eng = X_test_eng.reindex(columns=target_features, fill_value=0)

    return X_train_eng, y_train, X_test_eng, y_test


def _run_for_disease(disease: str, dataset_path: Path, target_column: str) -> dict[str, Any]:
    model_path = ROOT / "models" / "saved_models" / f"best_{disease}.joblib"
    meta_path = ROOT / "models" / "saved_models" / f"best_{disease}_meta.joblib"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing model artifacts for {disease}. Train the model first.")

    model = joblib.load(model_path)
    meta = joblib.load(meta_path)

    X_train, y_train, X_test, y_test = _prepare_features_for_analysis(
        dataset_path,
        target_column=target_column,
        meta=meta,
    )

    return analyze_overfitting_from_features(
        disease=disease,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect overfitting using learning and validation curves.")
    parser.add_argument(
        "--diabetes-path",
        default=str(ROOT / "data" / "raw" / "diabetes_merged_10k.csv"),
        help="Path to diabetes merged dataset.",
    )
    parser.add_argument(
        "--heart-path",
        default=str(ROOT / "data" / "raw" / "heart_merged_10k.csv"),
        help="Path to heart merged dataset.",
    )
    args = parser.parse_args()

    diabetes_path = Path(args.diabetes_path)
    heart_path = Path(args.heart_path)

    results = []

    if diabetes_path.exists():
        results.append(_run_for_disease("diabetes", diabetes_path, "Outcome"))
    else:
        print(f"Missing diabetes dataset: {diabetes_path}")

    if heart_path.exists():
        results.append(_run_for_disease("heart", heart_path, "target"))
    else:
        print(f"Missing heart dataset: {heart_path}")

    summary_path = REPORTS_DIR / "overfitting_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
