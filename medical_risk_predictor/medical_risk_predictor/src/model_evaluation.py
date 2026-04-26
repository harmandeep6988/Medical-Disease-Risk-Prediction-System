from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


PLOTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "generated"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_models(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str]:
    """Compute metrics for each model and return ranking by AUC and F1."""
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "auc_roc": roc_auc,
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(by=["auc_roc", "f1_score"], ascending=False).reset_index(drop=True)
    best_model_name = summary.iloc[0]["model"]
    return summary, best_model_name


def plot_roc_curves(models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Path:
    plt.figure(figsize=(9, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    path = PLOTS_DIR / "roc_curves.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_confusion_for_best_model(
    best_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    path = PLOTS_DIR / "best_model_confusion_matrix.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_model_comparison(summary: pd.DataFrame) -> Path:
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    plot_df = summary[["model"] + metrics].set_index("model")

    ax = plot_df.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Model Comparison Across Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()

    path = PLOTS_DIR / "model_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def evaluate_and_visualize(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str, dict[str, Path]]:
    summary, best_model_name = evaluate_models(models, X_test, y_test)
    artifacts = {
        "roc_curves": plot_roc_curves(models, X_test, y_test),
        "confusion_matrix": plot_confusion_for_best_model(models[best_model_name], X_test, y_test),
        "comparison_chart": plot_model_comparison(summary),
    }

    print(summary)
    return summary, best_model_name, artifacts
