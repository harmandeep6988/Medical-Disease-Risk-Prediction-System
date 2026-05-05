from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise ImportError("xgboost is required for this project. Install it via requirements.txt") from exc


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_model_grids(random_state: int = 42) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=4000, random_state=random_state),
            {
                "C": [0.1, 1.0, 5.0],
                "solver": ["lbfgs"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=random_state),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 8],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 4],
                "learning_rate": [0.03, 0.05],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "min_child_weight": [1, 5],
                "reg_lambda": [1.0, 5.0],
                "gamma": [0, 0.1],
            },
        ),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train and tune all models, save artifacts, and return CV summary."""
    trained_models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for model_name, (model, param_grid) in get_model_grids().items():
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring)

        trained_models[model_name] = best_model
        joblib.dump(best_model, MODEL_DIR / f"{model_name}.joblib")

        rows.append(
            {
                "model": model_name,
                "best_params": grid.best_params_,
                "grid_best_score": grid.best_score_,
                "cv_mean": float(np.mean(cv_scores)),
                "cv_std": float(np.std(cv_scores)),
            }
        )

    cv_summary = pd.DataFrame(rows).sort_values(by="cv_mean", ascending=False).reset_index(drop=True)
    return trained_models, cv_summary
