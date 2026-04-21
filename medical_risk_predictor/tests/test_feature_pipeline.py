from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from feature_engineering import engineer_features, transform_features_for_inference


def _sample_data(n: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 8, n),
            "Glucose": rng.normal(120, 20, n).clip(60, 220),
            "BloodPressure": rng.normal(72, 10, n).clip(40, 120),
            "SkinThickness": rng.normal(25, 8, n).clip(5, 60),
            "Insulin": rng.normal(80, 40, n).clip(10, 300),
            "BMI": rng.normal(30, 6, n).clip(16, 50),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 1.5, n),
            "Age": rng.integers(20, 75, n),
        }
    )
    y = ((X["Glucose"] + X["BMI"] * 2 + X["Age"] * 0.4) > 190).astype(int)
    return X, y


class TestFeaturePipeline(unittest.TestCase):
    def test_train_test_feature_transform_consistency(self) -> None:
        X, y = _sample_data()
        X_train, X_test = X.iloc[:60], X.iloc[60:]
        y_train = y.iloc[:60]

        X_train_eng, _, _, selector, dropped_cols, reduced_cols = engineer_features(X_train, y_train, k_best=8)
        X_test_eng = transform_features_for_inference(X_test, dropped_cols, reduced_cols, selector)

        self.assertEqual(list(X_train_eng.columns), list(X_test_eng.columns))
        self.assertEqual(X_train_eng.shape[1], X_test_eng.shape[1])
        self.assertEqual(int(X_test_eng.isna().sum().sum()), 0)

    def test_engineering_outputs_non_empty(self) -> None:
        X, y = _sample_data()
        X_eng, importance, corr, _, _, _ = engineer_features(X, y, k_best=6)

        self.assertFalse(X_eng.empty)
        self.assertFalse(importance.empty)
        self.assertFalse(corr.empty)


if __name__ == "__main__":
    unittest.main()
