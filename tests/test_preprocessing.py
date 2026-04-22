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

from data_preprocessing import iqr_outlier_clip, replace_zero_with_nan, scale_features


class TestPreprocessing(unittest.TestCase):
    def test_replace_zero_with_nan(self) -> None:
        df = pd.DataFrame({"Glucose": [0, 100], "BMI": [0.0, 22.5], "Outcome": [1, 0]})
        out = replace_zero_with_nan(df, ["Glucose", "BMI"])
        self.assertEqual(int(out["Glucose"].isna().sum()), 1)
        self.assertEqual(int(out["BMI"].isna().sum()), 1)

    def test_iqr_clip_reduces_extreme_values(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 2, 3, 1000], "Outcome": [0, 0, 1, 1, 1]})
        clipped = iqr_outlier_clip(df, exclude_columns=["Outcome"])
        self.assertLess(float(clipped["x"].max()), 1000)

    def test_scaling_range(self) -> None:
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        X_test = pd.DataFrame({"a": [1.5, 2.5], "b": [15.0, 25.0]})

        scaled_train, scaled_test, _ = scale_features(X_train, X_test)

        self.assertTrue(np.all(scaled_train.min(axis=0) >= 0))
        self.assertTrue(np.all(scaled_train.max(axis=0) <= 1))
        self.assertEqual(list(scaled_train.columns), ["a", "b"])
        self.assertEqual(list(scaled_test.columns), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
