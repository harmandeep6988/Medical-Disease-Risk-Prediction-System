from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from fetch_and_merge_datasets import (  # type: ignore
    DIABETES_COLUMNS,
    HEART_COLUMNS,
    _merge_frames,
    _normalize_diabetes_frame,
    _normalize_heart_frame,
)


class TestDatasetMerge(unittest.TestCase):
    def test_diabetes_normalization(self) -> None:
        df = pd.DataFrame(
            {
                "preg": [1, 2],
                "plas": [120, 140],
                "pres": [80, 82],
                "skin": [20, 21],
                "insu": [100, 110],
                "mass": [28.5, 31.0],
                "pedi": [0.5, 0.6],
                "age": [35, 42],
                "target": ["tested_positive", "tested_negative"],
            }
        )

        normalized = _normalize_diabetes_frame(df)

        self.assertEqual(list(normalized.columns), DIABETES_COLUMNS)
        self.assertEqual(normalized["Outcome"].tolist(), [1, 0])

    def test_heart_normalization(self) -> None:
        df = pd.DataFrame(
            {
                "Age": [45],
                "Sex": ["male"],
                "ChestPainType": ["asymptomatic"],
                "RestingBP": [130],
                "Cholesterol": [250],
                "FastingBS": ["yes"],
                "RestingECG": ["normal"],
                "MaxHR": [150],
                "ExerciseAngina": ["no"],
                "Oldpeak": [0.5],
                "ST_Slope": ["flat"],
                "ca": [0],
                "thal": ["normal"],
                "HeartDisease": [1],
            }
        )

        normalized = _normalize_heart_frame(df)

        self.assertEqual(list(normalized.columns), HEART_COLUMNS)
        self.assertEqual(int(normalized.loc[0, "target"]), 1)
        self.assertEqual(int(normalized.loc[0, "cp"]), 4)

    def test_merge_drops_duplicates(self) -> None:
        base = pd.DataFrame(
            {
                "Pregnancies": [1],
                "Glucose": [120],
                "BloodPressure": [80],
                "SkinThickness": [20],
                "Insulin": [100],
                "BMI": [28.5],
                "DiabetesPedigreeFunction": [0.5],
                "Age": [35],
                "Outcome": [1],
            }
        )

        merged = _merge_frames([base, base.copy()], DIABETES_COLUMNS, "Outcome")

        self.assertEqual(len(merged), 1)


if __name__ == "__main__":
    unittest.main()