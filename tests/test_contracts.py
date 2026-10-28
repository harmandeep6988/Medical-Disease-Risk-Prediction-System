from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from risk_policy import risk_band


class TestProjectContracts(unittest.TestCase):
    def test_risk_band_boundaries_are_canonical(self) -> None:
        self.assertEqual(risk_band(0.3999), "Low")
        self.assertEqual(risk_band(0.4), "Moderate")
        self.assertEqual(risk_band(0.7), "Moderate")
        self.assertEqual(risk_band(0.7001), "High")

    def test_shipped_metadata_matches_reference_datasets(self) -> None:
        expected = {
            "diabetes": "data/raw/diabetes_kaggle.csv",
            "heart": "data/raw/heart_kaggle.csv",
        }
        for disease, relative_path in expected.items():
            with self.subTest(disease=disease):
                meta = joblib.load(ROOT / "models" / "saved_models" / f"best_{disease}_meta.joblib")
                dataset_path = ROOT / relative_path
                dataset = pd.read_csv(dataset_path)
                self.assertEqual(meta["reference_data_path"], relative_path)
                self.assertEqual(meta["record_count"], len(dataset))
                self.assertEqual(
                    meta["dataset_sha256"],
                    hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(meta["raw_columns"], [c for c in dataset.columns if c != meta["target"]])
                self.assertTrue(set(meta["invalid_zero_columns"]).issubset(dataset.columns))


if __name__ == "__main__":
    unittest.main()
