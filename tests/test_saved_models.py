from __future__ import annotations

import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "saved_models"


class TestSavedModels(unittest.TestCase):
    """Guards against dependency upgrades that break loading the shipped models."""

    def test_best_models_load_and_predict(self) -> None:
        for disease in ["diabetes", "heart"]:
            with self.subTest(disease=disease):
                model = joblib.load(MODEL_DIR / f"best_{disease}.joblib")
                meta = joblib.load(MODEL_DIR / f"best_{disease}_meta.joblib")

                X = pd.DataFrame(np.full((2, len(meta["features"])), 0.5), columns=meta["features"])
                proba = model.predict_proba(X)

                self.assertEqual(proba.shape, (2, 2))
                self.assertTrue(np.all((proba >= 0) & (proba <= 1)))


if __name__ == "__main__":
    unittest.main()
