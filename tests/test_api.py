from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api.main import app


DIABETES_LOW = dict(pregnancies=0, glucose=85, blood_pressure=66, skin_thickness=20, insulin=60,
                    bmi=22.0, diabetes_pedigree_function=0.2, age=25)
DIABETES_HIGH = dict(pregnancies=6, glucose=190, blood_pressure=80, skin_thickness=35, insulin=250,
                     bmi=40.0, diabetes_pedigree_function=1.2, age=55)
HEART_LOW = dict(age=35, sex=0, cp=3, trestbps=115, chol=190, fbs=0, restecg=0, thalach=175,
                 exang=0, oldpeak=0.0, slope=1, ca=0, thal=3)
HEART_HIGH = dict(age=65, sex=1, cp=4, trestbps=160, chol=300, fbs=1, restecg=2, thalach=105,
                  exang=1, oldpeak=3.5, slope=2, ca=3, thal=7)


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)

    def test_health_reports_models_loaded(self) -> None:
        body = self.client.get("/health").json()
        self.assertEqual(body["models_loaded"], {"diabetes": True, "heart": True})

    def test_predictions_rank_high_risk_above_low_risk(self) -> None:
        for endpoint, low, high in [("diabetes", DIABETES_LOW, DIABETES_HIGH), ("heart", HEART_LOW, HEART_HIGH)]:
            with self.subTest(endpoint=endpoint):
                low_res = self.client.post(f"/predict/{endpoint}", json=low).json()
                high_res = self.client.post(f"/predict/{endpoint}", json=high).json()

                self.assertLess(low_res["risk_probability"], 0.5)
                self.assertGreater(high_res["risk_probability"], 0.5)
                self.assertEqual(len(high_res["top_factors"]), 3)


if __name__ == "__main__":
    unittest.main()
