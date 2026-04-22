from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except Exception:
            pass
    return out


def load_diabetes_dataset() -> pd.DataFrame:
    dataset = fetch_openml(name="diabetes", version=1, as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()

    y_map = {
        "tested_positive": 1,
        "tested_negative": 0,
        "1": 1,
        "0": 0,
        "True": 1,
        "False": 0,
    }
    y = y.astype(str).map(y_map).fillna(pd.to_numeric(y, errors="coerce")).astype(int)

    df = _coerce_numeric(X)
    df["Outcome"] = y

    rename_map = {
        "preg": "Pregnancies",
        "plas": "Glucose",
        "pres": "BloodPressure",
        "skin": "SkinThickness",
        "insu": "Insulin",
        "mass": "BMI",
        "pedi": "DiabetesPedigreeFunction",
        "age": "Age",
    }
    df = df.rename(columns=rename_map)
    return df


def load_heart_dataset() -> pd.DataFrame:
    candidate_names = ["heart-disease", "heart", "HeartDisease"]
    last_err = None

    for name in candidate_names:
        try:
            dataset = fetch_openml(name=name, as_frame=True)
            X = dataset.data.copy()
            y = dataset.target.copy()
            df = _coerce_numeric(X)
            y_num = pd.to_numeric(y, errors="coerce")
            if y_num.isna().all():
                y_num = y.astype(str).str.extract(r"(\d+)", expand=False).astype(float)
            # Convert multi-class severity to binary risk target.
            y_bin = (y_num.fillna(0) > 0).astype(int)
            df["target"] = y_bin
            return df
        except Exception as exc:  # pragma: no cover
            last_err = exc
            continue

    # Fallback to UCI Cleveland dataset.
    uci_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    ]

    col_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    for url in uci_urls:
        try:
            if "processed.cleveland.data" in url:
                df = pd.read_csv(url, header=None, names=col_names)
                df = df.replace("?", pd.NA).dropna()
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna().reset_index(drop=True)
                df["target"] = (df["target"] > 0).astype(int)
                return df

            df = pd.read_csv(url)
            df = _coerce_numeric(df)
            if "target" not in df.columns and "HeartDisease" in df.columns:
                df["target"] = pd.to_numeric(df["HeartDisease"], errors="coerce").fillna(0).astype(int)
                df = df.drop(columns=["HeartDisease"])
            if "target" in df.columns:
                df["target"] = (pd.to_numeric(df["target"], errors="coerce").fillna(0) > 0).astype(int)
                return df.dropna().reset_index(drop=True)
        except Exception:
            continue

    raise RuntimeError(f"Could not fetch heart disease dataset from OpenML or URL fallbacks: {last_err}")


def main() -> None:
    diabetes_df = load_diabetes_dataset()
    heart_df = load_heart_dataset()

    diabetes_raw = RAW_DIR / "diabetes.csv"
    heart_raw = RAW_DIR / "heart.csv"
    diabetes_proc = PROC_DIR / "diabetes_processed.csv"
    heart_proc = PROC_DIR / "heart_processed.csv"

    diabetes_df.to_csv(diabetes_raw, index=False)
    heart_df.to_csv(heart_raw, index=False)
    diabetes_df.to_csv(diabetes_proc, index=False)
    heart_df.to_csv(heart_proc, index=False)

    print(f"Saved: {diabetes_raw}")
    print(f"Saved: {heart_raw}")
    print(f"Saved: {diabetes_proc}")
    print(f"Saved: {heart_proc}")
    print(f"Diabetes shape: {diabetes_df.shape}")
    print(f"Heart shape: {heart_df.shape}")


if __name__ == "__main__":
    main()
