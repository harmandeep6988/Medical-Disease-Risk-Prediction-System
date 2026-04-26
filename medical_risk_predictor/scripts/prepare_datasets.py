from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
from sklearn.datasets import fetch_openml


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
KAGGLE_DIR = RAW_DIR / "kaggle"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

DIABETES_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

HEART_COLUMNS = [
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

KAGGLE_DIABETES_DATASETS = [
    "uciml/pima-indians-diabetes-database",
    "mathchi/diabetes-data-set",
    "akshaydattatraykhare/diabetes-dataset",
]

KAGGLE_HEART_DATASETS = [
    "johnsmith88/heart-disease-dataset",
    "fedesoriano/heart-failure-prediction",
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except Exception:
            pass
    return out


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace(" ", "")


def _kaggle_base_command() -> list[str] | None:
    venv_kaggle = Path(sys.executable).parent / "kaggle"
    if venv_kaggle.exists():
        return [str(venv_kaggle)]

    if shutil.which("kaggle"):
        return ["kaggle"]

    return None


def _get_kaggle_config_dir() -> Path | None:
    candidates = [Path.home() / ".kaggle", KAGGLE_DIR]
    for candidate in candidates:
        config_path = candidate / "kaggle.json"
        if config_path.exists() and config_path.is_file():
            return candidate
    return None


def _download_kaggle_dataset(dataset_slug: str, destination_root: Path) -> list[Path]:
    base_cmd = _kaggle_base_command()
    if base_cmd is None:
        print("Kaggle CLI not found. Install via: pip install kaggle")
        return []

    kaggle_config_dir = _get_kaggle_config_dir()
    if kaggle_config_dir is None:
        for check_dir in [Path.home() / ".kaggle", KAGGLE_DIR]:
            config_path = check_dir / "kaggle.json"
            if config_path.exists() and config_path.is_dir():
                print(f"Invalid Kaggle config: {config_path} is a directory, expected a JSON file.")
        print("Kaggle credentials missing. Place kaggle.json in ~/.kaggle or data/raw/kaggle.")
        return []

    safe_slug = dataset_slug.replace("/", "__")
    out_dir = destination_root / safe_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = base_cmd + ["datasets", "download", "-d", dataset_slug, "-p", str(out_dir), "--unzip", "--force"]
    env = dict(**os.environ)
    env["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Kaggle download failed for {dataset_slug}: {result.stderr.strip() or result.stdout.strip()}")
        return []

    return sorted(out_dir.rglob("*.csv"))


def _standardize_diabetes_from_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    rename_map = {}
    for col in df.columns:
        norm = _normalize_column_name(col)
        mapping = {
            "preg": "Pregnancies",
            "pregnancies": "Pregnancies",
            "plas": "Glucose",
            "glucose": "Glucose",
            "pres": "BloodPressure",
            "bloodpressure": "BloodPressure",
            "skin": "SkinThickness",
            "skinthickness": "SkinThickness",
            "insu": "Insulin",
            "insulin": "Insulin",
            "mass": "BMI",
            "bmi": "BMI",
            "pedi": "DiabetesPedigreeFunction",
            "diabetespedigreefunction": "DiabetesPedigreeFunction",
            "age": "Age",
            "outcome": "Outcome",
            "class": "Outcome",
        }
        if norm in mapping:
            rename_map[col] = mapping[norm]

    standardized = df.rename(columns=rename_map)
    if not set(DIABETES_COLUMNS).issubset(standardized.columns):
        return None

    standardized = standardized[DIABETES_COLUMNS].copy()
    standardized = _coerce_numeric(standardized).dropna().reset_index(drop=True)
    standardized["Outcome"] = (pd.to_numeric(standardized["Outcome"], errors="coerce").fillna(0) > 0).astype(int)
    return standardized


def _standardize_heart_from_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # Mapping for the common Kaggle heart-failure prediction schema.
    if {
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
        "HeartDisease",
    }.issubset(df.columns):
        cp_map = {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}
        ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
        exang_map = {"N": 0, "Y": 1}
        slope_map = {"Up": 0, "Flat": 1, "Down": 2}

        out = pd.DataFrame(
            {
                "age": df["Age"],
                "sex": df["Sex"].map({"F": 0, "M": 1}),
                "cp": df["ChestPainType"].map(cp_map),
                "trestbps": df["RestingBP"],
                "chol": df["Cholesterol"],
                "fbs": df["FastingBS"],
                "restecg": df["RestingECG"].map(ecg_map),
                "thalach": df["MaxHR"],
                "exang": df["ExerciseAngina"].map(exang_map),
                "oldpeak": df["Oldpeak"],
                "slope": df["ST_Slope"].map(slope_map),
                "ca": 0,
                "thal": 1,
                "target": df["HeartDisease"],
            }
        )
        out = _coerce_numeric(out).dropna().reset_index(drop=True)
        out["target"] = (pd.to_numeric(out["target"], errors="coerce").fillna(0) > 0).astype(int)
        return out[HEART_COLUMNS]

    rename_map = {}
    mapping = {
        "age": "age",
        "sex": "sex",
        "cp": "cp",
        "trestbps": "trestbps",
        "chol": "chol",
        "fbs": "fbs",
        "restecg": "restecg",
        "thalach": "thalach",
        "exang": "exang",
        "oldpeak": "oldpeak",
        "slope": "slope",
        "ca": "ca",
        "thal": "thal",
        "target": "target",
        "num": "target",
        "heartdisease": "target",
    }
    for col in df.columns:
        norm = _normalize_column_name(col)
        if norm in mapping:
            rename_map[col] = mapping[norm]

    standardized = df.rename(columns=rename_map)
    if not set(HEART_COLUMNS).issubset(standardized.columns):
        return None

    standardized = standardized[HEART_COLUMNS].copy()
    standardized = _coerce_numeric(standardized).dropna().reset_index(drop=True)
    standardized["target"] = (pd.to_numeric(standardized["target"], errors="coerce").fillna(0) > 0).astype(int)
    return standardized


def load_kaggle_diabetes_datasets() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for slug in KAGGLE_DIABETES_DATASETS:
        csv_paths = _download_kaggle_dataset(slug, KAGGLE_DIR)
        for csv_path in csv_paths:
            standardized = _standardize_diabetes_from_csv(csv_path)
            if standardized is not None and not standardized.empty:
                frames.append(standardized)
                print(f"Loaded Kaggle diabetes data from {slug}: {csv_path.name} ({len(standardized)} rows)")
                break
    return frames


def load_kaggle_heart_datasets() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for slug in KAGGLE_HEART_DATASETS:
        csv_paths = _download_kaggle_dataset(slug, KAGGLE_DIR)
        for csv_path in csv_paths:
            standardized = _standardize_heart_from_csv(csv_path)
            if standardized is not None and not standardized.empty:
                frames.append(standardized)
                print(f"Loaded Kaggle heart data from {slug}: {csv_path.name} ({len(standardized)} rows)")
                break
    return frames


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
    parser = argparse.ArgumentParser(description="Prepare base datasets, optionally augmenting with Kaggle sources.")
    parser.add_argument(
        "--include-kaggle",
        action="store_true",
        help="Attempt to download and merge additional compatible datasets from Kaggle.",
    )
    args = parser.parse_args()

    diabetes_df = load_diabetes_dataset()
    heart_df = load_heart_dataset()

    if args.include_kaggle:
        kaggle_diabetes = load_kaggle_diabetes_datasets()
        kaggle_heart = load_kaggle_heart_datasets()

        if kaggle_diabetes:
            diabetes_df = pd.concat([diabetes_df] + kaggle_diabetes, ignore_index=True).drop_duplicates().reset_index(drop=True)
        if kaggle_heart:
            heart_df = pd.concat([heart_df] + kaggle_heart, ignore_index=True).drop_duplicates().reset_index(drop=True)

        print(f"Merged diabetes rows after Kaggle augmentation: {len(diabetes_df)}")
        print(f"Merged heart rows after Kaggle augmentation: {len(heart_df)}")

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
