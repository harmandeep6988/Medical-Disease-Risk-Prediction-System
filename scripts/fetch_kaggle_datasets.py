from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
KAGGLE_CACHE_DIR = RAW_DIR / "kaggle"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DIABETES_DATASETS = [
    {
        "name": "Pima Indians Diabetes Database",
        "slug": "uciml/pima-indians-diabetes-database",
    },
    {
        "name": "Diabetes Health Indicators (rishidamarla)",
        "slug": "rishidamarla/diabetes-health-indicators-dataset",
    },
]

HEART_DATASETS = [
    {
        "name": "Heart Failure Prediction",
        "slug": "fedesoriano/heart-failure-prediction",
    },
    {
        "name": "Heart Disease Dataset (johnsmith88)",
        "slug": "johnsmith88/heart-disease-dataset",
    },
    {
        "name": "Heart Failure Clinical Data",
        "slug": "andrewmvd/heart-failure-clinical-data",
    },
]

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

DIABETES_TARGET_VALUES = {
    "tested_positive": 1,
    "tested_negative": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
}

HEART_CODE_MAPS = {
    "sex": {"female": 0, "f": 0, "woman": 0, "male": 1, "m": 1, "man": 1},
    "cp": {
        "typical_angina": 1,
        "typical": 1,
        "ta": 1,
        "atypical_angina": 2,
        "atypical": 2,
        "aa": 2,
        "non_anginal": 3,
        "non_anginal_pain": 3,
        "nap": 3,
        "asymptomatic": 4,
        "asy": 4,
    },
    "fbs": {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0},
    "restecg": {
        "normal": 0,
        "stt_abnormal": 1,
        "st-t_abnormal": 1,
        "st_abnormality": 1,
        "lv_hypertrophy": 2,
        "left_ventricular_hypertrophy": 2,
    },
    "exang": {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0},
    "slope": {"upsloping": 1, "up": 1, "flat": 2, "downsloping": 3, "down": 3},
    "thal": {
        "normal": 3,
        "fixed_defect": 6,
        "reversible_defect": 7,
        "fixed": 6,
        "reversible": 7,
    },
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def _normalize_diabetes_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
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
        "target": "Outcome",
        "outcome": "Outcome",
        "diabetes_binary": "Outcome",
    }

    out = df.copy().rename(columns=rename_map)
    if "Outcome" in out.columns:
        outcome_series = out["Outcome"].astype(str).str.strip().str.lower().map(DIABETES_TARGET_VALUES)
        out["Outcome"] = outcome_series.fillna(pd.to_numeric(out["Outcome"], errors="coerce"))

    out = _coerce_numeric(out)
    out = out.reindex(columns=DIABETES_COLUMNS)
    out = out.dropna(subset=["Outcome"]).reset_index(drop=True)
    out["Outcome"] = out["Outcome"].astype(int)
    return out


def _normalize_heart_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "HeartDisease": "target",
        "cardio": "target",
        "target": "target",
        "Age": "age",
        "age": "age",
        "Sex": "sex",
        "sex": "sex",
        "ChestPainType": "cp",
        "cp": "cp",
        "RestingBP": "trestbps",
        "trestbps": "trestbps",
        "Cholesterol": "chol",
        "chol": "chol",
        "FastingBS": "fbs",
        "fbs": "fbs",
        "RestingECG": "restecg",
        "restecg": "restecg",
        "MaxHR": "thalach",
        "thalach": "thalach",
        "ExerciseAngina": "exang",
        "exang": "exang",
        "Oldpeak": "oldpeak",
        "ST_Slope": "slope",
        "slope": "slope",
        "ca": "ca",
        "thal": "thal",
    }

    out = df.copy().rename(columns=rename_map)

    for column, mapping in HEART_CODE_MAPS.items():
        if column in out.columns:
            out[column] = out[column].map(lambda value: _map_value(value, mapping))

    if "target" in out.columns:
        out["target"] = pd.to_numeric(out["target"], errors="coerce")
        out["target"] = (out["target"] > 0).astype("Int64")

    out = _coerce_numeric(out)
    out = out.reindex(columns=HEART_COLUMNS)
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    return out


def _map_value(value: object, mapping: dict[str, int]) -> object:
    if pd.isna(value):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        return mapping.get(key, value)
    return value


def _drop_invalid_rows(df: pd.DataFrame, invalid_zero_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in invalid_zero_columns:
        if col in out.columns:
            out[col] = out[col].replace(0, np.nan)
    out = out.dropna(subset=[c for c in invalid_zero_columns if c in out.columns], how="any")
    return out


def _resolve_kaggle_config_dir() -> Path:
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        return Path(os.environ["KAGGLE_CONFIG_DIR"])
    project_kaggle = ROOT / "kaggle"
    if (project_kaggle / "kaggle.json").exists():
        return project_kaggle
    raw_kaggle = ROOT / "data" / "raw" / "kaggle"
    if (raw_kaggle / "kaggle.json").exists():
        return raw_kaggle
    return Path.home() / ".kaggle"


def _download_kaggle_dataset(dataset_slug: str, output_dir: Path) -> list[pd.DataFrame]:
    kaggle_config = _resolve_kaggle_config_dir()
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(kaggle_config))

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise RuntimeError("kaggle is not installed. Add it to requirements.txt and reinstall dependencies.") from exc

    dataset_dir = output_dir / dataset_slug.replace("/", "__")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(dataset_dir), unzip=True, quiet=True)

    return [pd.read_csv(path) for path in sorted(dataset_dir.rglob("*.csv"))]


def _load_existing_processed(path: Path, target_name: str) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Missing fallback processed dataset for %s: %s", target_name, path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _process_dataset_group(
    datasets: list[dict[str, str]],
    disease: str,
    target_column: str,
    invalid_zero_cols: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for dataset in datasets:
        name = dataset["name"]
        slug = dataset["slug"]
        try:
            frames_raw = _download_kaggle_dataset(slug, KAGGLE_CACHE_DIR)
            if not frames_raw:
                raise RuntimeError("No CSV files downloaded")
            df_raw = pd.concat(frames_raw, ignore_index=True)
            raw_rows = len(df_raw)

            if disease == "diabetes":
                df_clean = _normalize_diabetes_frame(df_raw)
            else:
                df_clean = _normalize_heart_frame(df_raw)

            df_clean = _drop_invalid_rows(df_clean, invalid_zero_cols)
            before_dedup = len(df_clean)
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            duplicates_removed = before_dedup - len(df_clean)

            frames.append(df_clean)
            summary.append(
                {
                    "dataset": name,
                    "rows_downloaded": raw_rows,
                    "final_rows": len(df_clean),
                    "status": "Success",
                    "duplicates_removed": duplicates_removed,
                }
            )
        except Exception as exc:
            summary.append(
                {
                    "dataset": name,
                    "rows_downloaded": 0,
                    "final_rows": 0,
                    "status": f"Failed ({exc})",
                    "duplicates_removed": 0,
                }
            )
            logger.warning("Failed to download %s: %s", slug, exc)

    if not frames:
        return pd.DataFrame(), summary

    merged = pd.concat(frames, ignore_index=True)
    before_dedup = len(merged)
    merged = merged.drop_duplicates().reset_index(drop=True)
    logger.info("Deduplicated across sources: removed %d rows", before_dedup - len(merged))
    merged = merged.dropna(subset=[target_column]).reset_index(drop=True)

    return merged, summary


def _print_summary(rows: list[dict[str, object]]) -> None:
    print("\nDataset Summary:")
    print("Dataset | Rows Downloaded | Final Count After Cleaning | Status")
    print("--- | ---:| ---:| ---")
    for row in rows:
        print(
            f"{row['dataset']} | {row['rows_downloaded']} | {row['final_rows']} | {row['status']}"
        )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    diabetes_df, diabetes_summary = _process_dataset_group(
        DIABETES_DATASETS,
        disease="diabetes",
        target_column="Outcome",
        invalid_zero_cols=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    )

    if diabetes_df.empty:
        logger.warning("No Kaggle diabetes datasets loaded. Using processed fallback.")
        diabetes_df = _load_existing_processed(PROC_DIR / "diabetes_processed.csv", "diabetes")
        if not diabetes_df.empty:
            diabetes_df = _normalize_diabetes_frame(diabetes_df)

    heart_df, heart_summary = _process_dataset_group(
        HEART_DATASETS,
        disease="heart",
        target_column="target",
        invalid_zero_cols=["trestbps", "chol", "thalach", "oldpeak"],
    )

    if heart_df.empty:
        logger.warning("No Kaggle heart datasets loaded. Using processed fallback.")
        heart_df = _load_existing_processed(PROC_DIR / "heart_processed.csv", "heart")
        if not heart_df.empty:
            heart_df = _normalize_heart_frame(heart_df)

    diabetes_out = RAW_DIR / "diabetes_kaggle.csv"
    heart_out = RAW_DIR / "heart_kaggle.csv"

    if not diabetes_df.empty:
        diabetes_df.to_csv(diabetes_out, index=False)
        logger.info("Saved %s (%d rows)", diabetes_out, len(diabetes_df))

    if not heart_df.empty:
        heart_df.to_csv(heart_out, index=False)
        logger.info("Saved %s (%d rows)", heart_out, len(heart_df))

    _print_summary(diabetes_summary + heart_summary)


if __name__ == "__main__":
    main()
