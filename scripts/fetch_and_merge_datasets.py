from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from prepare_datasets import load_diabetes_dataset as load_openml_diabetes_dataset
from prepare_datasets import load_heart_dataset as load_openml_heart_dataset


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
KAGGLE_DIR = RAW_DIR / "kaggle"

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
        try:
            out[col] = pd.to_numeric(out[col])
        except Exception:
            pass
    return out


def _map_series(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    lower_mapping = {key.lower(): value for key, value in mapping.items()}

    def _map_value(value: object) -> object:
        if pd.isna(value):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            return lower_mapping.get(key, value)
        return value

    return series.map(_map_value)


def _normalize_diabetes_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "preg": "Pregnancies",
        "plas": "Glucose",
        "pres": "BloodPressure",
        "skin": "SkinThickness",
        "insu": "Insulin",
        "mass": "BMI",
        "pedi": "DiabetesPedigreeFunction",
        "age": "Age",
        "target": "Outcome",
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
        "target": "target",
        "Age": "age",
        "Sex": "sex",
        "ChestPainType": "cp",
        "RestingBP": "trestbps",
        "Cholesterol": "chol",
        "FastingBS": "fbs",
        "RestingECG": "restecg",
        "MaxHR": "thalach",
        "ExerciseAngina": "exang",
        "Oldpeak": "oldpeak",
        "ST_Slope": "slope",
    }

    out = df.copy().rename(columns=rename_map)

    for column, mapping in HEART_CODE_MAPS.items():
        if column in out.columns:
            out[column] = _map_series(out[column], mapping)

    if "target" in out.columns:
        out["target"] = pd.to_numeric(out["target"], errors="coerce")
        out["target"] = (out["target"] > 0).astype("Int64")

    out = _coerce_numeric(out)
    out = out.reindex(columns=HEART_COLUMNS)
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    return out


def _load_csv_sources(base_dir: Path, disease: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    disease_dir = base_dir / disease
    if not disease_dir.exists():
        return frames

    for path in sorted(disease_dir.rglob("*.csv")):
        if path.name.endswith("_merged.csv"):
            continue
        frames.append(pd.read_csv(path))
    return frames


def _download_kaggle_dataset(dataset_slug: str, output_dir: Path) -> list[pd.DataFrame]:
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_DIR))

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("kaggle is not installed. Add it to requirements.txt and reinstall dependencies.") from exc

    dataset_dir = output_dir / dataset_slug.replace("/", "__")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(dataset_dir), unzip=True, quiet=True)

    return [pd.read_csv(path) for path in sorted(dataset_dir.rglob("*.csv"))]


def _merge_frames(frames: list[pd.DataFrame], canonical_columns: list[str], target_column: str) -> pd.DataFrame:
    normalized_frames: list[pd.DataFrame] = []

    for frame in frames:
        if frame.empty:
            continue

        normalized = frame.copy().reindex(columns=canonical_columns)
        normalized[target_column] = pd.to_numeric(normalized[target_column], errors="coerce")
        normalized = normalized.dropna(subset=[target_column])

        if normalized.empty:
            continue

        normalized[target_column] = normalized[target_column].astype(int)
        normalized_frames.append(normalized)

    if not normalized_frames:
        raise RuntimeError(f"No usable rows found for {target_column}.")

    merged = pd.concat(normalized_frames, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged


def _build_diabetes_sources(include_openml: bool, kaggle_dataset: str | None) -> list[pd.DataFrame]:
    sources: list[pd.DataFrame] = []

    if include_openml:
        sources.append(load_openml_diabetes_dataset())

    sources.extend(_load_csv_sources(RAW_DIR, "diabetes"))

    if kaggle_dataset:
        sources.extend(_download_kaggle_dataset(kaggle_dataset, KAGGLE_DIR))

    return [_normalize_diabetes_frame(frame) for frame in sources]


def _build_heart_sources(include_openml: bool, kaggle_dataset: str | None) -> list[pd.DataFrame]:
    sources: list[pd.DataFrame] = []

    if include_openml:
        sources.append(load_openml_heart_dataset())

    sources.extend(_load_csv_sources(RAW_DIR, "heart"))

    if kaggle_dataset:
        sources.extend(_download_kaggle_dataset(kaggle_dataset, KAGGLE_DIR))

    return [_normalize_heart_frame(frame) for frame in sources]


def _save_dataset(df: pd.DataFrame, raw_path: Path, processed_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch, merge, and persist augmented medical datasets.")
    parser.add_argument(
        "--kaggle-only",
        action="store_true",
        help="Use Kaggle/local Kaggle CSV sources only and skip OpenML/local baseline sources.",
    )
    parser.add_argument(
        "--diabetes-kaggle-dataset",
        default=os.environ.get("DIABETES_KAGGLE_DATASET"),
        help="Optional Kaggle dataset slug for diabetes, for example owner/dataset-name.",
    )
    parser.add_argument(
        "--heart-kaggle-dataset",
        default=os.environ.get("HEART_KAGGLE_DATASET"),
        help="Optional Kaggle dataset slug for heart disease, for example owner/dataset-name.",
    )
    args = parser.parse_args()

    include_openml = not args.kaggle_only

    diabetes_sources = _build_diabetes_sources(include_openml=include_openml, kaggle_dataset=args.diabetes_kaggle_dataset)
    heart_sources = _build_heart_sources(include_openml=include_openml, kaggle_dataset=args.heart_kaggle_dataset)

    if args.kaggle_only and not diabetes_sources and not heart_sources:
        raise RuntimeError(
            "No Kaggle sources were found. Provide local Kaggle CSVs or set DIABETES_KAGGLE_DATASET / HEART_KAGGLE_DATASET."
        )

    diabetes_merged = _merge_frames(diabetes_sources, DIABETES_COLUMNS, "Outcome")
    heart_merged = _merge_frames(heart_sources, HEART_COLUMNS, "target")

    _save_dataset(
        diabetes_merged,
        RAW_DIR / "diabetes_merged.csv",
        PROC_DIR / "diabetes_processed.csv",
    )
    _save_dataset(
        heart_merged,
        RAW_DIR / "heart_merged.csv",
        PROC_DIR / "heart_processed.csv",
    )

    print(f"Saved merged diabetes dataset: {RAW_DIR / 'diabetes_merged.csv'} ({len(diabetes_merged)} rows)")
    print(f"Saved merged heart dataset: {RAW_DIR / 'heart_merged.csv'} ({len(heart_merged)} rows)")


if __name__ == "__main__":
    main()