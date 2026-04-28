from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from anomaly_detection import detect_anomalies
from data_preprocessing import handle_missing_values, replace_zero_with_nan
from feature_engineering import HEART_CATEGORICAL_COLUMNS, build_feature_matrix


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prep_for_anomaly_detection(
    dataset: pd.DataFrame,
    target_col: str,
    zero_invalid_cols: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    raw = dataset.drop(columns=[target_col], errors="ignore")
    raw = _coerce_numeric(raw)
    raw = replace_zero_with_nan(raw, zero_invalid_cols)
    raw = handle_missing_values(raw)

    raw_columns = list(raw.columns)
    passthrough_cols = [c for c in categorical_columns if c in raw_columns]
    scaled_cols = [c for c in raw_columns if c not in passthrough_cols]

    scaled = raw.copy()
    if scaled_cols:
        scaled = scaled.astype({c: float for c in scaled_cols})
        scale_input = raw.loc[:, scaled_cols].astype(float)
        std = StandardScaler()
        mm = MinMaxScaler()
        scaled_std = std.fit_transform(scale_input)
        scaled_vals = mm.fit_transform(scaled_std)
        scaled.loc[:, scaled_cols] = scaled_vals

    return build_feature_matrix(scaled, categorical_columns=passthrough_cols)


def _run_one(
    dataset_path: Path,
    disease_key: str,
    target_col: str,
    zero_invalid_cols: list[str],
) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    categorical_cols = [c for c in HEART_CATEGORICAL_COLUMNS if c in df.columns]
    X = _prep_for_anomaly_detection(df, target_col, zero_invalid_cols, categorical_cols)

    anomalies, _, reasons, plot_path = detect_anomalies(
        X,
        verbose=False,
        artifact_prefix=f"{disease_key.lower()}_anomaly",
        save_csv=True,
    )

    out_csv = ROOT / "reports" / "generated" / f"{disease_key.lower()}_anomaly_anomalies.csv"

    top_features = (
        reasons.loc[anomalies, "most_unusual_feature"].value_counts().head(10).to_dict() if anomalies else {}
    )

    print(f"[{disease_key}] rows={len(df)} features_for_detection={X.shape[1]}")
    print(f"[{disease_key}] anomalies={len(anomalies)} contamination=0.05")
    print(f"[{disease_key}] plot={plot_path}")
    print(f"[{disease_key}] csv={out_csv}")
    if top_features:
        print(f"[{disease_key}] top_unusual_features={top_features}")


def main() -> None:
    diabetes_path = ROOT / "data" / "processed" / "diabetes_processed.csv"
    heart_path = ROOT / "data" / "processed" / "heart_processed.csv"

    _run_one(
        diabetes_path,
        disease_key="Diabetes",
        target_col="Outcome",
        zero_invalid_cols=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    )
    _run_one(
        heart_path,
        disease_key="Heart",
        target_col="target",
        zero_invalid_cols=[],
    )


if __name__ == "__main__":
    main()
