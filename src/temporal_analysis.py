from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TEMPORAL_DIR = Path(__file__).resolve().parents[1] / "reports" / "generated"
TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)


def compute_temporal_risk(
    records: pd.DataFrame,
    model,
    time_column: str = "timestamp",
    patient_id_column: str = "patient_id",
    spike_threshold: float = 0.15,
) -> tuple[pd.DataFrame, list[str], Path]:
    """Compute risk trajectory and detect sudden deteriorations over time."""
    required = {time_column, patient_id_column}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = records.copy().sort_values([patient_id_column, time_column])
    feature_cols = [c for c in df.columns if c not in [time_column, patient_id_column]]

    df["risk_probability"] = model.predict_proba(df[feature_cols])[:, 1]
    df["risk_delta"] = df.groupby(patient_id_column)["risk_probability"].diff().fillna(0.0)

    alerts = []
    spikes = df[df["risk_delta"] > spike_threshold]
    for _, row in spikes.iterrows():
        alerts.append(
            f"ALERT: Patient {row[patient_id_column]} risk increased by {row['risk_delta'] * 100:.1f}% on {row[time_column]}."
        )

    plt.figure(figsize=(10, 6))
    for patient_id, group in df.groupby(patient_id_column):
        plt.plot(group[time_column], group["risk_probability"], marker="o", label=f"Patient {patient_id}")

    plt.title("Temporal Risk Trend")
    plt.xlabel("Time")
    plt.ylabel("Risk Probability")
    plt.xticks(rotation=30)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    plot_path = TEMPORAL_DIR / "temporal_risk_trend.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    for msg in alerts:
        print(msg)

    return df[[patient_id_column, time_column, "risk_probability", "risk_delta"]], alerts, plot_path
