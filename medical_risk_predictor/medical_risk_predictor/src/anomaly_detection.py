from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


ANOMALY_DIR = Path(__file__).resolve().parents[1] / "reports" / "generated"
ANOMALY_DIR.mkdir(parents=True, exist_ok=True)


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[list[int], pd.Series, pd.DataFrame, Path]:
    """Detect anomalous patient profiles and provide per-record reasons."""
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(df)
    scores = pd.Series(model.decision_function(df), index=df.index, name="anomaly_score")

    anomaly_mask = preds == -1
    anomaly_indices = df.index[anomaly_mask].tolist()

    z_scores = ((df - df.mean()) / (df.std(ddof=0) + 1e-8)).abs()
    reasons = pd.DataFrame(index=df.index)
    reasons["most_unusual_feature"] = z_scores.idxmax(axis=1)
    reasons["feature_zscore"] = z_scores.max(axis=1)

    reduced = PCA(n_components=2, random_state=random_state).fit_transform(df)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=["red" if m else "blue" for m in anomaly_mask], alpha=0.7)
    plt.title("Anomaly Detection (IsolationForest + PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    plot_path = ANOMALY_DIR / "anomaly_scatter.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    if verbose:
        for idx in anomaly_indices:
            feature = reasons.loc[idx, "most_unusual_feature"]
            z = reasons.loc[idx, "feature_zscore"]
            print(f"Patient index {idx} anomalous due to unusual {feature} (|z|={z:.2f}).")

    return anomaly_indices, scores, reasons, plot_path
