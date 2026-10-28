"""Shared risk labels used by the dashboard, API, and generated reports."""

from __future__ import annotations


LOW_RISK_THRESHOLD = 0.4
HIGH_RISK_THRESHOLD = 0.7


def risk_band(probability: float) -> str:
    """Return the canonical three-level risk band for a probability."""
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    if probability <= HIGH_RISK_THRESHOLD:
        return "Moderate"
    return "High"
