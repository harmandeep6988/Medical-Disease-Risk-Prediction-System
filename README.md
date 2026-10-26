# Medical Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.12-blue)](#) [![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](#) [![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](#) [![FastAPI](https://img.shields.io/badge/FastAPI-available-009688?logo=fastapi&logoColor=white)](#) [![CI](https://github.com/harmandeep6988/Medical-Disease-Risk-Prediction-System/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/harmandeep6988/Medical-Disease-Risk-Prediction-System/actions/workflows/test.yml)
Predicts medical disease risk in seconds with interpretable ML and a clean, patient-friendly interface.
Built for real-world use: explainability, anomaly checks, and production-ready APIs.

[Try Live Demo](https://medical-disease.streamlit.app/)

⭐ Star this repo if you find it useful!

## Quick Start

1. Online (no installation): Click the live demo link above
2. Local (3 steps):
   1. `git clone https://github.com/harmandeep6988/Medical-Disease-Risk-Prediction-System.git`
   2. `cd Medical-Disease-Risk-Prediction-System && pip install -r requirements.txt`
   3. `streamlit run dashboard/app.py`

## Performance Table

| Disease | AUC | Accuracy | Status |
|---|---:|---:|---|
| Diabetes | 0.826 | 76.0% | ✓ Good |
| Heart Disease | 0.911 | 83.6% | ⭐ Excellent |

## Features Highlight

- ✨ 0.911 AUC for heart disease prediction
- 🎯 SHAP explainability (why predictions?)
- 🐳 Fully dockerized (5 services)
- ⚡ FastAPI REST endpoints
- 📊 Interactive Streamlit dashboard
- 🔍 Anomaly detection with IsolationForest
- 🩹 Patient self-assessment forms
- 📄 Auto-generated PDF reports
- 🧪 CI/CD with GitHub Actions

## Tech Stack

- Machine Learning: scikit-learn, XGBoost, SHAP
- Frontend: Streamlit, Plotly
- Backend: FastAPI, Pydantic
- DevOps: Docker, Docker Compose, GitHub Actions
- Python 3.12

## Architecture Diagram (ASCII)

```text
+---------------------+
|     Raw Datasets    |
+----------+----------+
           |
           v
+---------------------+
| Data Prep + Feature |
| Engineering         |
+----------+----------+
           |
           v
+---------------------+
| Train + Validate ML |
| Models              |
+----------+----------+
           |
           v
+---------------------+
| Saved Models + Meta |
+----------+----------+
           |
   +-------+-------+
   |               |
   v               v
+---------+    +-----------+
| FastAPI |    | Streamlit |
| REST    |    | Dashboard |
+----+----+    +-----+-----+
     |               |
     v               v
+---------+    +-----------+
| SHAP    |    | PDF Report|
| Explain |    | Generator |
+---------+    +-----------+
```
