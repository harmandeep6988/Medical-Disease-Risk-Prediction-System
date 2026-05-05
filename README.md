# Medical Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.12-blue)](#) [![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](#) [![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](#) [![FastAPI](https://img.shields.io/badge/FastAPI-available-009688?logo=fastapi&logoColor=white)](#) [![CI](https://github.com/OWNER/REPO/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/OWNER/REPO/actions/workflows/test.yml) [![License](https://img.shields.io/badge/License-MIT-green)](#license)

Predicts medical disease risk in seconds with interpretable ML and a clean, patient-friendly interface.
Built for real-world use: explainability, anomaly checks, and production-ready APIs.

[Try Live Demo](https://your-app-medical-risk-prediction.streamlit.app)

⭐ Star this repo if you find it useful!

## Quick Start

1. Online (no installation): Click the live demo link above
2. Local (3 steps):
   1. `git clone https://github.com/OWNER/REPO.git`
   2. `cd Medical-Disease-Risk-Prediction-System && pip install -r requirements.txt`
   3. `streamlit run dashboard/app.py`

## Performance Table

| Disease | AUC | Accuracy | Status |
|---|---:|---:|---|
| Diabetes | 0.826 | 76.0% | ✓ Good |
| Heart Disease | 0.938 | 88.3% | ⭐ Excellent |

## Features Highlight

- ✨ 0.911 AUC for heart disease prediction
- 🎯 SHAP explainability (why predictions?)
- 🐳 Fully dockerized (5 service profiles)
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

## License

MIT
