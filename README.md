# Badges

[![Python](https://img.shields.io/badge/Python-3.12-blue)](#) [![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](#) [![FastAPI](https://img.shields.io/badge/FastAPI-available-009688?logo=fastapi&logoColor=white)](#) [![License](https://img.shields.io/badge/License-MIT-green)](#license) [![CI](https://github.com/OWNER/REPO/actions/workflows/test.yml/badge.svg)](#)

# Medical Disease Risk Prediction System

Predicts risk for heart disease, diabetes, and liver disease using ML.

[Live Demo](https://your-app.streamlit.app)

## Performance

| Disease | AUC | Accuracy | F1 |
|---|---:|---:|---:|
| Heart Disease | 0.911 | 88.5% | 0.842 |
| Diabetes | 0.847 | 81.2% | — |
| Liver Disease | 0.823 | 79.4% | — |

## Key Features

- SHAP explainability for every prediction
- SMOTENC for class imbalance on mixed data types
- Anomaly detection with IsolationForest
- 5 Docker service profiles (dashboard, training, pipeline, data-prep, jupyter)
- Multi-stage Docker build (40% smaller image)
- Non-root user security for healthcare data
- FastAPI REST endpoints for all 3 diseases
- Automated CI via GitHub Actions
- PDF clinical report generation

## Architecture (High Level)

```text
	 +-------------------+
	 |  Raw Data (CSV)   |
	 +---------+---------+
		   |
		   v
     +-----------------------------+
     | Data Prep + Feature Eng.    |
     | (clean, encode, scale)      |
     +--------------+--------------+
		    |
		    v
     +-----------------------------+
     | Train + Validate Models     |
     | (CV, metrics, selection)    |
     +--------------+--------------+
		    |
		    v
     +-----------------------------+
     | Saved Models + Metadata     |
     +--------------+--------------+
		    |
	 +----------+----------+
	 |                     |
	 v                     v
+------------------+  +------------------+
| FastAPI REST API  |  | Streamlit UI     |
| (3 disease routes)|  | (inference + UX) |
+--------+---------+  +---------+--------+
	 |                      |
	 v                      v
 +----------------+     +-----------------+
 | SHAP Explainer |     | PDF Report Gen  |
 +----------------+     +-----------------+
```

## Quick Start

### Option A — Docker

```bash
cd docker
docker compose up dashboard
```

### Option B — API

```bash
uvicorn api.main:app --reload
```

### Option C — Local (Dashboard)

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Project Structure (2 Levels)

```text
Medical-Disease-Risk-Prediction-System/
├── dashboard/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   └── saved_models/
├── notebooks/
│   └── exploration.ipynb
├── reports/
│   └── generated/
├── scripts/
│   └── run_all.py
├── src/
│   └── model_training.py
├── tests/
│   └── test_preprocessing.py
└── requirements.txt
```

## Tech Stack

- **Language & Runtime**: Python 3.12
- **ML**: Scikit-learn, XGBoost, GradientBoosting
- **Explainability**: SHAP
- **Imbalance Handling**: SMOTENC
- **Anomaly Detection**: IsolationForest
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Apps & APIs**: Streamlit, FastAPI
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Data Source**: Kaggle API

## Key ML Concepts

- **SHAP**: Explains individual predictions with feature attributions.
- **SMOTENC**: Balances classes when data includes mixed numeric + categorical features.
- **IsolationForest**: Flags anomalous inputs that may be unreliable for inference.

## Screenshots

> Add images here:
>
> - `docs/images/dashboard.png`
> - `docs/images/api_swagger.png`

## Cite / Contact

**How to cite** (example):

```text
Medical Disease Risk Prediction System, 2026. GitHub repository: https://github.com/OWNER/REPO
```

**Contact**: Your Name — your.email@example.com

## License

MIT (update if different).
