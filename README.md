# Medical Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.10--3.13-blue)](#) [![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://medical-disease.streamlit.app/) [![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)](#rest-api) [![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](#run-with-docker) [![Tests](https://github.com/harmandeep6988/Medical-Disease-Risk-Prediction-System/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/harmandeep6988/Medical-Disease-Risk-Prediction-System/actions/workflows/test.yml)

Predicts diabetes and heart disease risk from routine clinical measurements, and explains each prediction with SHAP.
Includes a Streamlit dashboard, a FastAPI REST API, patient self-assessment forms with PDF reports, and anomaly detection.

**[Try the live demo](https://medical-disease.streamlit.app/)** — no installation needed.

> **Disclaimer:** This is an educational machine learning project, not a medical device. Its predictions are not a diagnosis and must not be used to make medical decisions. Always consult a qualified healthcare professional.

## Results

Scores of the models shipped in `models/saved_models/`, measured on a held-out 20% test split (stratified, `random_state=42`):

| Disease | Model | Records | AUC | Accuracy |
|---|---|---:|---:|---:|
| Diabetes | Logistic Regression | 392 | 0.871 | 75.9% |
| Heart Disease | Random Forest | 633 | 0.909 | 85.0% |

Four model types (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) are tuned with 5-fold `GridSearchCV`, and the one with the best test AUC is saved. Run `python scripts/train_and_evaluate.py` to reproduce these numbers.

## Features

- **Risk prediction** for diabetes and heart disease, with Low / Moderate / High risk bands (below 40%, 40-70%, above 70%)
- **Explainability:** SHAP shows which measurements pushed each prediction up or down
- **Patient self-assessment** forms in plain language, with a downloadable PDF report
- **Anomaly detection** (IsolationForest) to flag unusual patient profiles
- **Model comparison** across four algorithms with ROC curves and confusion matrices
- **REST API** (FastAPI) with interactive docs
- **Docker** setup for the dashboard, API and retraining
- **CI** with GitHub Actions on Python 3.10, 3.12 and 3.13

## Quick start

You need **Python 3.10, 3.11, 3.12 or 3.13** and Git.

**1. Clone the repo**

```bash
git clone https://github.com/harmandeep6988/Medical-Disease-Risk-Prediction-System.git
cd Medical-Disease-Risk-Prediction-System
```

**2. Create a virtual environment and install the requirements**

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**3. Start the dashboard**

```bash
streamlit run dashboard/app.py
```

It opens at <http://localhost:8501>. The trained models and datasets are included in the repo, so it works straight away.

## REST API

```bash
uvicorn api.main:app --reload
```

- Interactive docs: <http://localhost:8000/docs>
- Health check: `GET /health`
- Predictions: `POST /predict/diabetes` and `POST /predict/heart`

Example:

```bash
curl -X POST http://localhost:8000/predict/heart \
  -H "Content-Type: application/json" \
  -d '{"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 2,
       "thalach": 130, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 1, "thal": 7}'
```

The response contains the risk probability, risk category and the top 3 SHAP factors. See [docs/API_GUIDE.md](docs/API_GUIDE.md) for every field.

## Run with Docker

No local Python setup needed, only [Docker](https://docs.docker.com/get-docker/):

```bash
docker compose up dashboard        # http://localhost:8501
docker compose up api              # http://localhost:8000/docs
```

## Retrain the models

Install the extra tools (training, data download, tests):

```bash
pip install -r requirements-dev.txt
python scripts/train_and_evaluate.py
```

This trains on `data/raw/diabetes_kaggle.csv` and `data/raw/heart_kaggle.csv` and overwrites the models in `models/saved_models/`. Model metadata records the reference dataset path, row count, SHA-256 hash, selected model, and evaluation metrics. Charts are written to `reports/generated/`.

To download fresh data from Kaggle, create an API token at [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**, and save it to `~/.kaggle/` (on Windows: `C:\Users\<you>\.kaggle\`). Never put the token inside this repo. New-style tokens (starting with `KGAT_`) need Python 3.11 or newer. Then run:

```bash
python scripts/fetch_kaggle_datasets.py
```

With Docker, `docker compose --profile training run --rm training` retrains and writes the new models to your local folders.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Project structure

```text
├── dashboard/app.py          Streamlit dashboard
├── api/                      FastAPI app (main.py) and request/response schemas
├── src/                      Core ML code
│   ├── data_preprocessing.py   cleaning, imputation, scaling, SMOTE
│   ├── feature_engineering.py  derived features, one-hot encoding, feature selection
│   ├── model_training.py       GridSearchCV over 4 model types
│   ├── model_evaluation.py     metrics, ROC curves, confusion matrices
│   ├── explainability.py       SHAP values and plots
│   ├── inference.py            preprocessing shared by the dashboard and the API
│   ├── anomaly_detection.py    IsolationForest outlier detection
│   └── report_generator.py     patient PDF reports
├── scripts/                  Training, data download and helper scripts
├── models/saved_models/      Trained models (.joblib)
├── data/                     Raw and processed datasets
├── reports/generated/        Evaluation and SHAP charts
├── notebooks/                Exploration notebook
├── tests/                    Unit and API tests
├── docs/                     Project report, workflow and API guide
├── requirements.txt          Packages needed to run the app and API
└── requirements-dev.txt      Extra packages for training and tests
```

## Troubleshooting

**`pip install` fails while building numpy, pandas or pydantic**
Your Python version is probably too new or too old. Check it with `python --version`; it must be 3.10–3.13. On Windows you can install 3.12 from [python.org](https://www.python.org/downloads/) and create the environment with `py -3.12 -m venv .venv`.

**Windows: `ImportError: DLL load failed ... An Application Control policy has blocked this file`**
Windows **Smart App Control** is blocking numpy's compiled files. Either run the project with Docker (see above), use WSL, or turn Smart App Control off in *Windows Security → App & browser control → Smart App Control settings*.

**`streamlit` or `uvicorn` is not recognized**
The virtual environment is not active. Run the activate command from step 2 again, or use `python -m streamlit run dashboard/app.py`.

**PowerShell says running scripts is disabled**
Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate the environment again.

## Tech stack

- **Machine learning:** scikit-learn, XGBoost, imbalanced-learn, SHAP
- **App:** Streamlit, Plotly, fpdf2
- **API:** FastAPI, Pydantic, Uvicorn
- **DevOps:** Docker, Docker Compose, GitHub Actions

## How it works

```text
Raw datasets ──► Cleaning + feature engineering ──► Train 4 models ──► Best model saved
                                                                          │
                                         ┌────────────────────────────────┤
                                         ▼                                ▼
                              Streamlit dashboard                    FastAPI REST API
                     (prediction, SHAP, PDF reports,              (JSON predictions with
                      anomaly detection, self-assessment)             SHAP factors)
```

More detail: [docs/PROJECT_WORKFLOW.md](docs/PROJECT_WORKFLOW.md) and [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md).
