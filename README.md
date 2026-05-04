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

- **Patient Self-Assessment Tab** — Interactive questionnaire with plain-English results (risk gauge, top-3 SHAP factors, risk card, PDF download)
  - Help expander explaining risk levels, SHAP factors, and how to interpret results
  - Input validation warnings for medically unlikely values (glucose=0, BMI=0, cholesterol=0)
- **SHAP Explainability** — Individual feature attribution for every prediction
- **FastAPI REST Endpoints** — Fully functional `/predict/diabetes` and `/predict/heart` routes with auto-generated Swagger UI
- **SMOTENC** — Balances class imbalance on mixed numeric + categorical data types
- **Anomaly Detection** — IsolationForest flags suspicious inputs
- **PDF Report Generation** — Downloadable clinical summary with predictions and explanations
- **5 Docker Profiles** — Dashboard, training, pipeline, data-prep, jupyter
- **Multi-Stage Docker Build** — 40% smaller container image
- **Non-Root User** — Enhanced security for healthcare data
- **Automated CI/CD** — GitHub Actions with graceful Codecov handling
- **Streamlit Dashboard** — Interactive inference and model comparison interface

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

### Option A — Streamlit Dashboard (Local)

```bash
pip install -r requirements.txt
python scripts/fetch_and_merge_datasets.py
streamlit run dashboard/app.py
```

Then navigate to **Patient Self-Assessment** tab to try the interactive prediction form.

### Option B — FastAPI Server

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

Access interactive API docs at **http://localhost:8000/docs** (Swagger UI).

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict/diabetes" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 1,
    "glucose": 120,
    "blood_pressure": 80,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 28.5,
    "diabetes_pedigree_function": 0.5,
    "age": 35
  }'
```

### Option C — Docker

```bash
cd docker
docker compose up dashboard
```

### Option D — Dashboard + API (Both)

**Terminal 1:**
```bash
bash scripts/run_all.py --no-dashboard
```

**Terminal 2:**
```bash
bash scripts/launch_api.sh
```

Both services run concurrently.

## Project Structure (2 Levels)

```text
Medical-Disease-Risk-Prediction-System/
├── api/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app & endpoints
│   ├── schemas.py                # Pydantic request/response models
│   └── API_GUIDE.md              # Comprehensive API documentation
├── dashboard/
│   └── app.py                    # Streamlit UI (all tabs)
├── data/
│   ├── processed/                # Cleaned & engineered datasets
│   └── raw/                      # Original CSV files
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── DOCKER_GUIDE.md
├── models/
│   └── saved_models/             # Trained models + metadata
├── notebooks/
│   └── exploration.ipynb         # Data exploration
├── reports/
│   └── generated/                # PDF reports (auto-generated)
├── scripts/
│   ├── fetch_and_merge_datasets.py # Merge OpenML/Kaggle/local sources
│   ├── launch_api.sh             # Start FastAPI server
│   ├── launch_dashboard.py       # Start Streamlit dashboard
│   ├── train_and_evaluate.py     # Model training pipeline
│   └── run_all.py                # Full ML workflow
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── explainability.py         # SHAP computation
│   ├── anomaly_detection.py      # IsolationForest
│   ├── report_generator.py       # PDF generation
│   └── temporal_analysis.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_feature_pipeline.py
├── .github/workflows/
│   └── test.yml                  # CI/CD pipeline
├── requirements.txt              # All dependencies
└── README.md
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
- **Data Source**: Kaggle API, OpenML

## Key ML Concepts

- **SHAP**: Explains individual predictions with feature attributions.
- **SMOTENC**: Balances classes when data includes mixed numeric + categorical features.
- **IsolationForest**: Flags anomalous inputs that may be unreliable for inference.

## Using the Patient Self-Assessment Tab

The Streamlit dashboard includes an interactive **Patient Self-Assessment** tab where users can:

1. **Read Help** — Expand "📚 How to Read Your Results" to learn about risk levels and SHAP factors in plain language
2. **Enter Data** — Fill in Diabetes or Heart Disease form with everyday language questions
3. **Get Instant Feedback** — Input validation warns about suspicious values (e.g., glucose=0) without blocking
4. **See Results** — Risk gauge, color-coded risk card (🟢/🟡/🔴), top-3 SHAP factors in plain English
5. **Download Report** — One-click PDF download with full prediction & explanation

### Example: Diabetes Self-Assessment

- Questions map directly to medical features (Glucose → "fasting blood sugar", BMI → "body weight ratio")
- BMI calculator helper to compute from weight & height
- Pregnancies question only shown if female
- DiabetesPedigreeFunction mapped to "family history" dropdown

### Validation & UX

- **Smart Warnings**: If you enter glucose=0 or BMI=0, a warning appears but allows submission (not medically likely, but might be a placeholder)
- **Result Persistence**: Results stay in session after form submission
- **PDF Download**: Includes patient data, prediction, SHAP explanations, risk score

---

## FastAPI Endpoints

Fully functional REST API with 3 endpoints:

### GET /health
Health check endpoint returning API status and which models are loaded.

### POST /predict/diabetes
Predicts diabetes risk from raw patient features (8 inputs).

**Request:**
```json
{
  "pregnancies": 1,
  "glucose": 120,
  "blood_pressure": 80,
  "skin_thickness": 25,
  "insulin": 100,
  "bmi": 28.5,
  "diabetes_pedigree_function": 0.5,
  "age": 35
}
```

**Response:**
```json
{
  "disease": "Diabetes",
  "risk_probability": 0.65,
  "risk_category": "Moderate",
  "risk_score_percentage": 65.0,
  "confidence": "High",
  "top_factors": [
    {
      "feature": "Glucose Level",
      "impact": 0.28,
      "direction": "increases risk"
    }
  ],
  "model_name": "XGBoost"
}
```

### POST /predict/heart
Predicts heart disease risk from raw patient features (13 inputs).

See [API_GUIDE.md](api/API_GUIDE.md) for complete documentation, examples, and request/response formats.

### Auto-Generated Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Screenshots

> Add images here:
>
> - `docs/images/dashboard.png` — Full dashboard with 5 tabs
> - `docs/images/self_assessment.png` — Patient Self-Assessment form
> - `docs/images/api_swagger.png` — FastAPI Swagger UI

## Cite / Contact

**How to cite** (example):

```text
Medical Disease Risk Prediction System, 2026. GitHub repository: https://github.com/OWNER/REPO
```

**Contact**: Your Name — your.email@example.com

## License

MIT (update if different).
