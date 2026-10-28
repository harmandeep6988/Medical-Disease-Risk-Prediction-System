# FastAPI Documentation

## Overview

The FastAPI backend provides REST endpoints for **Diabetes** and **Heart Disease** risk predictions using the trained machine learning models. All predictions include:
- Risk probability (0-1)
- Risk category (Low/Moderate/High)
- Top 3 SHAP-based feature importance factors
- Model metadata

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (if not already done)
```bash
python scripts/train_and_evaluate.py
```

### 3. Prepare Datasets
```bash
python scripts/prepare_datasets.py
```

### 4. Launch API
```bash
bash scripts/launch_api.sh
```

Or directly:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### Health Check
```
GET /health
```
Check API status and which models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "diabetes": true,
    "heart": true
  }
}
```

---

### Diabetes Risk Prediction
```
POST /predict/diabetes
```

**Request Body:**
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
    },
    {
      "feature": "BMI",
      "impact": 0.15,
      "direction": "increases risk"
    },
    {
      "feature": "Age",
      "impact": 0.12,
      "direction": "increases risk"
    }
  ],
  "model_name": "LogisticRegression"
}
```

---

### Heart Disease Risk Prediction
```
POST /predict/heart
```

**Request Body:**
```json
{
  "age": 45,
  "sex": 1,
  "cp": 3,
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 0.5,
  "slope": 2,
  "ca": 0,
  "thal": 3
}
```

**Response:**
```json
{
  "disease": "Heart Disease",
  "risk_probability": 0.58,
  "risk_category": "Moderate",
  "risk_score_percentage": 58.0,
  "confidence": "High",
  "top_factors": [
    {
      "feature": "Age",
      "impact": 0.25,
      "direction": "increases risk"
    },
    {
      "feature": "Cholesterol Level",
      "impact": 0.18,
      "direction": "increases risk"
    },
    {
      "feature": "Max Heart Rate",
      "impact": 0.14,
      "direction": "decreases risk"
    }
  ],
  "model_name": "RandomForestClassifier"
}
```

---

## Request Parameters

### Diabetes Features

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `pregnancies` | int | 0-17 | Number of pregnancies |
| `glucose` | float | > 0 | Fasting blood glucose (mg/dL) |
| `blood_pressure` | float | ≥ 0 | Diastolic blood pressure (mmHg) |
| `skin_thickness` | float | ≥ 0 | Triceps skin fold thickness (mm) |
| `insulin` | float | ≥ 0 | 2-hour serum insulin (mU/mL) |
| `bmi` | float | > 0 | Body Mass Index (kg/m²) |
| `diabetes_pedigree_function` | float | ≥ 0 | Diabetes Pedigree Function (0-2.5) |
| `age` | int | 18-100 | Age in years |

### Heart Disease Features

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `age` | int | 18-120 | Age in years |
| `sex` | int | 0-1 | 0=Female, 1=Male |
| `cp` | int | 1-4 | Chest pain type (1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic) |
| `trestbps` | float | > 0 | Resting blood pressure (mmHg) |
| `chol` | float | ≥ 0 | Serum cholesterol (mg/dL) |
| `fbs` | int | 0-1 | Fasting blood sugar > 120 (0=No, 1=Yes) |
| `restecg` | int | 0-2 | Resting ECG (0=Normal, 1=ST-T abnormal, 2=LV hypertrophy) |
| `thalach` | float | > 0 | Maximum heart rate achieved |
| `exang` | int | 0-1 | Exercise-induced angina (0=No, 1=Yes) |
| `oldpeak` | float | ≥ 0 | ST depression induced by exercise |
| `slope` | int | 1-3 | Slope of ST segment (1=Upsloping, 2=Flat, 3=Downsloping) |
| `ca` | int | 0-3 | Major vessels colored (0-3) |
| `thal` | int | 3/6/7 | Thalassemia (3=Normal, 6=Fixed defect, 7=Reversible defect) |

---

## Risk Categories

| Category | Threshold | Interpretation |
|----------|-----------|-----------------|
| Low | < 0.4 | Risk probability below 40% |
| Moderate | 0.4 - 0.7 (inclusive) | Risk probability from 40% through 70% |
| High | > 0.7 | Risk probability above 70% |

---

## Response Fields

- **risk_probability**: Model's predicted probability (0-1)
- **risk_category**: Simplified risk level (Low/Moderate/High)
- **risk_score_percentage**: Risk as percentage (0-100)
- **confidence**: Model's confidence in prediction (Low/Moderate/High)
- **top_factors**: Top 3 SHAP-based feature influences
- **model_name**: Python classifier class used for the prediction

---

## Error Handling

### 503 - Service Unavailable
```json
{
  "detail": "Diabetes model not loaded"
}
```
Models haven't been loaded. Ensure:
- Models exist in `models/saved_models/`
- Processed data exists in `data/processed/`

### 400 - Bad Request
```json
{
  "detail": "Failed to process patient data"
}
```
Invalid input data or feature transformation failed.

### 422 - Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error.number.not_ge"
    }
  ]
}
```
Input validation failed. Check parameter types and ranges.

---

## Example Usage with cURL

### Diabetes Prediction
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

### Heart Disease Prediction
```bash
curl -X POST "http://localhost:8000/predict/heart" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "cp": 3,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 0.5,
    "slope": 2,
    "ca": 0,
    "thal": 3
  }'
```

---

## Example Usage with Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Diabetes Prediction
diabetes_data = {
    "pregnancies": 1,
    "glucose": 120,
    "blood_pressure": 80,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 28.5,
    "diabetes_pedigree_function": 0.5,
    "age": 35
}

response = requests.post(
    f"{BASE_URL}/predict/diabetes",
    json=diabetes_data
)

print(response.json())
```

---

## Architecture

The FastAPI backend:
1. **Loads models** from `models/saved_models/` on startup
2. **Transforms raw input** using the same feature engineering pipeline as training
3. **Makes predictions** using the trained ML model
4. **Computes SHAP values** for model explainability
5. **Returns structured response** with risk and factors

---

## Performance Notes

- **Model Loading**: ~2-3 seconds on startup
- **Single Prediction**: ~200-500ms (includes SHAP computation)
- **SHAP Background**: Uses up to 200 samples for efficiency
- **Concurrency**: Supports multiple concurrent requests

---

## Troubleshooting

### Models Not Loading
1. Ensure models exist: `ls models/saved_models/`
2. Run training: `python scripts/train_and_evaluate.py`
3. Check logs for import errors

### Slow Predictions
- SHAP computation can be slow with large background datasets
- API automatically limits background to 200 samples

### Feature Transformation Errors
- Verify input ranges match specification
- Check for NaN values in request
- Ensure categorical features match expected codes

---

## Integration with Streamlit Dashboard

Both the API and Streamlit dashboard share:
- Same trained models
- Same feature engineering pipeline
- Same SHAP computation logic

They can be run simultaneously:
```bash
# Terminal 1: Start API
bash scripts/launch_api.sh

# Terminal 2: Start Dashboard
bash scripts/launch_dashboard.py
```

---

## Contributing

To add new disease predictions:
1. Train model in `scripts/train_and_evaluate.py`
2. Save model and metadata in `models/saved_models/`
3. Add request/response schema in `api/schemas.py`
4. Add prediction endpoint in `api/main.py`
