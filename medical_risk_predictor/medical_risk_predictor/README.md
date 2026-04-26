# Medical Disease Risk Prediction System

An end-to-end Python project that predicts **Diabetes** and **Heart Disease** risk from patient health data, provides explainable AI insights, generates patient-friendly PDF reports, and serves a multi-tab interactive dashboard.

## Features

- 🧹 Data preprocessing pipeline (missing values, outliers, scaling, stratified split, SMOTE)
- 🧠 Feature engineering (BMI categories, age groups, interaction terms, correlation pruning, SelectKBest)
- 🤖 Multi-model training (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) with GridSearchCV
- 📊 Model evaluation (Accuracy, Precision, Recall, F1, AUC-ROC, ROC curves, confusion matrix, comparison chart)
- 🔍 Explainable AI with SHAP (beeswarm, waterfall, global importance, patient-specific explanation text)
- 📄 Automated PDF patient risk report generation
- 🚨 IsolationForest anomaly detection with reason tracing
- 📈 Temporal risk trend tracking with sudden spike alerts
- 🌐 Streamlit dashboard for prediction, explanation, model comparison, anomaly detection, and report download

## Project Structure

```text
medical_risk_predictor/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── explainability.py
│   ├── report_generator.py
│   ├── anomaly_detection.py
│   └── temporal_analysis.py
├── models/
│   └── saved_models/
├── reports/
│   └── generated/
├── dashboard/
│   └── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository and go to project root.
2. Create a Python 3.10+ environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

Use these datasets and place CSV files in `data/raw/`:

- Pima Indians Diabetes Dataset (Kaggle / UCI)
- UCI Heart Disease Dataset (Kaggle / UCI)

Suggested filenames:

- `diabetes.csv`
- `heart.csv`

## How To Run

### 1. Train Models (example script flow)

Use your preferred script/notebook to call:

1. `preprocess_data(...)` from `src/data_preprocessing.py`
2. `engineer_features(...)` from `src/feature_engineering.py`
3. `train_models(...)` from `src/model_training.py`
4. `evaluate_and_visualize(...)` from `src/model_evaluation.py`
5. `explain_model_predictions(...)` from `src/explainability.py`

Model artifacts are saved to `models/saved_models/`.

### 2. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Model Performance (Placeholder)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |

## Future Enhancements

- Add authenticated clinician login and role-based views
- Integrate live EHR/FHIR data ingestion
- Add calibration curves and threshold optimization
- Add drift monitoring and model retraining schedules
- Containerize with Docker and CI/CD pipeline

## CV / Resume Value

This project demonstrates:

- End-to-end ML system design (data to deployment)
- Healthcare-oriented risk modeling and explainability
- Practical MLOps artifact management and reporting
- Interactive product thinking via Streamlit dashboard
- Responsible analytics with anomaly and temporal monitoring
