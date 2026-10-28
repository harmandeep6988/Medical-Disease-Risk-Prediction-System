# Project Workflow — Medical Disease Risk Prediction System

This document describes the end-to-end workflow, which modules implement each step, and what to upgrade next.

## End-to-End Flow
1. **Dataset acquisition** — `scripts/prepare_datasets.py`
   - Fetches Diabetes and Heart datasets (OpenML first; URL fallback for heart).
   - Writes to `data/raw/` and `data/processed/`.
2. **Preprocessing** — `src/data_preprocessing.py`
   - Zero-invalid columns -> NaN, median imputation, IQR outlier clipping.
   - Stratified split; SMOTE/SMOTENC for imbalance.
   - Scaling: StandardScaler then MinMaxScaler (categoricals passthrough).
3. **Feature engineering** — `src/feature_engineering.py`
   - Derived features (BMI categories, age groups, interactions).
   - One-hot encoding + correlation pruning + SelectKBest.
4. **Model training** — `src/model_training.py`
   - GridSearchCV over LogisticRegression, RandomForest, GradientBoosting, XGBoost.
   - CV scoring uses ROC-AUC.
5. **Evaluation + artifacts** — `src/model_evaluation.py`
   - Accuracy/Precision/Recall/F1/AUC + ROC + confusion matrix + comparison chart.
6. **Explainability** — `src/explainability.py`
   - SHAP plots (beeswarm, waterfall, global bar) and human-readable explanation.
7. **Anomaly detection** — `src/anomaly_detection.py`
   - IsolationForest + PCA scatter for outlier profiles.
8. **UI dashboard** — `dashboard/app.py`
   - Streamlit app for inference, SHAP, anomaly view, and PDF report generation.
9. **PDF clinical report** — `src/report_generator.py`
   - Creates a patient-facing PDF with risk band, top SHAP factors, and next steps.

## Tech Used (What + How)
- **Pandas/NumPy**: dataset IO + vectorized feature work
- **Scikit-learn**: preprocessing, SMOTE/SMOTENC integration, GridSearchCV, evaluation metrics
- **XGBoost / GradientBoosting / RandomForest / LogisticRegression**: candidate classifiers
- **SHAP**: local + global feature attribution plots
- **IsolationForest**: unsupervised outlier detection for reliability
- **Matplotlib/Seaborn**: static evaluation plots
- **Streamlit**: interactive dashboard for inference + report download
- **Docker/Docker Compose**: containerized dashboard/training/pipeline/notebook workflows
- **GitHub Actions**: CI test run on Python 3.12 (`.github/workflows/test.yml`)

## Upgrade Checklist (Recommended)
- **Add Liver Disease module**: dataset, preprocessing config, model training/eval, Streamlit tab
- **FastAPI service**: implemented in `api/`, with Pydantic schemas and `/predict/{disease}` endpoints
- **Prevent artifact overwrites**: prefix plot/model filenames per disease in training scripts
- **Calibration & thresholds**: calibrate probabilities and validate the current 40%/70% display thresholds before any clinical use
- **Data validation**: schema checks, value ranges, and missingness alerts
- **Model registry/versioning**: track dataset hash + params + metrics per model build
- **Monitoring**: drift detection + periodic retraining triggers
