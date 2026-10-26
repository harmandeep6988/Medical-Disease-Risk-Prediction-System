# Project Run Report — Medical Disease Risk Prediction System

Generated: 2026-04-28 22:38:55

> **Note:** This report records earlier training runs from April 2026. The models now shipped in `models/saved_models/` were trained on `data/raw/diabetes_kaggle.csv` (392 records) and `data/raw/heart_kaggle.csv` (633 records) and score **Diabetes AUC 0.871 / 75.9% accuracy** (Logistic Regression) and **Heart Disease AUC 0.909 / 85.0% accuracy** (Random Forest). See the [README](../README.md#results) for the current numbers.

## What Was Run
- Data prep: `scripts/prepare_datasets.py` (OpenML/URL fallback)
- Training + evaluation: `scripts/train_and_evaluate.py` (grid search + ROC/AUC ranking)
- This report: `scripts/generate_project_docs.py` (evaluates saved best models + generates artifacts)

## Environment
- OS: `Linux 6.17.0-22-generic`
- Python: `3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]`
- Key packages:
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `scikit-learn==1.5.1`
  - `xgboost==2.1.1`
  - `shap==0.46.0`
  - `matplotlib==3.9.2`
  - `seaborn==0.13.2`
  - `imbalanced-learn==0.12.3`
  - `streamlit==1.39.0`
  - `fpdf2==2.8.1`

## Results (Best Saved Models)
| Disease | AUC | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Diabetes | 0.826 | 76.0% | 0.689 | 0.631 | 0.759 |
| Heart Disease | 0.938 | 88.3% | 0.873 | 0.889 | 0.857 |

## Parameters (Pipeline)
- Split: stratified `test_size=0.2`, `random_state=42`
- Missing values: median imputation (numeric)
- Medical 0 handling: selected columns -> `NaN` then imputed
- Outliers: IQR clipping (1.5×)
- Scaling: StandardScaler -> MinMaxScaler (categoricals passthrough)
- Imbalance: SMOTE or SMOTENC depending on categorical columns
- Feature engineering: BMI/age bucketing + interactions + one-hot encoding
- Training: `GridSearchCV` with 5-fold CV, scoring = `roc_auc`, `n_jobs=-1`

## Diabetes
- Dataset:
  - Path: `data/processed/diabetes_processed.csv`
  - Shape: `768 rows × 9 cols`
  - Target: `Outcome`
  - Test positive rate: `35.1%`
- Preprocessing:
  - Zero-invalid columns: `Glucose, BloodPressure, SkinThickness, Insulin, BMI`
  - Categorical passthrough: `(none)`
  - Balancing: `SMOTE`
- Best saved model:
  - Type: `GradientBoostingClassifier`
  - Artifact: `models/saved_models/best_diabetes.joblib`
  - Features used: `8`
  - Key hyperparameters:
    - `learning_rate`: `0.1`
    - `max_depth`: `3`
    - `min_samples_leaf`: `1`
    - `min_samples_split`: `2`
    - `n_estimators`: `100`
    - `random_state`: `42`
    - `subsample`: `1.0`
- Explainability:
  - Example explanation: Top risk factors for this patient: Glucose (up risk by 242.4%), BMI (down risk by 91.0%), Glucose_Insulin_Ratio (up risk by 20.2%). Snapshot: Glucose=0.7202797202797202, SkinThickness=0.5178571428571429, Insulin=0.5271739130434783, BMI=0.28705148205928227, Age=0.4175824175824177.
  - SHAP artifacts:
    - `beeswarm`: `reports/generated/shap_summary_beeswarm_diabetes.png`
    - `waterfall`: `reports/generated/shap_waterfall_patient_diabetes.png`
    - `bar`: `reports/generated/shap_global_bar_diabetes.png`
- Evaluation artifacts:
  - `roc_curves`: `reports/generated/roc_curves_diabetes.png`
  - `confusion_matrix`: `reports/generated/best_model_confusion_matrix_diabetes.png`
  - `comparison_chart`: `reports/generated/model_comparison_diabetes.png`
- Anomaly detection plot: `reports/generated/anomaly_scatter_diabetes.png`
- Sample PDF report: `reports/generated/diabetes_patient_report_20260428_223857.pdf`

## Heart Disease
- Dataset:
  - Path: `data/processed/heart_processed.csv`
  - Shape: `297 rows × 14 cols`
  - Target: `target`
  - Test positive rate: `46.7%`
- Preprocessing:
  - Zero-invalid columns: `chol, trestbps, thalach`
  - Categorical passthrough: `sex, cp, fbs, restecg, exang, slope, ca, thal`
  - Balancing: `SMOTENC`
- Best saved model:
  - Type: `XGBClassifier`
  - Artifact: `models/saved_models/best_heart.joblib`
  - Features used: `10`
  - Key hyperparameters:
    - `colsample_bytree`: `None`
    - `learning_rate`: `0.05`
    - `max_depth`: `3`
    - `n_estimators`: `100`
    - `random_state`: `42`
    - `subsample`: `0.8`
- Explainability:
  - Example explanation: Top risk factors for this patient: cp_4 (down risk by 77.6%), age (down risk by 76.9%), thal_7 (down risk by 67.9%). Snapshot: age=0.16279069767441862, thalach=0.8785046728971964, oldpeak=0.0, sex_1=True, cp_3=False.
  - SHAP artifacts:
    - `beeswarm`: `reports/generated/shap_summary_beeswarm_heart.png`
    - `waterfall`: `reports/generated/shap_waterfall_patient_heart.png`
    - `bar`: `reports/generated/shap_global_bar_heart.png`
- Evaluation artifacts:
  - `roc_curves`: `reports/generated/roc_curves_heart.png`
  - `confusion_matrix`: `reports/generated/best_model_confusion_matrix_heart.png`
  - `comparison_chart`: `reports/generated/model_comparison_heart.png`
- Anomaly detection plot: `reports/generated/anomaly_scatter_heart.png`
- Sample PDF report: `reports/generated/heart_patient_report_20260428_223859.pdf`

## Notes / Gaps
- This repo currently runs **Diabetes** + **Heart Disease**. Liver disease + FastAPI endpoints are not present in the codebase yet.
- Plot filenames in the training script are generic (risk of overwrite). This script renames per-disease artifacts to preserve both.

