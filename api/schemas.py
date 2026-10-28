"""
Pydantic schemas for FastAPI endpoints.
Defines request/response models for disease predictions.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DiabetesPredictionRequest(BaseModel):
    """Request schema for diabetes risk prediction."""
    pregnancies: int = Field(..., ge=0, le=17, description="Number of pregnancies (0-17)")
    glucose: float = Field(..., gt=0, description="Fasting blood glucose (mg/dL)")
    blood_pressure: float = Field(..., ge=0, description="Diastolic blood pressure (mmHg)")
    skin_thickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, description="2-hour serum insulin (mU/mL)")
    bmi: float = Field(..., gt=0, description="Body Mass Index (kg/m²)")
    diabetes_pedigree_function: float = Field(..., ge=0, description="Diabetes Pedigree Function (0-2.5)")
    age: int = Field(..., ge=18, le=100, description="Age in years (18-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 1,
                "glucose": 120,
                "blood_pressure": 80,
                "skin_thickness": 25,
                "insulin": 100,
                "bmi": 28.5,
                "diabetes_pedigree_function": 0.5,
                "age": 35
            }
        }


class HeartPredictionRequest(BaseModel):
    """Request schema for heart disease risk prediction."""
    age: int = Field(..., ge=18, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic)")
    trestbps: float = Field(..., gt=0, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dL)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 (0=No, 1=Yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG (0=Normal, 1=ST-T abnormal, 2=LV hypertrophy)")
    thalach: float = Field(..., gt=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (0=No, 1=Yes)")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of ST segment (1=Upsloping, 2=Flat, 3=Downsloping)")
    ca: int = Field(..., ge=0, le=3, description="Major vessels colored (0-3)")
    thal: int = Field(..., ge=3, le=7, description="Thalassemia (3=Normal, 6=Fixed defect, 7=Reversible defect)")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class SHAPFactor(BaseModel):
    """SHAP feature importance factor."""
    feature: str = Field(..., description="Feature name (plain English)")
    impact: float = Field(..., description="SHAP value (impact magnitude)")
    direction: str = Field(..., description="Direction of impact (increases/decreases risk)")

    class Config:
        json_schema_extra = {
            "example": {
                "feature": "Age",
                "impact": 0.25,
                "direction": "increases risk"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoints."""
    disease: str = Field(..., description="Disease name")
    risk_probability: float = Field(..., ge=0, le=1, description="Predicted risk probability (0-1)")
    risk_category: str = Field(..., description="Risk category (Low/Moderate/High)")
    risk_score_percentage: float = Field(..., ge=0, le=100, description="Risk as percentage (0-100)")
    confidence: str = Field(..., description="Confidence level in prediction")
    top_factors: List[SHAPFactor] = Field(..., description="Top 3 SHAP factors")
    model_name: str = Field(..., description="Model name used for prediction")

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
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
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
