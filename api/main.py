"""
FastAPI application for medical disease predictions.
Provides REST endpoints for Diabetes and Heart Disease risk assessment.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from explainability import compute_shap_values
from inference import (
    fit_reference_transform,
    transform_dataset_to_model_features,
    transform_raw_to_model_features,
)
from api.schemas import (
    DiabetesPredictionRequest,
    HeartPredictionRequest,
    PredictionResponse,
    SHAPFactor,
    HealthResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved_models"
DATA_DIR = BASE_DIR / "data" / "processed"


class ModelContext:
    """Context manager for loaded models and metadata."""

    def __init__(self):
        self.models = {}
        self.metadata = {}
        self.reference_data = {}
        self.transforms = {}
        self.backgrounds = {}

    def load_models(self):
        """Load all models and metadata."""
        try:
            # Load Diabetes model
            if (MODELS_DIR / "best_diabetes.joblib").exists():
                self.models["diabetes"] = joblib.load(
                    MODELS_DIR / "best_diabetes.joblib"
                )
                self.metadata["diabetes"] = joblib.load(
                    MODELS_DIR / "best_diabetes_meta.joblib"
                )
                logger.info("✓ Diabetes model loaded")
            else:
                logger.warning("✗ Diabetes model not found")

            # Load Heart model
            if (MODELS_DIR / "best_heart.joblib").exists():
                self.models["heart"] = joblib.load(MODELS_DIR / "best_heart.joblib")
                self.metadata["heart"] = joblib.load(
                    MODELS_DIR / "best_heart_meta.joblib"
                )
                logger.info("✓ Heart Disease model loaded")
            else:
                logger.warning("✗ Heart Disease model not found")

            # Load reference datasets for SHAP
            if (DATA_DIR / "diabetes_processed.csv").exists():
                df = pd.read_csv(DATA_DIR / "diabetes_processed.csv")
                self.reference_data["diabetes"] = df
                logger.info(f"✓ Diabetes reference data loaded ({len(df)} samples)")

            if (DATA_DIR / "heart_processed.csv").exists():
                df = pd.read_csv(DATA_DIR / "heart_processed.csv")
                self.reference_data["heart"] = df
                logger.info(f"✓ Heart reference data loaded ({len(df)} samples)")

            # Fit the same scaling/encoding the dashboard uses, once at startup
            for disease, meta in self.metadata.items():
                ref_data = self.reference_data.get(disease)
                if ref_data is None:
                    continue
                transform = fit_reference_transform(
                    ref_data,
                    meta["target"],
                    meta.get("invalid_zero_columns", []),
                    categorical_columns=meta.get("categorical_columns", []),
                )
                X_bg, _ = transform_dataset_to_model_features(
                    ref_data,
                    meta["target"],
                    transform,
                    meta["features"],
                    meta.get("invalid_zero_columns", []),
                )
                self.transforms[disease] = transform
                self.backgrounds[disease] = X_bg.sample(n=min(200, len(X_bg)), random_state=42)

            return len(self.models) > 0

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def are_models_ready(self) -> Dict[str, bool]:
        """Check which models are loaded."""
        return {
            "diabetes": "diabetes" in self.models,
            "heart": "heart" in self.models,
        }


# Global context
model_context = ModelContext()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup/shutdown."""
    # Startup
    logger.info("Starting Medical Disease Risk Prediction API...")
    model_context.load_models()
    yield
    # Shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="Medical Disease Risk Prediction API",
    description="REST API for predicting Diabetes and Heart Disease risks",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _transform_raw_to_model_features(
    raw_patient: Dict[str, float], disease: str
) -> Optional[pd.DataFrame]:
    """Transform raw patient data to engineered features matching model input."""
    try:
        meta = model_context.metadata.get(disease)
        transform = model_context.transforms.get(disease)

        if meta is None or transform is None:
            return None

        return transform_raw_to_model_features(
            pd.DataFrame([raw_patient]),
            transform,
            meta["features"],
            meta.get("invalid_zero_columns", []),
        )

    except Exception as e:
        logger.error(f"Error transforming features for {disease}: {e}")
        return None


def _get_shap_factors(
    disease: str, patient_engineered: pd.DataFrame
) -> Optional[list]:
    """Compute SHAP factors for top 3 features."""
    try:
        model = model_context.models.get(disease)
        meta = model_context.metadata.get(disease)
        X_bg = model_context.backgrounds.get(disease)

        if model is None or meta is None or X_bg is None:
            return None

        target_features = meta["features"]
        _, explanation = compute_shap_values(model, X_bg, patient_engineered)
        shap_values = np.asarray(explanation.values)

        # Handle 3D array (binary classification)
        if shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values

        # Get absolute impacts
        impacts = np.abs(shap_vals[0])
        top_indices = np.argsort(impacts)[-3:][::-1]

        factors = []
        for idx in top_indices:
            feature_name = target_features[idx]
            impact_value = float(shap_vals[0, idx])
            direction = "increases" if impact_value > 0 else "decreases"

            # Map engineered feature names to plain English
            plain_feature = _feature_to_english(feature_name, disease)

            factors.append(
                SHAPFactor(
                    feature=plain_feature,
                    impact=abs(impact_value),
                    direction=f"{direction} risk",
                )
            )

        return factors

    except Exception as e:
        logger.error(f"Error computing SHAP factors for {disease}: {e}")
        return None


def _feature_to_english(feature: str, disease: str) -> str:
    """Map engineered/raw feature names to plain English."""
    # Basic mapping
    diabetes_mapping = {
        "Pregnancies": "Number of Pregnancies",
        "Glucose": "Glucose Level",
        "BloodPressure": "Blood Pressure",
        "SkinThickness": "Skin Thickness",
        "Insulin": "Insulin Level",
        "BMI": "Body Mass Index (BMI)",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
        "Age": "Age",
        "BMI_Category_High": "BMI Category (High)",
        "BMI_Category_Normal": "BMI Category (Normal)",
        "BMI_Category_Obese": "BMI Category (Obese)",
        "BMI_Category_Overweight": "BMI Category (Overweight)",
        "Age_Group_Adult": "Age Group (Adult)",
        "Age_Group_MiddleAged": "Age Group (Middle-Aged)",
        "Age_Group_Senior": "Age Group (Senior)",
        "Glucose_Insulin_Interaction": "Glucose-Insulin Interaction",
    }

    heart_mapping = {
        "age": "Age",
        "sex": "Sex",
        "sex_1": "Sex (Male)",
        "cp": "Chest Pain Type",
        "cp_1": "Chest Pain (Typical Angina)",
        "cp_2": "Chest Pain (Atypical Angina)",
        "cp_3": "Chest Pain (Non-anginal)",
        "cp_4": "Chest Pain (Asymptomatic)",
        "trestbps": "Resting Blood Pressure",
        "chol": "Cholesterol Level",
        "fbs": "Fasting Blood Sugar > 120",
        "restecg": "Resting ECG",
        "thalach": "Max Heart Rate",
        "exang": "Exercise-Induced Angina",
        "exang_1": "Exercise-Induced Angina (Yes)",
        "oldpeak": "ST Depression (Oldpeak)",
        "slope": "ST Segment Slope",
        "slope_1": "ST Slope (Upsloping)",
        "slope_2": "ST Slope (Flat)",
        "slope_3": "ST Slope (Downsloping)",
        "ca": "Major Vessels",
        "thal": "Thalassemia Type",
    }

    mapping = diabetes_mapping if disease == "diabetes" else heart_mapping
    return mapping.get(feature, feature)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=model_context.are_models_ready(),
    )


@app.post("/predict/diabetes", response_model=PredictionResponse)
async def predict_diabetes(request: DiabetesPredictionRequest):
    """Predict Diabetes risk."""
    if model_context.models.get("diabetes") is None:
        raise HTTPException(status_code=503, detail="Diabetes model not loaded")

    try:
        # Convert request to dict
        patient_data = {
            "Pregnancies": request.pregnancies,
            "Glucose": request.glucose,
            "BloodPressure": request.blood_pressure,
            "SkinThickness": request.skin_thickness,
            "Insulin": request.insulin,
            "BMI": request.bmi,
            "DiabetesPedigreeFunction": request.diabetes_pedigree_function,
            "Age": request.age,
        }

        # Transform to model features
        X_patient = _transform_raw_to_model_features(patient_data, "diabetes")
        if X_patient is None:
            raise HTTPException(
                status_code=400, detail="Failed to process patient data"
            )

        # Make prediction
        model = model_context.models["diabetes"]
        pred_proba = model.predict_proba(X_patient)[0, 1]

        # Get SHAP factors
        shap_factors = _get_shap_factors("diabetes", X_patient)
        if shap_factors is None:
            shap_factors = []

        # Determine risk category
        if pred_proba < 0.4:
            risk_category = "Low"
        elif pred_proba < 0.7:
            risk_category = "Moderate"
        else:
            risk_category = "High"

        confidence = "High" if pred_proba > 0.8 else "Moderate" if pred_proba > 0.3 else "Low"

        # Get model name
        model_name = type(model).__name__

        return PredictionResponse(
            disease="Diabetes",
            risk_probability=float(pred_proba),
            risk_category=risk_category,
            risk_score_percentage=float(pred_proba * 100),
            confidence=confidence,
            top_factors=shap_factors if shap_factors else [],
            model_name=model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/heart", response_model=PredictionResponse)
async def predict_heart(request: HeartPredictionRequest):
    """Predict Heart Disease risk."""
    if model_context.models.get("heart") is None:
        raise HTTPException(status_code=503, detail="Heart Disease model not loaded")

    try:
        # Convert request to dict
        patient_data = {
            "age": request.age,
            "sex": request.sex,
            "cp": request.cp,
            "trestbps": request.trestbps,
            "chol": request.chol,
            "fbs": request.fbs,
            "restecg": request.restecg,
            "thalach": request.thalach,
            "exang": request.exang,
            "oldpeak": request.oldpeak,
            "slope": request.slope,
            "ca": request.ca,
            "thal": request.thal,
        }

        # Transform to model features
        X_patient = _transform_raw_to_model_features(patient_data, "heart")
        if X_patient is None:
            raise HTTPException(
                status_code=400, detail="Failed to process patient data"
            )

        # Make prediction
        model = model_context.models["heart"]
        pred_proba = model.predict_proba(X_patient)[0, 1]

        # Get SHAP factors
        shap_factors = _get_shap_factors("heart", X_patient)
        if shap_factors is None:
            shap_factors = []

        # Determine risk category
        if pred_proba < 0.4:
            risk_category = "Low"
        elif pred_proba < 0.7:
            risk_category = "Moderate"
        else:
            risk_category = "High"

        confidence = "High" if pred_proba > 0.8 else "Moderate" if pred_proba > 0.3 else "Low"

        # Get model name
        model_name = type(model).__name__

        return PredictionResponse(
            disease="Heart Disease",
            risk_probability=float(pred_proba),
            risk_category=risk_category,
            risk_score_percentage=float(pred_proba * 100),
            confidence=confidence,
            top_factors=shap_factors if shap_factors else [],
            model_name=model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
