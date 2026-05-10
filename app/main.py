"""Prediction module - FastAPI version with SHAP explainability"""

import os
import pickle

import mlflow
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from pymongo import MongoClient

import shap
import numpy as np

# -------------------------------------------------------------------
# Configuration — same environment variables as before
# -------------------------------------------------------------------
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "maternal-health-risk")
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "False") == "True"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_MODEL_ENABLED = os.getenv("DEFAULT_MODEL_ENABLED", "True") == "True"
MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "False") == "True"
EVIDENTLY_SERVICE_URI = os.getenv("EVIDENTLY_SERVICE_URI", "http://localhost:8085")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

if not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

if MONITORING_ENABLED:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client.get_database("prediction_service")
    collection = db.get_collection(EXPERIMENT_NAME)

# -------------------------------------------------------------------
# Pydantic input model — replaces validate_data() entirely
# FastAPI automatically returns a 422 error if any field is invalid
# -------------------------------------------------------------------
from pydantic import BaseModel, Field, model_validator

class PatientData(BaseModel):
    Age: int = Field(..., ge=13, le=50)
    SystolicBP: int = Field(..., ge=50, le=200)
    DiastolicBP: int = Field(..., ge=50, le=200)
    BS: float = Field(..., ge=0.1, le=15.0)
    BodyTemp: float = Field(..., ge=34.0, le=41.0)
    HeartRate: int = Field(..., ge=45, le=130)

    @model_validator(mode="after")
    def systolic_must_exceed_diastolic(self):
        if self.SystolicBP <= self.DiastolicBP:
            raise ValueError("Systolic BP must be higher than Diastolic BP")
        return self

# -------------------------------------------------------------------
# Model loading — identical logic to predict.py
# -------------------------------------------------------------------
def load_model_from_registry():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{EXPERIMENT_NAME}/latest"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Loaded model from MLflow registry")
    return loaded_model


def load_default_model():
    model_path = f"{os.path.dirname(os.path.abspath(__file__))}/model.bin"
    with open(model_path, "rb") as f_in:
        loaded_model = pickle.load(f_in)
    print(f"Loaded default model from disk")
    return loaded_model


def load_model():
    try:
        if MLFLOW_ENABLED:
            return load_model_from_registry()
        if DEFAULT_MODEL_ENABLED:
            return load_default_model()
    except:
        if DEFAULT_MODEL_ENABLED:
            return load_default_model()
    return None


# -------------------------------------------------------------------
# Prediction helpers — identical logic to predict.py
# -------------------------------------------------------------------
def predict(record: dict):
    df = pd.DataFrame([record])
    try:
        raw = model._predict(df, params=None)
        preds = [round(x) for x in raw.values.flatten()]
    except Exception:
        preds = [round(x) for x in model.predict(df)]
    return preds[0]


def convert_risk(pred: int):
    if pred == 0:
        return "high risk"
    if pred == 1:
        return "low risk"
    return "mid risk"


def save_to_db(record: dict, risk: str):
    rec = record.copy()
    rec["RiskLevel"] = risk
    collection.insert_one(rec)


def send_to_evidently_service(record: dict, risk: str):
    rec = record.copy()
    rec["RiskLevel"] = risk
    requests.post(
        f"{EVIDENTLY_SERVICE_URI}/iterate/maternal-health-risk",
        json=[rec]
    )


def calculate_risk(record: dict):
    pred = predict(record)
    risk = convert_risk(pred)
    explanation = explain_prediction(record, pred)
    interpretation = interpret_prediction(record, explanation)
    if MONITORING_ENABLED:
        save_to_db(record, risk)
        send_to_evidently_service(record, risk)
    return risk, explanation, interpretation


FEATURE_NAMES = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

def explain_prediction(record: dict, pred: int):
    df = pd.DataFrame([record])
    shap_values = explainer.shap_values(df)
    # For RandomForest multiclass, shap_values is a list of arrays
    # Shape: list of [n_samples x n_features] per class
    # Clamp pred index to available classes
    class_index = min(pred, len(shap_values) - 1)
    values = shap_values[class_index][0]
    explanation = {
        feature: round(float(value), 4)
        for feature, value in zip(FEATURE_NAMES, values)
    }
    explanation = dict(
        sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    return explanation


def interpret_prediction(record: dict, explanation: dict):
    """
    Converts raw SHAP values and actual feature values into
    clinically grounded plain language interpretation.
    Uses actual feature value against clinical thresholds
    combined with SHAP sign — not raw magnitude alone.
    """
    increasing = []
    decreasing = []

    for feature, shap_value in explanation.items():
        value = record[feature]
        thresh = CLINICAL_THRESHOLDS.get(feature)
        display = FEATURE_DISPLAY_NAMES.get(feature, feature)

        if thresh is None:
            continue

        # Determine clinical status from actual value
        if feature == "SystolicBP":
            if value >= thresh["high"]:
                status = thresh["high_label"]
            elif value >= thresh.get("elevated_max", thresh["normal_max"]):
                status = thresh.get("elevated_label", "elevated range")
            elif value < thresh["low"]:
                status = thresh["low_label"]
            else:
                status = "within normal range"

        elif feature == "BS":
            if value >= thresh["high"]:
                status = thresh["high_label"]
            elif value >= thresh.get("elevated_max", thresh["normal_max"]):
                status = thresh.get("elevated_label", "elevated range")
            elif value < thresh["low"]:
                status = thresh["low_label"]
            else:
                status = "within normal range"

        elif feature == "Age":
            if value > thresh["normal_max"]:
                status = thresh["high_label"]
            elif value <= thresh["low"] + 2:
                status = thresh["low_label"]
            else:
                status = "within normal range"

        else:
            if value > thresh["normal_max"]:
                status = thresh["high_label"]
            elif value < thresh["low"]:
                status = thresh["low_label"]
            else:
                status = "within normal range"

        # Use SHAP sign to determine direction of effect
        if shap_value > 0:
            increasing.append(
                f"{display} is {status} ({value} {thresh['unit']})"
            )
        elif shap_value < 0:
            decreasing.append(
                f"{display} is {status} ({value} {thresh['unit']})"
            )

    # Build summary from top increasing driver
    if increasing:
        top = increasing[0]
        summary = f"Risk is primarily driven by {top.lower()}."
    else:
        summary = "No single dominant risk factor identified."

    return {
        "increasing_risk": increasing,
        "decreasing_risk": decreasing,
        "summary": summary
    }


# Clinical reference ranges for interpretation
# Sources: WHO maternal health guidelines and standard clinical ranges
CLINICAL_THRESHOLDS = {
    "Age": {
        "low": 13,
        "normal_max": 35,
        "high": 50,
        "unit": "years",
        "low_label": "adolescent pregnancy",
        "high_label": "advanced maternal age"
    },
    "SystolicBP": {
        "low": 90,
        "normal_max": 120,
        "elevated_max": 129,
        "high": 130,
        "unit": "mmHg",
        "low_label": "hypotensive range",
        "elevated_label": "elevated range",
        "high_label": "hypertensive range"
    },
    "DiastolicBP": {
        "low": 60,
        "normal_max": 80,
        "high": 90,
        "unit": "mmHg",
        "low_label": "hypotensive range",
        "high_label": "hypertensive range"
    },
    "BS": {
        "low": 0.1,
        "normal_max": 6.1,
        "elevated_max": 7.8,
        "high": 7.8,
        "unit": "mmol/L",
        "low_label": "hypoglycemic range",
        "elevated_label": "pre-diabetic range",
        "high_label": "diabetic range"
    },
    "BodyTemp": {
        "low": 36.0,
        "normal_max": 37.5,
        "high": 37.5,
        "unit": "Celsius",
        "low_label": "below normal range",
        "high_label": "febrile range"
    },
    "HeartRate": {
        "low": 60,
        "normal_max": 100,
        "high": 100,
        "unit": "bpm",
        "low_label": "bradycardic range",
        "high_label": "tachycardic range"
    }
}

FEATURE_DISPLAY_NAMES = {
    "Age": "Age",
    "SystolicBP": "Systolic blood pressure",
    "DiastolicBP": "Diastolic blood pressure",
    "BS": "Blood glucose",
    "BodyTemp": "Body temperature",
    "HeartRate": "Heart rate"
}


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Maternal Health Risk Predictor",
    description="Predicts maternal health risk level based on patient vitals. Returns SHAP feature contributions and clinical interpretation per prediction.",
    version="1.0.0"
)

model = load_model()

explainer = shap.TreeExplainer(model)


@app.get("/", response_class=HTMLResponse)
def root():
    """Health check and welcome endpoint"""
    return """
    <html>
        <body>
            <h2>Maternal Health Risk Predictor API</h2>
            <p>Status: Running</p>
            <p><a href="/docs">View API Documentation</a></p>
        </body>
    </html>
    """


@app.post("/predict")
def predict_endpoint(patient: PatientData):
    """
    Accepts patient vitals and returns maternal health risk level
    with SHAP feature contributions and clinical interpretation.
    """
    record = patient.model_dump()
    risk, explanation, interpretation = calculate_risk(record)
    return {
        "RiskLevel": risk,
        "explanation": explanation,
        "interpretation": interpretation
    }


@app.get("/health")
def health_check():
    """Returns service health status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mlflow_enabled": MLFLOW_ENABLED,
        "monitoring_enabled": MONITORING_ENABLED
    }