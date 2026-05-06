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
    if MONITORING_ENABLED:
        save_to_db(record, risk)
        send_to_evidently_service(record, risk)
    return risk, explanation


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

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Maternal Health Risk Predictor",
    description="Predicts maternal health risk level based on patient vitals. SHAP explainability coming in v2.",
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
    with SHAP feature contribution explanation.
    """
    record = patient.model_dump()
    risk, explanation = calculate_risk(record)
    return {
        "RiskLevel": risk,
        "explanation": explanation
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