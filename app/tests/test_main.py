"""Unit tests for FastAPI maternal health risk predictor with SHAP"""

import pickle
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import app, PatientData

client = TestClient(app)

# -------------------------------------------------------------------
# Valid and invalid sample data
# -------------------------------------------------------------------
VALID_LOW_RISK = {
    "Age": 20,
    "SystolicBP": 120,
    "DiastolicBP": 70,
    "BS": 2.0,
    "BodyTemp": 36.0,
    "HeartRate": 60,
}

VALID_MID_RISK = {
    "Age": 35,
    "SystolicBP": 130,
    "DiastolicBP": 90,
    "BS": 7.0,
    "BodyTemp": 36.5,
    "HeartRate": 60,
}

VALID_HIGH_RISK = {
    "Age": 45,
    "SystolicBP": 160,
    "DiastolicBP": 90,
    "BS": 10.0,
    "BodyTemp": 38.0,
    "HeartRate": 70,
}

FEATURE_NAMES = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

# -------------------------------------------------------------------
# Test 1 — Input validation via Pydantic
# -------------------------------------------------------------------
class TestInputValidation:

    def test_valid_input_passes(self):
        """Valid input should not raise any error"""
        patient = PatientData(**VALID_LOW_RISK)
        assert patient.Age == 20

    def test_age_below_minimum_rejected(self):
        """Age below 13 should be rejected"""
        with pytest.raises(ValidationError):
            PatientData(**{**VALID_LOW_RISK, "Age": 10})

    def test_age_above_maximum_rejected(self):
        """Age above 50 should be rejected"""
        with pytest.raises(ValidationError):
            PatientData(**{**VALID_LOW_RISK, "Age": 60})

    def test_systolic_below_minimum_rejected(self):
        """SystolicBP below 50 should be rejected"""
        with pytest.raises(ValidationError):
            PatientData(**{**VALID_LOW_RISK, "SystolicBP": 30})

    def test_diastolic_above_maximum_rejected(self):
        """DiastolicBP above 200 should be rejected"""
        with pytest.raises(ValidationError):
            PatientData(**{**VALID_LOW_RISK, "DiastolicBP": 250})

    def test_blood_glucose_out_of_range_rejected(self):
        """BS above 15 should be rejected"""
        with pytest.raises(ValidationError):
            PatientData(**{**VALID_LOW_RISK, "BS": 20.0})

    def test_body_temp_out_of_range_rejected(self):
        """BodyTemp below 34 should be rejected"""
        with pytest.raises(ValidationError):
            PatientData(**{**VALID_LOW_RISK, "BodyTemp": 30.0})

    def test_heart_rate_out_of_range_rejected(self):
        """HeartRate above 130 should be rejected"""
        with pytest.raises(ValidationError):
            PatientData(**{**VALID_LOW_RISK, "HeartRate": 200})

    def test_systolic_must_exceed_diastolic(self):
        """SystolicBP equal to DiastolicBP should be rejected"""
        with pytest.raises(ValidationError):
            PatientData(**{**VALID_LOW_RISK, "SystolicBP": 70, "DiastolicBP": 70})


# -------------------------------------------------------------------
# Test 2 — Prediction output
# -------------------------------------------------------------------
class TestPredictionEndpoint:

    def test_predict_returns_200(self):
        """Valid input should return HTTP 200"""
        response = client.post("/predict", json=VALID_LOW_RISK)
        assert response.status_code == 200

    def test_predict_returns_risk_level(self):
        """Response must contain RiskLevel field"""
        response = client.post("/predict", json=VALID_LOW_RISK)
        assert "RiskLevel" in response.json()

    def test_predict_risk_level_is_valid_label(self):
        """RiskLevel must be one of the three valid labels"""
        response = client.post("/predict", json=VALID_LOW_RISK)
        assert response.json()["RiskLevel"] in ["low risk", "mid risk", "high risk"]

    def test_predict_returns_explanation(self):
        """Response must contain explanation field"""
        response = client.post("/predict", json=VALID_LOW_RISK)
        assert "explanation" in response.json()

    def test_invalid_input_returns_422(self):
        """Invalid input should return HTTP 422 Unprocessable Entity"""
        response = client.post("/predict", json={**VALID_LOW_RISK, "Age": 99})
        assert response.status_code == 422


# -------------------------------------------------------------------
# Test 3 — SHAP explanation structure
# -------------------------------------------------------------------
class TestSHAPExplanation:

    def test_explanation_contains_all_features(self):
        """Explanation must contain only valid feature names with numeric values"""
        response = client.post("/predict", json=VALID_LOW_RISK)
        explanation = response.json()["explanation"]
        assert len(explanation) > 0, "Explanation should not be empty"
        for feature in explanation.keys():
            assert feature in FEATURE_NAMES, f"Unexpected feature: {feature}"

    def test_explanation_values_are_numeric(self):
        """All SHAP values must be numeric"""
        response = client.post("/predict", json=VALID_LOW_RISK)
        explanation = response.json()["explanation"]
        for feature, value in explanation.items():
            assert isinstance(value, (int, float)), f"{feature} value is not numeric"

    def test_explanation_for_different_risk_levels(self):
        """SHAP explanation should work for all risk levels"""
        for data in [VALID_LOW_RISK, VALID_MID_RISK, VALID_HIGH_RISK]:
            response = client.post("/predict", json=data)
            assert response.status_code == 200
            assert "explanation" in response.json()


# -------------------------------------------------------------------
# Test 4 — Health check endpoint
# -------------------------------------------------------------------
class TestHealthEndpoint:

    def test_health_returns_200(self):
        """Health endpoint should return HTTP 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_shows_model_loaded(self):
        """Health endpoint should confirm model is loaded"""
        response = client.get("/health")
        assert response.json()["model_loaded"] is True

    def test_health_shows_status_healthy(self):
        """Health endpoint should show status as healthy"""
        response = client.get("/health")
        assert response.json()["status"] == "healthy"