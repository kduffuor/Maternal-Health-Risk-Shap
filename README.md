# Maternal Health Risk Predictor

![CI/CD Pipeline](https://github.com/kduffuor/Maternal-Health-Risk-Shap/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Docker](https://img.shields.io/badge/docker-28.0-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)

## Objective

This repository contains an MLOps pipeline for maternal health risk prediction during pregnancy. It extends the original pipeline provided by [Peco602](https://github.com/Peco602/maternal-health-risk) with three key contributions: a FastAPI prediction service replacing the original Flask app, SHAP explainability integrated into every prediction response, and a fully automated CI/CD pipeline via GitHub Actions.

## Live Demo

| Link | Description |
|------|-------------|
| [Clinical Frontend](https://kduffuor.github.io/Maternal-Health-Risk-Shap) | Web interface for clinicians and patients |
| [API Documentation](https://maternal-health-risk-shap.onrender.com/docs) | Swagger UI — interactive API docs |
| [Live API](https://maternal-health-risk-shap.onrender.com) | REST endpoint for programmatic access |

## Context

According to the World Health Organization (WHO):

> "*Maternal health refers to the health of women during pregnancy, childbirth and the post-natal period. Each stage should be a positive experience, ensuring women and their babies reach their full potential for health and well-being. Although important progress has been made in the last two decades, about 295 000 women died during and following pregnancy and childbirth in 2017. This number is unacceptably high.*"

The goal of this project is to build a production-ready MLOps pipeline that not only predicts maternal health risk but also explains which patient features drove each prediction – making the model interpretable for clinical use.

## Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/pyuxbhatt/maternal-health-risk) and contains data collected from hospitals, community clinics, and maternal health care centers through an IoT-based risk monitoring system.

| Feature | Description |
|---------|-------------|
| Age | Age of the woman during pregnancy |
| SystolicBP | Upper value of blood pressure in mmHg |
| DiastolicBP | Lower value of blood pressure in mmHg |
| BS | Blood glucose level in mmol/L |
| BodyTemp | Body temperature in Celsius |
| HeartRate | Resting heart rate in bpm |
| RiskLevel | Predicted risk level: low risk, mid risk, high risk |

## What This Version Adds Over the Original

| Feature | Original (Peco602) | This Version |
|---------|-------------------|--------------|
| Web framework | Flask | FastAPI |
| API documentation | None | Auto-generated Swagger UI at `/docs` |
| Input validation | Manual function | Pydantic models with automatic 422 errors |
| Explainability | None | SHAP TreeExplainer on every prediction |
| Clinical interpretation | None | Plain language risk drivers per prediction |
| Unit tests | Flask-based tests | 24 pytest tests across 5 test classes |
| CI/CD | GitHub Actions + SSH deploy | GitHub Actions + Docker Hub push |
| Server | Gunicorn | Uvicorn |

## API Response

Every prediction request returns three layers – a risk label, raw SHAP values for engineers, and a plain language clinical interpretation for clinicians:

```json
{
  "RiskLevel": "high risk",
  "explanation": {
    "BS": 0.1823,
    "SystolicBP": 0.1204,
    "Age": 0.0891,
    "DiastolicBP": -0.0423,
    "HeartRate": -0.0187,
    "BodyTemp": -0.0091
  },
  "interpretation": {
    "increasing_risk": [
      "Blood glucose is in diabetic range (10.0 mmol/L)",
      "Systolic blood pressure is in hypertensive range (160 mmHg)",
      "Age is advanced maternal age (45 years)"
    ],
    "decreasing_risk": [
      "Heart rate is within normal range (70 bpm)",
      "Body temperature is within normal range (38.0 Celsius)"
    ],
    "summary": "Risk is primarily driven by blood glucose in diabetic range (10.0 mmol/L)."
  }
}
```

The three layers serve different audiences:

| Layer | Field | Audience |
|-------|-------|----------|
| Risk label | `RiskLevel` | All users |
| Raw SHAP values | `explanation` | Engineers and researchers |
| Plain language drivers | `interpretation` | Clinicians and patients |

## Architecture

The full local MLOps pipeline runs via Docker Compose and includes 10 services:

![Architecture](images/architecture.png)

| Service | Port | Description |
|---------|------|-------------|
| Web Application | 80 | FastAPI prediction service with SHAP |
| MLflow | 5000 | Experiment tracking and model registry |
| Prefect | 4200 | Training workflow orchestration |
| MinIO | 9001 | S3-compatible model artifact storage |
| MongoDB | 27017 | Prediction logging database |
| Evidently | 8085 | Data and target drift monitoring |
| Grafana | 3001 | Real-time monitoring dashboards |
| Prometheus | 9090 | Time-series metrics database |
| PostgreSQL | 5433 | MLflow backend database |

## Local Setup

### Prerequisites

- Docker Desktop installed and running
- Git Bash or any terminal

### Steps

**1. Clone the repository:**

```bash
git clone https://github.com/kduffuor/Maternal-Health-Risk-Shap.git
cd Maternal-Health-Risk-Shap
```

**2. Pull all Docker images:**

```bash
docker compose pull
```

**3. Start the full pipeline:**

```bash
docker compose up -d
```

**4. Train the default model:**

```bash
docker compose exec web-app python3 retrain.py
docker compose restart web-app
```

**5. Access the services:**

> Note: The frontend at `https://kduffuor.github.io/Maternal-Health-Risk-Shap` calls the live Render API directly. No local setup is needed to use the web interface.

| Service | URL |
|---------|-----|
| API | http://localhost |
| API Docs | http://localhost/docs |
| MLflow | http://localhost:5000 |
| MinIO | http://localhost:9001 (admin / adminadmin) |
| Prefect | http://localhost:4200 |
| Grafana | http://localhost:3001 (admin / admin) |
| Evidently | http://localhost:8085/dashboard |
| Prometheus | http://localhost:9090 |

### Port Conflicts on Windows

If you encounter port conflicts on Windows, update the following in `docker-compose.yml`:

| Service | Default | Alternative |
|---------|---------|-------------|
| PostgreSQL | 5432 | 5433 |
| Grafana | 3000 | 3001 |

## Using the API

### Via Swagger UI

Go to `http://localhost/docs`, click `POST /predict`, then `Try it out` and submit patient data.

### Via curl

```bash
curl -X POST "http://localhost/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25,
    "SystolicBP": 130,
    "DiastolicBP": 80,
    "BS": 7.5,
    "BodyTemp": 37.0,
    "HeartRate": 80
  }'
```

### Via Docker Hub

```bash
docker pull kduffuor/maternal-health-risk-shap:latest
```

## Running Tests

```bash
docker compose exec web-app pip install pytest httpx
docker compose exec web-app python3 -m pytest tests/test_main.py -v
```

The test suite covers 24 tests across 5 classes:

| Class | Tests | Coverage |
|-------|-------|----------|
| TestInputValidation | 9 | Field ranges, cross-field BP validation |
| TestPredictionEndpoint | 5 | HTTP status, response structure, error handling |
| TestSHAPExplanation | 3 | Explanation structure, numeric values, all risk levels |
| TestHealthEndpoint | 3 | Health check status and model load confirmation |
| TestInterpretation | 4 | Interpretation structure, required keys, data types |

## CI/CD Pipeline

Every push to `main` triggers the GitHub Actions pipeline automatically:

```
Push to main
     ↓
Run 24 pytest tests
     ↓
Tests pass → Build Docker image
     ↓
Push to Docker Hub (kduffuor/maternal-health-risk-shap)
```

The pipeline will not push a broken image. If any test fails the build step is skipped entirely.

## Model Training

The default model is a Random Forest Classifier trained on the maternal health dataset. To retrain inside the running container:

```bash
docker compose exec web-app python3 retrain.py
docker compose restart web-app
```

For scheduled automated retraining using the full Prefect pipeline:

```bash
docker compose exec prefect python train.py
```

## Project Structure

```
Maternal-Health-Risk-Shap/
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD pipeline
├── app/
│   ├── main.py                   # FastAPI app with SHAP and clinical interpretation
│   ├── retrain.py                # Model retraining script
│   ├── train.py                  # Prefect training pipeline
│   ├── Dockerfile                # Container build instructions
│   ├── Pipfile                   # Python dependencies
│   └── tests/
│       └── test_main.py          # 24 pytest unit tests
├── data/
│   └── data.csv                  # Maternal health dataset
├── monitoring/                   # Grafana, Prometheus, Evidently config
├── index.html                    # Clinical frontend – hosted on GitHub Pages
├── docker-compose.yml            # Full 10-service stack definition
└── README.md
```

## Applied Technologies

| Name | Scope |
|------|-------|
| FastAPI | Prediction API server |
| Uvicorn | ASGI server for FastAPI |
| Pydantic | Input validation and schema definition |
| SHAP | Model explainability via TreeExplainer |
| scikit-learn | Random Forest classifier |
| XGBoost | Alternative model via Prefect training pipeline |
| Docker | Application containerization |
| Docker Compose | Multi-container orchestration |
| MLflow | Experiment tracking and model registry |
| Prefect | Training workflow orchestration |
| MinIO | S3-compatible artifact storage |
| MongoDB | Prediction logging |
| EvidentlyAI | Data and target drift monitoring |
| Prometheus | Time-series metrics |
| Grafana | Real-time monitoring dashboards |
| pytest | Unit testing |
| GitHub Actions | CI/CD pipeline |
| Docker Hub | Container image registry |

## Future Work

- **Calibrated risk probabilities** – Apply Platt scaling or isotonic regression to produce clinically defensible absolute risk percentages alongside the current classification output
- **Patient-facing UI** – A web interface that renders the interpretation layer in plain language with visual risk indicators
- **Clinical threshold validation** – Collaborate with maternal health clinicians to validate and refine the reference ranges used in the interpretation layer
- **Drift monitoring in production** – Extend Evidently and Grafana monitoring to the deployed Render service

## Disclaimer

This prediction service is intended for research and educational purposes only. It does not provide medical advice and cannot be used as a substitute for professional medical diagnosis or treatment. Never disregard professional medical advice because of something you have read here.

## Acknowledgements

Original MLOps pipeline by [Peco602](https://github.com/Peco602/maternal-health-risk), built as part of the MLOps Zoomcamp course by [DataTalks.Club](https://datatalks.club/). This version extends the original with FastAPI, SHAP explainability, clinically grounded interpretation, updated CI/CD, and a comprehensive pytest test suite.