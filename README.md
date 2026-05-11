# IMDB Sentiment Analysis — End-to-End MLOps Pipeline

A production-grade MLOps project that takes a sentiment classification model from notebook experimentation all the way to a containerized, auto-deployed Flask application on AWS EKS — with a fully automated CI/CD pipeline and real-time monitoring via Prometheus and Grafana.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## Overview

This project demonstrates a complete ML engineering workflow: structured experimentation, a reproducible DVC pipeline, MLflow model tracking and registry, automated testing, Docker containerization, Kubernetes deployment on AWS, and real-time monitoring with Prometheus and Grafana hosted on AWS EC2 — all tied together with GitHub Actions CI/CD.

**Task:** Binary sentiment classification (positive / negative) on IMDB movie reviews  
**Model:** Logistic Regression with TF-IDF / Bag-of-Words vectorization  
**Dataset:** IMDB Movie Reviews

---

## Architecture

```
Raw Data (S3 / Local CSV)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  DVC Pipeline                        │
│                                                      │
│  data_ingestion → data_preprocessing                │
│       → feature_engineering → model_building        │
│       → model_evaluation → model_registration       │
└─────────────────────────────────────────────────────┘
        │
        ▼
  MLflow (DagsHub) — Experiment Tracking + Model Registry
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              GitHub Actions CI/CD                    │
│                                                      │
│  dvc repro → model tests → promote to production    │
│  → flask app tests → build & push Docker to ECR     │
│  → deploy to AWS EKS                                │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Flask App on AWS EKS (Kubernetes)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│         Monitoring Stack (AWS EC2)                   │
│                                                      │
│  Flask /metrics → Prometheus (scrape) → Grafana     │
│                   (EC2 hosted)          (EC2 hosted) │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── data_ingestion.py        # Load, preprocess, train/test split
│   │   └── data_preprocessing.py   # Text cleaning pipeline
│   ├── features/
│   │   └── feature_engineering.py  # BoW vectorization, feature saving
│   ├── model/
│   │   ├── model_building.py       # Train Logistic Regression
│   │   ├── model_evaluation.py     # Evaluate + log to MLflow
│   │   └── register_model.py       # Register model to MLflow registry
│   ├── connections/
│   │   └── s3_connection.py        # AWS S3 data ingestion utility
│   └── logger/
│       └── __init__.py             # Rotating file + console logger
│
├── flask_app/
│   ├── app.py                      # Flask app with Prometheus metrics
│   ├── templates/index.html        # Frontend UI
│   └── static/style.css            # Styling
│
├── tests/
│   ├── test_model.py               # Model load, signature & performance tests
│   └── test_flask_app.py           # Flask endpoint tests
│
├── scripts/
│   └── promote_model.py            # Promote staging → production in MLflow
│
├── notebooks/
│   ├── exp1.ipynb                  # Baseline: Logistic Regression + BoW
│   ├── exp2_bow_vs_tfidf.ipynb     # BoW vs TF-IDF across 5 algorithms
│   └── exp3_lor_tfidf.ipynb        # Hyperparameter tuning with GridSearchCV
│
├── .github/workflows/ci.yaml       # Full CI/CD pipeline
├── dvc.yaml                        # DVC pipeline stages
├── params.yaml                     # Centralized pipeline parameters
├── Dockerfile                      # Container definition
└── deployment.yaml                 # Kubernetes deployment + service
```

---

## ML Experimentation

Three structured experiments were run before productionizing the model:

| Experiment | What was tested | Key finding |
|---|---|---|
| Exp 1 | Logistic Regression baseline with BoW | F1: 0.61 |
| Exp 2 | 5 algorithms × 2 vectorizers (BoW vs TF-IDF) | LR + TF-IDF best overall (F1: 0.83) |
| Exp 3 | Logistic Regression hyperparameter tuning (GridSearchCV) | Best CV F1: 0.75 with C=10, l2 penalty |

All experiments tracked on **DagsHub MLflow** with nested runs, logged parameters, metrics, and model artifacts.

---

## DVC Pipeline

The pipeline is defined in `dvc.yaml` and parameterized via `params.yaml`.

```bash
# Reproduce the entire pipeline
dvc repro
```

**Stages and data flow:**

```
data_ingestion     →  data/raw/
data_preprocessing →  data/interim/
feature_engineering→  data/processed/ + models/vectorizer.pkl
model_building     →  models/model.pkl
model_evaluation   →  reports/experiment_info.json
model_registration →  MLflow Model Registry (Staging)
```

**Configurable parameters (`params.yaml`):**
```yaml
data_ingestion:
  test_size: 0.2

feature_engineering:
  max_features: 50
```

---

## CI/CD Pipeline

Every push to the repository triggers the full pipeline via GitHub Actions:

```
1. Checkout + setup Python 3.10
2. Install dependencies (with pip cache)
3. dvc repro           — run the full ML pipeline
4. test_model.py       — model load, signature & performance checks
5. promote_model.py    — move model from Staging → Production in MLflow
6. test_flask_app.py   — test Flask endpoints
7. Login to AWS ECR
8. Build + tag + push Docker image
9. Update kubeconfig for EKS
10. Apply Kubernetes deployment
```

---

## Flask Application

The serving layer is a Flask app that loads the production model directly from the MLflow registry at startup.

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Home page with review input form |
| `/predict` | POST | Returns sentiment prediction |
| `/metrics` | GET | Prometheus metrics scrape endpoint |

---

## Monitoring — Prometheus & Grafana on AWS EC2

A dedicated monitoring stack was set up on an **AWS EC2 instance** to observe the live Flask application running on EKS.

### How it works

```
Flask App (EKS)
    │
    │  exposes /metrics endpoint
    ▼
Prometheus (EC2)
    │  scrapes /metrics every 15 seconds
    │  stores time-series data
    ▼
Grafana (EC2)
    │  queries Prometheus as data source
    │  visualizes dashboards in real time
    ▼
Live Dashboards
```

### Metrics tracked

| Metric | Type | Description |
|---|---|---|
| `app_request_count` | Counter | Total requests by method and endpoint |
| `app_request_latency_seconds` | Histogram | Response latency per endpoint |
| `model_prediction_count` | Counter | Prediction count split by class (positive / negative) |

### Setup on EC2

**Prometheus** was configured with a custom `prometheus.yml` pointing to the Flask `/metrics` endpoint on the EKS LoadBalancer URL:

```yaml
scrape_configs:
  - job_name: 'flask-app'
    scrape_interval: 15s
    static_configs:
      - targets: ['<EKS-LoadBalancer-URL>:5000']
```

**Grafana** was connected to Prometheus as a data source and dashboards were built to visualize:
- Request rate over time
- P95 / P99 latency
- Live positive vs negative prediction ratio
- Error rate monitoring

This setup enables real-time observability into model behavior and API health in production — a critical component of any production ML system.

---

## Infrastructure

| Component | Technology |
|---|---|
| Container Registry | AWS ECR |
| Orchestration | AWS EKS (Kubernetes) |
| Node type | t3.small, 1 node |
| Kubernetes version | 1.31 |
| Replicas | 2 pods |
| Model serving | Gunicorn + Flask |
| Experiment tracking | MLflow on DagsHub |
| Data versioning | DVC with S3 remote |
| Monitoring | Prometheus + Grafana on AWS EC2 |

---

## Getting Started

### Prerequisites

- Python 3.10
- Docker
- AWS CLI configured
- `eksctl` and `kubectl`
- DagsHub account with MLflow enabled

### Local Setup

```bash
# Clone the repo
git clone https://github.com/MLayush-dubey/MLOps-IMDB-Sentiment-Analysis.git
cd MLOps-IMDB-Sentiment-Analysis

# Create and activate environment
conda create -n imdb python=3.10
conda activate imdb

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DAGSHUB_TOKEN=your_token
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export S3_BUCKET_NAME=your_bucket

# Run the pipeline
dvc repro
```

### Run Flask App Locally

```bash
cd flask_app
python app.py
# Visit http://localhost:5000
```

### Run Tests

```bash
python -m unittest tests/test_model.py
python -m unittest tests/test_flask_app.py
```

---

## GitHub Secrets Required

Set these in your repository's Settings → Secrets before running CI/CD:

```
DAGSHUB_TOKEN
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_ACCOUNT_ID
ECR_REPOSITORY
CAPSTONE_TEST
```

---

## Key Design Decisions

**Why DVC?** Reproducibility — every pipeline run is tracked and the exact data/code/model combination that produced a result can be reproduced by anyone on the team.

**Why MLflow Model Registry?** Decouples training from serving — the Flask app always loads from the `Production` stage, so deploying a new model requires no code change, only a registry promotion.

**Why separate CI steps for model tests before deployment?** A model that fails performance thresholds or signature checks should never reach production. The pipeline hard-stops before building Docker if tests fail.

**Why Prometheus + Grafana on EC2?** Running the monitoring stack separately from the application cluster keeps concerns isolated. EC2 gives full control over Prometheus configuration and Grafana dashboards without adding overhead to the EKS cluster.

**Why a custom metrics registry in Flask?** Default Prometheus exporters include system-level noise. A custom `CollectorRegistry` exposes only ML-relevant signals — request counts, latency, and prediction class distribution — making dashboards clean and actionable.

---

## Author

**Ayush Dubey**  
End-to-End MLOps Project — 2026
