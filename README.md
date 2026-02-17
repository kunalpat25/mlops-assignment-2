# MLOps Assignment 2 – Cats vs Dogs Binary Classification

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform.

## Project Structure

```
├── data/                    # Data directory (tracked by DVC)
│   ├── raw/                 # Raw dataset from Kaggle
│   ├── processed/           # Preprocessed 224x224 images
│   └── splits/              # Train/Val/Test splits
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── api/                     # FastAPI inference service
│   ├── app.py
│   └── schemas.py
├── tests/                   # Unit tests
│   ├── test_preprocessing.py
│   └── test_inference.py
├── deployment/              # Deployment manifests
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── docker-compose.yml
├── monitoring/              # Monitoring scripts
│   └── simulate_requests.py
├── scripts/                 # Utility scripts
│   ├── smoke_test.py
│   └── download_data.py
├── .github/workflows/       # CI/CD pipelines
│   └── ci-cd.yml
├── .dvc/                    # DVC config
├── .dvcignore
├── data.dvc                 # DVC file tracking data
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── dvc.yaml                 # DVC pipeline
├── params.yaml              # Hyperparameters
└── README.md
```

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Download Dataset
```bash
python scripts/download_data.py
```

### 3. Preprocess Data
```bash
python src/data_preprocessing.py
```

### 4. Train Model (with MLflow tracking)
```bash
mlflow ui &                  # Start MLflow UI at localhost:5000
python src/train.py
```

### 5. Run Inference API Locally
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 6. Docker Build & Run
```bash
docker build -t catdog-classifier:latest .
docker run -p 8000:8000 catdog-classifier:latest
```

### 7. Test
```bash
pytest tests/ -v
```

### 8. Smoke Test (after deployment)
```bash
python scripts/smoke_test.py
```
