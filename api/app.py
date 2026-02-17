"""
FastAPI Inference Service for Cats vs Dogs Classifier.

Endpoints:
  GET  /health      - Health check
  POST /predict     - Predict cat or dog from uploaded image
  GET  /metrics     - Prometheus metrics
"""

import os
import io
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import SimpleCNN, predict_single
from schemas import HealthResponse, PredictionResponse, ErrorResponse

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_requests.log", mode="a"),
    ],
)
logger = logging.getLogger("catdog-api")

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Latency of prediction requests in seconds",
    ["endpoint"],
)
PREDICTION_CLASSES = Counter(
    "prediction_classes_total",
    "Count of predicted classes",
    ["class_label"],
)

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Binary image classification for a pet adoption platform",
    version="1.0.0",
)

# Global model reference
model = None
MODEL_PATH = os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models" / "best_model.pt"))

# Image preprocessing pipeline (must match training)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@app.on_event("startup")
def load_model_on_startup():
    """Load the trained model at application startup."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = SimpleCNN(num_classes=2)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Service will start without a model.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    REQUEST_COUNT.labels(endpoint="/health", status="success").inc()
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an uploaded image is a cat or dog.

    Accepts: image file (JPEG, PNG)
    Returns: predicted label, confidence, and class probabilities.
    """
    start_time = time.time()

    # Validate model is loaded
    if model is None:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Please check the model path.")

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Expected an image.")

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = inference_transform(image).unsqueeze(0)

        # Run inference
        result = predict_single(model, image_tensor)

        # Track metrics
        latency = time.time() - start_time
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        PREDICTION_CLASSES.labels(class_label=result["label"]).inc()

        # Log request (excluding file contents for privacy)
        logger.info(
            f"Prediction: file={file.filename}, "
            f"label={result['label']}, "
            f"confidence={result['confidence']:.4f}, "
            f"latency={latency:.4f}s"
        )

        return PredictionResponse(**result)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Upload image for classification",
            "GET /metrics": "Prometheus metrics",
        },
    }
