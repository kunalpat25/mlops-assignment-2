"""
Unit tests for model inference functions.
Tests: SimpleCNN forward pass, predict_single, API endpoints
"""

import os
import sys
import io
import tempfile
import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from fastapi.testclient import TestClient

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import SimpleCNN, predict_single


class TestSimpleCNN:
    """Tests for SimpleCNN model architecture."""

    @pytest.fixture
    def model(self):
        """Create a SimpleCNN model instance."""
        return SimpleCNN(num_classes=2, dropout=0.5)

    def test_model_output_shape(self, model):
        """Test that model output has correct shape."""
        batch = torch.randn(4, 3, 224, 224)
        output = model(batch)
        assert output.shape == (4, 2)

    def test_single_image_output(self, model):
        """Test model with a single image."""
        img = torch.randn(1, 3, 224, 224)
        output = model(img)
        assert output.shape == (1, 2)

    def test_model_parameters_exist(self, model):
        """Test that model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
        total_params = sum(p.numel() for p in params)
        assert total_params > 0

    def test_model_eval_mode(self, model):
        """Test that model can be switched to eval mode."""
        model.eval()
        img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(img)
        assert output.shape == (1, 2)

    def test_model_save_and_load(self, model):
        """Test that model can be saved and loaded."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            loaded_model = SimpleCNN(num_classes=2)
            loaded_model.load_state_dict(torch.load(f.name, map_location="cpu"))
        os.unlink(f.name)

        # Compare outputs
        model.eval()
        loaded_model.eval()
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(test_input)
            out2 = loaded_model(test_input)
        assert torch.allclose(out1, out2)


class TestPredictSingle:
    """Tests for predict_single function."""

    @pytest.fixture
    def model(self):
        model = SimpleCNN(num_classes=2)
        model.eval()
        return model

    def test_returns_dict(self, model):
        """Test that prediction returns a dictionary."""
        img = torch.randn(1, 3, 224, 224)
        result = predict_single(model, img)
        assert isinstance(result, dict)

    def test_has_required_keys(self, model):
        """Test that result contains required keys."""
        img = torch.randn(1, 3, 224, 224)
        result = predict_single(model, img)
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_label_is_valid(self, model):
        """Test that predicted label is cat or dog."""
        img = torch.randn(1, 3, 224, 224)
        result = predict_single(model, img)
        assert result["label"] in ["cat", "dog"]

    def test_confidence_range(self, model):
        """Test that confidence is between 0 and 1."""
        img = torch.randn(1, 3, 224, 224)
        result = predict_single(model, img)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, model):
        """Test that probabilities sum to approximately 1."""
        img = torch.randn(1, 3, 224, 224)
        result = predict_single(model, img)
        prob_sum = result["probabilities"]["cat"] + result["probabilities"]["dog"]
        assert abs(prob_sum - 1.0) < 0.01

    def test_probabilities_non_negative(self, model):
        """Test that probabilities are non-negative."""
        img = torch.randn(1, 3, 224, 224)
        result = predict_single(model, img)
        assert result["probabilities"]["cat"] >= 0.0
        assert result["probabilities"]["dog"] >= 0.0


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        from api.app import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "version" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_predict_no_file(self, client):
        """Test predict endpoint without file returns 422."""
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_with_image(self, client):
        """Test predict endpoint with a valid image."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        # Will be 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503]
