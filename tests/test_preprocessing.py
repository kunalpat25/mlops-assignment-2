"""
Unit tests for data preprocessing functions.
Tests: resize_image, normalize_image, validate_image, split_data
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_preprocessing import (
    resize_image,
    normalize_image,
    validate_image,
    split_data,
)


@pytest.fixture
def sample_image_path():
    """Create a temporary test image and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_image_path_grayscale():
    """Create a temporary grayscale test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L")
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_to_224(self, sample_image_path):
        """Test resizing image to 224x224."""
        result = resize_image(sample_image_path, target_size=(224, 224))
        assert result.shape == (224, 224, 3)

    def test_resize_to_custom_size(self, sample_image_path):
        """Test resizing to a custom size."""
        result = resize_image(sample_image_path, target_size=(128, 128))
        assert result.shape == (128, 128, 3)

    def test_output_is_numpy_array(self, sample_image_path):
        """Test that output is a numpy array."""
        result = resize_image(sample_image_path)
        assert isinstance(result, np.ndarray)

    def test_pixel_values_range(self, sample_image_path):
        """Test that pixel values are in [0, 255]."""
        result = resize_image(sample_image_path)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_grayscale_converted_to_rgb(self, sample_image_path_grayscale):
        """Test that grayscale images are converted to RGB (3 channels)."""
        result = resize_image(sample_image_path_grayscale, target_size=(224, 224))
        assert result.shape == (224, 224, 3)


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_output_range(self):
        """Test that normalized values are in [0, 1]."""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = normalize_image(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self):
        """Test that output is float32."""
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = normalize_image(img)
        assert result.dtype == np.float32

    def test_known_values(self):
        """Test normalization with known values."""
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = normalize_image(img)
        np.testing.assert_almost_equal(result[0, 0, 0], 0.0)
        np.testing.assert_almost_equal(result[0, 0, 2], 1.0)

    def test_shape_preserved(self):
        """Test that shape is preserved after normalization."""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = normalize_image(img)
        assert result.shape == img.shape


class TestValidateImage:
    """Tests for validate_image function."""

    def test_valid_image(self, sample_image_path):
        """Test that a valid image returns True."""
        assert validate_image(sample_image_path) is True

    def test_invalid_file(self):
        """Test that a non-image file returns False."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("not an image")
            f.flush()
            result = validate_image(f.name)
        os.unlink(f.name)
        assert result is False

    def test_nonexistent_file(self):
        """Test that a nonexistent file returns False."""
        assert validate_image("/nonexistent/path/image.jpg") is False


class TestSplitData:
    """Tests for split_data function."""

    def test_split_ratios(self):
        """Test that split produces correct approximate ratios."""
        paths = [f"img_{i}.jpg" for i in range(100)]
        train, val, test = split_data(paths, train_ratio=0.8, val_ratio=0.1, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        """Test that there is no overlap between splits."""
        paths = [f"img_{i}.jpg" for i in range(100)]
        train, val, test = split_data(paths, train_ratio=0.8, val_ratio=0.1, seed=42)
        all_items = set(train) | set(val) | set(test)
        assert len(all_items) == 100

    def test_reproducibility(self):
        """Test that same seed produces same split."""
        paths = [f"img_{i}.jpg" for i in range(100)]
        train1, val1, test1 = split_data(paths, seed=42)
        train2, val2, test2 = split_data(paths, seed=42)
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_different_seeds(self):
        """Test that different seeds produce different splits."""
        paths = [f"img_{i}.jpg" for i in range(100)]
        train1, _, _ = split_data(paths, seed=42)
        train2, _, _ = split_data(paths, seed=99)
        assert train1 != train2

    def test_all_items_included(self):
        """Test that all items are included in some split."""
        paths = [f"img_{i}.jpg" for i in range(50)]
        train, val, test = split_data(paths)
        assert len(train) + len(val) + len(test) == 50
