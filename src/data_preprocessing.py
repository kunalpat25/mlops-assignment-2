"""
Data Preprocessing Module
- Resizes images to 224x224 RGB
- Splits into train/val/test (80/10/10)
- Applies data augmentation
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List

import yaml
import numpy as np
from PIL import Image


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from YAML config."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def resize_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and resize an image to the target size.

    Args:
        image_path: Path to the input image.
        target_size: Tuple (width, height) for resizing.

    Returns:
        Resized image as numpy array (H, W, 3) with values in [0, 255].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img)


def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1].

    Args:
        image_array: Image as numpy array with values in [0, 255].

    Returns:
        Normalized image array with values in [0.0, 1.0].
    """
    return image_array.astype(np.float32) / 255.0


def validate_image(image_path: str) -> bool:
    """
    Check if a file is a valid image.

    Args:
        image_path: Path to the file.

    Returns:
        True if the file is a valid image, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_image_paths(raw_dir: str) -> Tuple[List[str], List[str]]:
    """
    Walk the raw data directory and collect cat/dog image paths.

    Args:
        raw_dir: Path to raw data directory.

    Returns:
        Tuple of (cat_paths, dog_paths).
    """
    cat_paths = []
    dog_paths = []

    raw_path = Path(raw_dir)
    for img_path in raw_path.rglob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            # Class is determined by parent folder name (Cat/ or Dog/)
            parent_lower = img_path.parent.name.lower()
            if "cat" in parent_lower:
                if validate_image(str(img_path)):
                    cat_paths.append(str(img_path))
            elif "dog" in parent_lower:
                if validate_image(str(img_path)):
                    dog_paths.append(str(img_path))

    return cat_paths, dog_paths


def split_data(
    file_paths: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split file paths into train/val/test sets.

    Args:
        file_paths: List of file paths to split.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) file path lists.
    """
    random.seed(seed)
    shuffled = file_paths.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def preprocess_and_save(
    image_paths: List[str],
    output_dir: str,
    target_size: Tuple[int, int] = (224, 224),
) -> int:
    """
    Resize images and save to the output directory.

    Args:
        image_paths: List of source image paths.
        output_dir: Directory to save processed images.
        target_size: Target size for resizing.

    Returns:
        Number of images successfully processed.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(target_size, Image.BILINEAR)
            filename = Path(img_path).name
            img.save(os.path.join(output_dir, filename))
            count += 1
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    return count


def run_preprocessing(params_path: str = "params.yaml") -> None:
    """Execute the full preprocessing pipeline."""
    params = load_params(params_path)
    img_size = params["data"]["image_size"]
    train_ratio = params["data"]["train_ratio"]
    val_ratio = params["data"]["val_ratio"]
    seed = params["data"]["random_seed"]
    target_size = (img_size, img_size)

    print("Collecting image paths...")
    cat_paths, dog_paths = collect_image_paths("data/raw")
    print(f"Found {len(cat_paths)} cat images, {len(dog_paths)} dog images")

    if len(cat_paths) == 0 or len(dog_paths) == 0:
        print("ERROR: No images found. Please download the dataset first.")
        print("Run: python scripts/download_data.py")
        return

    # Split each class
    cat_train, cat_val, cat_test = split_data(cat_paths, train_ratio, val_ratio, seed)
    dog_train, dog_val, dog_test = split_data(dog_paths, train_ratio, val_ratio, seed)

    print(f"Cats  -> Train: {len(cat_train)}, Val: {len(cat_val)}, Test: {len(cat_test)}")
    print(f"Dogs  -> Train: {len(dog_train)}, Val: {len(dog_val)}, Test: {len(dog_test)}")

    # Process and save
    splits = {
        "data/splits/train/cats": cat_train,
        "data/splits/train/dogs": dog_train,
        "data/splits/val/cats": cat_val,
        "data/splits/val/dogs": dog_val,
        "data/splits/test/cats": cat_test,
        "data/splits/test/dogs": dog_test,
    }

    for output_dir, paths in splits.items():
        n = preprocess_and_save(paths, output_dir, target_size)
        print(f"Saved {n} images to {output_dir}")

    print("Preprocessing complete!")


if __name__ == "__main__":
    run_preprocessing()
