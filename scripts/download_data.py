"""
Download the Cats vs Dogs dataset from Kaggle.
Requires: kaggle CLI configured with API key.

Alternative: manually download from
https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
and extract into data/raw/
"""

import os
import zipfile
import shutil
from pathlib import Path


def download_kaggle_dataset(dest_dir: str = "data/raw") -> None:
    """Download and extract the Cats vs Dogs dataset from Kaggle."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    os.system(
        f'kaggle datasets download -d karakaggle/kaggle-cat-vs-dog-dataset -p {dest}'
    )

    # Extract zip files
    for zip_file in dest.glob("*.zip"):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(dest)
        zip_file.unlink()

    print(f"Dataset downloaded and extracted to {dest}")
    print("Directory contents:")
    for item in sorted(dest.rglob("*"))[:20]:
        print(f"  {item}")


def setup_directory_structure() -> None:
    """Create necessary project directories."""
    dirs = [
        "data/raw",
        "data/processed/cats",
        "data/processed/dogs",
        "data/splits/train/cats",
        "data/splits/train/dogs",
        "data/splits/val/cats",
        "data/splits/val/dogs",
        "data/splits/test/cats",
        "data/splits/test/dogs",
        "models",
        "mlruns",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")


if __name__ == "__main__":
    setup_directory_structure()
    download_kaggle_dataset()
