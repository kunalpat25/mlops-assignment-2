"""
Post-deployment Smoke Test Script.

Calls the health endpoint and makes a prediction request.
Exits with code 1 if any test fails (for CI/CD pipeline integration).

Usage:
    python scripts/smoke_test.py --url http://localhost:8000
"""

import sys
import argparse
import io
import struct
import random

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)


def create_test_image() -> bytes:
    """Create a minimal valid BMP image using pure Python (no numpy/PIL needed)."""
    width, height = 224, 224
    # BMP file header (14 bytes) + DIB header (40 bytes) + pixel data
    row_size = (width * 3 + 3) & ~3  # rows padded to 4-byte boundary
    pixel_data_size = row_size * height
    file_size = 54 + pixel_data_size

    # BMP Header
    bmp = bytearray()
    bmp += b'BM'                              # Signature
    bmp += struct.pack('<I', file_size)        # File size
    bmp += struct.pack('<HH', 0, 0)            # Reserved
    bmp += struct.pack('<I', 54)               # Pixel data offset
    # DIB Header (BITMAPINFOHEADER)
    bmp += struct.pack('<I', 40)               # DIB header size
    bmp += struct.pack('<i', width)            # Width
    bmp += struct.pack('<i', height)           # Height
    bmp += struct.pack('<HH', 1, 24)           # Planes, bits per pixel
    bmp += struct.pack('<I', 0)                # Compression (none)
    bmp += struct.pack('<I', pixel_data_size)  # Image size
    bmp += struct.pack('<ii', 2835, 2835)      # Resolution (72 DPI)
    bmp += struct.pack('<II', 0, 0)            # Colors

    # Pixel data (random RGB)
    random.seed(42)
    for _ in range(height):
        row = bytes(random.randint(0, 255) for _ in range(width * 3))
        bmp += row + b'\x00' * (row_size - width * 3)

    return bytes(bmp)


def test_health(base_url: str) -> bool:
    """Test the health check endpoint."""
    print("=" * 50)
    print("SMOKE TEST: Health Check")
    print("=" * 50)

    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        print(f"  Status Code: {resp.status_code}")
        print(f"  Response: {resp.json()}")

        if resp.status_code != 200:
            print("FAILED: Expected status 200")
            return False

        data = resp.json()
        if data.get("status") != "healthy":
            print("FAILED: Status is not 'healthy'")
            return False

        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_prediction(base_url: str) -> bool:
    """Test the prediction endpoint with a sample image."""
    print("\n" + "=" * 50)
    print("SMOKE TEST: Prediction")
    print("=" * 50)

    try:
        image_bytes = create_test_image()
        files = {"file": ("test.bmp", image_bytes, "image/bmp")}
        resp = requests.post(f"{base_url}/predict", files=files, timeout=30)
        print(f"  Status Code: {resp.status_code}")
        print(f"  Response: {resp.json()}")

        if resp.status_code != 200:
            print(f"FAILED: Expected status 200, got {resp.status_code}")
            return False

        data = resp.json()
        if "label" not in data or "confidence" not in data:
            print("FAILED: Missing required fields in response")
            return False

        if data["label"] not in ["cat", "dog"]:
            print(f"FAILED: Invalid label '{data['label']}'")
            return False

        print(f"  Predicted: {data['label']} (confidence: {data['confidence']:.4f})")
        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_root(base_url: str) -> bool:
    """Test the root endpoint."""
    print("\n" + "=" * 50)
    print("SMOKE TEST: Root Endpoint")
    print("=" * 50)

    try:
        resp = requests.get(f"{base_url}/", timeout=10)
        print(f"  Status Code: {resp.status_code}")

        if resp.status_code != 200:
            print("FAILED")
            return False

        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Post-deployment smoke tests")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"\nRunning smoke tests against: {base_url}\n")

    results = []
    results.append(("Health Check", test_health(base_url)))
    results.append(("Root Endpoint", test_root(base_url)))
    results.append(("Prediction", test_prediction(base_url)))

    # Summary
    print("\n" + "=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll smoke tests passed!")
        sys.exit(0)
    else:
        print("\n Some smoke tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
