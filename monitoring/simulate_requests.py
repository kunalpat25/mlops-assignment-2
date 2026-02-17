"""
Simulate real or synthetic requests to the deployed model
and track model performance post-deployment.

Sends a batch of images, collects predictions, compares with
true labels, and generates a performance report.

Usage:
    python monitoring/simulate_requests.py --url http://localhost:8000 --data-dir data/splits/test
"""

import os
import sys
import json
import time
import argparse
import io
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. pip install requests")
    sys.exit(1)


def get_true_label(image_path: str) -> str:
    """Infer true label from directory structure (e.g., .../cats/img.jpg -> cat)."""
    parent = Path(image_path).parent.name.lower()
    if "cat" in parent:
        return "cat"
    elif "dog" in parent:
        return "dog"
    return "unknown"


def send_prediction_request(base_url: str, image_path: str) -> dict:
    """Send an image to the prediction API and return the result."""
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        start = time.time()
        resp = requests.post(f"{base_url}/predict", files=files, timeout=30)
        latency = time.time() - start

    result = resp.json() if resp.status_code == 200 else {"error": resp.text}
    result["latency_seconds"] = round(latency, 4)
    result["status_code"] = resp.status_code
    return result


def generate_synthetic_images(num_images: int = 20) -> list:
    """Generate synthetic test images (random noise) with random labels."""
    images = []
    for i in range(num_images):
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        label = "cat" if i % 2 == 0 else "dog"
        images.append({"image": img, "true_label": label, "name": f"synthetic_{i}.jpg"})
    return images


def run_simulation(base_url: str, data_dir: str = None, num_synthetic: int = 20):
    """Run simulation: send images and collect performance metrics."""
    print(f"\n{'='*60}")
    print(f"  Post-Deployment Model Performance Tracking")
    print(f"  Endpoint: {base_url}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    results = []
    true_labels = []
    pred_labels = []
    latencies = []

    if data_dir and os.path.exists(data_dir):
        # Use real test images
        print(f"Using real test images from: {data_dir}")
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(Path(data_dir).rglob(ext))

        image_paths = sorted(image_paths)[:50]  # Limit to 50
        print(f"Found {len(image_paths)} images\n")

        for img_path in image_paths:
            true_label = get_true_label(str(img_path))
            result = send_prediction_request(base_url, str(img_path))

            if result.get("status_code") == 200:
                pred_label = result.get("label", "unknown")
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                latencies.append(result["latency_seconds"])

                match = "✅" if true_label == pred_label else "❌"
                print(
                    f"  {match} {img_path.name}: "
                    f"true={true_label}, pred={pred_label}, "
                    f"conf={result.get('confidence', 0):.4f}, "
                    f"latency={result['latency_seconds']:.4f}s"
                )
            else:
                print(f"  ⚠️  {img_path.name}: Request failed - {result.get('error', 'Unknown')}")

            results.append({
                "image": str(img_path),
                "true_label": true_label,
                **result,
            })

    else:
        # Use synthetic images
        print(f"Using {num_synthetic} synthetic images\n")
        synthetic = generate_synthetic_images(num_synthetic)

        for item in synthetic:
            buf = io.BytesIO()
            item["image"].save(buf, format="JPEG")
            buf.seek(0)

            files = {"file": (item["name"], buf, "image/jpeg")}
            start = time.time()
            resp = requests.post(f"{base_url}/predict", files=files, timeout=30)
            latency = time.time() - start

            if resp.status_code == 200:
                result = resp.json()
                true_labels.append(item["true_label"])
                pred_labels.append(result["label"])
                latencies.append(latency)

                print(
                    f"  {item['name']}: "
                    f"true={item['true_label']}, pred={result['label']}, "
                    f"conf={result['confidence']:.4f}, "
                    f"latency={latency:.4f}s"
                )

            results.append({
                "image": item["name"],
                "true_label": item["true_label"],
                "latency_seconds": round(latency, 4),
                **(resp.json() if resp.status_code == 200 else {}),
            })

    # Compute metrics
    print(f"\n{'='*60}")
    print("  PERFORMANCE REPORT")
    print(f"{'='*60}")

    if true_labels and pred_labels:
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        total = len(true_labels)
        accuracy = correct / total if total > 0 else 0

        print(f"  Total Requests:     {total}")
        print(f"  Correct:            {correct}")
        print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  Avg Latency:        {np.mean(latencies):.4f}s")
        print(f"  P50 Latency:        {np.percentile(latencies, 50):.4f}s")
        print(f"  P95 Latency:        {np.percentile(latencies, 95):.4f}s")
        print(f"  P99 Latency:        {np.percentile(latencies, 99):.4f}s")
        print(f"  Max Latency:        {max(latencies):.4f}s")

        # Class distribution
        from collections import Counter
        pred_counts = Counter(pred_labels)
        true_counts = Counter(true_labels)
        print(f"\n  True Label Distribution:      {dict(true_counts)}")
        print(f"  Predicted Label Distribution: {dict(pred_counts)}")

    # Save results
    os.makedirs("monitoring", exist_ok=True)
    report_path = "monitoring/performance_report.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": base_url,
        "total_requests": len(results),
        "accuracy": accuracy if true_labels else None,
        "avg_latency": round(float(np.mean(latencies)), 4) if latencies else None,
        "results": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Simulate requests and track performance")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--data-dir", default=None, help="Path to test images directory")
    parser.add_argument("--num-synthetic", type=int, default=20, help="Number of synthetic images")
    args = parser.parse_args()

    run_simulation(args.url, args.data_dir, args.num_synthetic)


if __name__ == "__main__":
    main()
