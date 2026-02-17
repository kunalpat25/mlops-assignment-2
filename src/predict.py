"""
Standalone prediction script for inference.
"""

import sys
import torch
from torchvision import transforms
from PIL import Image

from model import SimpleCNN, predict_single


def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess a single image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/best_model.pt"

    # Load model
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Preprocess and predict
    image_tensor = preprocess_image(image_path)
    result = predict_single(model, image_tensor)

    print(f"Image: {image_path}")
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: Cat={result['probabilities']['cat']:.4f}, Dog={result['probabilities']['dog']:.4f}")


if __name__ == "__main__":
    main()
