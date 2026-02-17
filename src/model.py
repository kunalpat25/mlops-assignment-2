"""
CNN Model Definition for Cats vs Dogs Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary image classification (Cats vs Dogs).
    Input: 224x224 RGB images.
    Output: 2-class probabilities.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # After 4 pooling layers: 224 / 2^4 = 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path: str, device: str = "cpu") -> SimpleCNN:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved .pt file.
        device: Device to load the model on.

    Returns:
        Loaded SimpleCNN model in eval mode.
    """
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_single(model: SimpleCNN, image_tensor: torch.Tensor) -> dict:
    """
    Run inference on a single image tensor.

    Args:
        model: Trained SimpleCNN model.
        image_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).

    Returns:
        Dict with 'label', 'confidence', and 'probabilities'.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_names = ["cat", "dog"]
    label = class_names[predicted.item()]

    return {
        "label": label,
        "confidence": round(confidence.item(), 4),
        "probabilities": {
            "cat": round(probs[0][0].item(), 4),
            "dog": round(probs[0][1].item(), 4),
        },
    }
