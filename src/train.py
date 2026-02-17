"""
Training script with MLflow experiment tracking.
Logs parameters, metrics, artifacts (confusion matrix, loss curves).
"""

import os
import yaml
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import mlflow
import mlflow.pytorch

from model import SimpleCNN


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(params: dict, is_train: bool = True):
    """Get image transforms with optional augmentation."""
    img_size = params["data"]["image_size"]
    aug = params["augmentation"]

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if aug["horizontal_flip"] else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(aug["rotation_range"]),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model, return loss, accuracy, all predictions and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_accuracy_curves(train_accs, val_accs, save_path):
    """Plot and save training/validation accuracy curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    params = load_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Data loaders
    train_transform = get_transforms(params, is_train=True)
    val_transform = get_transforms(params, is_train=False)

    train_dataset = datasets.ImageFolder("data/splits/train", transform=train_transform)
    val_dataset = datasets.ImageFolder("data/splits/val", transform=val_transform)
    test_dataset = datasets.ImageFolder("data/splits/test", transform=val_transform)

    batch_size = params["data"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes  # ['cats', 'dogs']
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = SimpleCNN(
        num_classes=params["model"]["num_classes"],
        dropout=params["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["model"]["learning_rate"])

    # MLflow experiment tracking
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("cats-vs-dogs-classification")

    with mlflow.start_run(run_name="SimpleCNN_baseline"):
        # Log parameters
        mlflow.log_params({
            "architecture": params["model"]["architecture"],
            "learning_rate": params["model"]["learning_rate"],
            "epochs": params["model"]["epochs"],
            "batch_size": params["data"]["batch_size"],
            "optimizer": params["model"]["optimizer"],
            "dropout": params["model"]["dropout"],
            "image_size": params["data"]["image_size"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "augmentation_hflip": params["augmentation"]["horizontal_flip"],
            "augmentation_rotation": params["augmentation"]["rotation_range"],
        })

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        epochs = params["model"]["epochs"]
        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Log metrics per epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "models/best_model.pt")
                print(f"  -> Saved best model (val_acc={val_acc:.4f})")

        # Final evaluation on test set
        model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc,
        })

        # Classification report
        report = classification_report(test_labels, test_preds, target_names=class_names)
        print("\nClassification Report:")
        print(report)
        with open("artifacts/classification_report.txt", "w") as f:
            f.write(report)

        # Plot and log artifacts
        plot_loss_curves(train_losses, val_losses, "artifacts/loss_curves.png")
        plot_accuracy_curves(train_accs, val_accs, "artifacts/accuracy_curves.png")
        plot_confusion_matrix(
            test_labels, test_preds, class_names, "artifacts/confusion_matrix.png"
        )

        mlflow.log_artifact("artifacts/loss_curves.png")
        mlflow.log_artifact("artifacts/accuracy_curves.png")
        mlflow.log_artifact("artifacts/confusion_matrix.png")
        mlflow.log_artifact("artifacts/classification_report.txt")
        mlflow.log_artifact("models/best_model.pt")

        # Log the model with MLflow
        mlflow.pytorch.log_model(model, "model")

        print(f"\nMLflow run completed. Best Val Accuracy: {best_val_acc:.4f}")
        print("View results: mlflow ui")


if __name__ == "__main__":
    main()
