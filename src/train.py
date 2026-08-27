import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import CatDogCNN


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")

MODEL_PATH = MODEL_DIR / "model.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42

EXPERIMENT_NAME = "cats-dogs-classification"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


evaluation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATA LOADERS
# ============================================================

def create_dataloaders():

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=evaluation_transform
    )

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=evaluation_transform
    )

    print("\nClass mapping:")
    print(train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    predictions = []
    true_labels = []

    for images, labels in loader:

        images = images.to(device)

        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probabilities = torch.sigmoid(logits)

        predicted_labels = (
            probabilities >= 0.5
        ).int()

        predictions.extend(
            predicted_labels
            .cpu()
            .numpy()
            .flatten()
            .tolist()
        )

        true_labels.extend(
            labels
            .cpu()
            .numpy()
            .flatten()
            .astype(int)
            .tolist()
        )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        true_labels,
        predictions
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    predictions = []
    true_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = (
                labels
                .float()
                .unsqueeze(1)
                .to(device)
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predicted_labels = (
                probabilities >= 0.5
            ).int()

            predictions.extend(
                predicted_labels
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            )

            true_labels.extend(
                labels
                .cpu()
                .numpy()
                .flatten()
                .astype(int)
                .tolist()
            )

    loss = (
        running_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    return (
        loss,
        accuracy,
        true_labels,
        predictions
    )


# ============================================================
# TRAINING CURVES
# ============================================================

def save_training_curves(history):

    loss_path = (
        ARTIFACT_DIR /
        "training_loss_curve.png"
    )

    accuracy_path = (
        ARTIFACT_DIR /
        "training_accuracy_curve.png"
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # Loss plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(accuracy_path)
    plt.close()

    return loss_path, accuracy_path


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    true_labels,
    predictions
):

    matrix = confusion_matrix(
        true_labels,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Cat",
            "Dog"
        ]
    )

    display.plot()

    plt.title(
        "Cats vs Dogs Test Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        ARTIFACT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(output_path)
    plt.close()

    return output_path


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_dataloaders()

    print("\nDataset sizes:")
    print(
        f"Training:   {len(train_dataset)}"
    )
    print(
        f"Validation: {len(val_dataset)}"
    )
    print(
        f"Testing:    {len(test_dataset)}"
    )

    model = CatDogCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Local MLflow tracking
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")

    with mlflow.start_run():

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            "CatDogCNN"
        )

        mlflow.log_param(
            "image_size",
            IMAGE_SIZE
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE
        )

        mlflow.log_param(
            "epochs",
            EPOCHS
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE
        )

        mlflow.log_param(
            "optimizer",
            "Adam"
        )

        mlflow.log_param(
            "loss_function",
            "BCEWithLogitsLoss"
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_SEED
        )

        mlflow.log_param(
            "training_samples",
            len(train_dataset)
        )

        mlflow.log_param(
            "validation_samples",
            len(val_dataset)
        )

        mlflow.log_param(
            "test_samples",
            len(test_dataset)
        )

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------

        for epoch in range(EPOCHS):

            train_loss, train_accuracy = (
                train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device
                )
            )

            (
                val_loss,
                val_accuracy,
                _,
                _
            ) = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            history["train_loss"].append(
                train_loss
            )

            history["train_accuracy"].append(
                train_accuracy
            )

            history["val_loss"].append(
                val_loss
            )

            history["val_accuracy"].append(
                val_accuracy
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch + 1
            )

            mlflow.log_metric(
                "train_accuracy",
                train_accuracy,
                step=epoch + 1
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch + 1
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch + 1
            )

            print(
                f"\nEpoch "
                f"{epoch + 1}/{EPOCHS}"
            )

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy: "
                f"{train_accuracy:.4f}"
            )

            print(
                f"Validation Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Accuracy: "
                f"{val_accuracy:.4f}"
            )

            # Save best model based on validation loss
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                torch.save(
                    model.state_dict(),
                    MODEL_PATH
                )

                print(
                    "Best model saved."
                )

        # ----------------------------------------------------
        # Reload best model
        # ----------------------------------------------------

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=device
            )
        )

        # ----------------------------------------------------
        # Test evaluation
        # ----------------------------------------------------

        (
            test_loss,
            test_accuracy,
            true_labels,
            predictions
        ) = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        test_precision = precision_score(
            true_labels,
            predictions,
            zero_division=0
        )

        test_recall = recall_score(
            true_labels,
            predictions,
            zero_division=0
        )

        test_f1 = f1_score(
            true_labels,
            predictions,
            zero_division=0
        )

        print("\n==============================")
        print("TEST RESULTS")
        print("==============================")

        print(
            f"Test Loss:      "
            f"{test_loss:.4f}"
        )

        print(
            f"Test Accuracy:  "
            f"{test_accuracy:.4f}"
        )

        print(
            f"Test Precision: "
            f"{test_precision:.4f}"
        )

        print(
            f"Test Recall:    "
            f"{test_recall:.4f}"
        )

        print(
            f"Test F1 Score:  "
            f"{test_f1:.4f}"
        )

        # ----------------------------------------------------
        # Log final metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "test_loss",
            test_loss
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "test_precision",
            test_precision
        )

        mlflow.log_metric(
            "test_recall",
            test_recall
        )

        mlflow.log_metric(
            "test_f1",
            test_f1
        )

        # ----------------------------------------------------
        # Generate artifacts
        # ----------------------------------------------------

        (
            loss_curve,
            accuracy_curve
        ) = save_training_curves(
            history
        )

        confusion_matrix_path = (
            save_confusion_matrix(
                true_labels,
                predictions
            )
        )

        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }

        metrics_path = (
            ARTIFACT_DIR /
            "test_metrics.json"
        )

        with open(
            metrics_path,
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        history_path = (
            ARTIFACT_DIR /
            "training_history.json"
        )

        with open(
            history_path,
            "w"
        ) as file:

            json.dump(
                history,
                file,
                indent=4
            )

        # ----------------------------------------------------
        # Log artifacts in MLflow
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model"
        )

        mlflow.log_artifact(
            str(loss_curve),
            artifact_path="plots"
        )

        mlflow.log_artifact(
            str(accuracy_curve),
            artifact_path="plots"
        )

        mlflow.log_artifact(
            str(confusion_matrix_path),
            artifact_path="plots"
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="results"
        )

        mlflow.log_artifact(
            str(history_path),
            artifact_path="results"
        )

        print("\nTraining completed successfully.")

        print(
            f"Saved model: "
            f"{MODEL_PATH}"
        )

        print(
            f"Artifacts: "
            f"{ARTIFACT_DIR}"
        )


if __name__ == "__main__":
    main()