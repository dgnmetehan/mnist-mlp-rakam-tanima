
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data import get_data_loaders
from src.engine import evaluate, train_one_epoch
from src.model import MLP


# Model eğitimi için kullanılacak hiperparametreler
SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
# MLP modelindeki iki gizli katmanın nöron sayıları
HIDDEN1 = 256
HIDDEN2 = 128



BASE_DIR = Path(__file__).resolve().parent.parent

# Kaydedilecek sonuç dosyalarının konumları
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_PATH = RESULTS_DIR / "best_model.pth"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_history(history):
     # Her epoch için eğitim ve doğrulama metriklerini CSV dosyasına kaydet.
    csv_path = RESULTS_DIR / "training_history.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
        ])

        for index in range(len(history["train_loss"])):
            writer.writerow([
                index + 1,
                history["train_loss"][index],
                history["val_loss"][index],
                history["train_accuracy"][index],
                history["val_accuracy"][index],
            ])


def save_plots(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150)
    plt.close()


def main():
    set_seed(SEED)

    device = torch.device("cpu")

    print("Device:", device)
    print("Loading MNIST dataset...")

    # Eğitim ve doğrulama verilerini yükler.
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=str(BASE_DIR / "data"),
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # İki gizli katmanlı MLP modelini oluşturur.
    model = MLP(
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
    ).to(device)

    # Çok sınıflı sınıflandırma için kayıp fonksiyonunu belirlenmesi.
    criterion = nn.CrossEntropyLoss()

    # Model ağırlıklarını güncellemek için Adam optimizer kullanımı.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0

    print("\nTraining started")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 65)

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        # Doğrulama kaybı azaldıysa modeli en iyi model olarak kaydet
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "hidden1": HIDDEN1,
                    "hidden2": HIDDEN2,
                    "epochs": EPOCHS,
                },
                CHECKPOINT_PATH,
            )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )

    # Eğitim süresi.
    elapsed = time.perf_counter() - start_time
    
    # Eğitim geçmişini, loss ve accuracy grafiklerini kaydet
    save_history(history)
    save_plots(history)

    print("\nTraining completed")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {CHECKPOINT_PATH}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
