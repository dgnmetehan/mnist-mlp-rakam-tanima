
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.data import get_data_loaders
from src.model import MLP
from src.engine import evaluate


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_PATH = RESULTS_DIR / "best_model.pth"


def main():

    device = torch.device("cpu")

    # Test veri kümesini yükler.
    _, _, test_loader = get_data_loaders(
        data_dir=str(BASE_DIR / "data"),
        batch_size=128,
        seed=42,
    )

    # Doğrulama sonuçlarına göre kaydedilmiş en iyi modeli yükler.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    # Kaydedilen modelin mimarisini yeniden oluştur.
    model = MLP(
        hidden1=checkpoint["hidden1"],
        hidden2=checkpoint["hidden2"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Sınıflandırma için kayıp fonksiyonunu belirle.
    criterion = nn.CrossEntropyLoss()

    # Yalnızca test kümesinde  değerlendirme yapılır.
    metrics = evaluate(model, test_loader, criterion, device)

    print(f"Selected epoch: {checkpoint['epoch']}")
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.2f}%")

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Her rakam sınıfı için precision, recall ve F1-score hesapla.
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        digits=4,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sınıflandırma raporunu metin dosyasına kaydet.
    with open(
        RESULTS_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    # Gerçek ve tahmin edilen sınıflara göre karmaşıklık matrisini oluşturur.
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10)),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(10)),
    )

    # Karmaşıklık matrisini görsel olarak kaydet.
    fig, ax = plt.subplots(figsize=(9, 9))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("MLP - MNIST Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Temel test metriklerini CSV dosyasına kaydet.
    with open(
        RESULTS_DIR / "test_metrics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["selected_epoch", checkpoint["epoch"]])
        writer.writerow(["test_loss", metrics["loss"]])
        writer.writerow(["test_accuracy", metrics["accuracy"]])

    print("\nTest results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
