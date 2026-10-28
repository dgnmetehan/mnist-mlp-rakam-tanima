from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from src.model import MLP


BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "handwritten_digits"
CHECKPOINT_PATH = BASE_DIR / "results" / "best_model.pth"
OUTPUT_PATH = BASE_DIR / "results" / "handwriting_predictions.png"

# Görsellerin sıralaması ve gerçek rakam etiketleri
TRUE_LABELS = {
    "01_digit_2.png": 2,
    "02-digit_6.png": 6,
    "03-digit_9.png": 9,
    "04-digit_7.png": 7,
    "05-digit_9.png": 9,
    "06-digit_7.png": 7,
    "07-digit_0.png": 0,
    "08-digit_1.png": 1,
    "09-digit_2.png": 2,
}

TRANSFORM = transforms.ToTensor()


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {CHECKPOINT_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Görsel klasörü bulunamadı: {IMAGE_DIR}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = MLP(
        hidden1=checkpoint["hidden1"],
        hidden2=checkpoint["hidden2"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_paths = [
        IMAGE_DIR / filename
        for filename in TRUE_LABELS
    ]

    predictions = []
    confidences = []
    correct_count = 0

    with torch.no_grad():
        for image_path in image_paths:
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Görsel bulunamadı: {image_path}"
                )

            image = Image.open(image_path).convert("L")
            image_tensor = TRANSFORM(image).unsqueeze(0).to(device)

            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)

            predicted_label = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_label].item()

            true_label = TRUE_LABELS[image_path.name]
            correct = predicted_label == true_label
            correct_count += int(correct)

            predictions.append(predicted_label)
            confidences.append(confidence)

            print(
                f"{image_path.name} | "
                f"Gerçek: {true_label} | "
                f"Tahmin: {predicted_label} | "
                f"Güven skoru: {confidence:.2%} | "
                f"{'Doğru' if correct else 'Yanlış'}"
            )

    print(
        f"\nEl yazısı doğruluğu: "
        f"{correct_count}/{len(image_paths)} "
        f"({100 * correct_count / len(image_paths):.2f}%)"
    )

    # Görselleri ve tahmin sonuçlarını birlikte göster.
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))

    for index, (ax, image_path) in enumerate(
        zip(axes.flat, image_paths)
    ):
        image = Image.open(image_path).convert("L")
        true_label = TRUE_LABELS[image_path.name]
        predicted_label = predictions[index]

        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"Gerçek: {true_label} | Tahmin: {predicted_label}\n"
            f"Güven: {confidences[index]:.1%}"
        )
        ax.axis("off")

    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.show()
    plt.close(figure)

    print(f"Sonuç görseli kaydedildi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
