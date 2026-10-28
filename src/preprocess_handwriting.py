from pathlib import Path

from PIL import Image, ImageOps
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "raw_handwriting"
OUTPUT_DIR = BASE_DIR / "handwritten_digits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def convert_image(image_path):
    # 1. Görüntüyü gri tonlamaya dönüştür.
    image = Image.open(image_path).convert("L")

    # 2. Kenarlardan arka plan parlaklığını tahmin et.
    pixels = np.array(image)
    border = np.concatenate([
        pixels[0, :],
        pixels[-1, :],
        pixels[:, 0],
        pixels[:, -1],
    ])
    background_level = np.median(border)

    # 3. Arka planı siyah, rakamı beyaz yap.
    if background_level > 127:
        image = ImageOps.invert(image)

    # 4. Rakamı beyaz, arka planı siyah yap.
    image = image.point(lambda p: 255 if p > 100 else 0)

    # 5. Rakamın çevresindeki boşlukları kırp.
    bbox = image.getbbox()

    if bbox is None:
        raise ValueError("Görüntüde rakam bulunamadı.")

    digit = image.crop(bbox)

    # 6. En-boy oranını koruyarak 20x20 alana sığdır.
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 7. Siyah 28x28 tuvalin ortasına yerleştir.
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2
    canvas.paste(digit, (x, y))

    # 8. İşlenmiş görüntüyü kaydet.
    output_path = OUTPUT_DIR / f"{image_path.stem}.png"
    canvas.save(output_path)

    print(
        f"{image_path.name} -> {output_path.name} | "
        f"Boyut: {canvas.width}x{canvas.height} | "
        "Gri tonlama: L"
    )

    return canvas


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Orijinal fotoğraf klasörü bulunamadı: {INPUT_DIR}"
        )

    images = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("raw_handwriting klasöründe fotoğraf bulunamadı.")
        return

    for path in images:
        try:
            convert_image(path)
        except (ValueError, OSError) as error:
            print(f"Hata ({path.name}): {error}")

    print("\nGörüntü hazırlama tamamlandı.")


if __name__ == "__main__":
    main()
