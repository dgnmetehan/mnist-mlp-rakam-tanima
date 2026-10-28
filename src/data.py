
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir="data",
    batch_size=128,
    seed=42,
    num_workers=0,
):
    # Normalizasyon işlemi: Görüntüleri tensöre dönüştürür ve piksel değerlerini [0, 1] aralığına ölçekler

    transform = transforms.ToTensor()

    # Eğitim ve validasyon veri seti: 60.000 görüntü
    train_val_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    # Test veri seti: 10.000 görüntü
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    
    generator = torch.Generator().manual_seed(seed)

    # Eğitim verisini 55.000 train ve 5.000 validation olarak ayırır
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [55000, 5000],
        generator=generator,
    )
    # Eğitim sırasında verilerin karışık sırayla alınmasını sağlar.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    # Doğrulama verilerini karıştırmadan gruplar hâlinde sunar.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    # Test verilerini eğitimden ayrı tutarak değerlendirme için hazırlar.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
