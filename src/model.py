
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden1: int = 256,
        hidden2: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()

        # MLP modelinin katmanlarının sırayla tanımlanması.
        self.network = nn.Sequential(
            # 28x28 boyutundaki görüntüyü 784 elemanlı vektöre dönüştür.
            nn.Flatten(),
            # Birinci gizli katman: 784 girişten 256 nörona geçiş
            nn.Linear(input_size, hidden1),
            # Modele doğrusal olmayan öğrenme yeteneği kazandır.
            nn.ReLU(),
            # İkinci gizli katman: 256 nörondan 128 nörona geçiş.
            nn.Linear(hidden1, hidden2),
            # İkinci gizli katmanda da ReLU aktivasyonunu uygula.
            nn.ReLU(),
            # Çıkış katmanı: 10 rakam sınıfı için skor üret.
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        # Girdiyi tanımlanan katmanlardan geçir ve sınıf skorlarını döndür.
        return self.network(x)
