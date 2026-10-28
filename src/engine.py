
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Modelin tahminlerini üretilmesi ve kayıp değerinin hesaplanması.
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Grandyanların hesaplanması ve modelin ağırlıklarının güncellenmesi.
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        # Kayıp toplamının ve doğru tahmin sayısının güncellenmesi.
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    # Epoch boyunca ortalama kayıp ve doğruluk değerlerini döndürür.
    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Tahminlerin üretilmesi ve kayıp değerinin hesaplanması.
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            # Kayıp toplamının ve doğru tahmin sayısının güncellenmesi.
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size
            
    # Tüm veri kümesi için ortalama kayıp ve doğruluk değerlerini döndürür.
    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }
