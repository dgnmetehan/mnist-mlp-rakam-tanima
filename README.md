# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu projede, MNIST veri seti kullanılarak sıfırdan bir MLP modeli eğittim. Modelin performansını değerlendirdim ve kendi el yazımla yazdığım rakamlar üzerinde denedim.

## Proje Dosyaları

- `model.py`: MLP modelinin mimarisini tanımlar.
- `data.py`: MNIST veri setini yükler ve train, validation ve test kümelerini hazırlar.
- `engine.py`: Eğitim ve değerlendirme işlemlerini yürütür.
- `train.py`: Temel MLP modelini eğitir, her epoch sonunda eğitim ve doğrulama sonuçlarını hesaplar, en iyi modeli kaydeder ve grafikleri oluşturur.
- `train_small.py`: Daha düşük bir MLP modelini eğitir.
- `evaluate.py`: MLP modelinin daha önce eğitim sırasında kullanılmamış test veri seti üzerindeki başarısını ölçer.
- `preprocess_handwriting.py`: El yazısı görsellerini MNOST veri setine uygun hale getirir ve model için hazırlar.
- `predict_handwriting.py`: El yazısı rakamlarını tahmin eder.

## Kurulum

Sanal ortamı oluşturup etkinleştirin:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## Çalıştırma

Komutları proje ana klasöründe çalıştırın.

```bash
python -m src.train
python -m src.train_small
python -m src.evaluate
python -m src.preprocess_handwriting
python -m src.predict_handwriting
```

Kendi el yazım ile yazdığım öğrenci numaramın rakamlarının raw hali bu klasörde yer alıyor `raw_handwriting/`.
Gri tonlamalı,28 × 28 piksele dönüştürülmüş,arka plan ve rakam renkleri MNIST formatıyla uyumlu hali `handwritten_digits/`.

## Sonuçlar ve Rapor

Eğitim grafikleri, model ağırlıkları ve değerlendirme sonuçları `results/` klasöründedir. Deneyin sonuçlarını ve genel değerlendirmesini içeren rapora `269797012_MNIST_Rapor.pdf` dosyasından ulaşabilirsiniz.
