# 🚀 Kaggle Hızlı Başlangıç

## ⚡ En Hızlı Yöntem (3 Dakika)

### 1. Kaggle'da Yeni Notebook Aç
- Competition sayfası: https://www.kaggle.com/competitions/csiro-biomass
- **Code** → **New Notebook**
- **Settings**:
  - ✅ Accelerator: **GPU T4 x2** veya **GPU P100**
  - ✅ Internet: **ON**

### 2. Hazır Notebook'u Yükle
1. Bu repository'den `kaggle_notebook_standalone.ipynb` dosyasını indir
2. Kaggle'da **File** → **Import Notebook**
3. İndirdiğin dosyayı seç
4. **Run All** yap! 🎉

---

## 📝 Alternatif: Manuel Kurulum

Eğer notebook'u yüklemek istemezsen, aşağıdaki kodu Kaggle'da yeni bir hücreye yapıştır:

```python
# Tüm kodu tek seferde çalıştır
!pip install -q timm==0.9.12
!pip install -q albumentations==1.3.1 --no-deps
!pip install -q albucore==0.0.17

# Dizinleri oluştur
import os
os.makedirs('/kaggle/working/src', exist_ok=True)
os.makedirs('/kaggle/working/configs', exist_ok=True)
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

# Config dosyası oluştur
config_yaml = '''
data:
  data_dir: "/kaggle/input/csiro-biomass"
  train_csv: "train.csv"
  test_csv: "test.csv"
  image_dir: "images"
  img_size: 384
  n_folds: 5
  seed: 42

model:
  architectures: ["efficientnet_b3"]
  pretrained: true
  dropout: 0.3

training:
  batch_size: 16
  num_epochs: 15
  learning_rate: 0.0003
  mixed_precision: true
  patience: 7
'''

with open('/kaggle/working/configs/config.yaml', 'w') as f:
    f.write(config_yaml)

print("✅ Setup tamamlandı! Devam için kaggle_notebook_standalone.ipynb'deki diğer hücreleri çalıştır")
```

---

## ⚠️ Sorun Giderme

### Problem: GitHub'a erişilemiyor
**Çözüm**: `kaggle_notebook_standalone.ipynb` kullan (GitHub'sız çalışır)

### Problem: "Out of Memory"
**Çözüm**: Batch size'ı küçült:
```python
# Config'te
batch_size: 8  # 16'dan düşür
```

### Problem: "Could not resolve host"
**Çözüm**: Notebook Settings → Internet → **ON** yap

### Problem: Dependency conflicts
**Çözüm**: Versiyonları sabitle:
```bash
!pip install -q timm==0.9.12
!pip install -q albumentations==1.3.1 --no-deps
!pip install -q albucore==0.0.17
```

---

## 📊 Beklenen Süre

- **Setup**: ~2 dakika
- **Training** (5-fold, 15 epoch): ~2-3 saat (GPU T4)
- **Inference**: ~10 dakika
- **Toplam**: ~3 saat

---

## 📥 Submission İndirme

1. Notebook tamamlandıktan sonra sağ taraftaki **Output** sekmesine git
2. `submission.csv` dosyasını indir
3. Competition sayfasında **Submit Predictions** → CSV'yi yükle

---

## 🎯 Hızlı Test (5 Epoch)

Hızlı test için config'te:
```yaml
training:
  num_epochs: 5  # 15'ten düşür
  n_folds: 1     # 5'ten düşür
```

---

## 📚 Dosyalar

- `kaggle_notebook_standalone.ipynb` - **GitHub'sız standalone notebook (ÖNERİLEN)**
- `kaggle_notebook.ipynb` - GitHub'dan clone eden versiyon
- `KAGGLE_SETUP.md` - Detaylı setup kılavuzu
- `README.md` - Genel dökümantasyon

---

**Başarılar!** 🚀
