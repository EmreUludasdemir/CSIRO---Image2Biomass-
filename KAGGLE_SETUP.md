# CSIRO Image2Biomass - Kaggle Notebook Kullanım Kılavuzu

Bu kılavuz, projeyi Kaggle notebook ortamında çalıştırmak için gerekli tüm adımları içerir.

---

## 🚀 Hızlı Başlangıç (Kaggle Notebook)

### Adım 1: Yeni Kaggle Notebook Oluştur

1. Kaggle'da yarışma sayfasına git: https://www.kaggle.com/competitions/csiro-biomass
2. "Code" sekmesine tıkla
3. "New Notebook" butonuna tıkla
4. Notebook ayarlarını yap:
   - **Accelerator**: GPU T4 x2 (veya GPU P100)
   - **Internet**: ON (GitHub'dan kod çekmek için)
   - **Language**: Python

### Adım 2: GitHub'dan Kodu Çek

Notebook'un ilk hücresine şunu yaz:

```python
# GitHub'dan projeyi klonla
!git clone https://github.com/EmreUludasdemir/CSIRO---Image2Biomass-.git
%cd CSIRO---Image2Biomass-

# Gerekli kütüphaneleri yükle
!pip install -q timm albumentations pytorch-lightning

# Kurulumu kontrol et
!python quick_start.py
```

### Adım 3: Kaggle Veri Yapısını Ayarla

Kaggle'da veriler `/kaggle/input/` dizininde bulunur. Config dosyasını güncelle:

```python
import yaml

# Config'i oku
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Kaggle veri yollarını güncelle
config['data']['data_dir'] = '/kaggle/input/csiro-biomass'
config['data']['image_dir'] = 'images'  # veya verinin bulunduğu klasör

# Checkpoint ve submission yollarını güncelle
config['paths']['checkpoint_dir'] = '/kaggle/working/checkpoints'
config['paths']['submission_dir'] = '/kaggle/working/submissions'

# Config'i kaydet
with open('configs/config_kaggle.yaml', 'w') as f:
    yaml.dump(config, f)

print("✓ Kaggle config hazır!")
```

---

## 📊 Opsiyon 1: Hızlı Training (Tek Model)

```python
import sys
sys.path.append('/kaggle/working/CSIRO---Image2Biomass-/src')

import torch
from train import train_fold
import pandas as pd
from sklearn.model_selection import train_test_split

# Veriyi yükle
train_df = pd.read_csv('/kaggle/input/csiro-biomass/train.csv')

# Train/validation split
train_data, valid_data = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42
)

# Config'i yükle
with open('configs/config_kaggle.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Hızlı training için parametreleri ayarla
config['training']['num_epochs'] = 10  # Daha kısa test için
config['training']['batch_size'] = 16
config['data']['n_folds'] = 1  # Tek fold

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Training
best_score, model = train_fold(
    fold=0,
    train_df=train_data,
    valid_df=valid_data,
    config=config,
    device=device
)

print(f"\n✓ Training tamamlandı! Best RMSE: {best_score:.4f}")
```

---

## 🎯 Opsiyon 2: Full Training (5-Fold CV + Ensemble)

```python
# Full training pipeline
!python src/train.py --config configs/config_kaggle.yaml
```

---

## 🔮 Inference ve Submission Oluşturma

```python
import sys
sys.path.append('/kaggle/working/CSIRO---Image2Biomass-/src')

from inference import create_submission

# Submission oluştur
create_submission(
    config_path='configs/config_kaggle.yaml',
    test_csv_path='/kaggle/input/csiro-biomass/test.csv',
    output_path='/kaggle/working/submission.csv'
)

# Submission'ı kontrol et
import pandas as pd
submission = pd.read_csv('/kaggle/working/submission.csv')
print("\n✓ Submission hazır!")
print(submission.head())
print(f"\nShape: {submission.shape}")
print(f"Predictions range: [{submission['biomass'].min():.2f}, {submission['biomass'].max():.2f}]")
```

---

## 📝 Tek Hücreli Komple Pipeline (Kopyala-Yapıştır)

Aşağıdaki kodu tek bir Kaggle notebook hücresine kopyalayıp çalıştırabilirsiniz:

```python
# ============================================
# CSIRO Image2Biomass - Complete Pipeline
# ============================================

# 1. Setup
print("🔧 Setting up...")
!git clone -q https://github.com/EmreUludasdemir/CSIRO---Image2Biomass-.git
%cd CSIRO---Image2Biomass-
!pip install -q timm albumentations pytorch-lightning

# 2. Configure for Kaggle
print("\n⚙️ Configuring for Kaggle...")
import yaml
import sys
sys.path.append('/kaggle/working/CSIRO---Image2Biomass-/src')

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['data_dir'] = '/kaggle/input/csiro-biomass'
config['paths']['checkpoint_dir'] = '/kaggle/working/checkpoints'
config['paths']['submission_dir'] = '/kaggle/working/submissions'
config['training']['num_epochs'] = 15  # Adjust as needed
config['logging']['use_wandb'] = False  # Disable wandb in Kaggle

with open('configs/config_kaggle.yaml', 'w') as f:
    yaml.dump(config, f)

# 3. Quick EDA
print("\n📊 Running EDA...")
!python src/eda.py --data-dir /kaggle/input/csiro-biomass --save-dir /kaggle/working/eda_plots

# 4. Train Model
print("\n🏋️ Training models...")
!python src/train.py --config configs/config_kaggle.yaml

# 5. Generate Submission
print("\n🔮 Generating predictions...")
!python src/inference.py --config configs/config_kaggle.yaml

print("\n✅ Done! Submission saved to /kaggle/working/submissions/")
```

---

## 🎨 Kaggle-Specific Optimizasyonlar

### GPU Memory Optimization

```python
# Config'te şunları ayarla:
config['training']['batch_size'] = 8  # Daha küçük batch
config['training']['accumulation_steps'] = 4  # Gradient accumulation
config['training']['mixed_precision'] = True  # AMP kullan
```

### Kaggle Timeout (9 saat) için Strateji

```python
# Her fold'u ayrı notebook'ta çalıştır
config['data']['n_folds'] = 1
config['training']['num_epochs'] = 20

# Veya daha az epoch
config['training']['num_epochs'] = 10
config['training']['patience'] = 5  # Early stopping
```

---

## 📦 Kaggle Dataset Integration

Eğer kodu Kaggle Dataset olarak yüklemek istersen:

1. Projeyi Kaggle Dataset olarak yükle
2. Notebook'ta dataset'i ekle
3. Kod yolunu güncelle:

```python
import sys
sys.path.append('/kaggle/input/csiro-biomass-solution/src')

# Artık modülleri import edebilirsin
from models import create_model
from train import train_fold
```

---

## 🔍 Debug ve Monitoring

```python
# GPU durumunu kontrol et
!nvidia-smi

# Disk kullanımını kontrol et
!df -h

# Training loglarını izle
!tail -f logs/training.log  # Eğer log dosyası varsa

# Model checkpoint'lerini listele
!ls -lh /kaggle/working/checkpoints/
```

---

## 💾 Checkpoint Kaydetme (Session Kesintisi İçin)

```python
import os

# Her epoch sonrası checkpoint kaydet
# train.py'de zaten yapılıyor

# Manuel kaydetme:
checkpoint_dir = '/kaggle/working/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# En iyi modeli sakla
best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
```

---

## 🚨 Sık Karşılaşılan Sorunlar

### 1. Import Error
```python
# Çözüm: Path'i ekle
import sys
sys.path.append('/kaggle/working/CSIRO---Image2Biomass-/src')
```

### 2. CUDA Out of Memory
```python
# Çözüm: Batch size'ı küçült
config['training']['batch_size'] = 4
config['training']['accumulation_steps'] = 8
```

### 3. File Not Found
```python
# Veri yollarını kontrol et
import os
print("Available datasets:")
print(os.listdir('/kaggle/input/'))
```

### 4. Internet Connection
```python
# Notebook Settings -> Internet: ON olmalı
# GitHub clone için gerekli
```

---

## 📤 Submission İndirme

Kaggle notebook'tan submission'ı indir:

```python
# Submission dosyasını /kaggle/working/ altında oluştur
# Notebook'un sağ tarafındaki "Output" sekmesinden indir
```

---

## 🎯 Önerilen Kaggle Workflow

**Notebook 1: EDA & Baseline**
- Veriyi keşfet
- Basit model dene (EfficientNet-B0)
- Baseline submission

**Notebook 2: Full Training - Fold 0,1,2**
- İlk 3 fold'u train et
- Checkpoint'leri kaydet

**Notebook 3: Full Training - Fold 3,4**
- Son 2 fold'u train et

**Notebook 4: Ensemble & Final Submission**
- Tüm model checkpoint'lerini yükle
- Ensemble predictions
- TTA uygula
- Final submission

---

## 📚 Faydalı Kaggle Komutları

```python
# Tüm dosyaları listele
!find /kaggle/working -type f

# Büyük dosyaları bul
!du -sh /kaggle/working/*

# Checkpoint boyutlarını kontrol et
!ls -lh /kaggle/working/checkpoints/*.pth

# Submission formatını kontrol et
import pandas as pd
sub = pd.read_csv('/kaggle/working/submission.csv')
print(sub.head())
print(sub.info())
```

---

## ✅ Checklist

- [ ] Notebook ayarlarını yap (GPU, Internet ON)
- [ ] GitHub'dan kodu klonla
- [ ] Kaggle veri yollarını güncelle
- [ ] Quick start çalıştır
- [ ] EDA yap
- [ ] Training başlat
- [ ] Submission oluştur
- [ ] Submission'ı kontrol et (format, range)
- [ ] Submit et!

---

**Good luck!** 🚀
