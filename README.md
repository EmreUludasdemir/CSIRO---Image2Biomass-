# CSIRO - Image2Biomass Prediction

**Competition**: [CSIRO Image2Biomass Kaggle Competition](https://www.kaggle.com/competitions/csiro-biomass)

A state-of-the-art deep learning solution for predicting pasture biomass from images using multi-modal architecture with ensemble methods and Test Time Augmentation.

---

## Competition Overview

The CSIRO Image2Biomass competition, hosted by CSIRO, Meat & Livestock Australia (MLA), and Google Australia, challenges participants to build AI models that accurately predict pasture biomass from images, ground-truth measurements, and publicly available datasets.

- **Prize Pool**: US$75,000
- **Deadline**: January 28, 2026
- **Goal**: Improve accuracy and efficiency in estimating pasture biomass to support better grazing management decisions

---

## Solution Overview

This solution implements a **multi-modal deep learning approach** combining:

1. **CNN Backbone**: Multiple state-of-the-art architectures (EfficientNet, ConvNeXt, Swin Transformer)
2. **NDVI Integration**: Normalized Difference Vegetation Index features
3. **Metadata Features**: Season, region, plant height, species diversity
4. **Advanced Techniques**:
   - Heavy data augmentation (rotation, flip, color jitter, cutout, mixup, cutmix)
   - Test Time Augmentation (TTA)
   - Cross-validation with ensemble
   - Mixed precision training
   - Gradient accumulation

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Input Images                       │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    ┌────▼────┐         ┌────▼────┐
    │  Image  │         │  NDVI   │
    │Backbone │         │Features │
    │ (CNN)   │         │         │
    └────┬────┘         └────┬────┘
         │                   │
         └─────────┬─────────┘
                   │
              ┌────▼────┐
              │Metadata │
              │Features │
              └────┬────┘
                   │
            ┌──────▼──────┐
            │  Attention  │
            │   Fusion    │
            └──────┬──────┘
                   │
              ┌────▼────┐
              │  MLP    │
              │  Head   │
              └────┬────┘
                   │
            ┌──────▼──────┐
            │  Biomass    │
            │ Prediction  │
            └─────────────┘
```

---

## Project Structure

```
CSIRO-Image2Biomass/
├── configs/
│   └── config.yaml              # Main configuration file
├── data/                        # Data directory (not tracked)
│   ├── train.csv
│   ├── test.csv
│   └── images/
├── models/                      # Model checkpoints (not tracked)
│   └── checkpoints/
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Dataset and data loading
│   ├── models.py               # Model architectures
│   ├── train.py                # Training pipeline
│   ├── inference.py            # Inference and submission
│   ├── utils.py                # Utility functions
│   └── eda.py                  # Exploratory data analysis
├── notebooks/                  # Jupyter notebooks
├── submissions/                # Submission files
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended
- GPU with 8GB+ VRAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/CSIRO-Image2Biomass.git
cd CSIRO-Image2Biomass

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data Preparation

1. Download the competition data from [Kaggle](https://www.kaggle.com/competitions/csiro-biomass/data)

2. Extract and organize the data:
```
data/
├── train.csv
├── test.csv
└── images/
    ├── train/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    └── test/
        ├── image_test_001.jpg
        └── ...
```

---

## Usage

### 1. Exploratory Data Analysis

Run EDA to understand the dataset:

```bash
python src/eda.py \
    --data-dir ./data \
    --train-csv train.csv \
    --image-dir images \
    --save-dir ./eda_plots
```

This will generate:
- Target distribution plots
- Feature correlation analysis
- NDVI analysis
- Sample image visualizations

### 2. Training

Train the model with cross-validation:

```bash
python src/train.py --config configs/config.yaml
```

**Training options in `config.yaml`:**
- Model architectures: `efficientnet_b3`, `convnext_base`, `swin_transformer_base`
- Training hyperparameters: learning rate, batch size, epochs
- Augmentation settings
- Cross-validation folds

**Training features:**
- ✅ 5-fold cross-validation
- ✅ Mixed precision training (AMP)
- ✅ Gradient accumulation
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ WandB logging (optional)

### 3. Inference

Generate predictions on test set:

```bash
python src/inference.py \
    --config configs/config.yaml \
    --test-csv data/test.csv \
    --output submissions/submission.csv
```

**Inference features:**
- ✅ Test Time Augmentation (TTA)
- ✅ Model ensemble
- ✅ Uncertainty estimation
- ✅ Multiple merge strategies (mean, median, gmean)

---

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model settings
model:
  architectures:
    - efficientnet_b3
    - convnext_base
    - swin_transformer_base
  use_ndvi: true
  use_metadata: true

# Training settings
training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.0003
  mixed_precision: true

# Augmentation
augmentation:
  train:
    horizontal_flip: 0.5
    rotate_limit: 45
    mixup_alpha: 0.2

# Inference
inference:
  use_tta: true
  tta_transforms: 5
```

---

## Model Architecture Details

### Multi-Modal Fusion

The model combines three types of inputs:

1. **Image Features** (from CNN backbone):
   - EfficientNet-B3/B4: Efficient compound scaling
   - ConvNeXt: Modern ConvNet architecture
   - Swin Transformer: Hierarchical vision transformer

2. **NDVI Features**:
   - Normalized Difference Vegetation Index
   - Processed through dedicated MLP

3. **Metadata Features**:
   - Season, region, plant height, species diversity
   - Encoded and processed through BatchNorm + MLP

All features are fused using an **attention mechanism** before final prediction.

### Loss Functions

Supported loss functions:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Smooth L1 Loss (default)
- Huber Loss
- Root Mean Squared Error (RMSE)

---

## Advanced Features

### Data Augmentation

**Spatial Augmentations:**
- Horizontal/Vertical Flip
- Rotation (±45°)
- ShiftScaleRotate
- Coarse Dropout (Cutout)

**Color Augmentations:**
- Brightness/Contrast adjustment
- Hue/Saturation/Value shift
- Gaussian blur
- Gaussian noise

**Mixing Augmentations:**
- Mixup (α=0.2)
- CutMix (α=1.0)

### Test Time Augmentation (TTA)

Multiple augmentations applied during inference:
1. Original image
2. Horizontal flip
3. Vertical flip
4. Slight brightness adjustment
5. Both flips combined

Predictions are merged using mean/median/geometric mean.

### Ensemble Strategy

Multiple models are trained independently and their predictions are combined:
- Equal weighting
- Performance-based weighting
- Automatic weight optimization

---

## Performance Optimization

### Speed Optimizations
- Mixed precision training (AMP)
- Gradient accumulation
- Multi-worker data loading
- Pin memory for GPU transfer
- ONNX export (optional)

### Memory Optimizations
- Gradient checkpointing
- Efficient data augmentation pipeline
- Dynamic batch sizing
- Image preprocessing caching

---

## Monitoring and Logging

### WandB Integration

Enable Weights & Biases logging in `config.yaml`:

```yaml
logging:
  use_wandb: true
  project_name: "csiro-image2biomass"
  experiment_name: "multi_modal_ensemble"
```

Track:
- Training/validation loss
- RMSE, MAE, R² metrics
- Learning rate schedules
- Model predictions
- Confusion matrices

---

## Tips for Better Performance

1. **Data Quality**:
   - Ensure consistent image sizes
   - Handle missing NDVI values
   - Validate metadata integrity

2. **Training**:
   - Start with lower learning rate (3e-4)
   - Use cosine annealing scheduler
   - Monitor validation metrics closely
   - Enable early stopping

3. **Augmentation**:
   - Increase augmentation strength gradually
   - Use domain-specific augmentations
   - Balance spatial and color augmentations

4. **Ensemble**:
   - Train models with different architectures
   - Use different random seeds
   - Vary augmentation strategies
   - Optimize ensemble weights on validation set

5. **Inference**:
   - Always use TTA for final submission
   - Test different merge strategies
   - Consider pseudo-labeling for semi-supervised learning

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
```yaml
# Reduce in config.yaml
training:
  batch_size: 8  # Reduce from 16
  accumulation_steps: 4  # Increase to compensate
```

**Slow Training**:
```yaml
# Enable optimizations
training:
  mixed_precision: true  # Use AMP
data:
  num_workers: 4  # Increase data loading workers
```

**Overfitting**:
```yaml
model:
  dropout: 0.5  # Increase dropout
training:
  weight_decay: 0.001  # Increase regularization
augmentation:
  train:
    # Increase augmentation strength
```

**Underfitting**:
```yaml
training:
  num_epochs: 100  # Train longer
  learning_rate: 0.001  # Increase LR
model:
  dropout: 0.2  # Reduce dropout
```

---

## Results

### Cross-Validation Performance

| Fold | RMSE | MAE | R² |
|------|------|-----|-----|
| 0    | TBD  | TBD | TBD |
| 1    | TBD  | TBD | TBD |
| 2    | TBD  | TBD | TBD |
| 3    | TBD  | TBD | TBD |
| 4    | TBD  | TBD | TBD |
| **Mean** | **TBD** | **TBD** | **TBD** |

*Results will be updated after training on actual competition data*

---

## Citation

If you use this code in your research or competition submission, please cite:

```bibtex
@misc{csiro-image2biomass-solution,
  title={Multi-Modal Deep Learning Solution for CSIRO Image2Biomass Competition},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/CSIRO-Image2Biomass}
}
```

---

## Acknowledgments

- CSIRO, Meat & Livestock Australia (MLA), and Google Australia for organizing the competition
- FrontierSI for collaboration on the dataset
- PyTorch and Timm library communities
- Albumentations for efficient augmentation pipeline

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions or discussions about this solution:
- Open an issue on GitHub
- Competition discussion forum on Kaggle

---

## Roadmap

- [ ] Add pseudo-labeling for semi-supervised learning
- [ ] Implement knowledge distillation
- [ ] Add Vision Transformer (ViT) backbone
- [ ] Experiment with self-supervised pre-training
- [ ] Add gradient-weighted class activation mapping (Grad-CAM) visualization
- [ ] Implement automatic hyperparameter tuning (Optuna)
- [ ] Create Docker container for easy deployment
- [ ] Add model compression (quantization, pruning)

---

**Good luck with the competition!** 🚀
