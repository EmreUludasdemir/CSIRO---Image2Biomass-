"""
Quick start script for CSIRO Image2Biomass Competition
Run this to test the setup and train a simple model
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models import MultiModalBiomassModel, create_model
from utils import seed_everything, get_device, print_model_summary


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")

    required_packages = [
        'torch',
        'torchvision',
        'timm',
        'numpy',
        'pandas',
        'cv2',
        'albumentations',
        'sklearn',
        'tqdm',
        'yaml'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies installed!\n")
    return True


def check_data():
    """Check if data is properly set up"""
    print("Checking data setup...")

    data_dir = Path('./data')
    required_files = ['train.csv', 'test.csv']
    required_dirs = ['images']

    issues = []

    if not data_dir.exists():
        print("✗ Data directory not found")
        print("  Please create ./data directory and download competition data")
        return False

    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
            issues.append(file)

    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            issues.append(dir_name)

    if issues:
        print("\nPlease download the competition data from Kaggle:")
        print("https://www.kaggle.com/competitions/csiro-biomass/data")
        return False

    print("\n✓ Data setup complete!\n")
    return True


def check_gpu():
    """Check GPU availability"""
    print("Checking GPU...")

    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ No GPU available - will use CPU")
        print("  Training will be significantly slower on CPU")

    print()
    return True


def test_model():
    """Test model creation"""
    print("Testing model creation...")

    try:
        # Load config
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Create a simple model
        model = create_model(config, num_metadata_features=10)

        print(f"✓ Model created successfully")
        print(f"  Architecture: {config['model']['architectures'][0]}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        device = get_device()
        model = model.to(device)

        dummy_batch = {
            'image': torch.randn(2, 3, 224, 224).to(device),
            'ndvi': torch.randn(2, 1).to(device),
            'metadata': torch.randn(2, 10).to(device)
        }

        with torch.no_grad():
            output = model(dummy_batch)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_batch['image'].shape}")
        print(f"  Output shape: {output.shape}")

        print("\n✓ Model test passed!\n")
        return True

    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")
        return False


def create_dummy_data():
    """Create dummy data for testing"""
    print("Creating dummy data for testing...")

    import pandas as pd
    import numpy as np

    # Create dummy train.csv
    n_samples = 100
    dummy_train = pd.DataFrame({
        'id': range(n_samples),
        'image_path': [f'train/image_{i:04d}.jpg' for i in range(n_samples)],
        'biomass': np.random.uniform(100, 1000, n_samples),
        'ndvi': np.random.uniform(0.2, 0.9, n_samples),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'plant_height': np.random.uniform(5, 30, n_samples),
        'species_diversity': np.random.uniform(1, 10, n_samples)
    })

    # Create dummy test.csv
    n_test = 50
    dummy_test = pd.DataFrame({
        'id': range(n_test),
        'image_path': [f'test/image_{i:04d}.jpg' for i in range(n_test)],
        'ndvi': np.random.uniform(0.2, 0.9, n_test),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_test),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_test),
        'plant_height': np.random.uniform(5, 30, n_test),
        'species_diversity': np.random.uniform(1, 10, n_test)
    })

    # Save to data directory
    os.makedirs('./data', exist_ok=True)
    dummy_train.to_csv('./data/train_dummy.csv', index=False)
    dummy_test.to_csv('./data/test_dummy.csv', index=False)

    print("✓ Dummy data created")
    print(f"  Train samples: {len(dummy_train)}")
    print(f"  Test samples: {len(dummy_test)}")
    print(f"  Saved to: ./data/train_dummy.csv and ./data/test_dummy.csv")
    print()


def main():
    """Main quick start function"""
    print("=" * 70)
    print("CSIRO IMAGE2BIOMASS - QUICK START")
    print("=" * 70)
    print()

    seed_everything(42)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup incomplete - please install missing dependencies")
        return

    # Check GPU
    check_gpu()

    # Check data
    data_available = check_data()

    if not data_available:
        print("Would you like to create dummy data for testing? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            create_dummy_data()
        else:
            print("\n❌ Setup incomplete - please download competition data")
            return

    # Test model
    if not test_model():
        print("\n❌ Model test failed")
        return

    print("=" * 70)
    print("✓ QUICK START COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run EDA: python src/eda.py")
    print("2. Train model: python src/train.py --config configs/config.yaml")
    print("3. Generate predictions: python src/inference.py --config configs/config.yaml")
    print()
    print("For more information, see README.md")
    print()


if __name__ == '__main__':
    main()
