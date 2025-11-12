"""
Dataset and DataLoader implementations for CSIRO Image2Biomass Competition
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, LabelEncoder


class BiomassDataset(Dataset):
    """
    Dataset class for loading pasture images and associated metadata
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        img_size: int = 512,
        transform: Optional[A.Compose] = None,
        use_ndvi: bool = True,
        use_metadata: bool = True,
        metadata_cols: Optional[List[str]] = None,
        is_training: bool = True,
        scaler: Optional[StandardScaler] = None,
        label_encoders: Optional[Dict[str, LabelEncoder]] = None
    ):
        """
        Args:
            df: DataFrame with image paths and labels
            image_dir: Directory containing images
            img_size: Target image size
            transform: Albumentations transform pipeline
            use_ndvi: Whether to include NDVI values
            use_metadata: Whether to include metadata features
            metadata_cols: List of metadata column names
            is_training: Whether this is training data
            scaler: Fitted StandardScaler for numerical features
            label_encoders: Dictionary of fitted LabelEncoders for categorical features
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size
        self.transform = transform
        self.use_ndvi = use_ndvi
        self.use_metadata = use_metadata
        self.metadata_cols = metadata_cols or []
        self.is_training = is_training
        self.scaler = scaler
        self.label_encoders = label_encoders or {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        full_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _extract_ndvi(self, image: np.ndarray) -> float:
        """
        Calculate NDVI from RGB image (approximation)
        In real scenario, this would come from NIR and Red bands
        """
        # This is a simplified NDVI calculation
        # In competition, NDVI should be provided in the dataset
        r = image[:, :, 0].astype(float)
        g = image[:, :, 1].astype(float)
        b = image[:, :, 2].astype(float)

        # Approximate NDVI using visible bands
        # ExG (Excess Green Index) as proxy
        exg = 2 * g - r - b
        ndvi_proxy = np.mean(exg) / 255.0

        return ndvi_proxy

    def _get_metadata_features(self, idx: int) -> np.ndarray:
        """Extract and encode metadata features"""
        features = []

        for col in self.metadata_cols:
            if col not in self.df.columns:
                continue

            value = self.df.loc[idx, col]

            # Handle categorical features
            if col in self.label_encoders:
                if pd.isna(value):
                    value = -1
                else:
                    try:
                        value = self.label_encoders[col].transform([value])[0]
                    except:
                        value = -1
                features.append(value)
            # Handle numerical features
            else:
                if pd.isna(value):
                    value = 0.0
                features.append(float(value))

        features = np.array(features, dtype=np.float32)

        # Normalize if scaler is provided
        if self.scaler is not None and len(features) > 0:
            features = self.scaler.transform(features.reshape(1, -1))[0]

        return features

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        row = self.df.iloc[idx]

        # Load image
        image_path = row['image_path'] if 'image_path' in row else row['id'] + '.jpg'
        image = self._load_image(image_path)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        sample = {'image': image}

        # Add NDVI if available
        if self.use_ndvi:
            if 'ndvi' in row:
                ndvi = float(row['ndvi'])
            else:
                # Calculate from image if not provided
                ndvi = self._extract_ndvi(cv2.resize(image, (self.img_size, self.img_size)))
            sample['ndvi'] = torch.tensor([ndvi], dtype=torch.float32)

        # Add metadata features
        if self.use_metadata and len(self.metadata_cols) > 0:
            metadata = self._get_metadata_features(idx)
            sample['metadata'] = torch.from_numpy(metadata)

        # Add target if training
        if self.is_training and 'target' in row:
            sample['target'] = torch.tensor([row['target']], dtype=torch.float32)
        elif self.is_training and 'biomass' in row:
            sample['target'] = torch.tensor([row['biomass']], dtype=torch.float32)

        # Add image ID for tracking
        sample['id'] = row['id'] if 'id' in row else idx

        return sample


def get_train_transforms(img_size: int, config: Dict) -> A.Compose:
    """Get training augmentation pipeline"""
    aug_config = config.get('augmentation', {}).get('train', {})

    transforms = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=aug_config.get('vertical_flip', 0.5)),
        A.Rotate(limit=aug_config.get('rotate_limit', 45), p=0.7),
        A.RandomBrightnessContrast(
            brightness_limit=aug_config.get('brightness_limit', 0.2),
            contrast_limit=aug_config.get('contrast_limit', 0.2),
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=int(aug_config.get('hue_shift_limit', 0.1) * 100),
            sat_shift_limit=int(aug_config.get('saturation_shift_limit', 0.2) * 100),
            val_shift_limit=int(aug_config.get('brightness_limit', 0.2) * 100),
            p=0.7
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=aug_config.get('blur_limit', 3), p=1.0),
            A.MedianBlur(blur_limit=aug_config.get('blur_limit', 3), p=1.0),
            A.MotionBlur(blur_limit=aug_config.get('blur_limit', 3), p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.7
        ),
        A.CoarseDropout(
            max_holes=aug_config.get('cutout', {}).get('num_holes', 8),
            max_height=aug_config.get('cutout', {}).get('max_h_size', 32),
            max_width=aug_config.get('cutout', {}).get('max_w_size', 32),
            fill_value=0,
            p=aug_config.get('cutout', {}).get('p', 0.5)
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]

    return A.Compose(transforms)


def get_valid_transforms(img_size: int) -> A.Compose:
    """Get validation augmentation pipeline"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_tta_transforms(img_size: int, num_tta: int = 5) -> List[A.Compose]:
    """Get Test Time Augmentation transforms"""
    tta_transforms = []

    # Original
    tta_transforms.append(get_valid_transforms(img_size))

    if num_tta >= 2:
        # Horizontal flip
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))

    if num_tta >= 3:
        # Vertical flip
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))

    if num_tta >= 4:
        # Slight brightness adjustment
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))

    if num_tta >= 5:
        # Both flips
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))

    return tta_transforms


def prepare_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: Dict,
    scaler: Optional[StandardScaler] = None,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation dataloaders"""

    img_size = config['data']['img_size']
    batch_size = config['training']['batch_size']
    metadata_cols = config['model'].get('metadata_features', [])

    # Create datasets
    train_dataset = BiomassDataset(
        df=train_df,
        image_dir=os.path.join(config['data']['data_dir'], config['data']['image_dir']),
        img_size=img_size,
        transform=get_train_transforms(img_size, config),
        use_ndvi=config['model'].get('use_ndvi', True),
        use_metadata=config['model'].get('use_metadata', True),
        metadata_cols=metadata_cols,
        is_training=True,
        scaler=scaler,
        label_encoders=label_encoders
    )

    valid_dataset = BiomassDataset(
        df=valid_df,
        image_dir=os.path.join(config['data']['data_dir'], config['data']['image_dir']),
        img_size=img_size,
        transform=get_valid_transforms(img_size),
        use_ndvi=config['model'].get('use_ndvi', True),
        use_metadata=config['model'].get('use_metadata', True),
        metadata_cols=metadata_cols,
        is_training=True,
        scaler=scaler,
        label_encoders=label_encoders
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, valid_loader
