"""
Model architectures for CSIRO Image2Biomass Competition
Implements multi-modal models combining image features with NDVI and metadata
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional


class MultiModalBiomassModel(nn.Module):
    """
    Multi-modal model for biomass prediction
    Combines CNN backbone with NDVI and metadata features
    """

    def __init__(
        self,
        model_name: str = 'efficientnet_b3',
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        use_ndvi: bool = True,
        use_metadata: bool = True,
        num_metadata_features: int = 0
    ):
        """
        Args:
            model_name: Name of the backbone model from timm
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes (1 for regression)
            dropout: Dropout rate
            use_ndvi: Whether to use NDVI features
            use_metadata: Whether to use metadata features
            num_metadata_features: Number of metadata features
        """
        super(MultiModalBiomassModel, self).__init__()

        self.model_name = model_name
        self.use_ndvi = use_ndvi
        self.use_metadata = use_metadata

        # Create backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_features = self.backbone(dummy_input)
            self.backbone_dim = backbone_features.shape[1]

        # NDVI processing
        if use_ndvi:
            self.ndvi_fc = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
            ndvi_dim = 64
        else:
            ndvi_dim = 0

        # Metadata processing
        if use_metadata and num_metadata_features > 0:
            self.metadata_fc = nn.Sequential(
                nn.Linear(num_metadata_features, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout * 0.5)
            )
            metadata_dim = 128
        else:
            metadata_dim = 0

        # Fusion layer
        total_features = self.backbone_dim + ndvi_dim + metadata_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(total_features, total_features // 4),
            nn.ReLU(),
            nn.Linear(total_features // 4, total_features),
            nn.Sigmoid()
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Args:
            batch: Dictionary containing 'image', 'ndvi', 'metadata' tensors

        Returns:
            Predicted biomass values
        """
        # Extract image features
        image = batch['image']
        image_features = self.backbone(image)

        features = [image_features]

        # Process NDVI
        if self.use_ndvi and 'ndvi' in batch:
            ndvi = batch['ndvi']
            ndvi_features = self.ndvi_fc(ndvi)
            features.append(ndvi_features)

        # Process metadata
        if self.use_metadata and 'metadata' in batch:
            metadata = batch['metadata']
            if metadata.dim() == 1:
                metadata = metadata.unsqueeze(0)
            metadata_features = self.metadata_fc(metadata)
            features.append(metadata_features)

        # Concatenate all features
        combined_features = torch.cat(features, dim=1)

        # Apply attention
        attention_weights = self.attention(combined_features)
        combined_features = combined_features * attention_weights

        # Final prediction
        output = self.fusion(combined_features)

        return output


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            models: List of trained models
            weights: Optional weights for weighted averaging
        """
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        self.weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through all models and ensemble predictions

        Args:
            batch: Input batch

        Returns:
            Ensembled predictions
        """
        predictions = []

        for model in self.models:
            with torch.no_grad():
                pred = model(batch)
                predictions.append(pred)

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch_size, 1)

        # Weighted average
        weights = self.weights.view(-1, 1, 1)
        ensemble_pred = torch.sum(predictions * weights, dim=0)

        return ensemble_pred


def create_model(config: Dict, num_metadata_features: int = 0) -> nn.Module:
    """
    Factory function to create a model based on config

    Args:
        config: Configuration dictionary
        num_metadata_features: Number of metadata features

    Returns:
        Initialized model
    """
    model_config = config['model']

    model = MultiModalBiomassModel(
        model_name=model_config['architectures'][0] if isinstance(model_config['architectures'], list) else model_config['architectures'],
        pretrained=model_config.get('pretrained', True),
        num_classes=model_config.get('num_classes', 1),
        dropout=model_config.get('dropout', 0.3),
        use_ndvi=model_config.get('use_ndvi', True),
        use_metadata=model_config.get('use_metadata', True),
        num_metadata_features=num_metadata_features
    )

    return model


def create_ensemble(config: Dict, num_metadata_features: int = 0) -> List[nn.Module]:
    """
    Create multiple models for ensembling

    Args:
        config: Configuration dictionary
        num_metadata_features: Number of metadata features

    Returns:
        List of models
    """
    model_config = config['model']
    architectures = model_config.get('architectures', ['efficientnet_b3'])

    if not isinstance(architectures, list):
        architectures = [architectures]

    models = []
    for arch in architectures:
        model = MultiModalBiomassModel(
            model_name=arch,
            pretrained=model_config.get('pretrained', True),
            num_classes=model_config.get('num_classes', 1),
            dropout=model_config.get('dropout', 0.3),
            use_ndvi=model_config.get('use_ndvi', True),
            use_metadata=model_config.get('use_metadata', True),
            num_metadata_features=num_metadata_features
        )
        models.append(model)

    return models


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss"""

    def __init__(self, eps: float = 1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt(self.mse(pred, target) + self.eps)
        return loss


def get_loss_function(loss_name: str) -> nn.Module:
    """
    Get loss function by name

    Args:
        loss_name: Name of the loss function

    Returns:
        Loss function
    """
    loss_functions = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(),
        'huber': nn.HuberLoss(),
        'rmse': RMSELoss()
    }

    return loss_functions.get(loss_name, nn.SmoothL1Loss())


if __name__ == '__main__':
    # Test model creation
    print("Testing MultiModalBiomassModel...")

    model = MultiModalBiomassModel(
        model_name='efficientnet_b3',
        pretrained=False,
        use_ndvi=True,
        use_metadata=True,
        num_metadata_features=10
    )

    # Create dummy batch
    batch = {
        'image': torch.randn(4, 3, 224, 224),
        'ndvi': torch.randn(4, 1),
        'metadata': torch.randn(4, 10)
    }

    # Forward pass
    output = model(batch)
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
