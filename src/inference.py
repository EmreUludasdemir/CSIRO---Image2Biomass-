"""
Inference script for CSIRO Image2Biomass Competition
Supports Test Time Augmentation (TTA) and model ensembling
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models import create_model, create_ensemble, EnsembleModel
from dataset import BiomassDataset, get_tta_transforms, get_valid_transforms
from utils import seed_everything


class Predictor:
    """Predictor class for inference"""

    def __init__(
        self,
        models: List[torch.nn.Module],
        config: Dict,
        device: torch.device,
        use_tta: bool = True,
        ensemble_weights: Optional[List[float]] = None
    ):
        """
        Args:
            models: List of trained models
            config: Configuration dictionary
            device: Device to run inference on
            use_tta: Whether to use Test Time Augmentation
            ensemble_weights: Weights for ensemble averaging
        """
        self.models = [model.to(device).eval() for model in models]
        self.config = config
        self.device = device
        self.use_tta = use_tta

        if ensemble_weights is None:
            ensemble_weights = [1.0 / len(models)] * len(models)
        self.ensemble_weights = ensemble_weights

    @torch.no_grad()
    def predict(
        self,
        test_df: pd.DataFrame,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions on test data

        Args:
            test_df: Test dataframe
            batch_size: Batch size for inference

        Returns:
            Predictions array
        """
        all_predictions = []

        # Get TTA transforms if enabled
        if self.use_tta:
            tta_transforms = get_tta_transforms(
                self.config['data']['img_size'],
                num_tta=self.config['inference'].get('tta_transforms', 5)
            )
        else:
            tta_transforms = [get_valid_transforms(self.config['data']['img_size'])]

        # Iterate over TTA transforms
        for tta_idx, transform in enumerate(tta_transforms):
            print(f'\nTTA {tta_idx + 1}/{len(tta_transforms)}')

            # Create dataset
            test_dataset = BiomassDataset(
                df=test_df,
                image_dir=os.path.join(
                    self.config['data']['data_dir'],
                    self.config['data']['image_dir']
                ),
                img_size=self.config['data']['img_size'],
                transform=transform,
                use_ndvi=self.config['model'].get('use_ndvi', True),
                use_metadata=self.config['model'].get('use_metadata', True),
                metadata_cols=self.config['model'].get('metadata_features', []),
                is_training=False
            )

            # Create dataloader
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            # Collect predictions from all models
            tta_predictions = []

            for model_idx, model in enumerate(self.models):
                model_predictions = []

                for batch in tqdm(test_loader, desc=f'Model {model_idx + 1}/{len(self.models)}'):
                    # Move batch to device
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # Forward pass
                    predictions = model(batch)
                    model_predictions.append(predictions.cpu().numpy())

                # Concatenate predictions
                model_predictions = np.concatenate(model_predictions, axis=0)
                tta_predictions.append(model_predictions)

            # Ensemble predictions with weights
            tta_predictions = np.stack(tta_predictions, axis=0)
            weights = np.array(self.ensemble_weights).reshape(-1, 1, 1)
            weighted_predictions = np.sum(tta_predictions * weights, axis=0)

            all_predictions.append(weighted_predictions)

        # Average TTA predictions
        merge_mode = self.config['inference'].get('tta_merge_mode', 'mean')

        if merge_mode == 'mean':
            final_predictions = np.mean(all_predictions, axis=0)
        elif merge_mode == 'median':
            final_predictions = np.median(all_predictions, axis=0)
        elif merge_mode == 'gmean':
            final_predictions = np.exp(np.mean(np.log(np.array(all_predictions) + 1e-8), axis=0))
        else:
            final_predictions = np.mean(all_predictions, axis=0)

        return final_predictions.squeeze()

    def predict_with_uncertainty(
        self,
        test_df: pd.DataFrame,
        batch_size: int = 32
    ) -> tuple:
        """
        Make predictions with uncertainty estimates

        Args:
            test_df: Test dataframe
            batch_size: Batch size for inference

        Returns:
            Predictions and uncertainties (std)
        """
        all_predictions = []

        # Get TTA transforms
        if self.use_tta:
            tta_transforms = get_tta_transforms(
                self.config['data']['img_size'],
                num_tta=self.config['inference'].get('tta_transforms', 5)
            )
        else:
            tta_transforms = [get_valid_transforms(self.config['data']['img_size'])]

        # Collect all predictions
        for tta_idx, transform in enumerate(tta_transforms):
            test_dataset = BiomassDataset(
                df=test_df,
                image_dir=os.path.join(
                    self.config['data']['data_dir'],
                    self.config['data']['image_dir']
                ),
                img_size=self.config['data']['img_size'],
                transform=transform,
                use_ndvi=self.config['model'].get('use_ndvi', True),
                use_metadata=self.config['model'].get('use_metadata', True),
                metadata_cols=self.config['model'].get('metadata_features', []),
                is_training=False
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            for model in self.models:
                model_predictions = []

                for batch in test_loader:
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    predictions = model(batch)
                    model_predictions.append(predictions.cpu().numpy())

                model_predictions = np.concatenate(model_predictions, axis=0)
                all_predictions.append(model_predictions)

        # Calculate mean and std
        all_predictions = np.stack(all_predictions, axis=0)
        mean_predictions = np.mean(all_predictions, axis=0).squeeze()
        std_predictions = np.std(all_predictions, axis=0).squeeze()

        return mean_predictions, std_predictions


def load_trained_models(
    config: Dict,
    device: torch.device,
    num_metadata_features: int = 0
) -> List[torch.nn.Module]:
    """
    Load trained models from checkpoints

    Args:
        config: Configuration dictionary
        device: Device to load models on
        num_metadata_features: Number of metadata features

    Returns:
        List of loaded models
    """
    models = []
    checkpoint_dir = config['paths']['checkpoint_dir']

    # Get all checkpoint files
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('best_model_fold') and f.endswith('.pth')
    ]

    print(f'Found {len(checkpoint_files)} model checkpoints')

    for checkpoint_file in sorted(checkpoint_files):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model
        model = create_model(checkpoint['config'], num_metadata_features)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        models.append(model)

        print(f'Loaded {checkpoint_file} (RMSE: {checkpoint["score"]:.4f})')

    return models


def create_submission(
    config_path: str,
    test_csv_path: Optional[str] = None,
    output_path: Optional[str] = None
):
    """
    Create submission file

    Args:
        config_path: Path to configuration file
        test_csv_path: Path to test CSV file
        output_path: Path to save submission
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    seed_everything(config['data']['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load test data
    if test_csv_path is None:
        test_csv_path = os.path.join(
            config['data']['data_dir'],
            config['data']['test_csv']
        )

    test_df = pd.read_csv(test_csv_path)
    print(f'Loaded {len(test_df)} test samples')

    # Get number of metadata features
    metadata_cols = config['model'].get('metadata_features', [])
    num_metadata_features = len(metadata_cols)

    # Load trained models
    models = load_trained_models(config, device, num_metadata_features)

    if len(models) == 0:
        print('Error: No trained models found!')
        return

    # Create predictor
    predictor = Predictor(
        models=models,
        config=config,
        device=device,
        use_tta=config['inference'].get('use_tta', True)
    )

    # Make predictions
    print('\nGenerating predictions...')
    predictions = predictor.predict(
        test_df,
        batch_size=config['inference'].get('batch_size', 32)
    )

    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_df['id'],
        'biomass': predictions
    })

    # Save submission
    if output_path is None:
        output_path = os.path.join(
            config['paths']['submission_dir'],
            'submission.csv'
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f'\nSubmission saved to {output_path}')
    print(f'Prediction statistics:')
    print(submission['biomass'].describe())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate CSIRO Image2Biomass predictions')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default=None,
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save submission'
    )

    args = parser.parse_args()

    create_submission(args.config, args.test_csv, args.output)
