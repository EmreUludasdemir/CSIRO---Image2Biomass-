"""
Training pipeline for CSIRO Image2Biomass Competition
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models import create_model, get_loss_function
from dataset import prepare_dataloaders
from utils import AverageMeter, EarlyStopping, seed_everything, save_checkpoint


class Trainer:
    """Trainer class for model training"""

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        fold: int,
        device: torch.device
    ):
        """
        Args:
            model: PyTorch model
            config: Configuration dictionary
            fold: Current fold number
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.fold = fold
        self.device = device

        # Setup optimizer
        self.optimizer = self._get_optimizer()

        # Setup scheduler
        self.scheduler = None  # Will be set after knowing steps per epoch

        # Setup loss function
        self.criterion = get_loss_function(
            config['training'].get('loss_fn', 'smooth_l1')
        )

        # Setup mixed precision
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 10),
            min_delta=config['training'].get('min_delta', 0.001),
            mode='min'
        )

        # Best score tracking
        self.best_score = float('inf')

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        training_config = self.config['training']

        optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 0.0003),
            weight_decay=training_config.get('weight_decay', 0.0001)
        )

        return optimizer

    def _get_scheduler(self, steps_per_epoch: int):
        """Create learning rate scheduler"""
        training_config = self.config['training']
        scheduler_name = training_config.get('scheduler', 'CosineAnnealingLR')

        if scheduler_name == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('num_epochs', 50),
                eta_min=1e-6
            )
        elif scheduler_name == 'OneCycleLR':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=training_config.get('learning_rate', 0.0003),
                steps_per_epoch=steps_per_epoch,
                epochs=training_config.get('num_epochs', 50)
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        losses = AverageMeter()

        pbar = tqdm(train_loader, desc=f'Fold {self.fold} | Epoch {epoch}')

        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                predictions = self.model(batch)
                targets = batch['target']
                loss = self.criterion(predictions, targets)
                loss = loss / self.accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (step + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    if self.config['training'].get('gradient_clip', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['gradient_clip']
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.config['training'].get('gradient_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['gradient_clip']
                        )
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

            # Update metrics
            losses.update(loss.item() * self.accumulation_steps, batch['target'].size(0))

            # Update progress bar
            pbar.set_postfix({'loss': losses.avg})

        return losses.avg

    @torch.no_grad()
    def validate(self, valid_loader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        losses = AverageMeter()

        all_predictions = []
        all_targets = []

        for batch in tqdm(valid_loader, desc=f'Fold {self.fold} | Validation'):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            with autocast(enabled=self.use_amp):
                predictions = self.model(batch)
                targets = batch['target']
                loss = self.criterion(predictions, targets)

            # Update metrics
            losses.update(loss.item(), targets.size(0))

            # Collect predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Calculate RMSE
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

        return losses.avg, rmse

    def fit(
        self,
        train_loader,
        valid_loader,
        num_epochs: int
    ) -> Dict:
        """
        Train the model

        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            num_epochs: Number of epochs to train

        Returns:
            Training history
        """
        # Setup scheduler
        self.scheduler = self._get_scheduler(len(train_loader))

        history = {
            'train_loss': [],
            'valid_loss': [],
            'valid_rmse': []
        }

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            valid_loss, valid_rmse = self.validate(valid_loader)

            # Step scheduler
            if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # Update history
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['valid_rmse'].append(valid_rmse)

            # Log metrics
            print(f'\nEpoch {epoch}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_loss:.4f} | Valid RMSE: {valid_rmse:.4f}')

            # Log to wandb if enabled
            if self.config['logging'].get('use_wandb', False):
                wandb.log({
                    f'fold_{self.fold}/train_loss': train_loss,
                    f'fold_{self.fold}/valid_loss': valid_loss,
                    f'fold_{self.fold}/valid_rmse': valid_rmse,
                    f'fold_{self.fold}/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })

            # Save best model
            if valid_rmse < self.best_score:
                self.best_score = valid_rmse
                self.save_checkpoint(epoch, valid_rmse, is_best=True)
                print(f'Best model saved! RMSE: {valid_rmse:.4f}')

            # Early stopping
            self.early_stopping(valid_rmse)
            if self.early_stopping.early_stop:
                print(f'Early stopping triggered at epoch {epoch}')
                break

        return history

    def save_checkpoint(self, epoch: int, score: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'config': self.config
        }

        if is_best:
            path = os.path.join(
                checkpoint_dir,
                f'best_model_fold{self.fold}.pth'
            )
        else:
            path = os.path.join(
                checkpoint_dir,
                f'model_fold{self.fold}_epoch{epoch}.pth'
            )

        torch.save(checkpoint, path)


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: Dict,
    device: torch.device
) -> Tuple[float, nn.Module]:
    """
    Train a single fold

    Args:
        fold: Fold number
        train_df: Training dataframe
        valid_df: Validation dataframe
        config: Configuration dictionary
        device: Device to train on

    Returns:
        Best validation RMSE and trained model
    """
    print(f'\n{"="*50}')
    print(f'Training Fold {fold}')
    print(f'{"="*50}\n')

    # Prepare metadata preprocessing
    metadata_cols = config['model'].get('metadata_features', [])
    scaler = None
    label_encoders = {}

    if config['model'].get('use_metadata', False) and len(metadata_cols) > 0:
        # Fit scaler and encoders on training data
        scaler = StandardScaler()
        numerical_cols = []
        categorical_cols = []

        for col in metadata_cols:
            if col in train_df.columns:
                if train_df[col].dtype in ['object', 'category']:
                    categorical_cols.append(col)
                    le = LabelEncoder()
                    le.fit(train_df[col].dropna().astype(str))
                    label_encoders[col] = le
                else:
                    numerical_cols.append(col)

        if numerical_cols:
            scaler.fit(train_df[numerical_cols].fillna(0))

    # Get number of metadata features
    num_metadata_features = len(metadata_cols)

    # Create model
    model = create_model(config, num_metadata_features)

    # Create dataloaders
    train_loader, valid_loader = prepare_dataloaders(
        train_df, valid_df, config, scaler, label_encoders
    )

    # Create trainer
    trainer = Trainer(model, config, fold, device)

    # Train
    history = trainer.fit(
        train_loader,
        valid_loader,
        num_epochs=config['training']['num_epochs']
    )

    # Load best model
    checkpoint_path = os.path.join(
        config['paths']['checkpoint_dir'],
        f'best_model_fold{fold}.pth'
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return trainer.best_score, model


def train_cv(config_path: str):
    """
    Run cross-validation training

    Args:
        config_path: Path to configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    seed_everything(config['data']['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize wandb
    if config['logging'].get('use_wandb', False):
        wandb.init(
            project=config['logging']['project_name'],
            name=config['logging']['experiment_name'],
            config=config
        )

    # Load data
    data_dir = config['data']['data_dir']
    train_csv_path = os.path.join(data_dir, config['data']['train_csv'])
    train_df = pd.read_csv(train_csv_path)

    print(f'Loaded {len(train_df)} training samples')

    # Create folds
    n_folds = config['data']['n_folds']
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config['data']['seed'])

    # Track fold scores
    fold_scores = []

    # Train each fold
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[valid_idx].reset_index(drop=True)

        best_score, model = train_fold(
            fold=fold,
            train_df=fold_train_df,
            valid_df=fold_valid_df,
            config=config,
            device=device
        )

        fold_scores.append(best_score)

        print(f'\nFold {fold} Best RMSE: {best_score:.4f}')

    # Print final results
    print(f'\n{"="*50}')
    print('Cross-Validation Results')
    print(f'{"="*50}')
    for fold, score in enumerate(fold_scores):
        print(f'Fold {fold}: {score:.4f}')
    print(f'Mean RMSE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}')
    print(f'{"="*50}\n')

    # Log to wandb
    if config['logging'].get('use_wandb', False):
        wandb.log({
            'cv_mean_rmse': np.mean(fold_scores),
            'cv_std_rmse': np.std(fold_scores)
        })
        wandb.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train CSIRO Image2Biomass model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    train_cv(args.config)
