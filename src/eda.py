"""
Exploratory Data Analysis for CSIRO Image2Biomass Competition
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BiomassEDA:
    """Class for performing EDA on biomass dataset"""

    def __init__(self, data_dir: str, train_csv: str, image_dir: str):
        """
        Args:
            data_dir: Base data directory
            train_csv: Training CSV filename
            image_dir: Image directory name
        """
        self.data_dir = data_dir
        self.train_csv_path = os.path.join(data_dir, train_csv)
        self.image_dir = os.path.join(data_dir, image_dir)

        # Load data
        self.train_df = pd.read_csv(self.train_csv_path)
        print(f'Loaded {len(self.train_df)} training samples')

    def basic_info(self):
        """Print basic dataset information"""
        print('\n' + '=' * 50)
        print('BASIC DATASET INFORMATION')
        print('=' * 50)

        print(f'\nDataset shape: {self.train_df.shape}')
        print(f'\nColumns: {list(self.train_df.columns)}')

        print('\nData types:')
        print(self.train_df.dtypes)

        print('\nMissing values:')
        missing = self.train_df.isnull().sum()
        missing_pct = 100 * missing / len(self.train_df)
        missing_table = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_table[missing_table['Missing Count'] > 0])

        print('\nFirst few rows:')
        print(self.train_df.head())

    def target_distribution(self, save_dir: Optional[str] = None):
        """Analyze and plot target distribution"""
        print('\n' + '=' * 50)
        print('TARGET DISTRIBUTION ANALYSIS')
        print('=' * 50)

        # Assume target column is 'biomass' or 'target'
        target_col = 'biomass' if 'biomass' in self.train_df.columns else 'target'

        if target_col not in self.train_df.columns:
            print('Warning: Target column not found!')
            return

        target = self.train_df[target_col]

        print(f'\nTarget statistics:')
        print(target.describe())

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Histogram
        axes[0, 0].hist(target, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Biomass')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].axvline(target.mean(), color='r', linestyle='--', label=f'Mean: {target.mean():.2f}')
        axes[0, 0].axvline(target.median(), color='g', linestyle='--', label=f'Median: {target.median():.2f}')
        axes[0, 0].legend()

        # Box plot
        axes[0, 1].boxplot(target, vert=True)
        axes[0, 1].set_ylabel('Biomass')
        axes[0, 1].set_title('Target Box Plot')

        # Log-scale histogram
        axes[1, 0].hist(np.log1p(target), bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Log(Biomass + 1)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Log-Transformed Target Distribution')

        # Cumulative distribution
        sorted_target = np.sort(target)
        y = np.arange(len(sorted_target)) / float(len(sorted_target))
        axes[1, 1].plot(sorted_target, y)
        axes[1, 1].set_xlabel('Biomass')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Function')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'target_distribution.png'), dpi=300, bbox_inches='tight')
            print(f'\nSaved target distribution plot to {save_dir}')

        plt.show()

    def feature_analysis(self, save_dir: Optional[str] = None):
        """Analyze features and their correlation with target"""
        print('\n' + '=' * 50)
        print('FEATURE ANALYSIS')
        print('=' * 50)

        # Get numerical columns
        numerical_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) <= 1:
            print('Not enough numerical features for analysis')
            return

        # Correlation matrix
        corr_matrix = self.train_df[numerical_cols].corr()

        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            print(f'\nSaved correlation matrix to {save_dir}')

        plt.show()

        # Correlation with target
        target_col = 'biomass' if 'biomass' in self.train_df.columns else 'target'
        if target_col in numerical_cols:
            target_corr = corr_matrix[target_col].sort_values(ascending=False)
            print(f'\nCorrelation with {target_col}:')
            print(target_corr)

    def categorical_analysis(self, save_dir: Optional[str] = None):
        """Analyze categorical features"""
        print('\n' + '=' * 50)
        print('CATEGORICAL FEATURE ANALYSIS')
        print('=' * 50)

        # Get categorical columns
        categorical_cols = self.train_df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(categorical_cols) == 0:
            print('No categorical features found')
            return

        target_col = 'biomass' if 'biomass' in self.train_df.columns else 'target'

        n_cols = min(len(categorical_cols), 3)
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for idx, col in enumerate(categorical_cols):
            # Value counts
            value_counts = self.train_df[col].value_counts()
            print(f'\n{col} value counts:')
            print(value_counts)

            # Plot
            if target_col in self.train_df.columns:
                self.train_df.groupby(col)[target_col].mean().plot(kind='bar', ax=axes[idx])
                axes[idx].set_title(f'Mean {target_col} by {col}')
                axes[idx].set_ylabel(f'Mean {target_col}')
            else:
                value_counts.plot(kind='bar', ax=axes[idx])
                axes[idx].set_title(f'{col} Distribution')
                axes[idx].set_ylabel('Count')

            axes[idx].tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'categorical_analysis.png'), dpi=300, bbox_inches='tight')
            print(f'\nSaved categorical analysis to {save_dir}')

        plt.show()

    def image_analysis(self, n_samples: int = 12, save_dir: Optional[str] = None):
        """Analyze and visualize sample images"""
        print('\n' + '=' * 50)
        print('IMAGE ANALYSIS')
        print('=' * 50)

        # Sample images
        sample_df = self.train_df.sample(n=min(n_samples, len(self.train_df)), random_state=42)

        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        target_col = 'biomass' if 'biomass' in self.train_df.columns else 'target'

        for idx, (_, row) in enumerate(sample_df.iterrows()):
            if idx >= len(axes):
                break

            # Get image path
            image_path_col = 'image_path' if 'image_path' in row else 'id'
            if image_path_col == 'id':
                image_name = f"{row['id']}.jpg"
            else:
                image_name = row['image_path']

            image_path = os.path.join(self.image_dir, image_name)

            # Load and display image
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(image)
            else:
                axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')

            # Add title with target value
            if target_col in row:
                title = f'Biomass: {row[target_col]:.2f}'
            else:
                title = f'ID: {row["id"]}'

            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(len(sample_df), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
            print(f'\nSaved sample images to {save_dir}')

        plt.show()

    def ndvi_analysis(self, save_dir: Optional[str] = None):
        """Analyze NDVI values if available"""
        if 'ndvi' not in self.train_df.columns:
            print('\nNDVI column not found in dataset')
            return

        print('\n' + '=' * 50)
        print('NDVI ANALYSIS')
        print('=' * 50)

        ndvi = self.train_df['ndvi']
        print(f'\nNDVI statistics:')
        print(ndvi.describe())

        target_col = 'biomass' if 'biomass' in self.train_df.columns else 'target'

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # NDVI distribution
        axes[0].hist(ndvi, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('NDVI')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('NDVI Distribution')

        # NDVI vs Target
        if target_col in self.train_df.columns:
            axes[1].scatter(ndvi, self.train_df[target_col], alpha=0.5)
            axes[1].set_xlabel('NDVI')
            axes[1].set_ylabel(target_col)
            axes[1].set_title(f'NDVI vs {target_col}')

            # Calculate correlation
            corr = ndvi.corr(self.train_df[target_col])
            axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                        transform=axes[1].transAxes, va='top')

        # NDVI box plot
        axes[2].boxplot(ndvi, vert=True)
        axes[2].set_ylabel('NDVI')
        axes[2].set_title('NDVI Box Plot')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'ndvi_analysis.png'), dpi=300, bbox_inches='tight')
            print(f'\nSaved NDVI analysis to {save_dir}')

        plt.show()

    def run_full_eda(self, save_dir: Optional[str] = None):
        """Run complete EDA"""
        print('\n' + '=' * 70)
        print('CSIRO IMAGE2BIOMASS - EXPLORATORY DATA ANALYSIS')
        print('=' * 70)

        self.basic_info()
        self.target_distribution(save_dir)
        self.feature_analysis(save_dir)
        self.categorical_analysis(save_dir)
        self.ndvi_analysis(save_dir)
        self.image_analysis(save_dir=save_dir)

        print('\n' + '=' * 70)
        print('EDA COMPLETE')
        print('=' * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run EDA on CSIRO Image2Biomass dataset')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Data directory'
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default='train.csv',
        help='Training CSV filename'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='images',
        help='Image directory name'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./eda_plots',
        help='Directory to save plots'
    )

    args = parser.parse_args()

    eda = BiomassEDA(args.data_dir, args.train_csv, args.image_dir)
    eda.run_full_eda(args.save_dir)
