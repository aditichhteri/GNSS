"""Data preprocessing module for GNSS accuracy improvement.

This module handles loading, cleaning, and preprocessing GNSS data
for machine learning model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class GNSSDataPreprocessor:
    """Preprocesses GNSS data for model training."""

    def __init__(self, scaler_type='standard'):
        """Initialize the preprocessor with selected scaler type.
        
        Args:
            scaler_type: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.scaler_type = scaler_type
        self.feature_columns = None
        self.target_column = None

    def load_data(self, filepath):
        """Load GNSS data from CSV file.
        
        Args:
            filepath: Path to the CSV data file
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None

    def handle_missing_values(self, data, method='mean'):
        """Handle missing values in the dataset.
        
        Args:
            data: DataFrame with potential missing values
            method: 'mean', 'median', or 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        if method == 'mean':
            return data.fillna(data.mean())
        elif method == 'median':
            return data.fillna(data.median())
        elif method == 'drop':
            return data.dropna()
        return data

    def remove_outliers(self, data, method='iqr', threshold=1.5):
        """Remove outliers using IQR or Z-score method.
        
        Args:
            data: DataFrame with potential outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)].dropna()
        return data

    def split_features_target(self, data, target_column):
        """Split data into features and target.
        
        Args:
            data: DataFrame with all columns
            target_column: Name of the target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        self.target_column = target_column
        self.feature_columns = [col for col in data.columns if col != target_column]
        return data[self.feature_columns], data[target_column]

    def normalize_features(self, X_train, X_test=None):
        """Normalize features using fitted scaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Normalized features (X_train_scaled) or tuple (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled

    def preprocess_pipeline(self, filepath, target_column, test_size=0.2, 
                          val_size=0.1, random_state=42):
        """Complete preprocessing pipeline.
        
        Args:
            filepath: Path to data file
            target_column: Target column name
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            
        Returns:
            Dictionary with train, validation, and test sets
        """
        # Load data
        data = self.load_data(filepath)
        if data is None:
            return None

        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Remove outliers
        data = self.remove_outliers(data)

        # Split features and target
        X, y = self.split_features_target(data, target_column)

        # Split into train and temp (test+val)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state
        )

        # Split temp into test and validation
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state
        )

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_val': y_val.values,
            'y_test': y_test.values,
            'scaler': self.scaler
        }


if __name__ == '__main__':
    # Example usage
    preprocessor = GNSSDataPreprocessor(scaler_type='standard')
    # datasets = preprocessor.preprocess_pipeline('data/gnss_data.csv', 'error')
    print("Data preprocessing module loaded successfully.")
