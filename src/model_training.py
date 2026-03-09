"""Model training pipeline for GNSS accuracy improvement.

This module handles the training process including callbacks,
epoch training, validation, and model checkpointing.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model
import os
from datetime import datetime


class GNSSModelTrainer:
    """Trainer class for GNSS ML models."""

    def __init__(self, model, model_name='gnss_model'):
        """Initialize the trainer.
        
        Args:
            model: Compiled Keras model
            model_name: Name for saving model artifacts
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        self.trained = False
        
        # Create directories for model artifacts
        self.model_dir = 'models'
        self.log_dir = 'logs'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_callbacks(self, patience=20, model_path=None):
        """Setup training callbacks.
        
        Args:
            patience: Early stopping patience
            model_path: Path to save best model
            
        Returns:
            List of callbacks
        """
        if model_path is None:
            model_path = os.path.join(self.model_dir, f'{self.model_name}_best.h5')
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                update_freq='batch'
            )
        ]
        return callbacks

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
              patience=20, verbose=1):
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        print(f"Starting training for {self.model_name}...")
        print(f"Train set size: {X_train.shape}")
        print(f"Validation set size: {X_val.shape}")
        
        callbacks = self.setup_callbacks(patience=patience)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.trained = True
        print(f"Training completed for {self.model_name}")
        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trained:
            print("Warning: Model has not been trained yet.")
        
        test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_rmse': np.sqrt(test_mse)
        }
        
        print(f"\nTest Set Evaluation:")
        print(f"  Loss: {test_loss:.6f}")
        print(f"  MAE: {test_mae:.6f}")
        print(f"  MSE: {test_mse:.6f}")
        print(f"  RMSE: {metrics['test_rmse']:.6f}")
        
        return metrics

    def predict(self, X):
        """Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if not self.trained:
            print("Error: Model has not been trained yet.")
            return None
        
        predictions = self.model.predict(X)
        return predictions

    def save_model(self, filepath=None):
        """Save the trained model.
        
        Args:
            filepath: Path to save model
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, f'{self.model_name}_final.h5')
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model.
        
        Args:
            filepath: Path to load model from
        """
        self.model = load_model(filepath)
        self.trained = True
        print(f"Model loaded from {filepath}")

    def get_training_summary(self):
        """Get summary of training history.
        
        Returns:
            Dictionary with training summary
        """
        if self.history is None:
            return None
        
        summary = {
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'best_val_loss': float(np.min(self.history.history['val_loss'])),
            'best_epoch': int(np.argmin(self.history.history['val_loss'])) + 1
        }
        return summary


if __name__ == '__main__':
    print("Model training module loaded successfully.")
