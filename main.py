"""Main entry point for GNSS accuracy improvement project.

This script orchestrates the entire pipeline from data preprocessing
to model training and evaluation.
"""

import sys
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import GNSSDataPreprocessor
from model_architecture import GNSSModelBuilder
from model_training import GNSSModelTrainer


def main():
    """Main execution function."""
    print("="*60)
    print("GNSS Accuracy Improvement using AI/ML")
    print("="*60)
    
    # Configuration
    config = {
        'data_path': 'data/gnss_data.csv',
        'target_column': 'error',
        'test_size': 0.2,
        'val_size': 0.1,
        'epochs': 100,
        'batch_size': 32,
        'lstm_units': 64,
        'learning_rate': 0.001
    }
    
    print("\n[1/4] Loading and Preprocessing Data...")
    print("-" * 60)
    
    # Initialize preprocessor
    preprocessor = GNSSDataPreprocessor(scaler_type='standard')
    
    # Preprocess data (when data file is available)
    # datasets = preprocessor.preprocess_pipeline(
    #     config['data_path'],
    #     config['target_column'],
    #     test_size=config['test_size'],
    #     val_size=config['val_size']
    # )
    
    # For demonstration, create synthetic data
    print("Creating synthetic GNSS data for demonstration...")
    n_samples = 1000
    n_features = 10  # Number of GNSS features
    
    X_train = np.random.randn(int(n_samples * 0.7), n_features)
    y_train = np.random.randn(int(n_samples * 0.7), 1) * 5 + 10  # Simulated errors
    
    X_val = np.random.randn(int(n_samples * 0.1), n_features)
    y_val = np.random.randn(int(n_samples * 0.1), 1) * 5 + 10
    
    X_test = np.random.randn(int(n_samples * 0.2), n_features)
    y_test = np.random.randn(int(n_samples * 0.2), 1) * 5 + 10
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    print("\n[2/4] Building Neural Network Model...")
    print("-" * 60)
    
    # Build LSTM model
    builder = GNSSModelBuilder(input_shape=n_features)
    model = builder.build_lstm_model(
        lstm_units=config['lstm_units'],
        dropout_rate=0.2
    )
    model = builder.compile_model(model, learning_rate=config['learning_rate'])
    
    print("Model architecture:")
    model.summary()
    
    print("\n[3/4] Training the Model...")
    print("-" * 60)
    
    # Train the model
    trainer = GNSSModelTrainer(model, model_name='gnss_lstm_model')
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=20,
        verbose=1
    )
    
    print("\n[4/4] Evaluating the Model...")
    print("-" * 60)
    
    # Evaluate the model
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save the model
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nModel Summary:")
    summary = trainer.get_training_summary()
    if summary:
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    return trainer


if __name__ == '__main__':
    try:
        trainer = main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
