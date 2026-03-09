"""Model architecture definitions for GNSS accuracy improvement.

This module contains neural network models including LSTM, CNN, 
and hybrid architectures for GNSS error prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np


class GNSSModelBuilder:
    """Builder class for constructing GNSS ML models."""

    def __init__(self, input_shape):
        """Initialize model builder.
        
        Args:
            input_shape: Shape of input data (features,)
        """
        self.input_shape = input_shape
        self.model = None

    def build_lstm_model(self, lstm_units=64, dropout_rate=0.2):
        """Build LSTM model for time-series GNSS data.
        
        Args:
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(lstm_units, activation='relu', input_shape=(1, self.input_shape), 
                 return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, activation='relu', return_sequences=False),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model

    def build_cnn_model(self, filters=64, kernel_size=3, dropout_rate=0.2):
        """Build CNN model for spatial pattern recognition.
        
        Args:
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Conv1D(filters, kernel_size, activation='relu', input_shape=(1, self.input_shape)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters * 2, kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model

    def build_hybrid_cnn_lstm_model(self, cnn_filters=64, lstm_units=64, dropout_rate=0.2):
        """Build hybrid CNN-LSTM model combining spatial and temporal features.
        
        Args:
            cnn_filters: Number of convolutional filters
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Conv1D(cnn_filters, 3, activation='relu', input_shape=(1, self.input_shape)),
            MaxPooling1D(pool_size=2),
            Conv1D(cnn_filters * 2, 3, activation='relu'),
            LSTM(lstm_units, activation='relu', return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, activation='relu', return_sequences=False),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model

    def build_dense_neural_network(self, hidden_units=[256, 128, 64], dropout_rate=0.2):
        """Build fully connected neural network.
        
        Args:
            hidden_units: List of hidden layer units
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        model.add(Dense(hidden_units[0], activation='relu', input_dim=self.input_shape))
        model.add(Dropout(dropout_rate))
        
        for units in hidden_units[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1, activation='linear'))
        return model

    def compile_model(self, model, learning_rate=0.001):
        """Compile the model with optimizer and loss function.
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Compiled model
        """
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=MeanSquaredError(),
            metrics=['mae', 'mse']
        )
        return model

    def get_model_summary(self, model):
        """Print model summary.
        
        Args:
            model: Keras model
        """
        model.summary()


if __name__ == '__main__':
    # Example usage
    builder = GNSSModelBuilder(input_shape=10)  # 10 input features
    lstm_model = builder.build_lstm_model()
    lstm_model = builder.compile_model(lstm_model)
    print("Model architecture module loaded successfully.")
