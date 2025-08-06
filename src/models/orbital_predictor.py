"""LSTM-based orbital trajectory prediction model."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from ..utils.config import MODEL_CONFIG, SAVED_MODELS_DIR
from ..utils.orbital_mechanics import OrbitalMechanics


class OrbitalPredictor:
    """Predicts future orbital positions using LSTM networks."""
    
    def __init__(self, model_name: str = "orbital_lstm"):
        self.model_name = model_name
        self.model = None
        self.scaler_position = StandardScaler()
        self.scaler_velocity = StandardScaler()
        self.config = MODEL_CONFIG["lstm"]
        self.om = OrbitalMechanics()
        
    def prepare_sequences(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        
        sequences = []
        targets = []
        
        for i in range(len(positions) - sequence_length - 1):
            # Combine position and velocity into feature vector
            seq_features = []
            for j in range(sequence_length):
                features = np.concatenate([
                    positions[i + j],
                    velocities[i + j]
                ])
                seq_features.append(features)
            
            sequences.append(seq_features)
            # Target is next position
            targets.append(positions[i + sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model for trajectory prediction."""
        
        model = models.Sequential([
            layers.LSTM(
                self.config["units"],
                return_sequences=True,
                input_shape=input_shape
            ),
            layers.Dropout(self.config["dropout"]),
            layers.LSTM(
                self.config["units"] // 2,
                return_sequences=True
            ),
            layers.Dropout(self.config["dropout"]),
            layers.LSTM(
                self.config["units"] // 4
            ),
            layers.Dropout(self.config["dropout"]),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(3)  # Output: 3D position
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        training_data: List[Dict],
        validation_split: float = 0.2,
        epochs: Optional[int] = None
    ):
        """Train the LSTM model on historical orbital data."""
        
        if epochs is None:
            epochs = self.config["epochs"]
        
        print("Preparing training data...")
        
        # Collect all positions and velocities
        all_positions = []
        all_velocities = []
        
        for sat_data in training_data:
            if 'positions' in sat_data and 'velocities' in sat_data:
                # Remove NaN values
                mask = ~np.any(np.isnan(sat_data['positions']), axis=1)
                all_positions.extend(sat_data['positions'][mask])
                all_velocities.extend(sat_data['velocities'][mask])
        
        all_positions = np.array(all_positions)
        all_velocities = np.array(all_velocities)
        
        # Normalize data
        positions_norm = self.scaler_position.fit_transform(all_positions)
        velocities_norm = self.scaler_velocity.fit_transform(all_velocities)
        
        # Prepare sequences
        X, y = self.prepare_sequences(
            positions_norm,
            velocities_norm,
            self.config["sequence_length"]
        )
        
        # Normalize targets
        y = self.scaler_position.transform(y)
        
        print(f"Training data shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Build model
        self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5
            )
        ]
        
        # Train
        history = self.model.fit(
            X, y,
            batch_size=self.config["batch_size"],
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and scalers
        self.save_model()
        
        return history
    
    def predict_trajectory(
        self,
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
        prediction_steps: int
    ) -> np.ndarray:
        """Predict future trajectory from initial conditions."""
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Normalize initial data
        positions_norm = self.scaler_position.transform(initial_positions)
        velocities_norm = self.scaler_velocity.transform(initial_velocities)
        
        # Prepare input sequence
        sequence = []
        for i in range(len(positions_norm)):
            features = np.concatenate([positions_norm[i], velocities_norm[i]])
            sequence.append(features)
        
        sequence = np.array([sequence])  # Add batch dimension
        
        # Predict future positions
        predictions = []
        
        for _ in range(prediction_steps):
            # Predict next position
            next_pos_norm = self.model.predict(sequence, verbose=0)[0]
            next_pos = self.scaler_position.inverse_transform([next_pos_norm])[0]
            predictions.append(next_pos)
            
            # Estimate velocity (simplified - could be improved)
            if len(predictions) > 1:
                velocity = (predictions[-1] - predictions[-2]) * 60  # Assuming 1-minute steps
            else:
                velocity = initial_velocities[-1]
            
            velocity_norm = self.scaler_velocity.transform([velocity])[0]
            
            # Update sequence for next prediction
            new_features = np.concatenate([next_pos_norm, velocity_norm])
            sequence = np.concatenate([sequence[:, 1:, :], [[new_features]]], axis=1)
        
        return np.array(predictions)
    
    def predict_with_uncertainty(
        self,
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
        prediction_steps: int,
        num_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """Predict trajectory with uncertainty estimation using dropout."""
        
        # Enable dropout during inference for uncertainty estimation
        predictions_samples = []
        
        for _ in range(num_samples):
            pred = self.predict_trajectory(
                initial_positions,
                initial_velocities,
                prediction_steps
            )
            predictions_samples.append(pred)
        
        predictions_samples = np.array(predictions_samples)
        
        # Calculate statistics
        mean_prediction = np.mean(predictions_samples, axis=0)
        std_prediction = np.std(predictions_samples, axis=0)
        
        # Calculate confidence intervals
        lower_bound = mean_prediction - 2 * std_prediction
        upper_bound = mean_prediction + 2 * std_prediction
        
        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'lower_95': lower_bound,
            'upper_95': upper_bound,
            'samples': predictions_samples
        }
    
    def save_model(self):
        """Save model and scalers."""
        model_dir = SAVED_MODELS_DIR / self.model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save Keras model
        self.model.save(model_dir / "lstm_model.h5")
        
        # Save scalers
        joblib.dump(self.scaler_position, model_dir / "scaler_position.pkl")
        joblib.dump(self.scaler_velocity, model_dir / "scaler_velocity.pkl")
        
        # Save config
        joblib.dump(self.config, model_dir / "config.pkl")
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self):
        """Load saved model and scalers."""
        model_dir = SAVED_MODELS_DIR / self.model_name
        
        if not model_dir.exists():
            raise ValueError(f"Model directory {model_dir} not found")
        
        # Load Keras model
        self.model = tf.keras.models.load_model(model_dir / "lstm_model.h5")
        
        # Load scalers
        self.scaler_position = joblib.load(model_dir / "scaler_position.pkl")
        self.scaler_velocity = joblib.load(model_dir / "scaler_velocity.pkl")
        
        # Load config
        self.config = joblib.load(model_dir / "config.pkl")
        
        print(f"Model loaded from {model_dir}")
    
    def evaluate_predictions(
        self,
        true_positions: np.ndarray,
        predicted_positions: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate prediction accuracy."""
        
        # Calculate errors
        errors = np.linalg.norm(true_positions - predicted_positions, axis=1)
        
        metrics = {
            'mean_error_km': np.mean(errors),
            'std_error_km': np.std(errors),
            'max_error_km': np.max(errors),
            'min_error_km': np.min(errors),
            'median_error_km': np.median(errors),
            'percentile_95_error_km': np.percentile(errors, 95)
        }
        
        return metrics


def demo_orbital_predictor():
    """Demo the orbital predictor."""
    predictor = OrbitalPredictor()
    
    # Generate synthetic training data (in real use, this would come from TLE propagation)
    print("Generating synthetic training data...")
    
    # Simulate multiple satellite trajectories
    training_data = []
    
    for i in range(10):  # 10 satellites
        t = np.linspace(0, 2 * np.pi, 1000)
        
        # Simplified circular orbit
        radius = 6371 + 400 + i * 50  # Different altitudes
        positions = np.column_stack([
            radius * np.cos(t),
            radius * np.sin(t) * 0.8,  # Some inclination
            radius * np.sin(t) * 0.2
        ])
        
        # Approximate velocities
        velocities = np.gradient(positions, axis=0) * 60  # Convert to per minute
        
        training_data.append({
            'positions': positions,
            'velocities': velocities
        })
    
    # Train model
    print("\nTraining model...")
    history = predictor.train(training_data, epochs=5)  # Reduced for demo
    
    # Test prediction
    print("\nTesting predictions...")
    test_positions = training_data[0]['positions'][:24]  # 24 minutes of data
    test_velocities = training_data[0]['velocities'][:24]
    
    predictions = predictor.predict_trajectory(
        test_positions,
        test_velocities,
        prediction_steps=60  # Predict 1 hour ahead
    )
    
    print(f"Predicted {len(predictions)} future positions")
    print(f"First prediction: {predictions[0]}")
    print(f"Last prediction: {predictions[-1]}")


if __name__ == "__main__":
    demo_orbital_predictor()