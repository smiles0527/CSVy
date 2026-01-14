#!/usr/bin/env python3
"""
Neural Network Model for Hockey Prediction
Integrates with Ruby CSVy preprocessing pipeline
"""

import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

class HockeyNN:
    """Dense neural network for hockey goal prediction"""
    
    def __init__(self, params):
        self.params = params
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self, input_dim):
        """Build neural network architecture"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layer 1
        l1_reg = self.params.get('l1_penalty', 0.0)
        l2_reg = self.params.get('l2_penalty', 0.0)
        
        model.add(layers.Dense(
            self.params['layer1_units'],
            activation=self.params['activation'],
            kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(layers.Dropout(self.params['dropout_rate']))
        
        # Hidden layer 2
        model.add(layers.Dense(
            self.params['layer2_units'],
            activation=self.params['activation'],
            kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(layers.Dropout(self.params['dropout_rate']))
        
        # Optional hidden layer 3
        if self.params.get('layer3_units', 0) > 0:
            model.add(layers.Dense(
                self.params['layer3_units'],
                activation=self.params['activation'],
                kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(layers.Dropout(self.params['dropout_rate']))
        
        # Output layer (regression - single value)
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the neural network with early stopping"""
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model
        self.build_model(X_train.shape[1])
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.params.get('patience', 20),
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.params.get('epochs', 200),
            batch_size=self.params.get('batch_size', 32),
            callbacks=[early_stop],
            verbose=0
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
        return {
            'rmse': float(rmse),
            'r2': float(r2),
            'mae': float(mae)
        }
    
    def save_model(self, path):
        """Save trained model"""
        self.model.save(path)
        # Save scaler separately
        import pickle
        scaler_path = str(Path(path).parent / 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)


def load_data(csv_path, target_col='goals'):
    """Load preprocessed CSV from Ruby pipeline"""
    df = pd.read_csv(csv_path)
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Filter out non-numeric columns (e.g., team names, IDs)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < len(X.columns):
        dropped = set(X.columns) - set(numeric_cols)
        print(f"Dropping non-numeric columns: {dropped}")
        X = X[numeric_cols]
    
    if len(X.columns) == 0:
        raise ValueError("No numeric features found in dataset")
    
    return X, y


def hyperparameter_search(X, y, config_path, n_samples=50, output_csv='nn_results.csv'):
    """Random search over hyperparameter space"""
    # Pure Python implementation - independent of Ruby
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results = []
    
    for i in range(n_samples):
        # Sample hyperparameters
        params = {
            'layer1_units': np.random.choice(config['architecture']['layer1_units']),
            'layer2_units': np.random.choice(config['architecture']['layer2_units']),
            'layer3_units': np.random.choice(config['architecture']['layer3_units']),
            'dropout_rate': np.random.choice(config['architecture']['dropout_rate']),
            'activation': np.random.choice(config['architecture']['activation']),
            'learning_rate': np.random.choice(config['training']['learning_rate']),
            'batch_size': int(np.random.choice(config['training']['batch_size'])),
            'epochs': int(np.random.choice(config['training']['epochs'])),
            'patience': int(np.random.choice(config['training']['patience'])),
            'l1_penalty': np.random.choice(config['regularization']['l1_penalty']),
            'l2_penalty': np.random.choice(config['regularization']['l2_penalty'])
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + i
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + i
        )
        
        # Train model
        print(f"[{i+1}/{n_samples}] Training NN with {params['layer1_units']}-{params['layer2_units']} architecture...")
        
        nn = HockeyNN(params)
        nn.train(X_train.values, y_train.values, X_val.values, y_val.values)
        
        # Evaluate
        metrics = nn.evaluate(X_test.values, y_test.values)
        
        # Store results
        result = {**params, **metrics}
        results.append(result)
        
        print(f"  RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")
        
        # Save best model
        if len(results) == 1 or metrics['rmse'] < min(r['rmse'] for r in results[:-1]):
            nn.save_model('models/best_nn_model.keras')
            print(f"  ✓ New best model saved!")
    
    # Save results to CSV (Ruby-compatible format)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    return df_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train neural network for hockey prediction')
    parser.add_argument('data', help='CSV file with preprocessed features')
    parser.add_argument('--target', default='goals', help='Target column name')
    parser.add_argument('--config', default='config/hyperparams/model6_neural_network.yaml',
                       help='Hyperparameter config file')
    parser.add_argument('--search', type=int, metavar='N',
                       help='Run random search with N samples')
    parser.add_argument('--output', default='model6_neural_network_results.csv',
                       help='Output CSV for search results')
    
    args = parser.parse_args()
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}...")
    X, y = load_data(args.data, args.target)
    print(f"  Features: {X.shape[1]}, Samples: {len(y)}")
    
    if args.search:
        # Hyperparameter search
        print(f"\nStarting random search with {args.search} iterations...")
        results = hyperparameter_search(X, y, args.config, args.search, args.output)
        
        # Show best result
        best = results.loc[results['rmse'].idxmin()]
        print(f"\n🏆 Best Configuration:")
        print(f"  RMSE: {best['rmse']:.3f}")
        print(f"  Architecture: {int(best['layer1_units'])}-{int(best['layer2_units'])}-{int(best['layer3_units'])}")
        print(f"  Dropout: {best['dropout_rate']:.2f}, LR: {best['learning_rate']:.4f}")
        
    else:
        # Single training run with default params
        print("\nTraining single model with default hyperparameters...")
        params = {
            'layer1_units': 32,
            'layer2_units': 16,
            'layer3_units': 0,
            'dropout_rate': 0.4,
            'activation': 'relu',
            'learning_rate': 0.003,
            'batch_size': 32,
            'epochs': 200,
            'patience': 20,
            'l1_penalty': 0.0,
            'l2_penalty': 0.001
        }
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        nn = HockeyNN(params)
        nn.train(X_train.values, y_train.values, X_val.values, y_val.values)
        
        metrics = nn.evaluate(X_test.values, y_test.values)
        print(f"\nTest Results:")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  MAE: {metrics['mae']:.3f}")
        
        nn.save_model('models/nn_model.keras')


if __name__ == '__main__':
    main()
