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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

class HockeyNN:
    """Dense neural network for hockey goal prediction"""
    
    def __init__(self, params):
        self.params = params
        self.model = None
        # Select scaler based on params
        scaler_type = params.get('scaler', 'standard')
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.poly = None
        if params.get('polynomial_degree', 1) > 1:
            self.poly = PolynomialFeatures(degree=params['polynomial_degree'], include_bias=False)
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
        
        # Apply polynomial features if configured
        if self.poly:
            X_train = self.poly.fit_transform(X_train)
            X_val = self.poly.transform(X_val)
        
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
        if self.poly:
            X = self.poly.transform(X)
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
        # Save scaler and poly separately
        import pickle
        scaler_path = str(Path(path).parent / 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        if self.poly:
            poly_path = str(Path(path).parent / 'poly.pkl')
            with open(poly_path, 'wb') as f:
                pickle.dump(self.poly, f)


def engineer_hockey_features(df):
    """Add hockey-specific engineered features"""
    df = df.copy()
    
    # Recent form features (if game-by-game data)
    if 'game_number' in df.columns and 'team_id' in df.columns:
        df = df.sort_values(['team_id', 'game_number'])
        # 5-game rolling averages
        df['goals_last_5'] = df.groupby('team_id')['goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['goals_allowed_last_5'] = df.groupby('team_id')['goals_allowed'].transform(lambda x: x.rolling(5, min_periods=1).mean()) if 'goals_allowed' in df.columns else 0
    
    # Rest days (if date columns exist)
    if 'game_date' in df.columns and 'team_id' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['team_id', 'game_date'])
        df['rest_days'] = df.groupby('team_id')['game_date'].diff().dt.days.fillna(3)
    
    # Goal differential features
    if 'goals_for' in df.columns and 'goals_against' in df.columns:
        df['goal_differential'] = df['goals_for'] - df['goals_against']
        df['goal_differential_pct'] = df['goals_for'] / (df['goals_for'] + df['goals_against'] + 1e-8)
    
    return df


def add_interaction_features(X, max_interactions=20):
    """Add top feature interactions (product of pairs)"""
    # Select most important feature pairs (simple heuristic: highest variance)
    feature_vars = X.var().sort_values(ascending=False)
    top_features = feature_vars.head(min(10, len(feature_vars))).index.tolist()
    
    interactions_added = 0
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            if interactions_added >= max_interactions:
                break
            X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
            interactions_added += 1
    
    return X


def load_data(csv_path, target_col='goals', add_hockey_features=False, add_interactions=False):
    """Load preprocessed CSV from Ruby pipeline"""
    df = pd.read_csv(csv_path)
    
    # Add hockey-specific features
    if add_hockey_features:
        print("Adding hockey-specific features...")
        df = engineer_hockey_features(df)
    
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
    
    # Add interaction features
    if add_interactions:
        print("Adding interaction features...")
        X = add_interaction_features(X)
    
    return X, y


def validate_architecture_for_dataset(n_samples, n_features, params):
    """Validate NN architecture against dataset size to prevent overfitting"""
    warnings = []
    
    # Calculate total parameters
    layer1 = params['layer1_units']
    layer2 = params['layer2_units']
    layer3 = params.get('layer3_units', 0)
    
    total_params = (n_features * layer1) + layer1 + (layer1 * layer2) + layer2
    if layer3 > 0:
        total_params += (layer2 * layer3) + layer3 + layer3 + 1
    else:
        total_params += layer2 + 1
    
    # Rule of thumb: need 10x samples per parameter for good generalization
    recommended_samples = total_params * 10
    
    if n_samples < recommended_samples:
        ratio = n_samples / recommended_samples
        warnings.append(f"⚠️  Dataset too small for architecture: {n_samples} samples vs {total_params} params (need ~{recommended_samples})")
        warnings.append(f"   Risk: {(1-ratio)*100:.0f}% overfitting probability. Consider: reduce layers, add dropout, or get more data")
    
    # Recommend max layer sizes
    max_safe_layer1 = int(np.sqrt(n_samples / 10))
    if layer1 > max_safe_layer1:
        warnings.append(f"   Recommendation: layer1_units <= {max_safe_layer1} for dataset size")
    
    return warnings


def hyperparameter_search(X, y, config_path, n_samples=50, output_csv='nn_results.csv'):
    """Random search over hyperparameter space"""
    # Pure Python implementation - independent of Ruby
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Warn if dataset is small
    if len(y) < 500:
        print(f"\n⚠️  WARNING: Small dataset ({len(y)} samples) - high overfitting risk!")
        print("   Recommendations:")
        print("   - Use 2-layer network (layer3_units=0)")
        print("   - High dropout (0.4-0.5)")
        print("   - Strong regularization (l2_penalty=0.01)")
        print("   - Early stopping (patience=15-20)\n")
    
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
            'l2_penalty': np.random.choice(config['regularization']['l2_penalty']),
            'scaler': np.random.choice(config.get('feature_engineering', {}).get('scaler', ['standard'])),
            'polynomial_degree': int(np.random.choice(config.get('feature_engineering', {}).get('polynomial_degree', [1])))
        }
        
        # Validate architecture for dataset size
        arch_warnings = validate_architecture_for_dataset(len(y), X.shape[1], params)
        if arch_warnings and i == 0:  # Show once at start
            for warning in arch_warnings:
                print(warning)
        
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
    X, y = load_data(args.data, args.target, 
                     add_hockey_features=True,  # Enable hockey-specific features
                     add_interactions=False)     # Enable if beneficial
    print(f"  Features: {X.shape[1]}, Samples: {len(y)}")
    
    # Dataset size warning
    if len(y) < 500:
        print(f"  ⚠️  Small dataset! Recommend: simpler architecture, high dropout, strong regularization")
    
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
