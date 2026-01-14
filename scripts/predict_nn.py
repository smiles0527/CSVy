#!/usr/bin/env python3
"""
Simple prediction script for production use
Loads trained neural network and makes predictions on new data
"""

import sys
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def load_model_and_scaler(model_dir='models'):
    """Load trained model and scaler"""
    model_path = Path(model_dir) / 'best_nn_model.keras'
    scaler_path = Path(model_dir) / 'scaler.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    model = keras.models.load_model(str(model_path))
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict(data_file, target_col='goals', model_dir='models'):
    """Make predictions on new data"""
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_dir)
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Separate features and target (if present)
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        has_target = True
    else:
        X = df
        y = None
        has_target = False
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled, verbose=0).flatten()
    
    # Build result
    result = {
        'predictions': predictions.tolist(),
        'n_samples': len(predictions)
    }
    
    # Calculate metrics if target available
    if has_target:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        
        result['metrics'] = {
            'rmse': float(rmse),
            'r2': float(r2),
            'mae': float(mae)
        }
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_nn.py DATA_FILE [TARGET_COL] [MODEL_DIR]", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python predict_nn.py data/test.csv goals models", file=sys.stderr)
        sys.exit(1)
    
    data_file = sys.argv[1]
    target_col = sys.argv[2] if len(sys.argv) > 2 else 'goals'
    model_dir = sys.argv[3] if len(sys.argv) > 3 else 'models'
    
    try:
        result = predict(data_file, target_col, model_dir)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
