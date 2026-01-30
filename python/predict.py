#!/usr/bin/env python3
"""
Prediction Script - Load trained models and make predictions on new data.

Usage:
    python predict.py --model output/models/xgboost_best.json --data data/new_games.csv
    python predict.py --model output/models/xgboost_best.json --data data/new_games.csv --output predictions.csv
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.xgboost_model import XGBoostModel


def load_model(model_path: str) -> XGBoostModel:
    """Load a trained XGBoost model from disk."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = XGBoostModel()
    model.load_model(str(model_path))
    print("Model loaded successfully")

    return model


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df


def preprocess_data(df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
    """Preprocess data for prediction."""
    # Make a copy
    df = df.copy()

    # If feature columns specified, select only those
    if feature_columns:
        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
        available_cols = [c for c in feature_columns if c in df.columns]
        df = df[available_cols]

    # Select only numeric columns for prediction
    numeric_df = df.select_dtypes(include=[np.number])

    # Handle missing values
    if numeric_df.isnull().any().any():
        null_counts = numeric_df.isnull().sum()
        print(f"Warning: Missing values found in {(null_counts > 0).sum()} columns")
        numeric_df = numeric_df.fillna(numeric_df.median())

    print(f"Features for prediction: {list(numeric_df.columns)}")

    return numeric_df


def make_predictions(model: XGBoostModel, X: pd.DataFrame) -> np.ndarray:
    """Make predictions using the loaded model."""
    print(f"\nMaking predictions on {len(X)} samples...")
    predictions = model.predict(X)

    print(f"Predictions complete!")
    print(f"  Min: {predictions.min():.4f}")
    print(f"  Max: {predictions.max():.4f}")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std: {predictions.std():.4f}")

    return predictions


def save_predictions(
    original_df: pd.DataFrame,
    predictions: np.ndarray,
    output_path: str,
    target_col: str = "prediction",
):
    """Save predictions to CSV."""
    output_df = original_df.copy()
    output_df[target_col] = predictions

    # Add metadata
    output_df["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    return output_df


def list_available_models(models_dir: str = None):
    """List all available trained models."""
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "output" / "models"
    else:
        models_dir = Path(models_dir)

    print("\nAvailable Models:")
    print("-" * 50)

    if not models_dir.exists():
        print("No models directory found. Train a model first.")
        return []

    models = list(models_dir.glob("*.json")) + list(models_dir.glob("*.pkl"))

    if not models:
        print("No trained models found. Run hyperparameter search first.")
        return []

    for model_path in sorted(models):
        size = model_path.stat().st_size / 1024  # KB
        modified = datetime.fromtimestamp(model_path.stat().st_mtime)
        print(f"  {model_path.name:<30} {size:>8.1f} KB  ({modified:%Y-%m-%d %H:%M})")

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions using trained XGBoost models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python predict.py --list-models
  
  # Make predictions on new data
  python predict.py --model output/models/xgboost_best.json --data data/new_games.csv
  
  # Save predictions to specific file
  python predict.py --model output/models/xgboost_best.json --data data/new_games.csv --output my_predictions.csv
  
  # Specify target column name
  python predict.py --model output/models/xgboost_best.json --data data/new_games.csv --target-col expected_goals
        """,
    )

    parser.add_argument(
        "--model", "-m", type=str, help="Path to trained model file (.json)"
    )
    parser.add_argument(
        "--data", "-d", type=str, help="Path to CSV data file for predictions"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save predictions (default: output/predictions/predictions_<timestamp>.csv)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="prediction",
        help="Name for prediction column (default: prediction)",
    )
    parser.add_argument(
        "--features", "-f", type=str, nargs="+", help="Specific feature columns to use"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List all available trained models"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("XGBOOST PREDICTION SCRIPT")
    print("=" * 60)

    # List models if requested
    if args.list_models:
        list_available_models()
        return 0

    # Validate required arguments
    if not args.model or not args.data:
        print("\nError: --model and --data are required for predictions")
        print("Use --list-models to see available models")
        print("Use --help for usage information")
        return 1

    try:
        # Load model
        model = load_model(args.model)

        # Load data
        original_df = load_data(args.data)

        # Preprocess
        X = preprocess_data(original_df, args.features)

        # Predict
        predictions = make_predictions(model, X)

        # Generate output path if not specified
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(__file__).parent.parent / "output" / "predictions"
            output_path = output_dir / f"predictions_{timestamp}.csv"

        # Save
        save_predictions(original_df, predictions, output_path, args.target_col)

        print("\n" + "=" * 60)
        print("PREDICTION COMPLETE")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
