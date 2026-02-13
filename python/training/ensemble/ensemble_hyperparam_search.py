"""
Ensemble Model Hyperparameter Search

Performs grid search and random search for ensemble model configurations.
Outputs results to: output/hyperparams/model5_ensemble_grid_search.csv
                    output/hyperparams/model5_ensemble_random_search.csv

All runs are logged to MLflow for visualization at http://localhost:5000
"""

import sys
import os
import itertools
import random
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from utils.experiment_tracker import ExperimentTracker

# Initialize MLflow tracker - use absolute path with file:// URI
mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
tracking_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"
tracker = ExperimentTracker(
    experiment_name="ensemble_hyperparam_search",
    tracking_uri=tracking_uri
)


def generate_sample_data(n_games=500, n_teams=10):
    """Generate synthetic hockey data for testing."""
    np.random.seed(42)
    random.seed(42)
    teams = [f"Team_{i}" for i in range(n_teams)]
    
    team_strengths = {team: np.random.uniform(0.8, 1.2) for team in teams}
    
    games = []
    for game_idx in range(n_games):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        
        home_exp = 3.0 * team_strengths[home] * 1.05
        away_exp = 3.0 * team_strengths[away]
        
        home_goals = max(0, int(np.random.poisson(home_exp)))
        away_goals = max(0, int(np.random.poisson(away_exp)))
        
        games.append({
            'game_id': game_idx,
            'home_team': home,
            'away_team': away,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_elo': 1500 + team_strengths[home] * 100,
            'away_elo': 1500 + team_strengths[away] * 100,
            'home_wins_l10': random.randint(2, 8),
            'away_wins_l10': random.randint(2, 8),
            'home_goals_avg': np.random.uniform(2.5, 3.5),
            'away_goals_avg': np.random.uniform(2.5, 3.5),
            'home_goals_against_avg': np.random.uniform(2.5, 3.5),
            'away_goals_against_avg': np.random.uniform(2.5, 3.5),
            'home_pp_pct': random.uniform(0.15, 0.30),
            'away_pp_pct': random.uniform(0.15, 0.30),
            'home_pk_pct': random.uniform(0.75, 0.90),
            'away_pk_pct': random.uniform(0.75, 0.90),
            'home_rest_days': random.randint(1, 5),
            'away_rest_days': random.randint(1, 5),
        })
    
    return pd.DataFrame(games)


def prepare_features_target(df, target_col='home_goals'):
    """Prepare feature matrix X and target y from DataFrame."""
    feature_cols = [
        'home_elo', 'away_elo',
        'home_wins_l10', 'away_wins_l10',
        'home_goals_avg', 'away_goals_avg',
        'home_goals_against_avg', 'away_goals_against_avg',
        'home_pp_pct', 'away_pp_pct',
        'home_pk_pct', 'away_pk_pct',
        'home_rest_days', 'away_rest_days',
    ]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Add derived features
    X['elo_diff'] = X['home_elo'] - X['away_elo']
    
    return X, y


# Define base model configurations
BASE_MODEL_CONFIGS = {
    'ridge': lambda alpha=1.0: Ridge(alpha=alpha),
    'lasso': lambda alpha=0.1: Lasso(alpha=alpha, max_iter=5000),
    'elastic': lambda alpha=0.1, l1_ratio=0.5: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000),
    'rf_small': lambda: RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
    'rf_medium': lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'rf_large': lambda: RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'gbm_small': lambda: GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
    'gbm_medium': lambda: GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42),
    'gbm_large': lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
}


def create_weighted_ensemble(models, weights=None):
    """Create a weighted ensemble from fitted models."""
    class WeightedEnsemble:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights or [1/len(models)] * len(models)
        
        def predict(self, X):
            preds = np.column_stack([m.predict(X) for m in self.models])
            return np.average(preds, axis=1, weights=self.weights)
    
    return WeightedEnsemble(models, weights)


def evaluate_ensemble(model_names, weights, X_train, y_train, X_test, y_test, run_name=None):
    """Evaluate an ensemble configuration and log to MLflow."""
    # Create and fit models
    models = []
    for name in model_names:
        model = BASE_MODEL_CONFIGS[name]()
        model.fit(X_train, y_train)
        models.append(model)
    
    # Create ensemble
    ensemble = create_weighted_ensemble(models, weights)
    predictions = ensemble.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
    }
    
    # Log to MLflow
    with tracker.start_run(run_name=run_name or f"ensemble_{datetime.now().strftime('%H%M%S')}"):
        tracker.log_params({
            'models': '+'.join(model_names),
            'weights': ','.join([f"{w:.2f}" for w in weights]) if weights else 'equal'
        })
        tracker.log_metrics(metrics)
    
    return metrics


def evaluate_stacking(estimators, meta_model, X_train, y_train, X_test, y_test, passthrough=False, run_name=None):
    """Evaluate a stacking ensemble and log to MLflow."""
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=clone(meta_model),
        cv=5,
        passthrough=passthrough,
        n_jobs=-1
    )
    
    stacking.fit(X_train, y_train)
    predictions = stacking.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
    }
    
    # Log to MLflow
    with tracker.start_run(run_name=run_name or f"stacking_{datetime.now().strftime('%H%M%S')}"):
        tracker.log_params({
            'estimators': '+'.join([name for name, _ in estimators]),
            'meta_model': type(meta_model).__name__,
            'passthrough': passthrough
        })
        tracker.log_metrics(metrics)
    
    return metrics


def grid_search_weighted_ensemble(X_train, y_train, X_test, y_test, verbose=True):
    """
    Grid search over ensemble combinations and weights.
    """
    # Define model combinations to try
    model_combinations = [
        ['ridge', 'rf_medium'],
        ['ridge', 'gbm_medium'],
        ['rf_medium', 'gbm_medium'],
        ['ridge', 'rf_medium', 'gbm_medium'],
        ['elastic', 'rf_medium', 'gbm_medium'],
        ['ridge', 'rf_small', 'rf_large'],
        ['ridge', 'gbm_small', 'gbm_large'],
        ['ridge', 'elastic', 'rf_medium', 'gbm_medium'],
    ]
    
    # Weight options for 2, 3, 4 model ensembles
    weight_options = {
        2: [[0.5, 0.5], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4]],
        3: [[0.33, 0.33, 0.34], [0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [0.5, 0.25, 0.25]],
        4: [[0.25, 0.25, 0.25, 0.25], [0.4, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
    }
    
    if verbose:
        print(f"Grid Search: Testing {len(model_combinations)} model combinations")
    
    results = []
    for i, models in enumerate(model_combinations):
        n_models = len(models)
        weights_list = weight_options.get(n_models, [[1/n_models] * n_models])
        
        for weights in weights_list:
            try:
                metrics = evaluate_ensemble(models, weights, X_train, y_train, X_test, y_test)
                result = {
                    'models': '+'.join(models),
                    'n_models': n_models,
                    'weights': str(weights),
                    **metrics
                }
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Error with {models}, {weights}: {e}")
        
        if verbose and (i + 1) % 2 == 0:
            print(f"  Progress: {i + 1}/{len(model_combinations)} combinations")
    
    df = pd.DataFrame(results)
    df = df.sort_values('rmse').reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def grid_search_stacking(X_train, y_train, X_test, y_test, verbose=True):
    """
    Grid search over stacking configurations.
    """
    # Base model sets
    base_model_sets = [
        [('ridge', Ridge(alpha=1.0)), ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))],
        [('ridge', Ridge(alpha=1.0)), ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42))],
        [('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)), 
         ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42))],
        [('ridge', Ridge(alpha=1.0)), 
         ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
         ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42))],
        [('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
         ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
         ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42))],
    ]
    
    # Meta models
    meta_models = [
        ('ridge_0.1', Ridge(alpha=0.1)),
        ('ridge_1.0', Ridge(alpha=1.0)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ]
    
    # Passthrough options
    passthrough_options = [False, True]
    
    total = len(base_model_sets) * len(meta_models) * len(passthrough_options)
    if verbose:
        print(f"Grid Search: {total} stacking configurations")
    
    results = []
    count = 0
    
    for estimators in base_model_sets:
        base_names = '+'.join([name for name, _ in estimators])
        
        for meta_name, meta_model in meta_models:
            for passthrough in passthrough_options:
                count += 1
                
                try:
                    metrics = evaluate_stacking(
                        estimators, meta_model,
                        X_train, y_train, X_test, y_test,
                        passthrough=passthrough
                    )
                    result = {
                        'base_models': base_names,
                        'n_base_models': len(estimators),
                        'meta_model': meta_name,
                        'passthrough': passthrough,
                        **metrics
                    }
                    results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"  Error with {base_names}, {meta_name}: {e}")
                
                if verbose and count % 5 == 0:
                    print(f"  Progress: {count}/{total}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('rmse').reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def random_search_ensemble(X_train, y_train, X_test, y_test, n_iter=50, verbose=True):
    """
    Random search over ensemble configurations.
    """
    if verbose:
        print(f"Random Search: {n_iter} iterations")
    
    all_models = list(BASE_MODEL_CONFIGS.keys())
    results = []
    
    for i in range(n_iter):
        # Random number of models (2-4)
        n_models = random.randint(2, min(4, len(all_models)))
        models = random.sample(all_models, n_models)
        
        # Random weights
        raw_weights = [random.random() for _ in range(n_models)]
        weights = [w / sum(raw_weights) for w in raw_weights]
        
        try:
            metrics = evaluate_ensemble(models, weights, X_train, y_train, X_test, y_test)
            result = {
                'models': '+'.join(models),
                'n_models': n_models,
                'weights': str([round(w, 3) for w in weights]),
                **metrics
            }
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  Error with {models}: {e}")
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n_iter}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('rmse').reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def main():
    """Run hyperparameter search for ensemble models."""
    print("=" * 60)
    print("Ensemble Model Hyperparameter Search")
    print("=" * 60)
    
    # Generate or load data
    print("\nGenerating sample data...")
    df = generate_sample_data(n_games=500)
    X, y = prepare_features_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'hyperparams')
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================
    # Weighted Ensemble Grid Search
    # ========================================
    print("\n" + "=" * 40)
    print("Weighted Ensemble Grid Search")
    print("=" * 40)
    
    weighted_results = grid_search_weighted_ensemble(X_train, y_train, X_test, y_test)
    
    weighted_path = os.path.join(output_dir, 'model5_weighted_ensemble_grid_search.csv')
    weighted_results.to_csv(weighted_path, index=False)
    print(f"\nSaved to: {weighted_path}")
    
    print("\nTop 5 Weighted Ensemble configurations:")
    print(weighted_results.head()[['models', 'weights', 'rmse', 'mae']].to_string(index=False))
    
    # ========================================
    # Stacking Ensemble Grid Search
    # ========================================
    print("\n" + "=" * 40)
    print("Stacking Ensemble Grid Search")
    print("=" * 40)
    
    stacking_results = grid_search_stacking(X_train, y_train, X_test, y_test)
    
    stacking_path = os.path.join(output_dir, 'model5_stacking_grid_search.csv')
    stacking_results.to_csv(stacking_path, index=False)
    print(f"\nSaved to: {stacking_path}")
    
    print("\nTop 5 Stacking configurations:")
    print(stacking_results.head()[['base_models', 'meta_model', 'passthrough', 'rmse']].to_string(index=False))
    
    # ========================================
    # Random Search
    # ========================================
    print("\n" + "=" * 40)
    print("Ensemble Random Search")
    print("=" * 40)
    
    random_results = random_search_ensemble(X_train, y_train, X_test, y_test, n_iter=50)
    
    random_path = os.path.join(output_dir, 'model5_ensemble_random_search.csv')
    random_results.to_csv(random_path, index=False)
    print(f"\nSaved to: {random_path}")
    
    print("\nTop 5 Random Search configurations:")
    print(random_results.head()[['models', 'weights', 'rmse']].to_string(index=False))
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    best_weighted = weighted_results.iloc[0]
    best_stacking = stacking_results.iloc[0]
    best_random = random_results.iloc[0]
    
    print(f"\nBest Weighted:  RMSE={best_weighted['rmse']:.4f} ({best_weighted['models']})")
    print(f"Best Stacking:  RMSE={best_stacking['rmse']:.4f} ({best_stacking['base_models']} + {best_stacking['meta_model']})")
    print(f"Best Random:    RMSE={best_random['rmse']:.4f} ({best_random['models']})")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
