"""
Random Forest Model Hyperparameter Search

Performs grid search and random search for Random Forest model hyperparameters.
Outputs results to: output/hyperparams/model4_random_forest_grid_search.csv
                    output/hyperparams/model4_random_forest_random_search.csv

All runs are logged to MLflow for visualization at http://localhost:5000
"""

import sys
import os
import itertools
import random
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_tracker import ExperimentTracker

# Initialize MLflow tracker - use absolute path with file:// URI
mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
tracking_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"
tracker = ExperimentTracker(
    experiment_name="random_forest_hyperparam_search",
    tracking_uri=tracking_uri
)

# Handle scikit-learn version compatibility for RMSE
try:
    from sklearn.metrics import root_mean_squared_error
    def rmse_score(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)
except ImportError:
    def rmse_score(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


def generate_sample_data(n_games=500, n_teams=10):
    """Generate synthetic hockey data for testing with features for Random Forest."""
    np.random.seed(42)
    random.seed(42)
    teams = [f"Team_{i}" for i in range(n_teams)]
    
    # Create team strength factors
    team_strengths = {team: np.random.uniform(0.8, 1.2) for team in teams}
    
    games = []
    for game_idx in range(n_games):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        
        # Base expected goals with team strengths
        home_exp = 3.0 * team_strengths[home] * 1.05  # Home advantage
        away_exp = 3.0 * team_strengths[away]
        
        # Simulate realistic scores
        home_goals = max(0, int(np.random.poisson(home_exp)))
        away_goals = max(0, int(np.random.poisson(away_exp)))
        
        # Create features
        games.append({
            'game_id': game_idx,
            'home_team': home,
            'away_team': away,
            'home_goals': home_goals,
            'away_goals': away_goals,
            # Feature columns 
            'home_wins_l10': random.randint(2, 8),
            'away_wins_l10': random.randint(2, 8),
            'home_goals_l10': random.uniform(20, 40),
            'away_goals_l10': random.uniform(20, 40),
            'home_goals_against_l10': random.uniform(20, 40),
            'away_goals_against_l10': random.uniform(20, 40),
            'home_pp_pct': random.uniform(0.15, 0.30),
            'away_pp_pct': random.uniform(0.15, 0.30),
            'home_pk_pct': random.uniform(0.75, 0.90),
            'away_pk_pct': random.uniform(0.75, 0.90),
            'home_shot_pct': random.uniform(0.08, 0.12),
            'away_shot_pct': random.uniform(0.08, 0.12),
            'home_save_pct': random.uniform(0.88, 0.94),
            'away_save_pct': random.uniform(0.88, 0.94),
            'home_rest_days': random.randint(1, 5),
            'away_rest_days': random.randint(1, 5),
            'is_home_b2b': 1 if random.random() < 0.15 else 0,
            'is_away_b2b': 1 if random.random() < 0.15 else 0,
        })
    
    return pd.DataFrame(games)


def prepare_features_target(df, target_col='home_goals'):
    """Prepare feature matrix X and target y from DataFrame."""
    feature_cols = [
        'home_wins_l10', 'away_wins_l10',
        'home_goals_l10', 'away_goals_l10',
        'home_goals_against_l10', 'away_goals_against_l10',
        'home_pp_pct', 'away_pp_pct',
        'home_pk_pct', 'away_pk_pct',
        'home_shot_pct', 'away_shot_pct',
        'home_save_pct', 'away_save_pct',
        'home_rest_days', 'away_rest_days',
        'is_home_b2b', 'is_away_b2b',
    ]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y


def evaluate_params(params, X_train, y_train, X_test, y_test, run_name=None):
    """Evaluate a single parameter combination and log to MLflow."""
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    metrics = {
        'rmse': rmse_score(y_test, predictions),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
    }
    
    # Log to MLflow
    with tracker.start_run(run_name=run_name or f"rf_{datetime.now().strftime('%H%M%S')}"):
        tracker.log_params(params)
        tracker.log_metrics(metrics)
    
    return metrics


def grid_search_rf(X_train, y_train, X_test, y_test, param_grid, verbose=True):
    """
    Perform grid search over all parameter combinations.
    
    Returns DataFrame with all results sorted by RMSE.
    """
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    
    if verbose:
        print(f"Grid Search: {len(combinations)} combinations")
    
    results = []
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        
        try:
            metrics = evaluate_params(params, X_train, y_train, X_test, y_test)
            result = {**params, **metrics}
            results.append(result)
            
            if verbose and (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{len(combinations)}")
        except Exception as e:
            if verbose:
                print(f"  Error with params {params}: {e}")
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values('rmse').reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def random_search_rf(X_train, y_train, X_test, y_test, param_distributions, n_iter=50, verbose=True):
    """
    Perform random search sampling from parameter distributions.
    
    Returns DataFrame with results sorted by RMSE.
    """
    if verbose:
        print(f"Random Search: {n_iter} iterations")
    
    results = []
    for i in range(n_iter):
        params = {}
        for key, dist in param_distributions.items():
            if isinstance(dist, list):
                params[key] = random.choice(dist)
            elif isinstance(dist, tuple):
                if len(dist) == 3 and dist[2] == 'log':
                    # Log-uniform sampling
                    params[key] = np.exp(np.random.uniform(np.log(dist[0]), np.log(dist[1])))
                elif len(dist) == 3 and dist[2] == 'int':
                    # Integer uniform
                    params[key] = random.randint(int(dist[0]), int(dist[1]))
                else:
                    # Uniform sampling
                    params[key] = np.random.uniform(dist[0], dist[1])
        
        try:
            metrics = evaluate_params(params, X_train, y_train, X_test, y_test)
            result = {**params, **metrics}
            results.append(result)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_iter}")
        except Exception as e:
            if verbose:
                print(f"  Error with params {params}: {e}")
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values('rmse').reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def main():
    print("=" * 60)
    print("RANDOM FOREST MODEL HYPERPARAMETER SEARCH")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate sample data
    print("\nGenerating sample data...")
    df = generate_sample_data(n_games=800, n_teams=12)
    
    # Split into train/test
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Prepare features
    X_train, y_train = prepare_features_target(train_df)
    X_test, y_test = prepare_features_target(test_df)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'output', 'hyperparams')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define parameter grid 
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.5, None],
    }
    
    # Grid Search
    print("\n" + "-" * 40)
    print("GRID SEARCH")
    print("-" * 40)
    
    grid_results = grid_search_rf(X_train, y_train, X_test, y_test, param_grid)
    
    # Save grid results
    grid_file = os.path.join(output_dir, 'model4_random_forest_grid_search.csv')
    grid_results.to_csv(grid_file, index=False)
    
    print(f"\nGrid Search Results saved to: {grid_file}")
    print(f"Total combinations tested: {len(grid_results)}")
    print("\nTop 5 configurations:")
    display_cols = ['rank', 'n_estimators', 'max_depth', 'min_samples_split', 'rmse', 'mae', 'r2']
    print(grid_results[display_cols].head())
    
    # Random Search with continuous distributions
    print("\n" + "-" * 40)
    print("RANDOM SEARCH")
    print("-" * 40)
    
    param_distributions = {
        'n_estimators': (50, 500, 'int'),
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': (2, 20, 'int'),
        'min_samples_leaf': (1, 10, 'int'),
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None],
    }
    
    random_results = random_search_rf(X_train, y_train, X_test, y_test, 
                                       param_distributions, n_iter=100)
    
    # Save random results
    random_file = os.path.join(output_dir, 'model4_random_forest_random_search.csv')
    random_results.to_csv(random_file, index=False)
    
    print(f"\nRandom Search Results saved to: {random_file}")
    print(f"Total iterations: {len(random_results)}")
    print("\nTop 5 configurations:")
    display_cols = ['rank', 'n_estimators', 'max_depth', 'min_samples_split', 'rmse', 'mae', 'r2']
    print(random_results[display_cols].head())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(grid_results) > 0:
        best_grid = grid_results.iloc[0]
        print(f"\nBest Grid Search Config:")
        print(f"  N Estimators: {best_grid.get('n_estimators', 'N/A')}")
        print(f"  Max Depth: {best_grid.get('max_depth', 'None')}")
        print(f"  Min Samples Split: {best_grid.get('min_samples_split', 'N/A')}")
        print(f"  RMSE: {best_grid['rmse']:.4f}")
        print(f"  MAE: {best_grid['mae']:.4f}")
        print(f"  R²: {best_grid['r2']:.4f}")
    
    if len(random_results) > 0:
        best_random = random_results.iloc[0]
        print(f"\nBest Random Search Config:")
        print(f"  N Estimators: {int(best_random.get('n_estimators', 0))}")
        print(f"  Max Depth: {best_random.get('max_depth', 'None')}")
        print(f"  Min Samples Split: {int(best_random.get('min_samples_split', 0))}")
        print(f"  RMSE: {best_random['rmse']:.4f}")
        print(f"  MAE: {best_random['mae']:.4f}")
        print(f"  R²: {best_random['r2']:.4f}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
