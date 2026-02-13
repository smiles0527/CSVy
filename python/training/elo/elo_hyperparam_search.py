"""
Elo Model Hyperparameter Search

Performs grid search and random search for Elo model hyperparameters.
Outputs results to: output/hyperparams/model3_elo_grid_search.csv
                    output/hyperparams/model3_elo_random_search.csv

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

from utils.elo_model import EloModel
from utils.experiment_tracker import ExperimentTracker

# Initialize MLflow tracker - use absolute path with file:// URI
mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
tracking_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"
tracker = ExperimentTracker(
    experiment_name="elo_hyperparam_search",
    tracking_uri=tracking_uri
)


def generate_sample_data(n_games=500, n_teams=10):
    """Generate synthetic hockey data for testing."""
    np.random.seed(42)
    teams = [f"Team_{i}" for i in range(n_teams)]
    
    games = []
    for _ in range(n_games):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        
        # Simulate realistic scores
        home_goals = np.random.poisson(3.2)  # Home advantage
        away_goals = np.random.poisson(2.8)
        
        games.append({
            'home_team': home,
            'away_team': away,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_rest': random.randint(1, 5),
            'away_rest': random.randint(1, 5),
            'travel_distance': random.randint(0, 2000) if random.random() > 0.3 else 0,
            'home_injuries': random.randint(0, 2),
            'away_injuries': random.randint(0, 2),
        })
    
    return pd.DataFrame(games)


def evaluate_params(params, train_df, test_df, run_name=None):
    """Evaluate a single parameter combination and log to MLflow."""
    model = EloModel(params)
    model.fit(train_df)
    metrics = model.evaluate(test_df)
    
    # Log to MLflow
    with tracker.start_run(run_name=run_name or f"elo_{datetime.now().strftime('%H%M%S')}"):
        tracker.log_params(params)
        tracker.log_metrics(metrics)
    
    return metrics


def grid_search_elo(train_df, test_df, param_grid, verbose=True):
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
            metrics = evaluate_params(params, train_df, test_df)
            result = {**params, **metrics}
            results.append(result)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(combinations)}")
        except Exception as e:
            if verbose:
                print(f"  Error with params {params}: {e}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('rmse').reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def random_search_elo(train_df, test_df, param_distributions, n_iter=50, verbose=True):
    """
    Perform random search sampling from parameter distributions.
    
    param_distributions: dict where each value is either:
        - list: sample uniformly from list
        - tuple (min, max): sample uniformly from range
        - tuple (min, max, 'log'): sample log-uniformly
    
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
                else:
                    # Uniform sampling
                    if isinstance(dist[0], int) and isinstance(dist[1], int):
                        params[key] = random.randint(dist[0], dist[1])
                    else:
                        params[key] = random.uniform(dist[0], dist[1])
        
        try:
            metrics = evaluate_params(params, train_df, test_df)
            result = {**params, **metrics}
            results.append(result)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_iter}")
        except Exception as e:
            if verbose:
                print(f"  Error with params {params}: {e}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('rmse').reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def main():
    print("=" * 60)
    print("ELO MODEL HYPERPARAMETER SEARCH")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Generate sample data (or load real data if available)
    print("Generating sample data...")
    data = generate_sample_data(n_games=800, n_teams=12)
    
    # Train/test split (chronological)
    split_idx = int(len(data) * 0.7)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    
    print(f"  Training games: {len(train_df)}")
    print(f"  Test games: {len(test_df)}")
    print()
    
    # Define parameter grids
    # Grid Search: Reduced set for exhaustive search
    param_grid = {
        'k_factor': [20, 32, 40],
        'initial_rating': [1500],
        'home_advantage': [50, 100, 150],
        'mov_multiplier': [0.0, 1.0, 1.5],
        'mov_method': ['linear', 'logarithmic'],
        'season_carryover': [0.67, 0.75, 0.85],
        'ot_win_multiplier': [0.75, 1.0],
        'rest_advantage_per_day': [0, 10],
        'b2b_penalty': [0, 50],
    }
    
    # Random Search: Continuous distributions for fine-tuning
    param_distributions = {
        'k_factor': (15, 50),
        'initial_rating': [1500],
        'home_advantage': (30, 200),
        'mov_multiplier': (0.0, 2.0),
        'mov_method': ['linear', 'logarithmic'],
        'season_carryover': (0.5, 0.95),
        'ot_win_multiplier': (0.6, 1.0),
        'rest_advantage_per_day': (0, 20),
        'b2b_penalty': (0, 80),
    }
    
    # Output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'output', 'hyperparams'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Grid Search
    print("-" * 40)
    print("GRID SEARCH")
    print("-" * 40)
    grid_results = grid_search_elo(train_df, test_df, param_grid)
    
    grid_output = os.path.join(output_dir, 'model3_elo_grid_search.csv')
    grid_results.to_csv(grid_output, index=False)
    print(f"\nGrid Search Results saved to: {grid_output}")
    print(f"Total combinations tested: {len(grid_results)}")
    print("\nTop 5 configurations:")
    print(grid_results.head()[['rank', 'k_factor', 'home_advantage', 'mov_multiplier', 'rmse', 'mae', 'r2']])
    print()
    
    # Run Random Search
    print("-" * 40)
    print("RANDOM SEARCH")
    print("-" * 40)
    random_results = random_search_elo(train_df, test_df, param_distributions, n_iter=100)
    
    random_output = os.path.join(output_dir, 'model3_elo_random_search.csv')
    random_results.to_csv(random_output, index=False)
    print(f"\nRandom Search Results saved to: {random_output}")
    print(f"Total iterations: {len(random_results)}")
    print("\nTop 5 configurations:")
    print(random_results.head()[['rank', 'k_factor', 'home_advantage', 'mov_multiplier', 'rmse', 'mae', 'r2']])
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    best_grid = grid_results.iloc[0]
    best_random = random_results.iloc[0]
    
    print("\nBest Grid Search Config:")
    print(f"  K-factor: {best_grid['k_factor']}")
    print(f"  Home Advantage: {best_grid['home_advantage']}")
    print(f"  MOV Multiplier: {best_grid['mov_multiplier']}")
    print(f"  MOV Method: {best_grid['mov_method']}")
    print(f"  RMSE: {best_grid['rmse']:.4f}")
    print(f"  MAE: {best_grid['mae']:.4f}")
    print(f"  R²: {best_grid['r2']:.4f}")
    
    print("\nBest Random Search Config:")
    print(f"  K-factor: {best_random['k_factor']:.2f}")
    print(f"  Home Advantage: {best_random['home_advantage']:.2f}")
    print(f"  MOV Multiplier: {best_random['mov_multiplier']:.2f}")
    print(f"  MOV Method: {best_random['mov_method']}")
    print(f"  RMSE: {best_random['rmse']:.4f}")
    print(f"  MAE: {best_random['mae']:.4f}")
    print(f"  R²: {best_random['r2']:.4f}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
