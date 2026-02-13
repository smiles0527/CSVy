"""
Linear Regression Model Hyperparameter Search

Performs grid search and random search for linear model hyperparameters.
Outputs results to: output/hyperparams/model2_linear_grid_search.csv
                    output/hyperparams/model2_linear_random_search.csv

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

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.experiment_tracker import ExperimentTracker, compute_comprehensive_metrics

# Initialize MLflow tracker - use absolute path with file:// URI
mlruns_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
tracking_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"
tracker = ExperimentTracker(
    experiment_name="linear_hyperparam_search",
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
    ]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Add derived features
    X['elo_diff'] = X['home_elo'] - X['away_elo']
    X['goal_diff_avg'] = X['home_goals_avg'] - X['away_goals_avg']
    
    return X, y


def evaluate_linear_model(model_type, params, X_train, y_train, X_test, y_test, scaler, run_name=None):
    """Evaluate a linear model with given parameters and log to MLflow."""
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    if model_type == 'ridge':
        model = Ridge(**params)
    elif model_type == 'lasso':
        model = Lasso(**params)
    elif model_type == 'elastic':
        model = ElasticNet(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit and predict
    model.fit(X_train_scaled, y_train)
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate comprehensive metrics for competition advantage
    metrics = compute_comprehensive_metrics(
        y_true=y_test,
        y_pred=test_pred,
        y_train_true=y_train,
        y_train_pred=train_pred,
        is_classification=False
    )
    
    # Log to MLflow
    with tracker.start_run(run_name=run_name or f"{model_type}_{datetime.now().strftime('%H%M%S')}"):
        tracker.log_params({'model_type': model_type, **params})
        tracker.log_metrics(metrics)
    
    return metrics


def grid_search_linear(X_train, y_train, X_test, y_test, param_grid, model_type='ridge', verbose=True):
    """
    Perform grid search over all parameter combinations.
    
    Returns DataFrame with all results sorted by RMSE.
    """
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    
    if verbose:
        print(f"Grid Search ({model_type}): {len(combinations)} combinations")
    
    scaler = StandardScaler()
    results = []
    
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        
        try:
            metrics = evaluate_linear_model(model_type, params, X_train, y_train, X_test, y_test, scaler,
                                            run_name=f"{model_type}_grid_{i+1}")
            result = {'model_type': model_type, **params, **metrics}
            results.append(result)
            
            if verbose and (i + 1) % 25 == 0:
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


def random_search_linear(X_train, y_train, X_test, y_test, param_distributions, model_type='ridge', n_iter=50, verbose=True):
    """
    Perform random search sampling from parameter distributions.
    
    Returns DataFrame with results sorted by RMSE.
    """
    if verbose:
        print(f"Random Search ({model_type}): {n_iter} iterations")
    
    scaler = StandardScaler()
    results = []
    
    for i in range(n_iter):
        params = {}
        for key, values in param_distributions.items():
            if isinstance(values, list):
                params[key] = random.choice(values)
            elif hasattr(values, 'rvs'):
                params[key] = values.rvs()
            else:
                params[key] = values
        
        try:
            metrics = evaluate_linear_model(model_type, params, X_train, y_train, X_test, y_test, scaler)
            result = {'model_type': model_type, **params, **metrics}
            results.append(result)
            
            if verbose and (i + 1) % 25 == 0:
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
    """Run hyperparameter search for linear models."""
    print("=" * 60)
    print("Linear Model Hyperparameter Search")
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
    # Ridge Regression Grid Search
    # ========================================
    print("\n" + "=" * 40)
    print("Ridge Regression Grid Search")
    print("=" * 40)
    
    ridge_param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
    }
    
    ridge_results = grid_search_linear(X_train, y_train, X_test, y_test, ridge_param_grid, model_type='ridge')
    
    # Save results
    ridge_path = os.path.join(output_dir, 'model2_ridge_grid_search.csv')
    ridge_results.to_csv(ridge_path, index=False)
    print(f"\nSaved to: {ridge_path}")
    
    print("\nTop 5 Ridge configurations:")
    print(ridge_results.head()[['alpha', 'solver', 'rmse', 'mae', 'r2']].to_string(index=False))
    
    # ========================================
    # Lasso Regression Grid Search
    # ========================================
    print("\n" + "=" * 40)
    print("Lasso Regression Grid Search")
    print("=" * 40)
    
    lasso_param_grid = {
        'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        'max_iter': [1000, 5000],
    }
    
    lasso_results = grid_search_linear(X_train, y_train, X_test, y_test, lasso_param_grid, model_type='lasso')
    
    lasso_path = os.path.join(output_dir, 'model2_lasso_grid_search.csv')
    lasso_results.to_csv(lasso_path, index=False)
    print(f"\nSaved to: {lasso_path}")
    
    print("\nTop 5 Lasso configurations:")
    print(lasso_results.head()[['alpha', 'max_iter', 'rmse', 'mae', 'r2']].to_string(index=False))
    
    # ========================================
    # ElasticNet Random Search
    # ========================================
    print("\n" + "=" * 40)
    print("ElasticNet Random Search")
    print("=" * 40)
    
    try:
        from scipy.stats import loguniform, uniform
        
        elastic_distributions = {
            'alpha': loguniform(0.001, 10),
            'l1_ratio': uniform(0.1, 0.8),
            'max_iter': [1000, 5000, 10000],
        }
    except ImportError:
        elastic_distributions = {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [1000, 5000],
        }
    
    elastic_results = random_search_linear(
        X_train, y_train, X_test, y_test,
        elastic_distributions, model_type='elastic', n_iter=50
    )
    
    elastic_path = os.path.join(output_dir, 'model2_elastic_random_search.csv')
    elastic_results.to_csv(elastic_path, index=False)
    print(f"\nSaved to: {elastic_path}")
    
    print("\nTop 5 ElasticNet configurations:")
    print(elastic_results.head()[['alpha', 'l1_ratio', 'rmse', 'mae', 'r2']].to_string(index=False))
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    best_ridge = ridge_results.iloc[0]
    best_lasso = lasso_results.iloc[0]
    best_elastic = elastic_results.iloc[0]
    
    print(f"\nBest Ridge:     RMSE={best_ridge['rmse']:.4f} (alpha={best_ridge['alpha']})")
    print(f"Best Lasso:     RMSE={best_lasso['rmse']:.4f} (alpha={best_lasso['alpha']})")
    print(f"Best ElasticNet: RMSE={best_elastic['rmse']:.4f} (alpha={best_elastic['alpha']:.4f}, l1_ratio={best_elastic['l1_ratio']:.2f})")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
