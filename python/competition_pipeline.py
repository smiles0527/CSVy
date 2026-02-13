#!/usr/bin/env python3
"""
WHSDSC 2026 — Final Competition Pipeline
=========================================

Combines Elo ratings, XGBoost, and engineered hockey features to predict
16 Round 1 matchups. Uses ONLY pre-game features (no in-game leakage).

Pipeline:
  1. Aggregate shifts → games (1,312 games)
  2. Engineer 90+ hockey features (offensive, defensive, goalie, special teams,
     momentum, SOS, travel proxy, schedule density)
  3. Train Elo model for dynamic team ratings
  4. Train XGBoost on pre-game-only features with time-series CV
  5. Hyperparameter tuning via grid search
  6. Ensemble Elo + XGBoost for final predictions
  7. Output Round 1 predictions with confidence

Usage:
    python competition_pipeline.py
"""

import os, sys, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from utils.hockey_features import aggregate_to_games, engineer_features, get_model_features
from utils.elo_model import EloModel

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(42)

OUTPUT_DIR = Path('output/predictions')
MODEL_DIR = Path('output/models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fix_game_ids(df):
    """Convert game_id from 'game_123' strings to integers."""
    if df['game_id'].dtype == object and df['game_id'].str.contains('game_').any():
        df = df.copy()
        df['game_id'] = df['game_id'].str.replace('game_', '', regex=False).astype(int)
    return df


# ── PRE-GAME FEATURE EXTRACTION ──────────────────────────────────

def get_pregame_features(game_features: pd.DataFrame) -> tuple:
    """Extract ONLY pre-game features (no in-game leakage).
    
    Keeps: rolling stats, season averages, team splits, SOS, OT,
           travel proxy, schedule density, lineup diversity.
    Drops: actual game stats (goals, shots, xG, assists from THIS game).
    """
    # Columns that contain in-game information → LEAKAGE
    leakage_cols = {
        'game_id', 'home_team', 'away_team',
        # Actual game outcomes
        'home_goals', 'away_goals', 'home_win', 'goal_diff', 'total_goals',
        'went_ot',
        # In-game raw stats (from THIS game — not rolling)
        'home_assists', 'home_shots', 'home_xg', 'home_max_xg',
        'away_assists', 'away_shots', 'away_xg', 'away_max_xg',
        'home_penalties_committed', 'home_penalty_minutes',
        'away_penalties_committed', 'away_penalty_minutes',
        'toi', 'n_shifts',
        # Per-game derived metrics (from THIS game — leakage)
        'home_sh_pct', 'home_sv_pct', 'home_pdo', 'home_xg_conversion',
        'home_gsax', 'home_xg_diff', 'home_shot_share',
        'home_penalty_diff', 'home_pim_diff', 'home_assists_per_goal',
        'away_sh_pct', 'away_sv_pct', 'away_pdo', 'away_xg_conversion',
        'away_gsax', 'away_xg_diff', 'away_shot_share',
        'away_penalty_diff', 'away_pim_diff', 'away_assists_per_goal',
    }
    
    feature_cols = [c for c in game_features.columns 
                    if c not in leakage_cols 
                    and game_features[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X = game_features[feature_cols].copy()
    X = X.fillna(X.median())
    
    return X, feature_cols


# ── ELO TRAINING + TUNING ────────────────────────────────────────

def tune_elo(games: pd.DataFrame, verbose: bool = True) -> dict:
    """Grid search over Elo hyperparameters using time-series CV."""
    
    param_grid = {
        'k_factor': [20, 32, 40],
        'home_advantage': [50, 80, 100, 120],
        'mov_multiplier': [0.0, 0.3, 0.5, 0.8],
        'mov_method': ['logarithmic'],
        'initial_rating': [1500],
    }
    
    # Generate combinations
    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    
    if verbose:
        print(f"Elo grid search: {len(combos)} combinations...")
    
    # Time-series 5-fold
    n = len(games)
    fold_size = n // 6
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for combo in combos:
        params = dict(zip(keys, combo))
        
        fold_rmses = []
        fold_accs = []
        for fold in range(5):
            train_end = fold_size * (fold + 2)
            test_start = train_end
            test_end = min(train_end + fold_size, n)
            
            train = games.iloc[:train_end]
            test = games.iloc[test_start:test_end]
            if len(test) == 0:
                continue
            
            model = EloModel(params)
            model.fit(train)
            metrics = model.evaluate(test)
            fold_rmses.append(metrics['combined_rmse'])
            fold_accs.append(metrics['win_accuracy'])
        
        avg_rmse = np.mean(fold_rmses)
        avg_acc = np.mean(fold_accs)
        results.append({**params, 'cv_rmse': avg_rmse, 'cv_win_acc': avg_acc})
        
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params
    
    if verbose:
        print(f"Best Elo params: {best_params}")
        print(f"Best CV RMSE: {best_score:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('cv_rmse')
    results_df.to_csv(str(OUTPUT_DIR / 'elo_tuning_results.csv'), index=False)
    
    return best_params


# ── XGBOOST PRE-GAME TRAINING + TUNING ───────────────────────────

def tune_xgboost(X_train, y_train, target_name='home_goals', verbose=True):
    """Tune XGBoost on pre-game features using time-series CV."""
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [3, 5],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1.0],
    }
    
    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    
    if verbose:
        print(f"XGBoost grid search for {target_name}: {len(combos)} combinations...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    best_score = float('inf')
    best_params = None
    
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        
        fold_rmses = []
        for train_idx, test_idx in tscv.split(X_train):
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]
            
            if XGB_AVAILABLE:
                model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            else:
                model = GradientBoostingRegressor(**{k: v for k, v in params.items()
                                                     if k not in ['colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_weight']},
                                                   random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            fold_rmses.append(rmse(y_te, pred))
        
        avg = np.mean(fold_rmses)
        if avg < best_score:
            best_score = avg
            best_params = params
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)} tested, best RMSE so far: {best_score:.4f}")
    
    if verbose:
        print(f"Best XGBoost params for {target_name}: {best_params}")
        print(f"Best CV RMSE: {best_score:.4f}")
    
    return best_params


# ── ENSEMBLE PREDICTION ───────────────────────────────────────────

def ensemble_predict(elo_pred, xgb_pred, elo_weight=0.4):
    """Blend Elo and XGBoost predictions."""
    return elo_weight * elo_pred + (1 - elo_weight) * xgb_pred


# ── MAIN PIPELINE ────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("WHSDSC 2026 — COMPETITION PIPELINE")
    print("=" * 70)
    
    # ── 1. LOAD & PREPARE DATA ────────────────────────────────────
    print("\n[1] Loading data...")
    raw = pd.read_csv('data/whl_2025.csv')
    raw = fix_game_ids(raw)
    matchups = pd.read_excel('data/WHSDSC_Rnd1_matchups.xlsx')
    print(f"    Raw: {len(raw):,} shifts, {raw['game_id'].nunique()} games")
    print(f"    Matchups: {len(matchups)} Round 1 games")
    
    # ── 2. ENGINEER FEATURES ──────────────────────────────────────
    print("\n[2] Engineering features...")
    game_features, team_stats = engineer_features(raw, rolling_window=10, verbose=True)
    
    # Extract pre-game-only features
    X_all, feature_cols = get_pregame_features(game_features)
    y_home = game_features['home_goals']
    y_away = game_features['away_goals']
    y_win = game_features['home_win']
    
    # Sort by game_id for time-series integrity
    sort_idx = game_features['game_id'].argsort()
    X_all = X_all.iloc[sort_idx].reset_index(drop=True)
    y_home = y_home.iloc[sort_idx].reset_index(drop=True)
    y_away = y_away.iloc[sort_idx].reset_index(drop=True)
    y_win = y_win.iloc[sort_idx].reset_index(drop=True)
    game_features_sorted = game_features.iloc[sort_idx].reset_index(drop=True)
    
    print(f"    Pre-game features: {X_all.shape[1]} columns")
    print(f"    Feature list: {sorted(feature_cols)[:10]}... ({len(feature_cols)} total)")
    
    # ── 3. TUNE & TRAIN ELO ──────────────────────────────────────
    print("\n[3] Tuning Elo model...")
    games = aggregate_to_games(fix_game_ids(pd.read_csv('data/whl_2025.csv')))
    games = games.sort_values('game_id').reset_index(drop=True)
    
    best_elo_params = tune_elo(games, verbose=True)
    
    # Train final Elo on ALL data
    elo_model = EloModel(best_elo_params)
    elo_model.fit(games)
    elo_model.save_model(str(MODEL_DIR / 'elo_final'))
    
    # Evaluate on last 20%
    split = int(len(games) * 0.8)
    elo_eval = EloModel(best_elo_params)
    elo_eval.fit(games.iloc[:split])
    elo_metrics = elo_eval.evaluate(games.iloc[split:])
    print(f"    Elo test: RMSE={elo_metrics['combined_rmse']:.4f}, Win Acc={elo_metrics['win_accuracy']:.1%}")
    
    # ── 4. TUNE & TRAIN XGBOOST ──────────────────────────────────
    print("\n[4] Tuning XGBoost models...")
    
    # Home goals model
    best_home_params = tune_xgboost(X_all, y_home, 'home_goals', verbose=True)
    
    # Away goals model
    best_away_params = tune_xgboost(X_all, y_away, 'away_goals', verbose=True)
    
    # Train final models on all data
    print("\n    Training final XGBoost models on full data...")
    if XGB_AVAILABLE:
        home_model = xgb.XGBRegressor(**best_home_params, random_state=42, verbosity=0)
        away_model = xgb.XGBRegressor(**best_away_params, random_state=42, verbosity=0)
    else:
        home_model = GradientBoostingRegressor(**{k: v for k, v in best_home_params.items()
                                                  if k not in ['colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_weight']},
                                                random_state=42)
        away_model = GradientBoostingRegressor(**{k: v for k, v in best_away_params.items()
                                                  if k not in ['colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_weight']},
                                                random_state=42)
    
    home_model.fit(X_all, y_home)
    away_model.fit(X_all, y_away)
    
    # Win classifier
    print("    Training win classifier...")
    if XGB_AVAILABLE:
        win_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, eval_metric='logloss'
        )
    else:
        win_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
    win_model.fit(X_all, y_win)
    
    # Evaluate on holdout
    split = int(len(X_all) * 0.8)
    X_tr, X_te = X_all.iloc[:split], X_all.iloc[split:]
    
    if XGB_AVAILABLE:
        h_eval = xgb.XGBRegressor(**best_home_params, random_state=42, verbosity=0)
        a_eval = xgb.XGBRegressor(**best_away_params, random_state=42, verbosity=0)
    else:
        h_eval = GradientBoostingRegressor(**{k: v for k, v in best_home_params.items()
                                              if k not in ['colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_weight']},
                                            random_state=42)
        a_eval = GradientBoostingRegressor(**{k: v for k, v in best_away_params.items()
                                              if k not in ['colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_weight']},
                                            random_state=42)
    
    h_eval.fit(X_tr, y_home.iloc[:split])
    a_eval.fit(X_tr, y_away.iloc[:split])
    h_pred = h_eval.predict(X_te)
    a_pred = a_eval.predict(X_te)
    
    xgb_home_rmse = rmse(y_home.iloc[split:], h_pred)
    xgb_away_rmse = rmse(y_away.iloc[split:], a_pred)
    xgb_combined_rmse = rmse(
        list(y_home.iloc[split:]) + list(y_away.iloc[split:]),
        list(h_pred) + list(a_pred)
    )
    xgb_win_acc = ((h_pred > a_pred) == (y_home.iloc[split:].values > y_away.iloc[split:].values)).mean()
    
    print(f"    XGBoost test: Home RMSE={xgb_home_rmse:.4f}, Away RMSE={xgb_away_rmse:.4f}")
    print(f"    XGBoost test: Combined RMSE={xgb_combined_rmse:.4f}, Win Acc={xgb_win_acc:.1%}")
    
    # ── 5. FIND OPTIMAL ENSEMBLE WEIGHT ───────────────────────────
    print("\n[5] Optimizing ensemble weights...")
    
    # Get Elo predictions on test set
    elo_h_preds, elo_a_preds = [], []
    test_games = games.iloc[split:]
    for _, game in test_games.iterrows():
        h, a = elo_eval.predict_goals(game)
        elo_h_preds.append(h)
        elo_a_preds.append(a)
    elo_h_preds = np.array(elo_h_preds)
    elo_a_preds = np.array(elo_a_preds)
    
    best_weight = 0.5
    best_ens_rmse = float('inf')
    for w in np.arange(0.0, 1.05, 0.05):
        ens_h = ensemble_predict(elo_h_preds, h_pred, elo_weight=w)
        ens_a = ensemble_predict(elo_a_preds, a_pred, elo_weight=w)
        ens_rmse = rmse(
            list(y_home.iloc[split:]) + list(y_away.iloc[split:]),
            list(ens_h) + list(ens_a)
        )
        if ens_rmse < best_ens_rmse:
            best_ens_rmse = ens_rmse
            best_weight = w
    
    print(f"    Best ensemble weight: Elo={best_weight:.2f}, XGBoost={1-best_weight:.2f}")
    print(f"    Ensemble test RMSE: {best_ens_rmse:.4f}")
    
    # Ensemble win accuracy
    ens_h_test = ensemble_predict(elo_h_preds, h_pred, best_weight)
    ens_a_test = ensemble_predict(elo_a_preds, a_pred, best_weight)
    ens_win_acc = ((ens_h_test > ens_a_test) == (y_home.iloc[split:].values > y_away.iloc[split:].values)).mean()
    print(f"    Ensemble Win Acc: {ens_win_acc:.1%}")
    
    # ── 6. PREDICT ROUND 1 ───────────────────────────────────────
    print("\n[6] Predicting Round 1 matchups...")
    
    predictions = []
    for _, matchup in matchups.iterrows():
        ht = matchup['home_team']
        at = matchup['away_team']
        
        # Elo predictions
        game_row = pd.Series({'home_team': ht, 'away_team': at})
        elo_home, elo_away = elo_model.predict_goals(game_row)
        winner, win_prob = elo_model.predict_winner(game_row)
        
        home_elo = elo_model.ratings.get(ht, 1500)
        away_elo = elo_model.ratings.get(at, 1500)
        
        # XGBoost predictions — need to build a feature row for this matchup
        # Use the team's most recent stats from the game_features
        home_last = game_features_sorted[
            (game_features_sorted['home_team'] == ht) | (game_features_sorted['away_team'] == ht)
        ].iloc[-1:] if len(game_features_sorted[
            (game_features_sorted['home_team'] == ht) | (game_features_sorted['away_team'] == ht)
        ]) > 0 else None
        
        away_last = game_features_sorted[
            (game_features_sorted['home_team'] == at) | (game_features_sorted['away_team'] == at)
        ].iloc[-1:] if len(game_features_sorted[
            (game_features_sorted['home_team'] == at) | (game_features_sorted['away_team'] == at)
        ]) > 0 else None
        
        # Build feature vector from last known home game for this team
        home_as_home = game_features_sorted[game_features_sorted['home_team'] == ht]
        if len(home_as_home) > 0:
            feature_row = home_as_home.iloc[-1:]
            X_pred, _ = get_pregame_features(feature_row)
            
            # Ensure columns match
            for col in feature_cols:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            X_pred = X_pred[feature_cols].fillna(X_all.median())
            
            xgb_home = float(home_model.predict(X_pred)[0])
            xgb_away = float(away_model.predict(X_pred)[0])
            xgb_win_prob = float(win_model.predict_proba(X_pred)[0][1])
        else:
            xgb_home = 3.0
            xgb_away = 3.0
            xgb_win_prob = 0.5
        
        # Ensemble
        ens_home = ensemble_predict(elo_home, xgb_home, best_weight)
        ens_away = ensemble_predict(elo_away, xgb_away, best_weight)
        ens_win_prob = best_weight * (win_prob if winner == ht else 1 - win_prob) + (1 - best_weight) * xgb_win_prob
        
        pred_winner = ht if ens_home > ens_away else at
        confidence = max(ens_win_prob, 1 - ens_win_prob)
        
        predictions.append({
            'game': matchup['game'],
            'game_id': matchup['game_id'],
            'home_team': ht,
            'away_team': at,
            'home_elo': round(home_elo, 0),
            'away_elo': round(away_elo, 0),
            'elo_home_goals': round(elo_home, 2),
            'elo_away_goals': round(elo_away, 2),
            'xgb_home_goals': round(xgb_home, 2),
            'xgb_away_goals': round(xgb_away, 2),
            'ensemble_home_goals': round(ens_home, 2),
            'ensemble_away_goals': round(ens_away, 2),
            'home_win_prob': round(ens_win_prob, 3),
            'predicted_winner': pred_winner,
            'confidence': round(confidence, 3),
        })
    
    pred_df = pd.DataFrame(predictions)
    
    # Save predictions
    pred_path = OUTPUT_DIR / 'round1_predictions_final.csv'
    pred_df.to_csv(str(pred_path), index=False)
    
    print(f"\n{'='*70}")
    print("ROUND 1 PREDICTIONS")
    print(f"{'='*70}")
    print(f"\n{'Game':<5} {'Home':<15} {'Away':<15} {'Ens Home':>9} {'Ens Away':>9} {'Winner':<15} {'Conf':>6}")
    print("-" * 80)
    for _, p in pred_df.iterrows():
        print(f"{p['game']:<5} {p['home_team']:<15} {p['away_team']:<15} "
              f"{p['ensemble_home_goals']:>9.2f} {p['ensemble_away_goals']:>9.2f} "
              f"{p['predicted_winner']:<15} {p['confidence']:>5.1%}")
    
    home_picks = (pred_df['predicted_winner'] == pred_df['home_team']).sum()
    away_picks = 16 - home_picks
    print(f"\nHome picks: {home_picks}, Away picks: {away_picks}")
    print(f"Avg confidence: {pred_df['confidence'].mean():.1%}")
    
    # ── 7. SAVE METRICS SUMMARY ──────────────────────────────────
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data': {
            'total_shifts': len(raw),
            'total_games': len(games),
            'total_teams': games['home_team'].nunique(),
            'pregame_features': len(feature_cols),
        },
        'elo': {
            'params': best_elo_params,
            'test_combined_rmse': elo_metrics['combined_rmse'],
            'test_win_accuracy': elo_metrics['win_accuracy'],
        },
        'xgboost': {
            'home_params': best_home_params,
            'away_params': best_away_params,
            'test_home_rmse': xgb_home_rmse,
            'test_away_rmse': xgb_away_rmse,
            'test_combined_rmse': xgb_combined_rmse,
            'test_win_accuracy': float(xgb_win_acc),
        },
        'ensemble': {
            'elo_weight': best_weight,
            'xgb_weight': 1 - best_weight,
            'test_combined_rmse': best_ens_rmse,
            'test_win_accuracy': float(ens_win_acc),
        },
        'predictions': {
            'home_picks': int(home_picks),
            'away_picks': int(away_picks),
            'avg_confidence': float(pred_df['confidence'].mean()),
        }
    }
    
    summary_path = OUTPUT_DIR / 'pipeline_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Predictions: {pred_path}")
    print(f"Summary:     {summary_path}")
    print(f"Elo model:   {MODEL_DIR / 'elo_final.pkl'}")
    
    return pred_df, summary


if __name__ == '__main__':
    pred_df, summary = main()
