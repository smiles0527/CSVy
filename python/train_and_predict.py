#!/usr/bin/env python3
"""
WHSDSC 2026 - Training and Prediction Pipeline

Aggregates shift-level data, engineers team features, trains models,
and generates predictions for Round 1 matchups.

Usage:
    python train_and_predict.py
    python train_and_predict.py --data data/whl_2025.csv --matchups data/WHSDSC_Rnd1_matchups.xlsx
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not installed")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def aggregate_to_games(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shift-level data to game-level."""
    print("Aggregating shifts to games...")
    
    games = df.groupby('game_id').agg({
        'home_team': 'first',
        'away_team': 'first',
        'went_ot': 'first',
        'home_goals': 'sum',
        'away_goals': 'sum',
        'home_shots': 'sum',
        'away_shots': 'sum',
        'home_xg': 'sum',
        'away_xg': 'sum',
        'home_max_xg': 'max',
        'away_max_xg': 'max',
        'home_assists': 'sum',
        'away_assists': 'sum',
        'home_penalties_committed': 'sum',
        'away_penalties_committed': 'sum',
        'home_penalty_minutes': 'sum',
        'away_penalty_minutes': 'sum',
        'toi': 'sum'
    }).reset_index()
    
    # Extract game number for ordering
    games['game_num'] = games['game_id'].str.extract(r'game_(\d+)').astype(int)
    games = games.sort_values('game_num').reset_index(drop=True)
    
    # Derived features at game level
    games['home_shot_pct'] = games['home_goals'] / games['home_shots'].replace(0, 1)
    games['away_shot_pct'] = games['away_goals'] / games['away_shots'].replace(0, 1)
    games['home_xg_diff'] = games['home_xg'] - games['away_xg']
    games['goal_diff'] = games['home_goals'] - games['away_goals']
    games['home_win'] = (games['home_goals'] > games['away_goals']).astype(int)
    
    print(f"  Aggregated {len(df)} shifts -> {len(games)} games")
    return games


def build_team_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative team statistics from game history."""
    print("Building team statistics...")
    
    teams = set(games['home_team'].unique()) | set(games['away_team'].unique())
    
    # Initialize team stats
    team_stats = {team: {
        'games_played': 0,
        'wins': 0,
        'goals_for': 0,
        'goals_against': 0,
        'shots_for': 0,
        'shots_against': 0,
        'xg_for': 0,
        'xg_against': 0,
        'pim': 0,
        'ot_games': 0,
    } for team in teams}
    
    # Process games in order to build cumulative stats
    for _, game in games.iterrows():
        home = game['home_team']
        away = game['away_team']
        
        # Update stats after game
        team_stats[home]['games_played'] += 1
        team_stats[away]['games_played'] += 1
        
        team_stats[home]['goals_for'] += game['home_goals']
        team_stats[home]['goals_against'] += game['away_goals']
        team_stats[away]['goals_for'] += game['away_goals']
        team_stats[away]['goals_against'] += game['home_goals']
        
        team_stats[home]['shots_for'] += game['home_shots']
        team_stats[home]['shots_against'] += game['away_shots']
        team_stats[away]['shots_for'] += game['away_shots']
        team_stats[away]['shots_against'] += game['home_shots']
        
        team_stats[home]['xg_for'] += game['home_xg']
        team_stats[home]['xg_against'] += game['away_xg']
        team_stats[away]['xg_for'] += game['away_xg']
        team_stats[away]['xg_against'] += game['home_xg']
        
        team_stats[home]['pim'] += game['home_penalty_minutes']
        team_stats[away]['pim'] += game['away_penalty_minutes']
        
        if game['went_ot']:
            team_stats[home]['ot_games'] += 1
            team_stats[away]['ot_games'] += 1
        
        if game['home_goals'] > game['away_goals']:
            team_stats[home]['wins'] += 1
        else:
            team_stats[away]['wins'] += 1
    
    # Convert to DataFrame with derived metrics
    stats_list = []
    for team, stats in team_stats.items():
        gp = max(stats['games_played'], 1)
        stats_list.append({
            'team': team,
            'games_played': stats['games_played'],
            'wins': stats['wins'],
            'win_pct': stats['wins'] / gp,
            'goals_per_game': stats['goals_for'] / gp,
            'goals_against_per_game': stats['goals_against'] / gp,
            'goal_diff_per_game': (stats['goals_for'] - stats['goals_against']) / gp,
            'shots_per_game': stats['shots_for'] / gp,
            'shots_against_per_game': stats['shots_against'] / gp,
            'xg_per_game': stats['xg_for'] / gp,
            'xg_against_per_game': stats['xg_against'] / gp,
            'xg_diff_per_game': (stats['xg_for'] - stats['xg_against']) / gp,
            'pim_per_game': stats['pim'] / gp,
            'ot_pct': stats['ot_games'] / gp,
            'shot_pct': stats['goals_for'] / max(stats['shots_for'], 1),
            'save_pct': 1 - (stats['goals_against'] / max(stats['shots_against'], 1)),
        })
    
    team_df = pd.DataFrame(stats_list)
    print(f"  Built stats for {len(team_df)} teams")
    return team_df


def create_matchup_features(home_team: str, away_team: str, 
                            team_stats: pd.DataFrame) -> dict:
    """Create features for a matchup between two teams."""
    home = team_stats[team_stats['team'] == home_team].iloc[0]
    away = team_stats[team_stats['team'] == away_team].iloc[0]
    
    return {
        # Home team stats
        'home_win_pct': home['win_pct'],
        'home_goals_per_game': home['goals_per_game'],
        'home_goals_against_per_game': home['goals_against_per_game'],
        'home_xg_per_game': home['xg_per_game'],
        'home_xg_against_per_game': home['xg_against_per_game'],
        'home_shots_per_game': home['shots_per_game'],
        'home_shot_pct': home['shot_pct'],
        'home_save_pct': home['save_pct'],
        'home_pim_per_game': home['pim_per_game'],
        
        # Away team stats
        'away_win_pct': away['win_pct'],
        'away_goals_per_game': away['goals_per_game'],
        'away_goals_against_per_game': away['goals_against_per_game'],
        'away_xg_per_game': away['xg_per_game'],
        'away_xg_against_per_game': away['xg_against_per_game'],
        'away_shots_per_game': away['shots_per_game'],
        'away_shot_pct': away['shot_pct'],
        'away_save_pct': away['save_pct'],
        'away_pim_per_game': away['pim_per_game'],
        
        # Differential features
        'win_pct_diff': home['win_pct'] - away['win_pct'],
        'goals_diff': home['goal_diff_per_game'] - away['goal_diff_per_game'],
        'xg_diff': home['xg_diff_per_game'] - away['xg_diff_per_game'],
        'shot_pct_diff': home['shot_pct'] - away['shot_pct'],
        'save_pct_diff': home['save_pct'] - away['save_pct'],
    }


def build_training_data(games: pd.DataFrame, team_stats: pd.DataFrame) -> tuple:
    """Build training dataset with features and targets."""
    print("Building training features...")
    
    features = []
    targets_home = []
    targets_away = []
    
    for _, game in games.iterrows():
        try:
            feat = create_matchup_features(game['home_team'], game['away_team'], team_stats)
            features.append(feat)
            targets_home.append(game['home_goals'])
            targets_away.append(game['away_goals'])
        except IndexError:
            continue  # Skip if team not found
    
    X = pd.DataFrame(features)
    y_home = np.array(targets_home)
    y_away = np.array(targets_away)
    
    print(f"  Built {len(X)} training samples with {len(X.columns)} features")
    return X, y_home, y_away


def train_xgboost(X_train, y_train, X_val, y_val, target_name: str) -> xgb.XGBRegressor:
    """Train XGBoost model with early stopping."""
    print(f"\nTraining XGBoost for {target_name}...")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
    print(f"  Val RMSE:   {np.sqrt(mean_squared_error(y_val, val_pred)):.4f}")
    print(f"  Val MAE:    {mean_absolute_error(y_val, val_pred):.4f}")
    print(f"  Val R2:     {r2_score(y_val, val_pred):.4f}")
    
    return model


def predict_matchups(matchups: pd.DataFrame, team_stats: pd.DataFrame,
                     model_home: xgb.XGBRegressor, model_away: xgb.XGBRegressor) -> pd.DataFrame:
    """Generate predictions for matchups."""
    print("\nGenerating predictions...")
    
    predictions = []
    
    for _, match in matchups.iterrows():
        home = match['home_team']
        away = match['away_team']
        
        try:
            feat = create_matchup_features(home, away, team_stats)
            X = pd.DataFrame([feat])
            
            home_goals = model_home.predict(X)[0]
            away_goals = model_away.predict(X)[0]
            
            predictions.append({
                'game': match.get('game', match.get('game_id', '')),
                'game_id': match.get('game_id', ''),
                'home_team': home,
                'away_team': away,
                'pred_home_goals': round(home_goals, 2),
                'pred_away_goals': round(away_goals, 2),
                'pred_winner': home if home_goals > away_goals else away,
                'pred_goal_diff': round(home_goals - away_goals, 2),
            })
        except Exception as e:
            print(f"  Warning: Could not predict {home} vs {away}: {e}")
            predictions.append({
                'game': match.get('game', ''),
                'game_id': match.get('game_id', ''),
                'home_team': home,
                'away_team': away,
                'pred_home_goals': None,
                'pred_away_goals': None,
                'pred_winner': None,
                'pred_goal_diff': None,
            })
    
    return pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(description='WHSDSC 2026 Training Pipeline')
    parser.add_argument('--data', default='data/whl_2025.csv', help='Path to training data')
    parser.add_argument('--matchups', default='data/WHSDSC_Rnd1_matchups.xlsx', help='Path to matchups file')
    parser.add_argument('--output', default='output/predictions/round1_predictions.csv', help='Output path')
    args = parser.parse_args()
    
    print("=" * 60)
    print("WHSDSC 2026 - TRAINING AND PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Resolve paths
    base_dir = Path(__file__).parent
    data_path = base_dir / args.data
    matchups_path = base_dir / args.matchups
    output_path = base_dir / args.output
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} records")
    
    # Aggregate to games
    games = aggregate_to_games(df)
    
    # Build team statistics
    team_stats = build_team_stats(games)
    
    # Show top teams
    print("\nTop 10 teams by win %:")
    print(team_stats.nlargest(10, 'win_pct')[['team', 'games_played', 'win_pct', 'goals_per_game', 'xg_per_game']].to_string(index=False))
    
    # Build training data
    X, y_home, y_away = build_training_data(games, team_stats)
    
    # Split data
    X_train, X_val, y_home_train, y_home_val, y_away_train, y_away_val = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Train models
    if not XGB_AVAILABLE:
        print("ERROR: XGBoost not available. Install with: pip install xgboost")
        return 1
    
    model_home = train_xgboost(X_train, y_home_train, X_val, y_home_val, "home_goals")
    model_away = train_xgboost(X_train, y_away_train, X_val, y_away_val, "away_goals")
    
    # Feature importance
    print("\nTop 10 features (home goals model):")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model_home.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10).to_string(index=False))
    
    # Save models
    models_dir = base_dir / 'output' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_home.save_model(str(models_dir / 'xgboost_home_goals.json'))
    model_away.save_model(str(models_dir / 'xgboost_away_goals.json'))
    print(f"\nModels saved to {models_dir}")
    
    # Load matchups
    print(f"\nLoading matchups from {matchups_path}...")
    matchups = pd.read_excel(matchups_path)
    print(f"  Loaded {len(matchups)} matchups")
    
    # Generate predictions
    predictions = predict_matchups(matchups, team_stats, model_home, model_away)
    
    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Display predictions
    print("\n" + "=" * 60)
    print("ROUND 1 PREDICTIONS")
    print("=" * 60)
    print(predictions.to_string(index=False))
    
    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    home_wins = (predictions['pred_home_goals'] > predictions['pred_away_goals']).sum()
    print(f"Predicted home wins: {home_wins}/{len(predictions)}")
    print(f"Avg predicted home goals: {predictions['pred_home_goals'].mean():.2f}")
    print(f"Avg predicted away goals: {predictions['pred_away_goals'].mean():.2f}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
