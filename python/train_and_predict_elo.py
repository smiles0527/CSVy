#!/usr/bin/env python3
"""
WHSDSC 2026 - Dynamic Elo + XGBoost Pipeline

Uses dynamic Elo ratings that update after each game, combined with
XGBoost for goal prediction. Outputs win probabilities, not just picks.

Usage:
    python train_and_predict_elo.py
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class DynamicElo:
    """
    Dynamic Elo rating system that updates after each game.
    """
    def __init__(self, k_factor: float = 32, home_advantage: float = 50, 
                 initial_rating: float = 1500, mov_factor: float = 1.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.mov_factor = mov_factor  # Margin of victory multiplier
        self.ratings: Dict[str, float] = {}
        self.history: list = []
    
    def get_rating(self, team: str) -> float:
        """Get current rating for a team."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]
    
    def expected_score(self, home_team: str, away_team: str) -> float:
        """Calculate expected score (win probability) for home team."""
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)
        
        exp_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        return exp_home
    
    def update(self, home_team: str, away_team: str, 
               home_goals: int, away_goals: int) -> Tuple[float, float]:
        """Update ratings after a game. Returns rating changes."""
        # Get pre-game ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Expected scores
        exp_home = self.expected_score(home_team, away_team)
        exp_away = 1 - exp_home
        
        # Actual scores (1 = win, 0.5 = tie, 0 = loss)
        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Margin of victory adjustment
        goal_diff = abs(home_goals - away_goals)
        mov_mult = np.log(goal_diff + 1) * self.mov_factor + 1
        
        # Update ratings
        k = self.k_factor * mov_mult
        home_change = k * (actual_home - exp_home)
        away_change = k * (actual_away - exp_away)
        
        self.ratings[home_team] = home_rating + home_change
        self.ratings[away_team] = away_rating + away_change
        
        # Store history
        self.history.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_rating_before': home_rating,
            'away_rating_before': away_rating,
            'home_rating_after': self.ratings[home_team],
            'away_rating_after': self.ratings[away_team],
            'home_goals': home_goals,
            'away_goals': away_goals,
            'expected_home_win': exp_home,
        })
        
        return home_change, away_change
    
    def get_rankings(self) -> pd.DataFrame:
        """Get current team rankings."""
        rankings = pd.DataFrame([
            {'team': team, 'rating': rating}
            for team, rating in self.ratings.items()
        ]).sort_values('rating', ascending=False).reset_index(drop=True)
        rankings['rank'] = range(1, len(rankings) + 1)
        return rankings


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
    
    print(f"  Aggregated {len(df)} shifts -> {len(games)} games")
    return games


def build_features_with_elo(games: pd.DataFrame, elo: DynamicElo) -> pd.DataFrame:
    """
    Process games chronologically, building features with current Elo
    BEFORE each game, then updating Elo AFTER.
    """
    print("Building features with dynamic Elo...")
    
    # Track rolling stats per team
    team_stats = {}
    
    def get_team_stats(team):
        if team not in team_stats:
            team_stats[team] = {
                'games': 0, 'wins': 0, 'goals_for': 0, 'goals_against': 0,
                'shots_for': 0, 'shots_against': 0, 'xg_for': 0, 'xg_against': 0,
                'recent_goals': [], 'recent_xg': []  # Last 5 games
            }
        return team_stats[team]
    
    features_list = []
    
    for idx, game in games.iterrows():
        home = game['home_team']
        away = game['away_team']
        
        # Get CURRENT ratings before game
        home_elo = elo.get_rating(home)
        away_elo = elo.get_rating(away)
        elo_diff = home_elo - away_elo
        home_win_prob = elo.expected_score(home, away)
        
        # Get current team stats
        home_stats = get_team_stats(home)
        away_stats = get_team_stats(away)
        
        # Calculate rolling averages
        h_games = max(home_stats['games'], 1)
        a_games = max(away_stats['games'], 1)
        
        features = {
            # Elo features
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'elo_win_prob': home_win_prob,
            
            # Home team rolling stats
            'home_win_rate': home_stats['wins'] / h_games,
            'home_gpg': home_stats['goals_for'] / h_games,
            'home_gapg': home_stats['goals_against'] / h_games,
            'home_xgpg': home_stats['xg_for'] / h_games,
            'home_spg': home_stats['shots_for'] / h_games,
            
            # Away team rolling stats
            'away_win_rate': away_stats['wins'] / a_games,
            'away_gpg': away_stats['goals_for'] / a_games,
            'away_gapg': away_stats['goals_against'] / a_games,
            'away_xgpg': away_stats['xg_for'] / a_games,
            'away_spg': away_stats['shots_for'] / a_games,
            
            # Recent form (last 5 games)
            'home_recent_gpg': np.mean(home_stats['recent_goals'][-5:]) if home_stats['recent_goals'] else 0,
            'away_recent_gpg': np.mean(away_stats['recent_goals'][-5:]) if away_stats['recent_goals'] else 0,
            
            # Differentials
            'gpg_diff': (home_stats['goals_for'] / h_games) - (away_stats['goals_for'] / a_games),
            'xgpg_diff': (home_stats['xg_for'] / h_games) - (away_stats['xg_for'] / a_games),
            
            # Targets
            'home_goals': game['home_goals'],
            'away_goals': game['away_goals'],
            'home_win': 1 if game['home_goals'] > game['away_goals'] else 0,
        }
        features_list.append(features)
        
        # NOW update Elo and stats AFTER the game
        elo.update(home, away, game['home_goals'], game['away_goals'])
        
        # Update home team stats
        home_stats['games'] += 1
        home_stats['goals_for'] += game['home_goals']
        home_stats['goals_against'] += game['away_goals']
        home_stats['shots_for'] += game['home_shots']
        home_stats['shots_against'] += game['away_shots']
        home_stats['xg_for'] += game['home_xg']
        home_stats['xg_against'] += game['away_xg']
        home_stats['recent_goals'].append(game['home_goals'])
        home_stats['recent_xg'].append(game['home_xg'])
        if game['home_goals'] > game['away_goals']:
            home_stats['wins'] += 1
        
        # Update away team stats
        away_stats['games'] += 1
        away_stats['goals_for'] += game['away_goals']
        away_stats['goals_against'] += game['home_goals']
        away_stats['shots_for'] += game['away_shots']
        away_stats['shots_against'] += game['home_shots']
        away_stats['xg_for'] += game['away_xg']
        away_stats['xg_against'] += game['home_xg']
        away_stats['recent_goals'].append(game['away_goals'])
        away_stats['recent_xg'].append(game['away_xg'])
        if game['away_goals'] > game['home_goals']:
            away_stats['wins'] += 1
    
    df_features = pd.DataFrame(features_list)
    print(f"  Built {len(df_features)} samples with {len(df_features.columns)} columns")
    
    return df_features, team_stats


def train_models(df: pd.DataFrame) -> Tuple:
    """Train XGBoost models for goals and win probability."""
    print("\nTraining models...")
    
    # Feature columns (exclude targets)
    feature_cols = [c for c in df.columns if c not in ['home_goals', 'away_goals', 'home_win']]
    
    X = df[feature_cols]
    y_home = df['home_goals']
    y_away = df['away_goals']
    y_win = df['home_win']
    
    # Train/val split (use last 20% as validation to respect time order)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_home_train, y_home_val = y_home.iloc[:split_idx], y_home.iloc[split_idx:]
    y_away_train, y_away_val = y_away.iloc[:split_idx], y_away.iloc[split_idx:]
    y_win_train, y_win_val = y_win.iloc[:split_idx], y_win.iloc[split_idx:]
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train home goals model
    print("\n  Training home goals model...")
    model_home = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, random_state=42, verbosity=0
    )
    model_home.fit(X_train, y_home_train, eval_set=[(X_val, y_home_val)], verbose=False)
    
    home_pred = model_home.predict(X_val)
    print(f"    RMSE: {np.sqrt(mean_squared_error(y_home_val, home_pred)):.3f}")
    print(f"    MAE:  {mean_absolute_error(y_home_val, home_pred):.3f}")
    
    # Train away goals model
    print("\n  Training away goals model...")
    model_away = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, random_state=42, verbosity=0
    )
    model_away.fit(X_train, y_away_train, eval_set=[(X_val, y_away_val)], verbose=False)
    
    away_pred = model_away.predict(X_val)
    print(f"    RMSE: {np.sqrt(mean_squared_error(y_away_val, away_pred)):.3f}")
    print(f"    MAE:  {mean_absolute_error(y_away_val, away_pred):.3f}")
    
    # Train win probability model (classifier)
    print("\n  Training win probability model...")
    model_win = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, random_state=42, verbosity=0,
        use_label_encoder=False, eval_metric='logloss'
    )
    model_win.fit(X_train, y_win_train, eval_set=[(X_val, y_win_val)], verbose=False)
    
    win_proba = model_win.predict_proba(X_val)[:, 1]
    win_pred = model_win.predict(X_val)
    print(f"    Accuracy: {accuracy_score(y_win_val, win_pred):.3f}")
    print(f"    Log Loss: {log_loss(y_win_val, win_proba):.3f}")
    
    # Compare to Elo baseline
    elo_proba = X_val['elo_win_prob'].values
    elo_pred = (elo_proba > 0.5).astype(int)
    print(f"\n  Elo baseline accuracy: {accuracy_score(y_win_val, elo_pred):.3f}")
    print(f"  Elo baseline log loss: {log_loss(y_win_val, elo_proba):.3f}")
    
    return model_home, model_away, model_win, feature_cols


def predict_matchups(matchups: pd.DataFrame, elo: DynamicElo, 
                     team_stats: dict, feature_cols: list,
                     model_home, model_away, model_win) -> pd.DataFrame:
    """Generate predictions for Round 1 matchups."""
    print("\nGenerating predictions for matchups...")
    
    predictions = []
    
    for _, match in matchups.iterrows():
        home = match['home_team']
        away = match['away_team']
        
        # Get current Elo ratings
        home_elo = elo.get_rating(home)
        away_elo = elo.get_rating(away)
        elo_win_prob = elo.expected_score(home, away)
        
        # Get team stats
        h_stats = team_stats.get(home, {'games': 0, 'wins': 0, 'goals_for': 0, 'goals_against': 0,
                                         'shots_for': 0, 'xg_for': 0, 'recent_goals': []})
        a_stats = team_stats.get(away, {'games': 0, 'wins': 0, 'goals_for': 0, 'goals_against': 0,
                                         'shots_for': 0, 'xg_for': 0, 'recent_goals': []})
        
        h_games = max(h_stats['games'], 1)
        a_games = max(a_stats['games'], 1)
        
        # Build feature vector
        features = {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            'elo_win_prob': elo_win_prob,
            'home_win_rate': h_stats['wins'] / h_games,
            'home_gpg': h_stats['goals_for'] / h_games,
            'home_gapg': h_stats['goals_against'] / h_games,
            'home_xgpg': h_stats['xg_for'] / h_games,
            'home_spg': h_stats['shots_for'] / h_games,
            'away_win_rate': a_stats['wins'] / a_games,
            'away_gpg': a_stats['goals_for'] / a_games,
            'away_gapg': a_stats['goals_against'] / a_games,
            'away_xgpg': a_stats['xg_for'] / a_games,
            'away_spg': a_stats['shots_for'] / a_games,
            'home_recent_gpg': np.mean(h_stats['recent_goals'][-5:]) if h_stats['recent_goals'] else 0,
            'away_recent_gpg': np.mean(a_stats['recent_goals'][-5:]) if a_stats['recent_goals'] else 0,
            'gpg_diff': (h_stats['goals_for'] / h_games) - (a_stats['goals_for'] / a_games),
            'xgpg_diff': (h_stats['xg_for'] / h_games) - (a_stats['xg_for'] / a_games),
        }
        
        X = pd.DataFrame([features])[feature_cols]
        
        # Predict
        pred_home_goals = model_home.predict(X)[0]
        pred_away_goals = model_away.predict(X)[0]
        xgb_win_prob = model_win.predict_proba(X)[0][1]
        
        # Ensemble: average Elo and XGBoost probabilities
        ensemble_prob = (elo_win_prob + xgb_win_prob) / 2
        
        predictions.append({
            'game': match.get('game', ''),
            'game_id': match.get('game_id', ''),
            'home_team': home,
            'away_team': away,
            'home_elo': round(home_elo, 0),
            'away_elo': round(away_elo, 0),
            'pred_home_goals': round(pred_home_goals, 2),
            'pred_away_goals': round(pred_away_goals, 2),
            'elo_win_prob': round(elo_win_prob, 3),
            'xgb_win_prob': round(xgb_win_prob, 3),
            'ensemble_prob': round(ensemble_prob, 3),
            'predicted_winner': home if ensemble_prob > 0.5 else away,
            'confidence': round(abs(ensemble_prob - 0.5) * 2, 3),  # 0-1 scale
        })
    
    return pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(description='WHSDSC 2026 Dynamic Elo Pipeline')
    parser.add_argument('--data', default='data/whl_2025.csv')
    parser.add_argument('--matchups', default='data/WHSDSC_Rnd1_matchups.xlsx')
    parser.add_argument('--output', default='output/predictions/round1_predictions_elo.csv')
    parser.add_argument('--k-factor', type=float, default=32, help='Elo K-factor')
    parser.add_argument('--home-advantage', type=float, default=50, help='Elo home advantage')
    args = parser.parse_args()
    
    print("=" * 70)
    print("WHSDSC 2026 - DYNAMIC ELO + XGBOOST PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elo K-factor: {args.k_factor}, Home advantage: {args.home_advantage}")
    
    base_dir = Path(__file__).parent
    
    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv(base_dir / args.data)
    print(f"  Loaded {len(df)} records")
    
    # Aggregate to games
    games = aggregate_to_games(df)
    
    # Initialize Elo system
    elo = DynamicElo(k_factor=args.k_factor, home_advantage=args.home_advantage)
    
    # Build features with dynamic Elo
    features_df, team_stats = build_features_with_elo(games, elo)
    
    # Show final Elo rankings
    print("\nFinal Elo Rankings (Top 10):")
    rankings = elo.get_rankings()
    print(rankings.head(10).to_string(index=False))
    
    # Train models
    model_home, model_away, model_win, feature_cols = train_models(features_df)
    
    # Save models
    models_dir = base_dir / 'output' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    model_home.save_model(str(models_dir / 'elo_xgb_home.json'))
    model_away.save_model(str(models_dir / 'elo_xgb_away.json'))
    model_win.save_model(str(models_dir / 'elo_xgb_win.json'))
    print(f"\nModels saved to {models_dir}")
    
    # Load matchups
    print(f"\nLoading matchups...")
    matchups = pd.read_excel(base_dir / args.matchups)
    print(f"  Loaded {len(matchups)} matchups")
    
    # Generate predictions
    predictions = predict_matchups(matchups, elo, team_stats, feature_cols,
                                   model_home, model_away, model_win)
    
    # Save predictions
    output_path = base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Display predictions
    print("\n" + "=" * 70)
    print("ROUND 1 PREDICTIONS")
    print("=" * 70)
    display_cols = ['game', 'home_team', 'away_team', 'home_elo', 'away_elo', 
                    'ensemble_prob', 'predicted_winner', 'confidence']
    print(predictions[display_cols].to_string(index=False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    home_picks = (predictions['predicted_winner'] == predictions['home_team']).sum()
    print(f"Home team picks: {home_picks}/16")
    print(f"Away team picks: {16 - home_picks}/16")
    print(f"Avg confidence: {predictions['confidence'].mean():.1%}")
    print(f"Highest confidence: {predictions.loc[predictions['confidence'].idxmax(), 'home_team']} vs {predictions.loc[predictions['confidence'].idxmax(), 'away_team']} ({predictions['confidence'].max():.1%})")
    print(f"Lowest confidence: {predictions.loc[predictions['confidence'].idxmin(), 'home_team']} vs {predictions.loc[predictions['confidence'].idxmin(), 'away_team']} ({predictions['confidence'].min():.1%})")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
