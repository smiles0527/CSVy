"""
Baseline Models - Reference Implementations for Hockey Prediction

This module provides multiple baseline models to benchmark against.
All models follow the same interface as EloModel for consistency.

Available Models:
    - GlobalMeanBaseline: Predicts league-wide average goals
    - TeamMeanBaseline: Predicts based on team-specific averages
    - HomeAwayBaseline: Accounts for home/away goal differentials
    - MovingAverageBaseline: Uses recent game window for predictions
    - WeightedHistoryBaseline: Weights recent games more heavily

Usage:
    from utils.baseline_model import TeamMeanBaseline
    
    model = TeamMeanBaseline()
    model.fit(games_df)
    metrics = model.evaluate(test_df)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Column name mappings - same as ELO model for consistency
COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_goals': ['home_goals', 'home_score', 'h_goals', 'goals_home', 'home_pts'],
    'away_goals': ['away_goals', 'away_score', 'a_goals', 'goals_away', 'away_pts', 'visitor_goals'],
    'game_date': ['game_date', 'date', 'Date', 'game_datetime', 'datetime', 'game_time'],
}


def get_value(game, field, default=None):
    """Get a value from a game record, checking multiple possible column names."""
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in game:
            val = game[alias]
            if pd.isna(val):
                return default
            return val
    return default


def get_column(df, field):
    """Find the correct column name in a DataFrame."""
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


class BaselineModel:
    """
    Abstract base class for all baseline models.
    
    All baseline models must implement:
        - fit(games_df): Train on historical data
        - predict_goals(game): Predict home and away goals
        - evaluate(games_df): Calculate performance metrics
    """
    
    def __init__(self, params=None):
        """
        Initialize baseline model.
        
        Parameters
        ----------
        params : dict, optional
            Model-specific parameters (most baselines need none)
        """
        self.params = params or {}
        self.is_fitted = False
    
    def fit(self, games_df):
        """Train the model on historical games."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict_goals(self, game):
        """
        Predict goals for both teams.
        
        Parameters
        ----------
        game : dict or Series
            Game record with team identifiers
        
        Returns
        -------
        tuple
            (home_goals_pred, away_goals_pred)
        """
        raise NotImplementedError("Subclasses must implement predict_goals()")
    
    def evaluate(self, games_df):
        """
        Evaluate model on test set.
        
        Parameters
        ----------
        games_df : DataFrame
            Test games with actual outcomes
        
        Returns
        -------
        dict
            {'rmse': float, 'mae': float, 'r2': float}
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        home_preds = []
        away_preds = []
        home_actuals = []
        away_actuals = []
        
        for _, game in games_df.iterrows():
            home_pred, away_pred = self.predict_goals(game)
            home_preds.append(home_pred)
            away_preds.append(away_pred)
            home_actuals.append(get_value(game, 'home_goals', 0))
            away_actuals.append(get_value(game, 'away_goals', 0))
        
        # Calculate metrics for home goals
        rmse = mean_squared_error(home_actuals, home_preds, squared=False)
        mae = mean_absolute_error(home_actuals, home_preds)
        r2 = r2_score(home_actuals, home_preds) if len(set(home_actuals)) > 1 else 0.0
        
        # Also calculate combined metrics (both home and away)
        all_preds = home_preds + away_preds
        all_actuals = home_actuals + away_actuals
        combined_rmse = mean_squared_error(all_actuals, all_preds, squared=False)
        combined_mae = mean_absolute_error(all_actuals, all_preds)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'combined_rmse': combined_rmse,
            'combined_mae': combined_mae
        }
    
    def get_summary(self):
        """Get model summary statistics."""
        raise NotImplementedError("Subclasses must implement get_summary()")


class GlobalMeanBaseline(BaselineModel):
    """
    Simplest baseline: predict league-wide average goals for all games.
    
    This model predicts the same value for every game, regardless of teams.
    Useful as a "sanity check" - any useful model should beat this.
    
    Example:
        If league average is 3.1 goals per team, every prediction is (3.1, 3.1).
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.global_mean_home = None
        self.global_mean_away = None
    
    def fit(self, games_df):
        """Calculate league-wide average goals."""
        home_col = get_column(games_df, 'home_goals')
        away_col = get_column(games_df, 'away_goals')
        
        if home_col is None or away_col is None:
            raise ValueError("DataFrame must contain home_goals and away_goals columns")
        
        self.global_mean_home = games_df[home_col].mean()
        self.global_mean_away = games_df[away_col].mean()
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def predict_goals(self, game):
        """Return global averages for all predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.global_mean_home, self.global_mean_away
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'GlobalMeanBaseline',
            'global_mean_home': round(self.global_mean_home, 3),
            'global_mean_away': round(self.global_mean_away, 3),
            'n_games_trained': self.n_games
        }


class TeamMeanBaseline(BaselineModel):
    """
    Team-specific baseline: predict based on each team's historical averages.
    
    Calculates:
        - Average goals scored by each team (offense)
        - Average goals allowed by each team (defense)
    
    Prediction formula:
        home_goals = (home_team_offense + away_team_defense_allowed) / 2
        away_goals = (away_team_offense + home_team_defense_allowed) / 2
    
    This is a standard baseline in sports analytics.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.team_offense = {}  # Goals scored
        self.team_defense = {}  # Goals allowed
        self.global_mean = None
    
    def fit(self, games_df):
        """Calculate per-team offensive and defensive averages."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')
        
        # Track goals for and against for each team
        goals_for = {}
        goals_against = {}
        games_played = {}
        
        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]
            
            # Home team stats
            goals_for[home_team] = goals_for.get(home_team, 0) + home_goals
            goals_against[home_team] = goals_against.get(home_team, 0) + away_goals
            games_played[home_team] = games_played.get(home_team, 0) + 1
            
            # Away team stats
            goals_for[away_team] = goals_for.get(away_team, 0) + away_goals
            goals_against[away_team] = goals_against.get(away_team, 0) + home_goals
            games_played[away_team] = games_played.get(away_team, 0) + 1
        
        # Calculate averages
        for team in games_played:
            self.team_offense[team] = goals_for[team] / games_played[team]
            self.team_defense[team] = goals_against[team] / games_played[team]
        
        # Global fallback for unseen teams
        self.global_mean = games_df[home_goals_col].mean()
        self.n_teams = len(games_played)
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def predict_goals(self, game):
        """Predict based on team offensive/defensive averages."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        # Get offensive and defensive ratings (with fallback)
        home_offense = self.team_offense.get(home_team, self.global_mean)
        home_defense = self.team_defense.get(home_team, self.global_mean)
        away_offense = self.team_offense.get(away_team, self.global_mean)
        away_defense = self.team_defense.get(away_team, self.global_mean)
        
        # Prediction: average of attacker strength and defender weakness
        home_goals = (home_offense + away_defense) / 2
        away_goals = (away_offense + home_defense) / 2
        
        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'TeamMeanBaseline',
            'n_teams': self.n_teams,
            'n_games_trained': self.n_games,
            'global_mean': round(self.global_mean, 3),
            'top_offense': sorted(self.team_offense.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_defense': sorted(self.team_defense.items(), key=lambda x: x[1])[:5]  # Lower is better
        }


class HomeAwayBaseline(BaselineModel):
    """
    Home/Away adjusted baseline: accounts for home ice advantage.
    
    Tracks separate averages for:
        - Goals scored at home vs away
        - Goals allowed at home vs away
    
    This captures the well-known home advantage effect in hockey.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.home_offense = {}  # Goals scored at home
        self.away_offense = {}  # Goals scored away
        self.home_defense = {}  # Goals allowed at home
        self.away_defense = {}  # Goals allowed away
        self.global_home_mean = None
        self.global_away_mean = None
    
    def fit(self, games_df):
        """Calculate home/away specific averages per team."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')
        
        # Track by team and location
        home_goals_for = {}
        home_goals_against = {}
        home_games = {}
        away_goals_for = {}
        away_goals_against = {}
        away_games = {}
        
        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]
            
            # Home team at home
            home_goals_for[home_team] = home_goals_for.get(home_team, 0) + home_goals
            home_goals_against[home_team] = home_goals_against.get(home_team, 0) + away_goals
            home_games[home_team] = home_games.get(home_team, 0) + 1
            
            # Away team on the road
            away_goals_for[away_team] = away_goals_for.get(away_team, 0) + away_goals
            away_goals_against[away_team] = away_goals_against.get(away_team, 0) + home_goals
            away_games[away_team] = away_games.get(away_team, 0) + 1
        
        # Calculate averages
        for team in set(list(home_games.keys()) + list(away_games.keys())):
            if team in home_games and home_games[team] > 0:
                self.home_offense[team] = home_goals_for[team] / home_games[team]
                self.home_defense[team] = home_goals_against[team] / home_games[team]
            
            if team in away_games and away_games[team] > 0:
                self.away_offense[team] = away_goals_for[team] / away_games[team]
                self.away_defense[team] = away_goals_against[team] / away_games[team]
        
        # Global fallbacks
        self.global_home_mean = games_df[home_goals_col].mean()
        self.global_away_mean = games_df[away_goals_col].mean()
        self.home_advantage = self.global_home_mean - self.global_away_mean
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def predict_goals(self, game):
        """Predict using home/away specific averages."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        # Home team's offense at home + away team's defense on road
        home_off = self.home_offense.get(home_team, self.global_home_mean)
        away_def = self.away_defense.get(away_team, self.global_home_mean)
        
        # Away team's offense on road + home team's defense at home
        away_off = self.away_offense.get(away_team, self.global_away_mean)
        home_def = self.home_defense.get(home_team, self.global_away_mean)
        
        home_goals = (home_off + away_def) / 2
        away_goals = (away_off + home_def) / 2
        
        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'HomeAwayBaseline',
            'n_games_trained': self.n_games,
            'global_home_mean': round(self.global_home_mean, 3),
            'global_away_mean': round(self.global_away_mean, 3),
            'home_advantage': round(self.home_advantage, 3)
        }


class MovingAverageBaseline(BaselineModel):
    """
    Recent form baseline: uses only last N games for predictions.
    
    Parameters
    ----------
    params : dict
        window : int
            Number of recent games to consider (default: 5)
    
    This captures team momentum and recent form.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.window = self.params.get('window', 5)
        self.team_history = {}  # List of (goals_for, goals_against) per team
        self.global_mean = None
    
    def fit(self, games_df):
        """Build game history for each team."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')
        
        # Process games chronologically
        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]
            
            if home_team not in self.team_history:
                self.team_history[home_team] = []
            if away_team not in self.team_history:
                self.team_history[away_team] = []
            
            self.team_history[home_team].append((home_goals, away_goals))
            self.team_history[away_team].append((away_goals, home_goals))
        
        self.global_mean = games_df[home_goals_col].mean()
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def _get_recent_avg(self, team):
        """Get average from last N games."""
        if team not in self.team_history or len(self.team_history[team]) == 0:
            return self.global_mean, self.global_mean
        
        recent = self.team_history[team][-self.window:]
        avg_for = np.mean([g[0] for g in recent])
        avg_against = np.mean([g[1] for g in recent])
        
        return avg_for, avg_against
    
    def predict_goals(self, game):
        """Predict based on recent game window."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        home_off, home_def = self._get_recent_avg(home_team)
        away_off, away_def = self._get_recent_avg(away_team)
        
        home_goals = (home_off + away_def) / 2
        away_goals = (away_off + home_def) / 2
        
        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'MovingAverageBaseline',
            'window': self.window,
            'n_games_trained': self.n_games,
            'n_teams': len(self.team_history),
            'global_mean': round(self.global_mean, 3)
        }


class WeightedHistoryBaseline(BaselineModel):
    """
    Weighted baseline: recent games count more than older games.
    
    Parameters
    ----------
    params : dict
        decay : float
            Weight decay factor (default: 0.9)
            Each older game is weighted by decay^n
    
    Example with decay=0.9:
        Most recent game: weight 1.0
        Second most recent: weight 0.9
        Third: weight 0.81
        etc.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.decay = self.params.get('decay', 0.9)
        self.team_history = {}
        self.global_mean = None
    
    def fit(self, games_df):
        """Build weighted game history for each team."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')
        
        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]
            
            if home_team not in self.team_history:
                self.team_history[home_team] = []
            if away_team not in self.team_history:
                self.team_history[away_team] = []
            
            self.team_history[home_team].append((home_goals, away_goals))
            self.team_history[away_team].append((away_goals, home_goals))
        
        self.global_mean = games_df[home_goals_col].mean()
        self.n_games = len(games_df)
        self.is_fitted = True
        
        return self
    
    def _get_weighted_avg(self, team):
        """Get exponentially weighted average."""
        if team not in self.team_history or len(self.team_history[team]) == 0:
            return self.global_mean, self.global_mean
        
        history = self.team_history[team]
        n = len(history)
        
        weighted_for = 0
        weighted_against = 0
        total_weight = 0
        
        for i, (gf, ga) in enumerate(history):
            weight = self.decay ** (n - 1 - i)  # Most recent has highest weight
            weighted_for += gf * weight
            weighted_against += ga * weight
            total_weight += weight
        
        return weighted_for / total_weight, weighted_against / total_weight
    
    def predict_goals(self, game):
        """Predict using weighted historical averages."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        home_off, home_def = self._get_weighted_avg(home_team)
        away_off, away_def = self._get_weighted_avg(away_team)
        
        home_goals = (home_off + away_def) / 2
        away_goals = (away_off + home_def) / 2
        
        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'WeightedHistoryBaseline',
            'decay': self.decay,
            'n_games_trained': self.n_games,
            'n_teams': len(self.team_history),
            'global_mean': round(self.global_mean, 3)
        }


class PoissonBaseline(BaselineModel):
    """
    Poisson regression baseline: models goals as Poisson-distributed.
    
    Assumes goals follow a Poisson distribution and estimates:
        - Attack strength for each team
        - Defense strength for each team
        - Home advantage factor
    
    This is a statistically principled approach used in academic research.
    
    Parameters
    ----------
    params : dict
        home_advantage : float
            Fixed home advantage multiplier (default: fitted from data)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.attack_strength = {}
        self.defense_strength = {}
        self.league_avg = None
        self.home_factor = None
    
    def fit(self, games_df):
        """Estimate Poisson parameters from data."""
        home_team_col = get_column(games_df, 'home_team')
        away_team_col = get_column(games_df, 'away_team')
        home_goals_col = get_column(games_df, 'home_goals')
        away_goals_col = get_column(games_df, 'away_goals')
        
        # Calculate league averages
        self.league_avg = games_df[home_goals_col].mean()
        self.home_factor = games_df[home_goals_col].mean() / games_df[away_goals_col].mean()
        
        # Track goals for and against
        goals_for = {}
        goals_against = {}
        games_played = {}
        
        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]
            home_goals = game[home_goals_col]
            away_goals = game[away_goals_col]
            
            # Home team
            goals_for[home_team] = goals_for.get(home_team, 0) + home_goals
            goals_against[home_team] = goals_against.get(home_team, 0) + away_goals
            games_played[home_team] = games_played.get(home_team, 0) + 1
            
            # Away team
            goals_for[away_team] = goals_for.get(away_team, 0) + away_goals
            goals_against[away_team] = goals_against.get(away_team, 0) + home_goals
            games_played[away_team] = games_played.get(away_team, 0) + 1
        
        # Calculate strength parameters
        for team in games_played:
            avg_for = goals_for[team] / games_played[team]
            avg_against = goals_against[team] / games_played[team]
            
            self.attack_strength[team] = avg_for / self.league_avg
            self.defense_strength[team] = avg_against / self.league_avg
        
        self.n_games = len(games_df)
        self.n_teams = len(games_played)
        self.is_fitted = True
        
        return self
    
    def predict_goals(self, game):
        """Predict using Poisson model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        # Get strength parameters (default to 1.0 for unknown teams)
        home_attack = self.attack_strength.get(home_team, 1.0)
        home_defense = self.defense_strength.get(home_team, 1.0)
        away_attack = self.attack_strength.get(away_team, 1.0)
        away_defense = self.defense_strength.get(away_team, 1.0)
        
        # Expected goals = league_avg * attack_strength * opponent_defense * home_factor
        home_goals = self.league_avg * home_attack * away_defense * self.home_factor
        away_goals = self.league_avg * away_attack * home_defense / self.home_factor
        
        return home_goals, away_goals
    
    def get_summary(self):
        """Get model summary."""
        return {
            'model': 'PoissonBaseline',
            'league_avg': round(self.league_avg, 3),
            'home_factor': round(self.home_factor, 3),
            'n_teams': self.n_teams,
            'n_games_trained': self.n_games,
            'top_attack': sorted(self.attack_strength.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_defense': sorted(self.defense_strength.items(), key=lambda x: x[1])[:5]
        }


def compare_baselines(games_df, test_df=None, models=None):
    """
    Compare multiple baseline models on the same data.
    
    Parameters
    ----------
    games_df : DataFrame
        Training data
    test_df : DataFrame, optional
        Test data (if None, uses last 20% of games_df)
    models : list, optional
        List of model instances (if None, uses all baselines)
    
    Returns
    -------
    DataFrame
        Comparison results with metrics for each model
    """
    if test_df is None:
        split_idx = int(len(games_df) * 0.8)
        train_df = games_df.iloc[:split_idx]
        test_df = games_df.iloc[split_idx:]
    else:
        train_df = games_df
    
    if models is None:
        models = [
            GlobalMeanBaseline(),
            TeamMeanBaseline(),
            HomeAwayBaseline(),
            MovingAverageBaseline({'window': 5}),
            MovingAverageBaseline({'window': 10}),
            WeightedHistoryBaseline({'decay': 0.9}),
            WeightedHistoryBaseline({'decay': 0.95}),
            PoissonBaseline()
        ]
    
    results = []
    for model in models:
        model.fit(train_df)
        metrics = model.evaluate(test_df)
        summary = model.get_summary()
        
        results.append({
            'model': summary['model'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'combined_rmse': metrics['combined_rmse'],
            **{k: v for k, v in summary.items() if k != 'model'}
        })
    
    return pd.DataFrame(results).sort_values('rmse')


# Convenience aliases
Baseline = TeamMeanBaseline  # Default baseline
