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

import json
import pickle
from pathlib import Path

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


def poisson_win_confidence(home_lam, away_lam):
    """
    Compute win probability from expected goals (Poisson model).
    Useful for averaged predictions where we have (avg_home, avg_away) but no model.

    Returns
    -------
    float
        Confidence (max of home_win_prob, away_win_prob) in [0.5, 1.0]
    """
    from scipy.stats import poisson
    lam_h = max(float(home_lam), 0.3)
    lam_a = max(float(away_lam), 0.3)
    max_goals = 12
    home_win_prob = 0.0
    draw_prob = 0.0
    for h in range(max_goals + 1):
        ph = poisson.pmf(h, lam_h)
        for a in range(max_goals + 1):
            pa = poisson.pmf(a, lam_a)
            if h > a:
                home_win_prob += ph * pa
            elif h == a:
                draw_prob += ph * pa
    home_win_prob += draw_prob / 2.0
    return max(home_win_prob, 1.0 - home_win_prob)


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

    def predict_winner(self, game):
        """
        Predict the winner and confidence using Poisson-based win probability.

        Models each team's goals as independent Poisson random variables,
        then sums P(home_goals > away_goals) over the outcome grid.
        This gives a proper probabilistic confidence instead of a simple
        goal ratio.

        Returns
        -------
        tuple
            (winner_team_name, win_probability)
        """
        from scipy.stats import poisson

        home_goals, away_goals = self.predict_goals(game)
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        # Clamp predicted goals to avoid degenerate Poisson (lambda must be > 0)
        lam_h = max(home_goals, 0.3)
        lam_a = max(away_goals, 0.3)

        # Compute P(home wins) by summing over Poisson outcome grid
        max_goals = 12
        home_win_prob = 0.0
        draw_prob = 0.0
        for h in range(max_goals + 1):
            ph = poisson.pmf(h, lam_h)
            for a in range(max_goals + 1):
                pa = poisson.pmf(a, lam_a)
                if h > a:
                    home_win_prob += ph * pa
                elif h == a:
                    draw_prob += ph * pa

        # Split draws evenly (hockey has OT/SO so no real draws)
        home_win_prob += draw_prob / 2.0

        if home_win_prob >= 0.5:
            return home_team, home_win_prob
        else:
            return away_team, 1.0 - home_win_prob
    
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
        home_rmse = np.sqrt(mean_squared_error(home_actuals, home_preds))
        home_mae = mean_absolute_error(home_actuals, home_preds)
        home_r2 = r2_score(home_actuals, home_preds) if len(set(home_actuals)) > 1 else 0.0
        
        # Calculate metrics for away goals
        away_rmse = np.sqrt(mean_squared_error(away_actuals, away_preds))
        away_mae = mean_absolute_error(away_actuals, away_preds)
        away_r2 = r2_score(away_actuals, away_preds) if len(set(away_actuals)) > 1 else 0.0
        
        # Combined metrics (both home and away)
        all_preds = home_preds + away_preds
        all_actuals = home_actuals + away_actuals
        combined_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
        combined_mae = mean_absolute_error(all_actuals, all_preds)
        combined_r2 = r2_score(all_actuals, all_preds) if len(set(all_actuals)) > 1 else 0.0
        
        # Win prediction accuracy
        correct_wins = sum(
            1 for hp, ap, ha, aa in zip(home_preds, away_preds, home_actuals, away_actuals)
            if (hp > ap) == (ha > aa)
        )
        win_accuracy = correct_wins / len(home_preds) if home_preds else 0.0
        
        return {
            'rmse': home_rmse,
            'mae': home_mae,
            'r2': home_r2,
            'away_rmse': away_rmse,
            'away_mae': away_mae,
            'away_r2': away_r2,
            'combined_rmse': combined_rmse,
            'combined_mae': combined_mae,
            'combined_r2': combined_r2,
            'win_accuracy': win_accuracy,
        }
    
    def get_summary(self):
        """Get model summary statistics."""
        raise NotImplementedError("Subclasses must implement get_summary()")
    
    def save_model(self, filepath):
        """
        Save model state to disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save file (.pkl)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'class': self.__class__.__name__,
            'params': self.params,
            'is_fitted': self.is_fitted,
        }
        # Save all instance attributes (team dicts, means, etc.)
        for key, value in self.__dict__.items():
            if key not in state:
                state[key] = value
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load model state from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to saved model (.pkl)
        
        Returns
        -------
        BaselineModel
            Restored model instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Look up the correct subclass
        class_name = state.pop('class', cls.__name__)
        class_map = {
            'GlobalMeanBaseline': GlobalMeanBaseline,
            'TeamMeanBaseline': TeamMeanBaseline,
            'HomeAwayBaseline': HomeAwayBaseline,
            'MovingAverageBaseline': MovingAverageBaseline,
            'WeightedHistoryBaseline': WeightedHistoryBaseline,
            'PoissonBaseline': PoissonBaseline,
            'DixonColesBaseline': DixonColesBaseline,
            'BayesianTeamBaseline': BayesianTeamBaseline,
            'EnsembleBaseline': EnsembleBaseline,
        }
        model_cls = class_map.get(class_name, cls)
        
        model = model_cls.__new__(model_cls)
        for key, value in state.items():
            setattr(model, key, value)
        
        return model


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
            PoissonBaseline(),
            DixonColesBaseline(),
            DixonColesBaseline({'decay': 0.98}),
            BayesianTeamBaseline(),
            BayesianTeamBaseline({'prior_weight': 10}),
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


# ── ADVANCED BASELINES ────────────────────────────────────────────

class DixonColesBaseline(BaselineModel):
    """
    Dixon-Coles Poisson model — the gold standard for match outcome
    prediction in soccer and hockey analytics.

    Jointly estimates per-team attack/defense strengths via iterative
    maximum-likelihood with optional time-decay weighting (recent games
    matter more).  Produces calibrated Poisson means for each side.

    Parameters
    ----------
    params : dict
        max_iter : int   – EM-style iterations (default 50)
        tol      : float – convergence tolerance (default 1e-6)
        decay    : float – per-game exponential decay weight, 0-1
                           (1.0 = no decay, 0.95 = recent-heavy)
        home_adv : float – initial home-advantage multiplier (fitted)
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.max_iter = self.params.get('max_iter', 50)
        self.tol = self.params.get('tol', 1e-6)
        self.decay = self.params.get('decay', 1.0)
        self.attack = {}
        self.defense = {}
        self.home_adv = self.params.get('home_adv', 1.15)
        self.league_avg = None

    def fit(self, games_df):
        """Iteratively estimate attack / defense strengths."""
        ht_col = get_column(games_df, 'home_team')
        at_col = get_column(games_df, 'away_team')
        hg_col = get_column(games_df, 'home_goals')
        ag_col = get_column(games_df, 'away_goals')

        hteams = games_df[ht_col].values
        ateams = games_df[at_col].values
        hgoals = games_df[hg_col].values.astype(float)
        agoals = games_df[ag_col].values.astype(float)
        n = len(games_df)

        # Time-decay weights (last game has weight 1, older ones decay)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])

        teams = sorted(set(hteams) | set(ateams))
        # Initialise
        atk = {t: 1.0 for t in teams}
        dfn = {t: 1.0 for t in teams}
        home_adv = self.home_adv
        self.league_avg = float(np.mean(np.concatenate([hgoals, agoals])))

        for iteration in range(self.max_iter):
            old_atk = dict(atk)
            old_dfn = dict(dfn)
            old_ha = home_adv

            # --- update attack strengths ---
            for t in teams:
                # Games where t is home
                mask_h = hteams == t
                # Games where t is away
                mask_a = ateams == t

                numerator = float(
                    np.sum(hgoals[mask_h] * weights[mask_h]) +
                    np.sum(agoals[mask_a] * weights[mask_a])
                )

                denom = 0.0
                if mask_h.any():
                    opp_def = np.array([dfn[a] for a in ateams[mask_h]])
                    denom += float(np.sum(home_adv * opp_def * weights[mask_h]))
                if mask_a.any():
                    opp_def = np.array([dfn[h] for h in hteams[mask_a]])
                    denom += float(np.sum(opp_def / 1.0 * weights[mask_a]))  # away

                atk[t] = (numerator / denom) if denom > 0 else 1.0

            # --- update defense strengths ---
            for t in teams:
                mask_h = hteams == t
                mask_a = ateams == t

                numerator = float(
                    np.sum(agoals[mask_h] * weights[mask_h]) +
                    np.sum(hgoals[mask_a] * weights[mask_a])
                )

                denom = 0.0
                if mask_h.any():
                    opp_atk = np.array([atk[a] for a in ateams[mask_h]])
                    denom += float(np.sum(opp_atk * weights[mask_h]))  # they're away
                if mask_a.any():
                    opp_atk = np.array([atk[h] for h in hteams[mask_a]])
                    denom += float(np.sum(home_adv * opp_atk * weights[mask_a]))

                dfn[t] = (numerator / denom) if denom > 0 else 1.0

            # --- update home advantage ---
            num_ha = float(np.sum(hgoals * weights))
            den_ha = 0.0
            for i in range(n):
                den_ha += atk[hteams[i]] * dfn[ateams[i]] * weights[i]
            home_adv = (num_ha / den_ha) if den_ha > 0 else 1.15

            # --- normalise attack strengths (mean=1) ---
            mean_atk = np.mean(list(atk.values()))
            if mean_atk > 0:
                for t in teams:
                    atk[t] /= mean_atk

            # convergence check
            delta = max(
                max(abs(atk[t] - old_atk[t]) for t in teams),
                max(abs(dfn[t] - old_dfn[t]) for t in teams),
                abs(home_adv - old_ha),
            )
            if delta < self.tol:
                break

        self.attack = atk
        self.defense = dfn
        self.home_adv = home_adv
        self.n_games = n
        self.n_teams = len(teams)
        self.is_fitted = True
        return self

    def predict_goals(self, game):
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        h_atk = self.attack.get(home_team, 1.0)
        h_def = self.defense.get(home_team, 1.0)
        a_atk = self.attack.get(away_team, 1.0)
        a_def = self.defense.get(away_team, 1.0)

        home_goals = self.league_avg * h_atk * a_def * self.home_adv
        away_goals = self.league_avg * a_atk * h_def
        return home_goals, away_goals

    def get_summary(self):
        top_atk = sorted(self.attack.items(), key=lambda x: x[1], reverse=True)[:5]
        top_def = sorted(self.defense.items(), key=lambda x: x[1])[:5]
        return {
            'model': 'DixonColesBaseline',
            'decay': self.decay,
            'home_adv': round(self.home_adv, 4),
            'league_avg': round(self.league_avg, 3),
            'n_teams': self.n_teams,
            'n_games_trained': self.n_games,
            'iterations': self.max_iter,
            'top_attack': top_atk,
            'top_defense': top_def,
        }


class BayesianTeamBaseline(BaselineModel):
    """
    Bayesian-regularised team baseline.

    Shrinks per-team attack/defense estimates toward the league average
    using a simple conjugate-prior approach.  Teams with fewer games are
    pulled more strongly toward the mean, preventing overfitting on
    small sample sizes.

    Parameters
    ----------
    params : dict
        prior_weight : float – equivalent number of pseudo-games at
                               league average (default 5)
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.prior_weight = self.params.get('prior_weight', 5)
        self.attack = {}
        self.defense = {}
        self.league_home_avg = None
        self.league_away_avg = None

    def fit(self, games_df):
        ht_col = get_column(games_df, 'home_team')
        at_col = get_column(games_df, 'away_team')
        hg_col = get_column(games_df, 'home_goals')
        ag_col = get_column(games_df, 'away_goals')

        self.league_home_avg = float(games_df[hg_col].mean())
        self.league_away_avg = float(games_df[ag_col].mean())
        self.league_avg = (self.league_home_avg + self.league_away_avg) / 2

        goals_for = {}
        goals_against = {}
        gp = {}

        for _, game in games_df.iterrows():
            ht = game[ht_col]; at = game[at_col]
            hg = game[hg_col]; ag = game[ag_col]

            for team, gf, ga in [(ht, hg, ag), (at, ag, hg)]:
                goals_for.setdefault(team, 0.0)
                goals_against.setdefault(team, 0.0)
                gp.setdefault(team, 0)
                goals_for[team] += gf
                goals_against[team] += ga
                gp[team] += 1

        pw = self.prior_weight
        for team in gp:
            n = gp[team]
            raw_atk = goals_for[team] / n
            raw_def = goals_against[team] / n
            # Bayesian shrinkage: weighted average of team rate and league rate
            self.attack[team] = (raw_atk * n + self.league_avg * pw) / (n + pw)
            self.defense[team] = (raw_def * n + self.league_avg * pw) / (n + pw)

        self.n_games = len(games_df)
        self.n_teams = len(gp)
        self.is_fitted = True
        return self

    def predict_goals(self, game):
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        h_atk = self.attack.get(home_team, self.league_avg)
        h_def = self.defense.get(home_team, self.league_avg)
        a_atk = self.attack.get(away_team, self.league_avg)
        a_def = self.defense.get(away_team, self.league_avg)

        # Home advantage baked into league averages
        home_factor = self.league_home_avg / self.league_avg
        away_factor = self.league_away_avg / self.league_avg

        home_goals = (h_atk * (a_def / self.league_avg)) * home_factor
        away_goals = (a_atk * (h_def / self.league_avg)) * away_factor

        return home_goals, away_goals

    def get_summary(self):
        return {
            'model': 'BayesianTeamBaseline',
            'prior_weight': self.prior_weight,
            'league_avg': round(self.league_avg, 3),
            'n_teams': self.n_teams,
            'n_games_trained': self.n_games,
            'top_attack': sorted(self.attack.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_defense': sorted(self.defense.items(), key=lambda x: x[1])[:5],
        }


class EnsembleBaseline(BaselineModel):
    """
    Blends multiple baseline models' predictions.

    Uses a simple weighted average.  Weights can be uniform, inverse-RMSE,
    or explicitly provided.

    Parameters
    ----------
    params : dict
        models  : list[BaselineModel]  – sub-models (must be fitted)
        weights : list[float] | None   – explicit blend weights
        method  : str – 'uniform' | 'inverse_rmse' (default 'inverse_rmse')
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.sub_models = self.params.get('models', [])
        self.weights = self.params.get('weights', None)
        self.method = self.params.get('method', 'inverse_rmse')

    def fit(self, games_df):
        """Fit sub-models and calibrate weights if needed."""
        # Split a small calibration set from training data
        cal_split = int(len(games_df) * 0.75)
        train_part = games_df.iloc[:cal_split]
        cal_part = games_df.iloc[cal_split:]

        for m in self.sub_models:
            m.fit(train_part)

        if self.weights is None:
            if self.method == 'inverse_rmse':
                rmses = []
                for m in self.sub_models:
                    metrics = m.evaluate(cal_part)
                    rmses.append(metrics['combined_rmse'])
                inv = [1.0 / r for r in rmses]
                total = sum(inv)
                self.weights = [w / total for w in inv]
            else:
                n = len(self.sub_models)
                self.weights = [1.0 / n] * n

        # Re-fit on full training data
        for m in self.sub_models:
            m.fit(games_df)

        self.n_games = len(games_df)
        self.is_fitted = True
        return self

    def predict_goals(self, game):
        h_preds, a_preds = [], []
        for m in self.sub_models:
            h, a = m.predict_goals(game)
            h_preds.append(h)
            a_preds.append(a)

        home_goals = sum(w * h for w, h in zip(self.weights, h_preds))
        away_goals = sum(w * a for w, a in zip(self.weights, a_preds))
        return home_goals, away_goals

    def get_summary(self):
        sub_names = []
        for m in self.sub_models:
            try:
                sub_names.append(m.get_summary()['model'])
            except Exception:
                sub_names.append(type(m).__name__)
        return {
            'model': 'EnsembleBaseline',
            'sub_models': sub_names,
            'weights': [round(w, 4) for w in (self.weights or [])],
            'method': self.method,
            'n_games_trained': self.n_games,
        }


# Convenience aliases
Baseline = TeamMeanBaseline  # Default baseline
