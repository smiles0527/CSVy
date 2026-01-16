"""
ELO Model - Reusable Python Module

This is the same EloModel class from the notebook, 
extracted as a .py file for easy importing.

Usage:
    from utils.elo_model import EloModel
    
    model = EloModel(params)
    model.fit(games_df)
    metrics = model.evaluate(test_df)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Column name mappings - checks these alternatives in order
COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_goals': ['home_goals', 'home_score', 'h_goals', 'goals_home', 'home_pts'],
    'away_goals': ['away_goals', 'away_score', 'a_goals', 'goals_away', 'away_pts', 'visitor_goals'],
    'home_rest': ['home_rest', 'rest_time', 'home_days_rest', 'h_rest', 'home_rest_days', 'rest_home'],
    'away_rest': ['away_rest', 'away_days_rest', 'a_rest', 'away_rest_days', 'rest_away'],
    'travel_distance': ['travel_distance', 'away_travel_dist', 'travel_dist', 'distance', 'travel_miles'],
    'travel_time': ['travel_time', 'travel_hours', 'travel_duration'],
    'home_injuries': ['home_injuries', 'injuries', 'h_injuries', 'home_injury_count'],
    'away_injuries': ['away_injuries', 'a_injuries', 'away_injury_count'],
    'division': ['division', 'div', 'tier', 'league_division'],
    'game_date': ['game_date', 'date', 'Date', 'game_datetime', 'datetime', 'game_time'],
    'home_outcome': ['home_outcome', 'outcome', 'result', 'home_result'],
    'home_win': ['home_win', 'h_win', 'home_victory'],
}


def get_value(game, field, default=None):
    """
    Get a value from a game record, checking multiple possible column names.
    
    Parameters
    ----------
    game : dict or Series
        Game record
    field : str
        Logical field name (e.g., 'home_rest')
    default : any
        Value to return if no matching column found
    
    Returns
    -------
    Value from the first matching column, or default
    """
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in game:
            val = game[alias]
            # Handle NaN values
            if pd.isna(val):
                return default
            return val
    return default


class EloModel:
    def __init__(self, params):
        """
        Initialize ELO model with hyperparameters.
        
        params: dict with keys:
            - k_factor: rating change rate (20-40)
            - home_advantage: home ice boost (50-150)
            - initial_rating: starting rating (1500)
            - mov_multiplier: margin of victory weight (0-1.5)
            - mov_method: 'linear' or 'logarithmic'
            - season_carryover: year-to-year retention (0.67-0.85)
            - ot_win_multiplier: OT win value (0.75-1.0)
            - rest_advantage_per_day: rating boost per rest day (0-10)
            - b2b_penalty: back-to-back penalty (0-50)
        
        Supported column names (checks alternatives automatically):
            - home_team: home_team, home, team_home, h_team
            - away_team: away_team, away, visitor, visiting_team
            - home_goals: home_goals, home_score, h_goals
            - away_goals: away_goals, away_score, a_goals, visitor_goals
            - home_rest: home_rest, rest_time, home_days_rest
            - away_rest: away_rest, away_days_rest
            - travel_distance: travel_distance, travel_dist, distance
            - injuries: home_injuries, away_injuries, injuries
            - division: division, div, tier
        """
        self.params = params
        self.ratings = {}
        self.rating_history = []
    
    def initialize_ratings(self, teams, divisions=None):
        """Initialize team ratings based on division tier."""
        initial = self.params.get('initial_rating', 1500)
        division_ratings = {
            'D1': initial + 100,
            'D2': initial,
            'D3': initial - 100
        }
        
        for i, team in enumerate(teams):
            if divisions is not None and i < len(divisions):
                div = divisions.iloc[i] if hasattr(divisions, 'iloc') else divisions[i]
                self.ratings[team] = division_ratings.get(div, initial)
            else:
                self.ratings[team] = initial
    
    def calculate_expected_score(self, team_elo, opponent_elo):
        """Calculate expected win probability."""
        return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))
    
    def calculate_mov_multiplier(self, goal_diff):
        """Calculate margin of victory multiplier."""
        mov = self.params.get('mov_multiplier', 0)
        if mov == 0:
            return 1.0
        
        if self.params.get('mov_method', 'logarithmic') == 'linear':
            return 1 + (abs(goal_diff) * mov)
        return 1 + (np.log(abs(goal_diff) + 1) * mov)
    
    def get_actual_score(self, outcome):
        """Convert game outcome to actual score (0-1)."""
        if outcome in ['RW', 'W', 1]:  # Regulation win
            return 1.0
        elif outcome == 'OTW':  # Overtime win
            return self.params.get('ot_win_multiplier', 0.75)
        elif outcome == 'OTL':  # Overtime loss
            return 1 - self.params.get('ot_win_multiplier', 0.75)
        return 0.0  # Regulation loss
    
    def adjust_for_context(self, team_elo, is_home, rest_time, travel_dist, injuries):
        """Apply contextual adjustments to ELO rating."""
        adjusted = team_elo
        
        # Home advantage
        if is_home:
            adjusted += self.params.get('home_advantage', 0)
        
        # Back-to-back penalty
        if rest_time <= 1:
            adjusted -= self.params.get('b2b_penalty', 0)
        
        # Travel fatigue (15 points per 1000 miles)
        if not is_home and travel_dist > 0:
            adjusted -= (travel_dist / 1000) * 15
        
        # Injury penalty (25 points per key injury)
        adjusted -= injuries * 25
        
        return adjusted
    
    def update_ratings(self, game):
        """Update team ratings after a game."""
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        # Get base ratings (default to 1500 for new teams)
        home_elo = self.ratings.get(home_team, 1500)
        away_elo = self.ratings.get(away_team, 1500)
        
        # Get context values with flexible column name lookup
        home_rest = get_value(game, 'home_rest', 2)
        away_rest = get_value(game, 'away_rest', 2)
        travel_dist = get_value(game, 'travel_distance', 0)
        travel_time = get_value(game, 'travel_time', 0)
        home_injuries = get_value(game, 'home_injuries', 0)
        away_injuries = get_value(game, 'away_injuries', 0)
        
        # Use travel_time as fallback (convert hours to equivalent miles)
        if travel_dist == 0 and travel_time > 0:
            travel_dist = travel_time * 60  # Rough conversion: 60 mph average
        
        # Apply contextual adjustments
        home_adj = self.adjust_for_context(home_elo, True, home_rest, 0, home_injuries)
        away_adj = self.adjust_for_context(away_elo, False, away_rest, travel_dist, away_injuries)
        
        # Rest differential advantage
        rest_diff = home_rest - away_rest
        home_adj += rest_diff * self.params.get('rest_advantage_per_day', 0)
        
        # Calculate expected scores
        home_expected = self.calculate_expected_score(home_adj, away_adj)
        
        # Handle different outcome column names
        home_outcome = get_value(game, 'home_outcome')
        home_win = get_value(game, 'home_win')
        home_goals = get_value(game, 'home_goals', 0)
        away_goals = get_value(game, 'away_goals', 0)
        
        if home_outcome is not None:
            home_actual = self.get_actual_score(home_outcome)
        elif home_win is not None:
            home_actual = 1.0 if home_win else 0.0
        else:
            home_actual = 1.0 if home_goals > away_goals else 0.0
        
        # Calculate margin of victory multiplier
        goal_diff = home_goals - away_goals
        mov_mult = self.calculate_mov_multiplier(goal_diff)
        
        # Update ratings
        k = self.params.get('k_factor', 32) * mov_mult
        self.ratings[home_team] = home_elo + k * (home_actual - home_expected)
        self.ratings[away_team] = away_elo + k * ((1 - home_actual) - (1 - home_expected))
        
        # Store history
        self.rating_history.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_rating': self.ratings[home_team],
            'away_rating': self.ratings[away_team]
        })
    
    def predict_goals(self, game):
        """Predict goals for both teams."""
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        # Get adjusted ratings
        home_elo = self.ratings.get(home_team, 1500)
        away_elo = self.ratings.get(away_team, 1500)
        
        # Get context values with flexible column lookup
        home_rest = get_value(game, 'home_rest', 2)
        away_rest = get_value(game, 'away_rest', 2)
        travel_dist = get_value(game, 'travel_distance', 0)
        travel_time = get_value(game, 'travel_time', 0)
        home_injuries = get_value(game, 'home_injuries', 0)
        away_injuries = get_value(game, 'away_injuries', 0)
        
        # Use travel_time as fallback
        if travel_dist == 0 and travel_time > 0:
            travel_dist = travel_time * 60
        
        home_adj = self.adjust_for_context(home_elo, True, home_rest, 0, home_injuries)
        away_adj = self.adjust_for_context(away_elo, False, away_rest, travel_dist, away_injuries)
        
        # Rest differential
        rest_diff = home_rest - away_rest
        home_adj += rest_diff * self.params.get('rest_advantage_per_day', 0)
        
        # Calculate win probability
        home_win_prob = self.calculate_expected_score(home_adj, away_adj)
        
        # Convert to expected goal differential
        # Scale: 50% win prob = 0 goal diff, 100% = +6 goals, 0% = -6 goals
        expected_diff = (home_win_prob - 0.5) * 12
        
        # League average is ~3 goals per team
        home_goals = 3.0 + (expected_diff / 2)
        away_goals = 3.0 - (expected_diff / 2)
        
        return home_goals, away_goals
    
    def predict_winner(self, game):
        """Predict winner and win probability."""
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        
        home_elo = self.ratings.get(home_team, 1500)
        away_elo = self.ratings.get(away_team, 1500)
        
        # Get context values with flexible column lookup
        home_rest = get_value(game, 'home_rest', 2)
        away_rest = get_value(game, 'away_rest', 2)
        travel_dist = get_value(game, 'travel_distance', 0)
        travel_time = get_value(game, 'travel_time', 0)
        home_injuries = get_value(game, 'home_injuries', 0)
        away_injuries = get_value(game, 'away_injuries', 0)
        
        if travel_dist == 0 and travel_time > 0:
            travel_dist = travel_time * 60
        
        home_adj = self.adjust_for_context(home_elo, True, home_rest, 0, home_injuries)
        away_adj = self.adjust_for_context(away_elo, False, away_rest, travel_dist, away_injuries)
        
        rest_diff = home_rest - away_rest
        home_adj += rest_diff * self.params.get('rest_advantage_per_day', 0)
        
        home_win_prob = self.calculate_expected_score(home_adj, away_adj)
        
        if home_win_prob > 0.5:
            return home_team, home_win_prob
        else:
            return away_team, 1 - home_win_prob
    
    def _get_team_column(self, df, field):
        """Find the correct column name for a field in a DataFrame."""
        aliases = COLUMN_ALIASES.get(field, [field])
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None
    
    def fit(self, games_df):
        """Train the model on historical games."""
        # Find correct column names
        home_col = self._get_team_column(games_df, 'home_team') or 'home_team'
        away_col = self._get_team_column(games_df, 'away_team') or 'away_team'
        div_col = self._get_team_column(games_df, 'division')
        
        # Initialize ratings
        teams = pd.concat([games_df[home_col], games_df[away_col]]).unique()
        
        if div_col:
            divisions = games_df.groupby(home_col)[div_col].first()
            self.initialize_ratings(teams, divisions)
        else:
            self.initialize_ratings(teams)
        
        # Update ratings game-by-game
        for _, game in games_df.iterrows():
            self.update_ratings(game)
    
    def evaluate(self, games_df):
        """Evaluate model on test set."""
        predictions = []
        actuals = []
        
        for _, game in games_df.iterrows():
            home_pred, _ = self.predict_goals(game)
            predictions.append(home_pred)
            actuals.append(get_value(game, 'home_goals', 0))
        
        rmse = mean_squared_error(actuals, predictions, squared=False)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions) if len(set(actuals)) > 1 else 0.0
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def get_rankings(self, top_n=None):
        """Get team rankings sorted by ELO rating."""
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            return sorted_ratings[:top_n]
        return sorted_ratings
    
    def get_rating_history_df(self):
        """Get rating history as a DataFrame."""
        return pd.DataFrame(self.rating_history)