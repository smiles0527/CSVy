"""
ELO Model - Reusable Python Module

Same EloModel from the notebook, extracted for easy importing. Uses the classic
Elo rating system (like chess) adapted for hockey: each team has a rating, and
games update ratings based on expected vs actual outcome. We also fold in home
advantage, rest, travel, injuries—stuff that actually affects who wins.

Usage:
    from utils.elo_model import EloModel
    model = EloModel(params)
    model.fit(games_df)
    metrics = model.evaluate(test_df)
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse_score(y_true, y_pred):
    """Root mean squared error—how far off our predictions are on average."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Different datasets name columns differently; we try these in order until one sticks
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
    Grab a value from a game row. Tries each possible column name (home_team, home, etc.)
    and returns the first match. Falls back to default if missing or NaN.
    """
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in game:
            val = game[alias]
            if pd.isna(val):
                return default
            return val
    return default


class EloModel:
    def __init__(self, params):
        """
        Set up the model with your tuning knobs. params is a dict—here's what matters:
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
        self.ratings = {}           # team name -> current Elo. Filled in during fit.
        self.rating_history = []    # log of rating changes after each game
    
    def initialize_ratings(self, teams, divisions=None):
        """
        Give every team a starting rating. If we know divisions (D1/D2/D3), we
        bump D1 up and D3 down a bit—they're not all equal coming in.
        """
        initial = self.params.get('initial_rating', 1500)
        division_ratings = {
            'D1': initial + 100,   # Top tier gets a head start
            'D2': initial,
            'D3': initial - 100    # Lower tier starts behind
        }

        for team in teams:
            if divisions is not None and team in divisions:
                div = divisions[team] if not hasattr(divisions, 'get') else divisions.get(team)   # Handle both dict and Series
                self.ratings[team] = division_ratings.get(div, initial)
            else:
                self.ratings[team] = initial

    def calculate_expected_score(self, team_elo, opponent_elo):
        """Classic Elo: what's the chance this team wins? 0-1 scale."""
        return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))
    
    def calculate_mov_multiplier(self, goal_diff):
        """
        Blowouts should move ratings more than close games. Linear = proportional
        to goal diff. Log dampens it so a 6-goal rout doesn't swing ratings crazy.
        """
        mov = self.params.get('mov_multiplier', 0)
        if mov == 0:
            return 1.0

        if self.params.get('mov_method', 'logarithmic') == 'linear':
            return 1 + (abs(goal_diff) * mov)
        return 1 + (np.log(abs(goal_diff) + 1) * mov)
    
    def get_actual_score(self, outcome):
        """
        Turn outcome into a 0-1 "score" for Elo. Regulation win = 1, regulation loss = 0.
        OT win/loss is often worth less—you didn't dominate, so we scale it down (e.g. 0.75).
        """
        if outcome in ['RW', 'W', 1]:
            return 1.0
        elif outcome == 'OTW':
            return self.params.get('ot_win_multiplier', 0.75)
        elif outcome == 'OTL':
            return 1 - self.params.get('ot_win_multiplier', 0.75)
        return 0.0
    
    def adjust_for_context(self, team_elo, is_home, rest_time, travel_dist, injuries):
        """
        Raw Elo isn't enough—home ice, rest, travel, injuries all matter. We bump
        or drop the effective rating before computing expected score.
        """
        adjusted = team_elo

        if is_home:
            adjusted += self.params.get('home_advantage', 0)

        # Back-to-back = tired. rest_time <= 1 means they played yesterday or today
        if rest_time <= 1:
            adjusted -= self.params.get('b2b_penalty', 0)

        # Long flight = jet lag. Only hits away teams
        if not is_home and travel_dist > 0:
            adjusted -= (travel_dist / 1000) * 15

        adjusted -= injuries * 25   # Key players out = weaker

        return adjusted
    
    def update_ratings(self, game):
        """
        Core Elo update: compare expected vs actual, move ratings. New teams get 1500.
        """
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')

        home_elo = self.ratings.get(home_team, 1500)
        away_elo = self.ratings.get(away_team, 1500)
        home_rest = get_value(game, 'home_rest', 2)
        away_rest = get_value(game, 'away_rest', 2)
        travel_dist = get_value(game, 'travel_distance', 0)
        travel_time = get_value(game, 'travel_time', 0)
        home_injuries = get_value(game, 'home_injuries', 0)
        away_injuries = get_value(game, 'away_injuries', 0)

        if travel_dist == 0 and travel_time > 0:
            travel_dist = travel_time * 60   # Assume ~60 mph if we only have hours
        home_adj = self.adjust_for_context(home_elo, True, home_rest, 0, home_injuries)
        away_adj = self.adjust_for_context(away_elo, False, away_rest, travel_dist, away_injuries)

        rest_diff = home_rest - away_rest   # Home rested more? Give them a boost
        home_adj += rest_diff * self.params.get('rest_advantage_per_day', 0)
        home_expected = self.calculate_expected_score(home_adj, away_adj)
        home_outcome = get_value(game, 'home_outcome')   # Sometimes it's RW, OTW, etc.
        home_win = get_value(game, 'home_win')           # Or a simple True/False
        home_goals = get_value(game, 'home_goals', 0)
        away_goals = get_value(game, 'away_goals', 0)

        if home_outcome is not None:
            home_actual = self.get_actual_score(home_outcome)
        elif home_win is not None:
            home_actual = 1.0 if home_win else 0.0
        else:
            home_actual = 1.0 if home_goals > away_goals else 0.0   # Infer from goals
        goal_diff = home_goals - away_goals
        mov_mult = self.calculate_mov_multiplier(goal_diff)

        k = self.params.get('k_factor', 32) * mov_mult
        # Elo formula: new = old + k * (actual - expected). Upset = big move.
        self.ratings[home_team] = home_elo + k * (home_actual - home_expected)
        self.ratings[away_team] = away_elo + k * ((1 - home_actual) - (1 - home_expected))
        self.rating_history.append({   # Save a snapshot so we can inspect later
            'home_team': home_team,
            'away_team': away_team,
            'home_rating': self.ratings[home_team],
            'away_rating': self.ratings[away_team]
        })
    
    def predict_goals(self, game):
        """
        Turn Elo win probability into expected goals. We don't predict "3-2" exactly—
        we give expected values like 3.1 and 2.7. Win prob 50% → 3–3. Higher → home
        scores more.
        """
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

        # Map win prob to goal differential: 50% → 0 diff, 100% → +6, 0% → -6
        expected_diff = (home_win_prob - 0.5) * 12

        # League average ~3 goals/team. Split the diff between home and away.
        home_goals = 3.0 + (expected_diff / 2)
        away_goals = 3.0 - (expected_diff / 2)
        
        return home_goals, away_goals
    
    def predict_winner(self, game):
        """Who wins and how confident are we? Same logic as predict_goals, just return team + prob."""
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
        """Same idea as get_value but for a whole table—which column has this data?"""
        aliases = COLUMN_ALIASES.get(field, [field])
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None
    
    def fit(self, games_df):
        """
        Process games in order. Start everyone at initial rating (or by division),
        then update ratings game by game. Chronological order matters—we're
        simulating how ratings would have evolved.
        """
        home_col = self._get_team_column(games_df, 'home_team') or 'home_team'
        away_col = self._get_team_column(games_df, 'away_team') or 'away_team'
        div_col = self._get_team_column(games_df, 'division')

        teams = pd.concat([games_df[home_col], games_df[away_col]]).unique()

        if div_col:
            divisions = games_df.groupby(home_col)[div_col].first()   # One division per team
            self.initialize_ratings(teams, divisions)
        else:
            self.initialize_ratings(teams)

        for _, game in games_df.iterrows():
            self.update_ratings(game)

    def evaluate(self, games_df):
        """
        Predict every game, compare to actuals, spit back RMSE, MAE, R², win accuracy.
        Returns dict with:
            home_rmse, away_rmse, combined_rmse, home_mae, away_mae, home_r2, away_r2, win_accuracy, n_games
        """
        home_preds, away_preds = [], []
        home_actuals, away_actuals = [], []
        correct_wins = 0
        
        for _, game in games_df.iterrows():
            h_pred, a_pred = self.predict_goals(game)
            h_actual = get_value(game, 'home_goals', 0)
            a_actual = get_value(game, 'away_goals', 0)
            
            home_preds.append(h_pred)
            away_preds.append(a_pred)
            home_actuals.append(h_actual)
            away_actuals.append(a_actual)

            pred_home_win = h_pred > a_pred
            actual_home_win = h_actual > a_actual
            if pred_home_win == actual_home_win:
                correct_wins += 1
        
        n = len(home_actuals)
        all_actuals = home_actuals + away_actuals
        all_preds = home_preds + away_preds
        
        return {
            'home_rmse': rmse_score(home_actuals, home_preds),
            'away_rmse': rmse_score(away_actuals, away_preds),
            'combined_rmse': rmse_score(all_actuals, all_preds),
            'home_mae': float(mean_absolute_error(home_actuals, home_preds)),
            'away_mae': float(mean_absolute_error(away_actuals, away_preds)),
            'home_r2': float(r2_score(home_actuals, home_preds)) if len(set(home_actuals)) > 1 else 0.0,
            'away_r2': float(r2_score(away_actuals, away_preds)) if len(set(away_actuals)) > 1 else 0.0,
            'win_accuracy': correct_wins / n if n > 0 else 0.0,
            'n_games': n,
        }
    
    def get_rankings(self, top_n=None):
        """Sorted list of (team, rating). top_n limits to top N if you only want that."""
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            return sorted_ratings[:top_n]
        return sorted_ratings
    
    def get_rating_history_df(self):
        """Rating after each game—useful for plotting how teams moved over time."""
        return pd.DataFrame(self.rating_history)
    
    # ── Save / Load ──────────────────────────────────────────────

    def save_model(self, path):
        """
        Dump to disk. Creates both a .pkl (exact restore) and .json (human-readable
        so you can peek at ratings without loading Python).
        Saves two files:
            {path}.pkl  – full pickle (ratings + history + params)
            {path}.json – human-readable params + final ratings
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        state = {
            'params': self.params,
            'ratings': self.ratings,
            'rating_history': self.rating_history,
        }

        pkl_path = path if path.endswith('.pkl') else path + '.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(state, f)
        json_path = path.replace('.pkl', '') + '.json' if path.endswith('.pkl') else path + '.json'
        json_state = {   # Floats for JSON (numpy types don't serialize nicely)
            'params': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in self.params.items()},
            'ratings': {k: float(v) for k, v in self.ratings.items()},
            'n_games_trained': len(self.rating_history),
        }
        with open(json_path, 'w') as f:
            json.dump(json_state, f, indent=2)
        
        print(f"Elo model saved to {pkl_path} ({len(self.ratings)} teams, {len(self.rating_history)} games)")
        return pkl_path
    
    @classmethod
    def load_model(cls, path):
        """Load from .pkl. Get back a full model—ratings, history, params—ready to predict."""
        pkl_path = path if path.endswith('.pkl') else path + '.pkl'
        with open(pkl_path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(state['params'])
        model.ratings = state['ratings']
        model.rating_history = state['rating_history']
        print(f"Elo model loaded: {len(model.ratings)} teams, {len(model.rating_history)} games")
        return model