"""
Baseline Elo Offensive/Defensive Model - Iteration 2.0
======================================================
Separate offensive (O) and defensive (D) Elo ratings per team. Updates driven by
observed xG vs expected xG. Same interface as BaselineEloXGModel for comparison.

Formulas:
  multi = 10^((O_attacker - D_defender) / 400)
  expected_xG = league_avg_xG * multi * time
  O += k * (observed_xG - expected_xG) / ln(10)
  D -= k * (observed_xG - expected_xG) / ln(10)

Usage:
    from utils.baseline_elo_offdef import BaselineEloOffDefModel
    model = BaselineEloOffDefModel(params)
    model.fit(games_df)  # must have home_xg, away_xg
    metrics = model.evaluate(test_df)
    h, a = model.predict_goals(game)
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LN10 = np.log(10)


COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_xg': ['home_xg', 'home_xG', 'xg_home', 'h_xg'],
    'away_xg': ['away_xg', 'away_xG', 'xg_away', 'a_xg', 'visitor_xg'],
}


def get_value(game, field, default=None):
    """Grab value from game row with column alias support."""
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in game:
            val = game[alias]
            if pd.isna(val):
                return default
            return val
    return default


def rmse_score(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class BaselineEloOffDefModel:
    """Offensive/Defensive Elo model. Updates driven by observed vs expected xG."""

    def __init__(self, params=None):
        self.params = params or {}
        self.k = float(self.params.get('k_factor', 32))
        self.base_elo = float(self.params.get('initial_rating', 1200))
        self.elo_scale = float(self.params.get('elo_scale', 400))
        self.goal_diff_half_range = float(self.params.get('goal_diff_half_range', 6.0))
        self.time_factor = float(self.params.get('time_factor', 1.0))
        self.league_avg_xg: Optional[float] = self.params.get('league_avg_xg')
        self.O: Dict[str, float] = {}
        self.D: Dict[str, float] = {}

    def _add_team(self, team: str) -> None:
        if team not in self.O:
            self.O[team] = float(self.base_elo)
            self.D[team] = float(self.base_elo)

    def _get_team_column(self, df: pd.DataFrame, field: str) -> Optional[str]:
        aliases = COLUMN_ALIASES.get(field, [field])
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None

    def fit(self, games_df: pd.DataFrame) -> None:
        """Train on game-level DataFrame with home_xg, away_xg."""
        home_col = self._get_team_column(games_df, 'home_team') or 'home_team'
        away_col = self._get_team_column(games_df, 'away_team') or 'away_team'
        hxg_col = self._get_team_column(games_df, 'home_xg') or 'home_xg'
        axg_col = self._get_team_column(games_df, 'away_xg') or 'away_xg'

        if self.league_avg_xg is None:
            league_xg_per_team = (games_df[hxg_col].fillna(0) + games_df[axg_col].fillna(0)).sum() / (2 * max(1, len(games_df)))
            self.league_avg_xg = float(league_xg_per_team)

        for _, row in games_df.iterrows():
            ht = str(row[home_col]) if not pd.isna(row[home_col]) else 'Unknown_Home'
            at = str(row[away_col]) if not pd.isna(row[away_col]) else 'Unknown_Away'
            if ht == at:
                continue
            obs_home = 0 if pd.isna(row[hxg_col]) else float(row[hxg_col])
            obs_away = 0 if pd.isna(row[axg_col]) else float(row[axg_col])

            self._add_team(ht)
            self._add_team(at)

            O_home = self.O[ht]
            D_home = self.D[ht]
            O_away = self.O[at]
            D_away = self.D[at]

            multi_home = 10.0 ** ((O_home - D_away) / self.elo_scale)
            multi_away = 10.0 ** ((O_away - D_home) / self.elo_scale)
            exp_home = self.league_avg_xg * multi_home * self.time_factor
            exp_away = self.league_avg_xg * multi_away * self.time_factor

            delta_home = self.k * (obs_home - exp_home) / LN10
            delta_away = self.k * (obs_away - exp_away) / LN10

            self.O[ht] += delta_home
            self.D[at] -= delta_home
            self.O[at] += delta_away
            self.D[ht] -= delta_away

    def predict_goals(self, game) -> Tuple[float, float]:
        """Return (expected_home_xg, expected_away_xg)."""
        home_team = get_value(game, 'home_team') or 'Unknown_Home'
        away_team = get_value(game, 'away_team') or 'Unknown_Away'
        self._add_team(home_team)
        self._add_team(away_team)
        O_home = self.O.get(home_team, self.base_elo)
        D_home = self.D.get(home_team, self.base_elo)
        O_away = self.O.get(away_team, self.base_elo)
        D_away = self.D.get(away_team, self.base_elo)

        multi_home = 10.0 ** ((O_home - D_away) / self.elo_scale)
        multi_away = 10.0 ** ((O_away - D_home) / self.elo_scale)
        league = self.league_avg_xg or 3.0
        home_xg = max(0.0, league * multi_home * self.time_factor)
        away_xg = max(0.0, league * multi_away * self.time_factor)
        return home_xg, away_xg

    def predict_winner(self, game) -> Tuple[str, float]:
        """Return (winning_team, confidence). Confidence from xG-share win prob."""
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        h_xg, a_xg = self.predict_goals(game)
        total = h_xg + a_xg
        if total <= 0:
            home_win_prob = 0.5
        else:
            home_win_prob = h_xg / total
        if home_win_prob > 0.5:
            return home_team, home_win_prob
        return away_team, 1.0 - home_win_prob

    def evaluate(self, games_df: pd.DataFrame) -> Dict[str, float]:
        """Compute RMSE, MAE, R², win_accuracy on xG."""
        home_preds, away_preds = [], []
        home_actuals, away_actuals = [], []
        correct_wins = 0

        for _, game in games_df.iterrows():
            h_pred, a_pred = self.predict_goals(game)
            h_actual = get_value(game, 'home_xg', 0)
            a_actual = get_value(game, 'away_xg', 0)

            home_preds.append(h_pred)
            away_preds.append(a_pred)
            home_actuals.append(h_actual)
            away_actuals.append(a_actual)

            pred_home_win = h_pred > a_pred
            actual_home_win = h_actual > a_actual
            if pred_home_win == actual_home_win:
                correct_wins += 1

        n = len(home_actuals)
        if n == 0:
            return {
                'home_rmse': 0.0, 'away_rmse': 0.0, 'combined_rmse': 0.0,
                'home_mae': 0.0, 'away_mae': 0.0,
                'home_r2': 0.0, 'away_r2': 0.0,
                'win_accuracy': 0.0, 'n_games': 0,
            }
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

    def get_rankings(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Sorted list of (team, net_strength). net_strength = O - D (higher = stronger)."""
        ratings = {t: self.O.get(t, self.base_elo) - self.D.get(t, self.base_elo) for t in set(self.O) | set(self.D)}
        sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            return sorted_ratings[:top_n]
        return sorted_ratings

    @staticmethod
    def compute_brier_logloss(model: 'BaselineEloOffDefModel', test_df: pd.DataFrame, eps: float = 1e-10) -> Tuple[float, float]:
        """Compute Brier and Log loss for win probability predictions (xG outcome)."""
        brier_sum, logloss_sum = 0.0, 0.0
        for _, game in test_df.iterrows():
            home_team = get_value(game, 'home_team')
            home_xg = get_value(game, 'home_xg', 0)
            away_xg = get_value(game, 'away_xg', 0)
            winner, conf = model.predict_winner(game)
            EA = conf if winner == home_team else (1 - conf)
            EB = 1 - EA
            OA = 1 if home_xg > away_xg else 0
            OB = 1 - OA
            EA_c = np.clip(EA, eps, 1 - eps)
            EB_c = np.clip(EB, eps, 1 - eps)
            brier_sum += (EA - OA) ** 2
            logloss_sum += -(OA * np.log(EA_c) + OB * np.log(EB_c))
        n = len(test_df)
        return (brier_sum / n if n else 0.0, logloss_sum / n if n else 0.0)
