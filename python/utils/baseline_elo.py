"""
Baseline Elo Model - Classic Elo for Hockey
==========================================
A minimal Elo rating system (no home advantage, MOV, rest, etc.) used as a
simple baseline. Same interface as EloModel and baseline_model for easy
comparison in Jupyter notebooks and pipelines.

Usage:
    from utils.baseline_elo import BaselineEloModel
    model = BaselineEloModel(params)
    model.fit(games_df)
    metrics = model.evaluate(test_df)
    h, a = model.predict_goals(game)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Column aliases for flexible CSV/DataFrame column naming
COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_goals': ['home_goals', 'home_score', 'h_goals', 'goals_home', 'home_pts'],
    'away_goals': ['away_goals', 'away_score', 'a_goals', 'goals_away', 'away_pts', 'visitor_goals'],
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


# ── Core Elo dataclasses ────────────────────────────────────────────────

@dataclass
class MatchResult:
    team_a: str
    team_b: str
    outcome_a: int
    ra_before: float
    rb_before: float
    ea: float
    eb: float
    delta_a: float
    delta_b: float
    ra_after: float
    rb_after: float


@dataclass
class EloSystem:
    k: float = 32.0
    base_elo: float = 1200.0
    elo_scale: float = 400.0  # divisor in expected-score: 1/(1+10^((rb-ra)/scale))
    use_doc_ob_same_as_oa: bool = False
    ratings: Dict[str, float] = field(default_factory=dict)
    history: List[MatchResult] = field(default_factory=list)

    def add_team(self, team: str) -> None:
        if team not in self.ratings:
            self.ratings[team] = float(self.base_elo)

    def add_teams(self, teams: Iterable[str]) -> None:
        for team in teams:
            self.add_team(team)

    def get_rating(self, team: str) -> float:
        self.add_team(team)
        return self.ratings[team]

    def expected_score(self, ra: float, rb: float) -> float:
        if self.elo_scale <= 0:
            return 0.5  # fallback to 50/50 to avoid division by zero
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / self.elo_scale))

    def expected_scores(self, team_a: str, team_b: str) -> Tuple[float, float]:
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea
        return ea, eb

    def _validate_outcome(self, outcome_a: int) -> None:
        if outcome_a not in (0, 1):
            raise ValueError("outcome_a must be 0 (A loses) or 1 (A wins)")

    def update_match(self, team_a: str, team_b: str, outcome_a: int) -> MatchResult:
        if team_a == team_b:
            raise ValueError("team_a and team_b must be different")
        self._validate_outcome(outcome_a)

        self.add_team(team_a)
        self.add_team(team_b)

        ra = self.ratings[team_a]
        rb = self.ratings[team_b]

        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea

        oa = int(outcome_a)
        ob = oa if self.use_doc_ob_same_as_oa else 1 - oa

        delta_a = self.k * (oa - ea)
        delta_b = self.k * (ob - eb)

        ra_new = ra + delta_a
        rb_new = rb + delta_b

        self.ratings[team_a] = ra_new
        self.ratings[team_b] = rb_new

        record = MatchResult(
            team_a=team_a,
            team_b=team_b,
            outcome_a=oa,
            ra_before=ra,
            rb_before=rb,
            ea=ea,
            eb=eb,
            delta_a=delta_a,
            delta_b=delta_b,
            ra_after=ra_new,
            rb_after=rb_new,
        )
        self.history.append(record)
        return record

    def process_matches(
        self,
        matches: Iterable[Tuple[str, str, int]]
    ) -> List[MatchResult]:
        results = []
        for team_a, team_b, outcome_a in matches:
            results.append(self.update_match(team_a, team_b, outcome_a))
        return results

    def leaderboard(self, descending: bool = True) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=descending)

    def rating_table(self, digits: int = 2) -> List[Dict[str, Any]]:
        board = self.leaderboard()
        return [
            {"rank": i + 1, "team": team, "elo": round(rating, digits)}
            for i, (team, rating) in enumerate(board)
        ]

    def last_match_summary(self, digits: int = 4) -> Optional[Dict[str, Any]]:
        if not self.history:
            return None
        r = self.history[-1]
        return {
            "team_a": r.team_a,
            "team_b": r.team_b,
            "outcome_a": r.outcome_a,
            "ra_before": round(r.ra_before, digits),
            "rb_before": round(r.rb_before, digits),
            "ea": round(r.ea, digits),
            "eb": round(r.eb, digits),
            "delta_a": round(r.delta_a, digits),
            "delta_b": round(r.delta_b, digits),
            "ra_after": round(r.ra_after, digits),
            "rb_after": round(r.rb_after, digits),
        }


# ── BaselineEloModel: EloModel-compatible interface for notebooks ──────────

class BaselineEloModel:
    """Classic Elo model with fit/predict/evaluate interface for CSV/DataFrame data."""

    def __init__(self, params=None):
        self.params = params or {}
        self.k = self.params.get('k_factor', 32)
        self.base_elo = self.params.get('initial_rating', 1200)
        self.elo_scale = self.params.get('elo_scale', 400)
        self.league_avg_goals = self.params.get('league_avg_goals', 3.0)
        self.goal_diff_half_range = self.params.get('goal_diff_half_range', 6.0)
        self.elo = EloSystem(k=self.k, base_elo=self.base_elo, elo_scale=self.elo_scale)
        self.rating_history = []

    def _get_team_column(self, df, field):
        aliases = COLUMN_ALIASES.get(field, [field])
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None

      def fit(self, games_df: pd.DataFrame) -> None:
        """Train on game-level DataFrame (home_team, away_team, home_goals, away_goals)."""
        home_col = self._get_team_column(games_df, 'home_team') or 'home_team'
        away_col = self._get_team_column(games_df, 'away_team') or 'away_team'
        hg_col = self._get_team_column(games_df, 'home_goals') or 'home_goals'
        ag_col = self._get_team_column(games_df, 'away_goals') or 'away_goals'

        matches = []
        for _, row in games_df.iterrows():
            ht = row[home_col]
            at = row[away_col]
            if pd.isna(ht):
                ht = 'Unknown_Home'
            if pd.isna(at):
                at = 'Unknown_Away'
            hg = row[hg_col] if hg_col in row else 0
            ag = row[ag_col] if ag_col in row else 0
            hg = 0 if pd.isna(hg) else float(hg)
            ag = 0 if pd.isna(ag) else float(ag)
            outcome_a = 1 if hg > ag else 0
            if ht == at:  # skip self-play
                continue
            matches.append((str(ht), str(at), outcome_a))

        self.elo.process_matches(matches)
        self.rating_history = [
            {'home_team': r.team_a, 'away_team': r.team_b,
             'home_rating': r.ra_after, 'away_rating': r.rb_after}
            for r in self.elo.history
        ]

    def predict_goals(self, game) -> Tuple[float, float]:
        """Return (expected_home_goals, expected_away_goals) for a game row."""
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        home_elo = self.elo.ratings.get(home_team, self.base_elo)
        away_elo = self.elo.ratings.get(away_team, self.base_elo)
        home_win_prob = self.elo.expected_score(home_elo, away_elo)
        # goals = league_avg +/- (goal_diff_half_range * (win_prob - 0.5))
        adj = self.goal_diff_half_range * (home_win_prob - 0.5)
        home_goals = max(0.0, self.league_avg_goals + adj)
        away_goals = max(0.0, self.league_avg_goals - adj)
        return home_goals, away_goals

    def predict_winner(self, game) -> Tuple[str, float]:
        """Return (winning_team, confidence) for a game row."""
        home_team = get_value(game, 'home_team')
        away_team = get_value(game, 'away_team')
        home_elo = self.elo.ratings.get(home_team, self.base_elo)
        away_elo = self.elo.ratings.get(away_team, self.base_elo)
        home_win_prob = self.elo.expected_score(home_elo, away_elo)
        if home_win_prob > 0.5:
            return home_team, home_win_prob
        return away_team, 1 - home_win_prob

    def evaluate(self, games_df: pd.DataFrame) -> Dict[str, float]:
        """Compute RMSE, MAE, R², win_accuracy on a DataFrame of games."""
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

    def get_rankings(self, top_n=None) -> List[Tuple[str, float]]:
        """Sorted list of (team, rating). top_n limits to top N if given."""
        sorted_ratings = sorted(self.elo.ratings.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            return sorted_ratings[:top_n]
        return sorted_ratings

    @staticmethod
    def compute_brier_logloss(model: 'BaselineEloModel', test_df: pd.DataFrame, eps: float = 1e-10) -> Tuple[float, float]:
        """
        Compute Brier loss and Log loss for win probability predictions.
        Returns (brier_mean, log_loss_mean).
        """
        brier_sum, logloss_sum = 0.0, 0.0
        for _, game in test_df.iterrows():
            home_team = get_value(game, 'home_team')
            home_goals = get_value(game, 'home_goals', 0)
            away_goals = get_value(game, 'away_goals', 0)
            winner, conf = model.predict_winner(game)
            EA = conf if winner == home_team else (1 - conf)
            EB = 1 - EA
            OA = 1 if home_goals > away_goals else 0
            OB = 1 - OA
            EA_c = np.clip(EA, eps, 1 - eps)
            EB_c = np.clip(EB, eps, 1 - eps)
            brier_sum += (EA - OA) ** 2
            logloss_sum += -(OA * np.log(EA_c) + OB * np.log(EB_c))
        n = len(test_df)
        return (brier_sum / n if n else 0.0, logloss_sum / n if n else 0.0)
