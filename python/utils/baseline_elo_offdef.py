"""
Baseline Elo Offensive/Defensive Model - Iteration 2.0
======================================================
Separate offensive (O) and defensive (D) Elo ratings per team, per line (line 1 and line 2).
Updates driven by observed xG vs expected xG. Uses expected wins/losses (xG) not actual goals.

2 sets per team: line 1 (O1, D1) and line 2 (O2, D2).

Formulas:
  multi = 10^((O - D) / 400)
  expected_xG = league_avg * multi * time
  O += k * (observed_xG - expected_xG) / ln(10)
  D -= k * (observed_xG - expected_xG) / ln(10)

Usage:
    from utils.baseline_elo_offdef import BaselineEloOffDefModel
    model = BaselineEloOffDefModel(params)
    model.fit(shifts_df)  # shift-level with home_off_line, away_off_line, toi
    # or model.fit(games_df) for game-level fallback (treats as single line)
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LN10 = np.log(10)

# Line identifiers in data
LINE1 = 'first_off'
LINE2 = 'second_off'
VALID_LINES = {LINE1, LINE2}

COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_xg': ['home_xg', 'home_xG', 'xg_home', 'h_xg'],
    'away_xg': ['away_xg', 'away_xG', 'xg_away', 'a_xg', 'visitor_xg'],
    'home_off_line': ['home_off_line'],
    'away_off_line': ['away_off_line'],
    'toi': ['toi', 'time_on_ice'],
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


def _parse_line(val) -> Optional[str]:
    """Map off_line value to LINE1 or LINE2, else None."""
    if pd.isna(val):
        return None
    v = str(val).strip().lower()
    if v in ('first_off', 'first'):
        return LINE1
    if v in ('second_off', 'second'):
        return LINE2
    return None


def rmse_score(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class BaselineEloOffDefModel:
    """
    Offensive/Defensive Elo model with line 1 and line 2 per team.
    Updates driven by observed vs expected xG. Uses xG-based wins (expected), not goals.
    """

    def __init__(self, params=None):
        self.params = params or {}
        self.k = float(self.params.get('k_factor', 32))
        self.base_elo = float(self.params.get('initial_rating', 1200))
        self.elo_scale = float(self.params.get('elo_scale', 400))
        self.goal_diff_half_range = float(self.params.get('goal_diff_half_range', 6.0))
        self.time_factor = float(self.params.get('time_factor', 1.0))
        self.league_avg_xg: Optional[float] = self.params.get('league_avg_xg')
        # Per team, per line: O[team][line] and D[team][line]
        self.O: Dict[str, Dict[str, float]] = {}
        self.D: Dict[str, Dict[str, float]] = {}

    def _add_team_line(self, team: str, line: str) -> None:
        if team not in self.O:
            self.O[team] = {LINE1: float(self.base_elo), LINE2: float(self.base_elo)}
            self.D[team] = {LINE1: float(self.base_elo), LINE2: float(self.base_elo)}
        if line not in self.O[team]:
            self.O[team][line] = float(self.base_elo)
            self.D[team][line] = float(self.base_elo)

    def _get_col(self, df: pd.DataFrame, field: str) -> Optional[str]:
        aliases = COLUMN_ALIASES.get(field, [field])
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train on shift-level or game-level DataFrame.

        Shift-level (preferred): must have home_team, away_team, home_xg, away_xg,
        home_off_line, away_off_line, toi. Uses only first_off and second_off shifts.

        Game-level fallback: home_team, away_team, home_xg, away_xg.
        Treats entire game as line 1 (no line 2 data).
        """
        home_col = self._get_col(df, 'home_team') or 'home_team'
        away_col = self._get_col(df, 'away_team') or 'away_team'
        hxg_col = self._get_col(df, 'home_xg') or 'home_xg'
        axg_col = self._get_col(df, 'away_xg') or 'away_xg'
        hline_col = self._get_col(df, 'home_off_line')
        aline_col = self._get_col(df, 'away_off_line')
        toi_col = self._get_col(df, 'toi')

        has_lines = hline_col and aline_col and toi_col
        if has_lines:
            self._fit_shifts(df, home_col, away_col, hxg_col, axg_col, hline_col, aline_col, toi_col)
        else:
            self._fit_games(df, home_col, away_col, hxg_col, axg_col)

    def _fit_games(self, games_df: pd.DataFrame, home_col: str, away_col: str, hxg_col: str, axg_col: str) -> None:
        """Game-level fallback: treat each game as line 1."""
        if self.league_avg_xg is None:
            total_xg = games_df[hxg_col].fillna(0).sum() + games_df[axg_col].fillna(0).sum()
            self.league_avg_xg = float(total_xg / (2 * max(1, len(games_df))))

        for _, row in games_df.iterrows():
            ht = str(row[home_col]) if not pd.isna(row[home_col]) else 'Unknown_Home'
            at = str(row[away_col]) if not pd.isna(row[away_col]) else 'Unknown_Away'
            if ht == at:
                continue
            obs_home = 0 if pd.isna(row[hxg_col]) else float(row[hxg_col])
            obs_away = 0 if pd.isna(row[axg_col]) else float(row[axg_col])

            self._add_team_line(ht, LINE1)
            self._add_team_line(at, LINE1)

            O_h = self.O[ht][LINE1]
            D_h = self.D[ht][LINE1]
            O_a = self.O[at][LINE1]
            D_a = self.D[at][LINE1]

            multi_h = 10.0 ** ((O_h - D_a) / self.elo_scale)
            multi_a = 10.0 ** ((O_a - D_h) / self.elo_scale)
            exp_h = self.league_avg_xg * multi_h * self.time_factor
            exp_a = self.league_avg_xg * multi_a * self.time_factor

            delta_h = self.k * (obs_home - exp_h) / LN10
            delta_a = self.k * (obs_away - exp_a) / LN10

            self.O[ht][LINE1] += delta_h
            self.D[at][LINE1] -= delta_h
            self.O[at][LINE1] += delta_a
            self.D[ht][LINE1] -= delta_a

    def _fit_shifts(
        self, shifts_df: pd.DataFrame,
        home_col: str, away_col: str, hxg_col: str, axg_col: str,
        hline_col: str, aline_col: str, toi_col: str,
    ) -> None:
        """Shift-level: update O/D per line using first_off and second_off shifts only.
        Uses toi_delta (change in TOI) per segment: toi_at_this_row - toi_at_previous_row.
        Assumes cumulative toi within each game; first row per game uses toi as delta."""
        shifts = shifts_df[
            shifts_df[hline_col].isin(VALID_LINES) & shifts_df[aline_col].isin(VALID_LINES)
        ].copy()
        if len(shifts) == 0:
            gb = shifts_df.groupby(['game_id', home_col, away_col]).agg({hxg_col: 'sum', axg_col: 'sum'}).reset_index()
            self._fit_games(gb, home_col, away_col, hxg_col, axg_col)
            return

        # Sort by game and time order; use record_id if available
        game_col = 'game_id' if 'game_id' in shifts.columns else home_col
        sort_cols = [game_col]
        if 'record_id' in shifts.columns:
            sort_cols.append('record_id')
        shifts = shifts.sort_values(sort_cols).reset_index(drop=True)

        # Compute toi_delta = toi - toi_prev within each game
        shifts['_toi_raw'] = shifts[toi_col].fillna(0).astype(float)
        shifts['_toi_prev'] = shifts.groupby(game_col)['_toi_raw'].shift(1).fillna(0)
        shifts['_toi_delta'] = (shifts['_toi_raw'] - shifts['_toi_prev']).clip(lower=0)

        total_xg = shifts[hxg_col].fillna(0).sum() + shifts[axg_col].fillna(0).sum()
        total_toi = shifts['_toi_delta'].sum()
        if total_toi <= 0:
            total_toi = shifts['_toi_raw'].sum()
        if total_toi <= 0:
            total_toi = len(shifts)
        self.league_avg_xg = float(total_xg / max(total_toi / 3600.0, 0.01))

        for _, row in shifts.iterrows():
            ht = str(row[home_col]) if not pd.isna(row[home_col]) else 'Unknown_Home'
            at = str(row[away_col]) if not pd.isna(row[away_col]) else 'Unknown_Away'
            if ht == at:
                continue

            hl = _parse_line(row[hline_col]) or LINE1
            al = _parse_line(row[aline_col]) or LINE1
            toi_delta = float(row['_toi_delta'])
            if toi_delta <= 0:
                toi_delta = float(row['_toi_raw']) if row['_toi_raw'] > 0 else 1.0
            time_frac = max(toi_delta / 3600.0, 0.001)

            obs_home = 0 if pd.isna(row[hxg_col]) else float(row[hxg_col])
            obs_away = 0 if pd.isna(row[axg_col]) else float(row[axg_col])

            self._add_team_line(ht, hl)
            self._add_team_line(at, al)

            O_h = self.O[ht][hl]
            D_a = self.D[at][al]
            O_a = self.O[at][al]
            D_h = self.D[ht][hl]

            multi_h = 10.0 ** ((O_h - D_a) / self.elo_scale)
            multi_a = 10.0 ** ((O_a - D_h) / self.elo_scale)
            exp_h = self.league_avg_xg * multi_h * time_frac
            exp_a = self.league_avg_xg * multi_a * time_frac

            delta_h = self.k * (obs_home - exp_h) / LN10
            delta_a = self.k * (obs_away - exp_a) / LN10

            self.O[ht][hl] += delta_h
            self.D[at][al] -= delta_h
            self.O[at][al] += delta_a
            self.D[ht][hl] -= delta_a

    def _team_net(self, team: str) -> float:
        """Net strength = average of (O - D) over lines."""
        if team not in self.O:
            return 0.0
        vals = [self.O[team].get(l, self.base_elo) - self.D[team].get(l, self.base_elo) for l in VALID_LINES]
        return sum(vals) / len(vals) if vals else 0.0

    def predict_goals(self, game) -> Tuple[float, float]:
        """Return (expected_home_xg, expected_away_xg) using average line ratings."""
        home_team = get_value(game, 'home_team') or 'Unknown_Home'
        away_team = get_value(game, 'away_team') or 'Unknown_Away'
        self._add_team_line(home_team, LINE1)
        self._add_team_line(away_team, LINE1)
        O_h = np.mean([self.O.get(home_team, {}).get(l, self.base_elo) for l in VALID_LINES])
        D_h = np.mean([self.D.get(home_team, {}).get(l, self.base_elo) for l in VALID_LINES])
        O_a = np.mean([self.O.get(away_team, {}).get(l, self.base_elo) for l in VALID_LINES])
        D_a = np.mean([self.D.get(away_team, {}).get(l, self.base_elo) for l in VALID_LINES])

        multi_h = 10.0 ** ((O_h - D_a) / self.elo_scale)
        multi_a = 10.0 ** ((O_a - D_h) / self.elo_scale)
        league = self.league_avg_xg or 3.0
        home_xg = max(0.0, league * multi_h * self.time_factor)
        away_xg = max(0.0, league * multi_a * self.time_factor)
        return home_xg, away_xg

    def predict_winner(self, game) -> Tuple[str, float]:
        """Return (winning_team, confidence). Uses rating-diff in Elo probability: P = 1/(1+10^(-diff/400))."""
        home_team = get_value(game, 'home_team') or 'Unknown_Home'
        away_team = get_value(game, 'away_team') or 'Unknown_Away'
        self._add_team_line(home_team, LINE1)
        self._add_team_line(away_team, LINE1)
        O_h = np.mean([self.O.get(home_team, {}).get(l, self.base_elo) for l in VALID_LINES])
        D_h = np.mean([self.D.get(home_team, {}).get(l, self.base_elo) for l in VALID_LINES])
        O_a = np.mean([self.O.get(away_team, {}).get(l, self.base_elo) for l in VALID_LINES])
        D_a = np.mean([self.D.get(away_team, {}).get(l, self.base_elo) for l in VALID_LINES])
        # Rating diff: home attacking vs away defending minus away attacking vs home defending
        diff = (O_h - D_a) - (O_a - D_h)
        home_win_prob = 1.0 / (1.0 + 10.0 ** (-diff / self.elo_scale))
        if home_win_prob > 0.5:
            return home_team, home_win_prob
        return away_team, 1.0 - home_win_prob

    def evaluate(self, games_df: pd.DataFrame) -> Dict[str, float]:
        """Compute RMSE, MAE, R², win_accuracy. Win accuracy uses xG (expected win), not goals."""
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

    def get_line_ratings(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Full O and D per team per line. Returns {team: {'O': {line: rating}, 'D': {line: rating}}}."""
        teams = set(self.O.keys()) | set(self.D.keys())
        out = {}
        for t in teams:
            out[t] = {
                'O': {line: round(self.O.get(t, {}).get(line, self.base_elo), 1) for line in VALID_LINES},
                'D': {line: round(self.D.get(t, {}).get(line, self.base_elo), 1) for line in VALID_LINES},
            }
        return out

    def get_rankings(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Sorted list of (team, net_strength). net_strength = avg(O - D) over lines.
        Values can be negative (O and D start at base_elo; league-wide D often > O).
        Higher net = stronger. Used for ranking only; P uses Elo formula on diff."""
        teams = set(self.O.keys()) | set(self.D.keys())
        ratings = {t: self._team_net(t) for t in teams}
        sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            return sorted_ratings[:top_n]
        return sorted_ratings

    @staticmethod
    def compute_brier_logloss(model: 'BaselineEloOffDefModel', test_df: pd.DataFrame, eps: float = 1e-10) -> Tuple[float, float]:
        """Brier and Log loss for win probability. Outcome = xG win (expected), not goals."""
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
