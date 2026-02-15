"""
Enhanced ELO Model — Hockey-Specific Rating System
===================================================

Improvements over basic Elo:
1. xG-based margin of victory (shot quality, not just goals)
2. Dynamic K-factor (higher early season, decays as ratings stabilize)
3. Team-specific scoring rate predictions (not constant 6.0 total)
4. Shot share (Corsi) adjustment to Elo updates
5. Penalty differential correction
6. Separate offensive & defensive Elo ratings
7. Rolling team statistics tracked internally

Usage:
    from utils.enhanced_elo_model import EnhancedEloModel

    model = EnhancedEloModel(params)
    model.fit(games_df)           # games_df must have shots, xG, penalties
    metrics = model.evaluate(test_df)
    h, a = model.predict_goals(game)
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class EnhancedEloModel:
    """
    Hockey-specific Elo with xG, shots, penalties, and team-specific baselines.

    Parameters
    ----------
    params : dict
        k_factor : float (5-50) — base rating change per game
        home_advantage : float (0-200) — home ice Elo boost
        mov_multiplier : float (0-2) — margin of victory weight
        xg_weight : float (0-1) — blend of xG vs actual goals for MOV
        k_decay : float (0-0.05) — how much K shrinks per game played (dynamic K)
        k_min : float — minimum K-factor after decay
        shot_share_weight : float (0-50) — Elo adjustment for Corsi/shot share
        penalty_weight : float (0-20) — Elo adjustment for penalty differential
        scoring_baseline : str — 'league' or 'team' (venue-scaled additive)
        xg_pred_weight : float (0-1) — blend of xG rate into goal predictions
        elo_shift_scale : float (0-5) — how Elo diff maps to predicted goal shift
        rolling_window : int — games for rolling average stats
    """

    def __init__(self, params: dict):
        self.params = params
        self.ratings = {}            # team → overall Elo
        self.off_ratings = {}        # team → offensive Elo
        self.def_ratings = {}        # team → defensive Elo
        self.rating_history = []
        self.team_stats = {}         # team → rolling stat tracker
        self.league_avg_home = 3.0   # updated during fit
        self.league_avg_away = 3.0
        self.games_played = {}       # team → count

    def _ensure_team(self, team):
        """Initialize a team if we haven't seen it."""
        init = self.params.get('initial_rating', 1500)
        if team not in self.ratings:
            self.ratings[team] = init
            self.off_ratings[team] = init
            self.def_ratings[team] = init
            self.games_played[team] = 0
            self.team_stats[team] = {
                'goals_for': [], 'goals_against': [],
                'shots_for': [], 'shots_against': [],
                'xg_for': [], 'xg_against': [],
                'pen_min_for': [], 'pen_min_against': [],
                'wins': [],
                # Home/away venue splits for asymmetric prediction
                'home_gf': [], 'home_ga': [],
                'away_gf': [], 'away_ga': [],
            }

    def _rolling_mean(self, team, stat, default=None):
        """Get rolling mean of a team stat over the last N games."""
        w = self.params.get('rolling_window', 15)
        vals = self.team_stats.get(team, {}).get(stat, [])
        if not vals:
            return default
        recent = vals[-w:]
        return np.mean(recent)

    def _team_game_count(self, team):
        return self.games_played.get(team, 0)

    # ── Core Elo math ──

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def dynamic_k(self, team):
        """K-factor that decays as a team plays more games."""
        base_k = self.params.get('k_factor', 32)
        k_decay = self.params.get('k_decay', 0.0)
        k_min = self.params.get('k_min', 8)
        n = self._team_game_count(team)
        if k_decay > 0 and n > 0:
            return max(k_min, base_k * (1 - k_decay) ** n)
        return base_k

    def mov_factor(self, game):
        """Margin of victory multiplier, blending goals and xG."""
        mov_mult = self.params.get('mov_multiplier', 1.0)
        if mov_mult == 0:
            return 1.0

        home_goals = game.get('home_goals', 0)
        away_goals = game.get('away_goals', 0)
        goal_diff = abs(home_goals - away_goals)

        # xG-blended margin
        xg_weight = self.params.get('xg_weight', 0.0)
        if xg_weight > 0 and 'home_xg' in game and 'away_xg' in game:
            home_xg = game.get('home_xg', home_goals)
            away_xg = game.get('away_xg', away_goals)
            xg_diff = abs(home_xg - away_xg)
            effective_diff = (1 - xg_weight) * goal_diff + xg_weight * xg_diff
        else:
            effective_diff = goal_diff

        # Logarithmic scaling (FiveThirtyEight-style)
        return 1.0 + mov_mult * np.log(effective_diff + 1)

    def shot_share_adjustment(self, game):
        """Corsi-style adjustment: reward teams that dominate shot attempts."""
        weight = self.params.get('shot_share_weight', 0.0)
        if weight == 0:
            return 0.0, 0.0
        hs = game.get('home_shots', 0)
        avs = game.get('away_shots', 0)
        total = hs + avs
        if total == 0:
            return 0.0, 0.0
        home_share = hs / total  # 0.5 = even, >0.5 = home dominated
        # Convert to Elo adjustment: +weight if 60% share, -weight if 40%
        home_adj = (home_share - 0.5) * 2 * weight
        return home_adj, -home_adj

    def penalty_adjustment(self, game):
        """Penalize undisciplined teams (more PIM = negative adjustment)."""
        weight = self.params.get('penalty_weight', 0.0)
        if weight == 0:
            return 0.0, 0.0
        h_pim = game.get('home_pen_min', 0)
        a_pim = game.get('away_pen_min', 0)
        # Negative for the team with more penalties
        diff = a_pim - h_pim  # positive means away had more penalties (good for home)
        home_adj = np.clip(diff / 10, -3, 3) * weight
        return home_adj, -home_adj

    # ── Training ──

    def update_ratings(self, game):
        """Update all ratings after a single game."""
        home = game['home_team']
        away = game['away_team']
        self._ensure_team(home)
        self._ensure_team(away)

        # Current ratings
        home_elo = self.ratings[home]
        away_elo = self.ratings[away]

        # Home ice advantage
        ha = self.params.get('home_advantage', 100)
        home_adj = home_elo + ha

        # Expected scores
        home_exp = self.expected_score(home_adj, away_elo)

        # Actual outcome
        home_goals = game.get('home_goals', 0)
        away_goals = game.get('away_goals', 0)
        home_actual = 1.0 if home_goals > away_goals else 0.0

        # Margin of victory
        mov = self.mov_factor(game)

        # Dynamic K
        k_home = self.dynamic_k(home) * mov
        k_away = self.dynamic_k(away) * mov

        # Base Elo update
        home_delta = k_home * (home_actual - home_exp)
        away_delta = k_away * ((1 - home_actual) - (1 - home_exp))

        # Shot share bonus/penalty
        ss_h, ss_a = self.shot_share_adjustment(game)
        home_delta += ss_h
        away_delta += ss_a

        # Penalty adjustment
        pen_h, pen_a = self.penalty_adjustment(game)
        home_delta += pen_h
        away_delta += pen_a

        # Apply updates to overall rating
        self.ratings[home] = home_elo + home_delta
        self.ratings[away] = away_elo + away_delta

        # Update offensive/defensive Elo with INDEPENDENT signals:
        # - Offensive: goals scored vs league average (high-scoring = positive)
        # - Defensive: goals allowed vs league average (stingy = positive)
        # These are decorrelated: a 6-5 win → big off boost + big def penalty.
        avg_gpg = (self.league_avg_home + self.league_avg_away) / 2
        off_k_scale = 0.5 * (k_home + k_away) / 2

        if avg_gpg > 0:
            self.off_ratings[home] += off_k_scale * (home_goals / avg_gpg - 1.0)
            self.def_ratings[home] += off_k_scale * (1.0 - away_goals / avg_gpg)
            self.off_ratings[away] += off_k_scale * (away_goals / avg_gpg - 1.0)
            self.def_ratings[away] += off_k_scale * (1.0 - home_goals / avg_gpg)

        # Update team stats
        self.games_played[home] = self.games_played.get(home, 0) + 1
        self.games_played[away] = self.games_played.get(away, 0) + 1

        for stat, h_val, a_val in [
            ('goals_for', home_goals, away_goals),
            ('goals_against', away_goals, home_goals),
            ('shots_for', game.get('home_shots', 0), game.get('away_shots', 0)),
            ('shots_against', game.get('away_shots', 0), game.get('home_shots', 0)),
            ('xg_for', game.get('home_xg', 0), game.get('away_xg', 0)),
            ('xg_against', game.get('away_xg', 0), game.get('home_xg', 0)),
            ('pen_min_for', game.get('home_pen_min', 0), game.get('away_pen_min', 0)),
            ('pen_min_against', game.get('away_pen_min', 0), game.get('home_pen_min', 0)),
            ('wins', int(home_goals > away_goals), int(away_goals > home_goals)),
            # Venue-specific stats
            ('home_gf', home_goals, None),
            ('home_ga', away_goals, None),
            ('away_gf', None, away_goals),
            ('away_ga', None, home_goals),
        ]:
            if h_val is not None:
                self.team_stats[home][stat].append(h_val)
            if a_val is not None:
                self.team_stats[away][stat].append(a_val)

        # Record history
        self.rating_history.append({
            'home_team': home, 'away_team': away,
            'home_rating': self.ratings[home],
            'away_rating': self.ratings[away],
            'home_off': self.off_ratings[home],
            'away_off': self.off_ratings[away],
            'home_def': self.def_ratings[home],
            'away_def': self.def_ratings[away],
            'home_delta': home_delta, 'away_delta': away_delta,
        })

    def fit(self, games_df):
        """Train on a DataFrame of games (must be chronologically sorted)."""
        # Compute league averages from training data
        if 'home_goals' in games_df.columns:
            self.league_avg_home = games_df['home_goals'].mean()
            self.league_avg_away = games_df['away_goals'].mean()

        for _, game in games_df.iterrows():
            self.update_ratings(game)

    # ── Prediction ──

    def predict_goals(self, game):
        """
        Predict goals using venue-scaled additive team strength model.

        Uses the standard averaging approach (home_GF + away_GA) / 2, then
        scales by venue factors (league_avg_home / league_avg and
        league_avg_away / league_avg) to break total-goals symmetry.

        When teams swap home/away, the totals differ because the venue
        scale factors redistribute goals between the two sides.

        Elo-based shift is then applied on top.
        """
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        self._ensure_team(home)
        self._ensure_team(away)

        # Rating-based win probability
        ha = self.params.get('home_advantage', 100)
        home_wp = self.expected_score(self.ratings[home] + ha, self.ratings[away])

        # --- Team-specific baselines ---
        use_team = self.params.get('scoring_baseline', 'team')
        if use_team == 'team':
            league_avg_gf = (self.league_avg_home + self.league_avg_away) / 2

            # Team offensive strength (goals scored per game)
            home_gf = self._rolling_mean(home, 'goals_for', league_avg_gf)
            away_gf = self._rolling_mean(away, 'goals_for', league_avg_gf)

            # Team defensive weakness (goals allowed per game)
            home_ga = self._rolling_mean(home, 'goals_against', league_avg_gf)
            away_ga = self._rolling_mean(away, 'goals_against', league_avg_gf)

            # xG-informed adjustment (blend actual scoring with expected)
            xg_blend = self.params.get('xg_pred_weight', 0.0)
            if xg_blend > 0:
                home_xg_gf = self._rolling_mean(home, 'xg_for', home_gf)
                away_xg_gf = self._rolling_mean(away, 'xg_for', away_gf)
                home_gf = (1 - xg_blend) * home_gf + xg_blend * home_xg_gf
                away_gf = (1 - xg_blend) * away_gf + xg_blend * away_xg_gf

            # Venue-scaled additive model:
            # 1. Average each team's offense with opponent's defense (additive, conservative)
            # 2. Scale by venue factors to break home/away symmetry
            # This avoids the multiplicative model's tendency to over-amplify differences.
            home_strength = (home_gf + away_ga) / 2  # how much home team should score
            away_strength = (away_gf + home_ga) / 2  # how much away team should score

            if league_avg_gf > 0:
                home_venue_factor = self.league_avg_home / league_avg_gf
                away_venue_factor = self.league_avg_away / league_avg_gf
            else:
                home_venue_factor = 1.0
                away_venue_factor = 1.0

            home_raw = home_strength * home_venue_factor
            away_raw = away_strength * away_venue_factor
        else:
            home_raw = self.league_avg_home
            away_raw = self.league_avg_away

        # Elo-based shift: winning team gets boosted, losing team gets reduced
        elo_shift_scale = self.params.get('elo_shift_scale', 2.0)
        elo_shift = (home_wp - 0.5) * elo_shift_scale

        home_pred = max(0.3, home_raw + elo_shift)
        away_pred = max(0.3, away_raw - elo_shift)

        return home_pred, away_pred

    def predict_winner(self, game):
        """Predict winner and confidence."""
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        self._ensure_team(home)
        self._ensure_team(away)

        ha = self.params.get('home_advantage', 100)
        home_wp = self.expected_score(self.ratings[home] + ha, self.ratings[away])

        if home_wp >= 0.5:
            return home, home_wp
        else:
            return away, 1 - home_wp

    # ── Evaluation ──

    def evaluate(self, games_df):
        """Evaluate on test set. Returns dict with combined_rmse etc."""
        home_preds, away_preds = [], []
        home_actuals, away_actuals = [], []
        correct_wins = 0

        for _, game in games_df.iterrows():
            hp, ap = self.predict_goals(game)
            ha = game.get('home_goals', 0)
            aa = game.get('away_goals', 0)

            home_preds.append(hp)
            away_preds.append(ap)
            home_actuals.append(ha)
            away_actuals.append(aa)

            if (hp > ap) == (ha > aa):
                correct_wins += 1

        n = len(home_actuals)
        all_act = home_actuals + away_actuals
        all_pred = home_preds + away_preds

        return {
            'home_rmse': rmse_score(home_actuals, home_preds),
            'away_rmse': rmse_score(away_actuals, away_preds),
            'combined_rmse': rmse_score(all_act, all_pred),
            'home_mae': float(mean_absolute_error(home_actuals, home_preds)),
            'away_mae': float(mean_absolute_error(away_actuals, away_preds)),
            'home_r2': float(r2_score(home_actuals, home_preds)) if len(set(home_actuals)) > 1 else 0.0,
            'away_r2': float(r2_score(away_actuals, away_preds)) if len(set(away_actuals)) > 1 else 0.0,
            'win_accuracy': correct_wins / n if n > 0 else 0.0,
            'pred_home_std': float(np.std(home_preds)),
            'pred_away_std': float(np.std(away_preds)),
            'pred_total_std': float(np.std([hp + ap for hp, ap in zip(home_preds, away_preds)])),
            'n_games': n,
        }

    # ── Rankings & History ──

    def get_rankings(self, top_n=None):
        sorted_r = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_r[:top_n] if top_n else sorted_r

    def get_off_rankings(self, top_n=None):
        sorted_r = sorted(self.off_ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_r[:top_n] if top_n else sorted_r

    def get_def_rankings(self, top_n=None):
        """Lower defensive rating = better defense."""
        sorted_r = sorted(self.def_ratings.items(), key=lambda x: x[1])
        return sorted_r[:top_n] if top_n else sorted_r

    def get_rating_history_df(self):
        return pd.DataFrame(self.rating_history)

    # ── Save / Load ──

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        state = {
            'params': self.params,
            'ratings': self.ratings,
            'off_ratings': self.off_ratings,
            'def_ratings': self.def_ratings,
            'rating_history': self.rating_history,
            'team_stats': self.team_stats,
            'league_avg_home': self.league_avg_home,
            'league_avg_away': self.league_avg_away,
            'games_played': self.games_played,
        }

        pkl_path = path if path.endswith('.pkl') else path + '.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(state, f)

        json_path = pkl_path.replace('.pkl', '.json')
        json_state = {
            'params': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in self.params.items()},
            'ratings': {k: round(float(v), 1) for k, v in self.ratings.items()},
            'off_ratings': {k: round(float(v), 1) for k, v in self.off_ratings.items()},
            'def_ratings': {k: round(float(v), 1) for k, v in self.def_ratings.items()},
            'n_games_trained': len(self.rating_history),
        }
        with open(json_path, 'w') as f:
            json.dump(json_state, f, indent=2)

        print(f"Enhanced Elo saved to {pkl_path} ({len(self.ratings)} teams, "
              f"{len(self.rating_history)} games)")
        return pkl_path

    @classmethod
    def load_model(cls, path):
        pkl_path = path if path.endswith('.pkl') else path + '.pkl'
        with open(pkl_path, 'rb') as f:
            state = pickle.load(f)
        model = cls(state['params'])
        for attr in ['ratings', 'off_ratings', 'def_ratings', 'rating_history',
                     'team_stats', 'league_avg_home', 'league_avg_away', 'games_played']:
            if attr in state:
                setattr(model, attr, state[attr])
        print(f"Enhanced Elo loaded: {len(model.ratings)} teams, {len(model.rating_history)} games")
        return model
