"""
Comprehensive Game Goal Predictor
==================================
Combines six untapped signals that other teams will miss:

1. GOALIE QUALITY — 1.70 GA/g spread across goalies (GSAX metric)
2. xG FINISHING RATE — teams convert xG at 0.69x to 1.31x
3. HEAD-TO-HEAD HISTORY — direct matchup results for each pairing
4. LINE DEPLOYMENT QUALITY — PP/PK effectiveness, first-line xG rate
5. POISSON REGRESSION — proper count-data modeling (log link)
6. CROSS-VALIDATION — random K-fold (no temporal ordering per glossary)

Usage:
    from utils.game_predictor import GamePredictor, grid_search_cv

    best_cfg = grid_search_cv(raw_df)           # find best model type + alpha
    model = GamePredictor(**best_cfg)
    model.fit(raw_df)
    h, a = model.predict(game_row)
    cv = model.cross_validate(raw_df, k=5)      # honest K-fold evaluation
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ═══════════════════════════════════════════════════════════════
# DATA AGGREGATION
# ═══════════════════════════════════════════════════════════════

def aggregate_games(raw_df):
    """Aggregate shift-level rows → one row per game."""
    games = raw_df.groupby('game_id').agg(
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first'),
        home_goals=('home_goals', 'sum'),
        away_goals=('away_goals', 'sum'),
        home_shots=('home_shots', 'sum'),
        away_shots=('away_shots', 'sum'),
        home_xg=('home_xg', 'sum'),
        away_xg=('away_xg', 'sum'),
        home_max_xg=('home_max_xg', 'max'),
        away_max_xg=('away_max_xg', 'max'),
        home_assists=('home_assists', 'sum'),
        away_assists=('away_assists', 'sum'),
        home_pen_comm=('home_penalties_committed', 'sum'),
        away_pen_comm=('away_penalties_committed', 'sum'),
        home_pen_min=('home_penalty_minutes', 'sum'),
        away_pen_min=('away_penalty_minutes', 'sum'),
        home_goalie=('home_goalie', 'first'),
        away_goalie=('away_goalie', 'first'),
        went_ot=('went_ot', 'max'),
    ).reset_index()
    return games


# ═══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_stats(raw_df, games_df):
    """
    Compute ALL statistics needed for prediction from a set of games.

    Returns dict with keys:
        team      — {team_name: {gf, ga, xgf, xga, shots, ...}}
        goalie    — {goalie_id: {ga_avg, save_pct, gsax, gp}}
        h2h       — {(home, away): {home_goals_avg, away_goals_avg, games}}
        line      — {team_name: {pp_eff, pk_eff, first_line_xg_rate}}
        goalies   — {team_name: primary_goalie_id}
        league    — {home_avg, away_avg, avg, save_pct}
    """
    stats = {}

    # ── League Averages ──
    total_sa = games_df['away_shots'].sum() + games_df['home_shots'].sum()
    total_ga = games_df['away_goals'].sum() + games_df['home_goals'].sum()
    stats['league'] = {
        'home_avg': float(games_df['home_goals'].mean()),
        'away_avg': float(games_df['away_goals'].mean()),
        'avg': float((games_df['home_goals'].mean() + games_df['away_goals'].mean()) / 2),
        'save_pct': float(1 - total_ga / max(total_sa, 1)),
    }

    # ── Team Stats ──
    teams = sorted(set(games_df['home_team']) | set(games_df['away_team']))
    team_stats = {}

    for team in teams:
        h = games_df[games_df['home_team'] == team]  # games as home
        a = games_df[games_df['away_team'] == team]  # games as away
        n = len(h) + len(a)
        if n == 0:
            team_stats[team] = {k: 0.0 for k in
                ['gf','ga','xgf','xga','shots','shots_ag','finish','save_rate',
                 'win_rate','max_xg','assists','pen_comm']}
            continue

        gf_total = h['home_goals'].sum() + a['away_goals'].sum()
        ga_total = h['away_goals'].sum() + a['home_goals'].sum()
        xgf_total = h['home_xg'].sum() + a['away_xg'].sum()
        xga_total = h['away_xg'].sum() + a['home_xg'].sum()
        shots_total = h['home_shots'].sum() + a['away_shots'].sum()
        sa_total = h['away_shots'].sum() + a['home_shots'].sum()
        max_xg_total = h['home_max_xg'].sum() + a['away_max_xg'].sum()
        assists_total = h['home_assists'].sum() + a['away_assists'].sum()
        pen_total = h['home_pen_comm'].sum() + a['away_pen_comm'].sum()
        wins = int((h['home_goals'] > h['away_goals']).sum() +
                   (a['away_goals'] > a['home_goals']).sum())

        team_stats[team] = {
            'gf':        gf_total / n,
            'ga':        ga_total / n,
            'xgf':       xgf_total / n,
            'xga':       xga_total / n,
            'shots':     shots_total / n,
            'shots_ag':  sa_total / n,
            'finish':    gf_total / max(xgf_total, 0.1),
            'save_rate': 1 - ga_total / max(sa_total, 1),
            'win_rate':  wins / n,
            'max_xg':    max_xg_total / n,
            'assists':   assists_total / n,
            'pen_comm':  pen_total / n,
        }
    stats['team'] = team_stats

    # ── Goalie Stats (GSAX = Goals Saved Above Expected) ──
    goalie_stats = {}
    for side, opp_goals, opp_shots, opp_xg in [
        ('home_goalie', 'away_goals', 'away_shots', 'away_xg'),
        ('away_goalie', 'home_goals', 'home_shots', 'home_xg'),
    ]:
        for goalie, grp in games_df.groupby(side):
            g = goalie_stats.setdefault(goalie, {'ga':0, 'sa':0, 'xga':0, 'gp':0})
            g['ga']  += grp[opp_goals].sum()
            g['sa']  += grp[opp_shots].sum()
            g['xga'] += grp[opp_xg].sum()
            g['gp']  += len(grp)

    for goalie, g in goalie_stats.items():
        gp = max(g['gp'], 1)
        g['ga_avg']   = g['ga'] / gp
        g['save_pct'] = 1 - g['ga'] / max(g['sa'], 1)
        g['gsax']     = (g['xga'] - g['ga']) / gp  # positive = better goalie
    stats['goalie'] = goalie_stats

    # ── Head-to-Head History ──
    h2h = {}
    for (ht, at), grp in games_df.groupby(['home_team', 'away_team']):
        h2h[(ht, at)] = {
            'home_goals_avg': float(grp['home_goals'].mean()),
            'away_goals_avg': float(grp['away_goals'].mean()),
            'games': len(grp),
        }
    stats['h2h'] = h2h

    # ── Line Deployment Stats (from shift-level data) ──
    line_stats = {}
    for team in teams:
        # Shifts where this team attacks
        h_shifts = raw_df[raw_df['home_team'] == team]
        a_shifts = raw_df[raw_df['away_team'] == team]

        # Power Play: when this team has PP_up
        pp_h = h_shifts[h_shifts['home_off_line'] == 'PP_up']
        pp_a = a_shifts[a_shifts['away_off_line'] == 'PP_up']
        pp_xg = pp_h['home_xg'].sum() + pp_a['away_xg'].sum()
        pp_goals = pp_h['home_goals'].sum() + pp_a['away_goals'].sum()
        pp_n = len(pp_h) + len(pp_a)

        # Penalty Kill: when this team has PP_kill_dwn
        pk_h = h_shifts[h_shifts['home_off_line'] == 'PP_kill_dwn']
        pk_a = a_shifts[a_shifts['away_off_line'] == 'PP_kill_dwn']
        pk_xga = pk_h['away_xg'].sum() + pk_a['home_xg'].sum()
        pk_ga = pk_h['away_goals'].sum() + pk_a['home_goals'].sum()
        pk_n = len(pk_h) + len(pk_a)

        # First line quality
        fl_h = h_shifts[h_shifts['home_off_line'] == 'first_off']
        fl_a = a_shifts[a_shifts['away_off_line'] == 'first_off']
        fl_xg = fl_h['home_xg'].sum() + fl_a['away_xg'].sum()
        fl_toi = fl_h['toi'].sum() + fl_a['toi'].sum()

        line_stats[team] = {
            'pp_eff':  pp_xg / max(pp_n, 1),           # PP xG per shift
            'pp_goal_rate': pp_goals / max(pp_n, 1),    # PP goals per shift
            'pk_eff':  pk_xga / max(pk_n, 1),           # PK xG against per shift (lower = better)
            'pk_save_rate': 1 - pk_ga / max(pk_n, 1),   # PK save rate
            'first_line_xg_rate': fl_xg / max(fl_toi / 3600, 0.01),  # xG per 60 min
        }
    stats['line'] = line_stats

    # ── Team → Primary Goalie Mapping ──
    team_goalies = {}
    for team in teams:
        counts = {}
        for _, row in games_df[games_df['home_team'] == team].iterrows():
            counts[row['home_goalie']] = counts.get(row['home_goalie'], 0) + 1
        for _, row in games_df[games_df['away_team'] == team].iterrows():
            counts[row['away_goalie']] = counts.get(row['away_goalie'], 0) + 1
        team_goalies[team] = max(counts, key=counts.get) if counts else ''
    stats['goalies'] = team_goalies

    return stats


# ═══════════════════════════════════════════════════════════════
# FEATURE VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'att_gf', 'att_xgf', 'att_finish', 'att_shots', 'att_max_xg', 'att_win_rate',
    'def_ga', 'def_xga', 'def_save_rate', 'def_shots_ag', 'def_win_rate',
    'goalie_gsax', 'goalie_sv',
    'att_pp_eff', 'def_pk_eff', 'att_1st_xg', 'def_pen',
    'h2h_goals', 'h2h_n',
    'is_home', 'venue_avg',
]


def build_features(game, stats, side='home'):
    """
    Build a feature vector for one attack side of a game.

    Parameters
    ----------
    game : dict-like row (from DataFrame or dict)
    stats : dict from compute_stats()
    side : 'home' or 'away'

    Returns
    -------
    dict of feature_name → value
    """
    ts = stats['team']
    gs = stats['goalie']
    h2h = stats['h2h']
    ls = stats['line']
    tg = stats['goalies']
    la = stats['league']

    if side == 'home':
        att_team = game['home_team']
        def_team = game['away_team']
        is_home = 1
        venue_avg = la['home_avg']
        def_goalie = game.get('away_goalie', tg.get(def_team, ''))
        # H2H: look for this exact pairing (att home vs def away)
        h2h_data = h2h.get((att_team, def_team), None)
        h2h_goals_key = 'home_goals_avg'
    else:
        att_team = game['away_team']
        def_team = game['home_team']
        is_home = 0
        venue_avg = la['away_avg']
        def_goalie = game.get('home_goalie', tg.get(def_team, ''))
        # H2H: look for (def home vs att away) — att scored as away
        h2h_data = h2h.get((def_team, att_team), None)
        h2h_goals_key = 'away_goals_avg'

    att = ts.get(att_team, {})
    dfn = ts.get(def_team, {})
    att_line = ls.get(att_team, {})
    def_line = ls.get(def_team, {})
    g = gs.get(def_goalie, {})

    if h2h_data is not None:
        h2h_goals = h2h_data[h2h_goals_key]
        h2h_n = h2h_data['games']
    else:
        h2h_goals = att.get('gf', venue_avg)
        h2h_n = 0

    return {
        'att_gf':       att.get('gf', venue_avg),
        'att_xgf':      att.get('xgf', venue_avg),
        'att_finish':   att.get('finish', 1.0),
        'att_shots':    att.get('shots', 25),
        'att_max_xg':   att.get('max_xg', 0.2),
        'att_win_rate': att.get('win_rate', 0.5),
        'def_ga':       dfn.get('ga', venue_avg),
        'def_xga':      dfn.get('xga', venue_avg),
        'def_save_rate':dfn.get('save_rate', la.get('save_pct', 0.9)),
        'def_shots_ag': dfn.get('shots_ag', 25),
        'def_win_rate': dfn.get('win_rate', 0.5),
        'goalie_gsax':  g.get('gsax', 0.0),
        'goalie_sv':    g.get('save_pct', la.get('save_pct', 0.9)),
        'att_pp_eff':   att_line.get('pp_eff', 0.1),
        'def_pk_eff':   def_line.get('pk_eff', 0.1),
        'att_1st_xg':   att_line.get('first_line_xg_rate', 3.0),
        'def_pen':      dfn.get('pen_comm', 3.0),
        'h2h_goals':    h2h_goals,
        'h2h_n':        h2h_n,
        'is_home':      is_home,
        'venue_avg':    venue_avg,
    }


def build_dataset(games_df, stats):
    """
    Build (X, y) matrix from games.
    Each game produces 2 rows: home attack and away attack.
    """
    rows = []
    targets = []

    for _, game in games_df.iterrows():
        rows.append(build_features(game, stats, 'home'))
        targets.append(game['home_goals'])

        rows.append(build_features(game, stats, 'away'))
        targets.append(game['away_goals'])

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = np.array(targets, dtype=float)
    return X, y


# ═══════════════════════════════════════════════════════════════
# GAME PREDICTOR MODEL
# ═══════════════════════════════════════════════════════════════

class GamePredictor:
    """
    Comprehensive hockey goal predictor.

    Combines team stats, goalie GSAX, xG finishing, H2H history,
    line deployment quality, and Poisson/Ridge/GBR regression.

    Parameters
    ----------
    model_type : str — 'poisson', 'ridge', or 'gbr'
    alpha : float — regularization strength (Poisson/Ridge) or learning_rate (GBR)
    """

    def __init__(self, model_type='poisson', alpha=1.0):
        self.model_type = model_type
        self.alpha = alpha
        self.model = None
        self.scaler = StandardScaler()
        self.stats = None
        self.feature_names = FEATURE_NAMES
        self.is_fitted = False

    def _make_model(self):
        if self.model_type == 'poisson':
            return PoissonRegressor(alpha=self.alpha, max_iter=2000)
        elif self.model_type == 'ridge':
            return Ridge(alpha=self.alpha)
        elif self.model_type == 'gbr':
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=3,
                learning_rate=self.alpha, min_samples_leaf=10,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, raw_df, games_df=None):
        """Train on raw shift-level data (or pre-aggregated games)."""
        if games_df is None:
            games_df = aggregate_games(raw_df)

        self.stats = compute_stats(raw_df, games_df)
        X, y = build_dataset(games_df, self.stats)

        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X)

        self.model = self._make_model()
        self.model.fit(X_sc, y)
        self.is_fitted = True
        return self

    def predict(self, game):
        """Predict (home_goals, away_goals) for a single game."""
        fh = build_features(game, self.stats, 'home')
        fa = build_features(game, self.stats, 'away')

        X = pd.DataFrame([fh, fa], columns=self.feature_names)
        X_sc = self.scaler.transform(X)
        preds = self.model.predict(X_sc)

        return float(max(0.3, preds[0])), float(max(0.3, preds[1]))

    def predict_winner(self, game):
        """Predict winner and confidence."""
        hp, ap = self.predict(game)
        if hp >= ap:
            return game['home_team'], hp / (hp + ap)
        else:
            return game['away_team'], ap / (hp + ap)

    def evaluate(self, games_df):
        """Evaluate predictions on a set of games."""
        hp_list, ap_list, ha_list, aa_list = [], [], [], []
        correct = 0
        for _, g in games_df.iterrows():
            hp, ap = self.predict(g)
            hp_list.append(hp)
            ap_list.append(ap)
            ha_list.append(g['home_goals'])
            aa_list.append(g['away_goals'])
            if (hp > ap) == (g['home_goals'] > g['away_goals']):
                correct += 1

        n = len(ha_list)
        all_act = ha_list + aa_list
        all_pred = hp_list + ap_list
        return {
            'combined_rmse': float(np.sqrt(mean_squared_error(all_act, all_pred))),
            'home_rmse':     float(np.sqrt(mean_squared_error(ha_list, hp_list))),
            'away_rmse':     float(np.sqrt(mean_squared_error(aa_list, ap_list))),
            'combined_mae':  float(mean_absolute_error(all_act, all_pred)),
            'home_r2':       float(r2_score(ha_list, hp_list)),
            'away_r2':       float(r2_score(aa_list, ap_list)),
            'win_accuracy':  correct / n if n > 0 else 0,
            'pred_home_std': float(np.std(hp_list)),
            'pred_away_std': float(np.std(ap_list)),
            'n_games': n,
        }

    def cross_validate(self, raw_df, k=5, seed=42):
        """
        K-fold cross-validation with proper fold isolation.

        For each fold:
        1. Compute stats from TRAINING folds only (no leakage)
        2. Build features for validation games using training stats
        3. Predict and evaluate
        """
        games_df = aggregate_games(raw_df)
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        all_hp, all_ap, all_ha, all_aa = [], [], [], []
        fold_results = []

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(games_df)):
            train_g = games_df.iloc[train_idx]
            val_g = games_df.iloc[val_idx]
            train_ids = set(train_g['game_id'])
            train_raw = raw_df[raw_df['game_id'].isin(train_ids)]

            # Stats from training data only
            fold_stats = compute_stats(train_raw, train_g)

            # Build training features and fit
            X_train, y_train = build_dataset(train_g, fold_stats)
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            model = self._make_model()
            model.fit(X_tr_sc, y_train)

            # Predict validation games
            fold_correct = 0
            fold_hp, fold_ap, fold_ha, fold_aa = [], [], [], []
            for _, game in val_g.iterrows():
                fh = build_features(game, fold_stats, 'home')
                fa = build_features(game, fold_stats, 'away')
                X_val = pd.DataFrame([fh, fa], columns=self.feature_names)
                X_val_sc = scaler.transform(X_val)
                preds = model.predict(X_val_sc)
                hp, ap = max(0.3, preds[0]), max(0.3, preds[1])

                fold_hp.append(hp); fold_ap.append(ap)
                fold_ha.append(game['home_goals']); fold_aa.append(game['away_goals'])
                if (hp > ap) == (game['home_goals'] > game['away_goals']):
                    fold_correct += 1

            all_hp.extend(fold_hp); all_ap.extend(fold_ap)
            all_ha.extend(fold_ha); all_aa.extend(fold_aa)

            fold_all_act = fold_ha + fold_aa
            fold_all_pred = fold_hp + fold_ap
            fold_rmse = float(np.sqrt(mean_squared_error(fold_all_act, fold_all_pred)))
            fold_wa = fold_correct / len(val_g) if len(val_g) > 0 else 0
            fold_results.append({'fold': fold_i+1, 'rmse': fold_rmse,
                                 'win_accuracy': fold_wa, 'n': len(val_g)})

        all_act = all_ha + all_aa
        all_pred = all_hp + all_ap
        overall_correct = sum(1 for h,a,ah,aa in zip(all_hp,all_ap,all_ha,all_aa)
                              if (h>a)==(ah>aa))

        return {
            'cv_rmse':          float(np.sqrt(mean_squared_error(all_act, all_pred))),
            'cv_home_rmse':     float(np.sqrt(mean_squared_error(all_ha, all_hp))),
            'cv_away_rmse':     float(np.sqrt(mean_squared_error(all_aa, all_ap))),
            'cv_mae':           float(mean_absolute_error(all_act, all_pred)),
            'cv_win_accuracy':  overall_correct / len(all_ha),
            'cv_home_r2':       float(r2_score(all_ha, all_hp)),
            'cv_away_r2':       float(r2_score(all_aa, all_ap)),
            'fold_results':     fold_results,
            'k': k,
            'n_games': len(all_ha),
            # Return raw predictions for ensemble blending
            '_preds': {'hp': all_hp, 'ap': all_ap, 'ha': all_ha, 'aa': all_aa},
        }

    def feature_importance(self):
        """Return feature importances (from fitted model)."""
        if not self.is_fitted:
            return {}
        if self.model_type in ('poisson', 'ridge'):
            coefs = self.model.coef_
            imp = sorted(zip(self.feature_names, coefs),
                         key=lambda x: abs(x[1]), reverse=True)
            return imp
        elif self.model_type == 'gbr':
            importances = self.model.feature_importances_
            imp = sorted(zip(self.feature_names, importances),
                         key=lambda x: x[1], reverse=True)
            return imp
        return {}

    def save(self, path):
        """Save trained model."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        pkl_path = path if path.endswith('.pkl') else path + '.pkl'
        state = {
            'model_type': self.model_type,
            'alpha': self.alpha,
            'model': self.model,
            'scaler': self.scaler,
            'stats': self.stats,
            'feature_names': self.feature_names,
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"GamePredictor saved to {pkl_path}")
        return pkl_path

    @classmethod
    def load(cls, path):
        """Load a saved model."""
        pkl_path = path if path.endswith('.pkl') else path + '.pkl'
        with open(pkl_path, 'rb') as f:
            state = pickle.load(f)
        obj = cls(state['model_type'], state['alpha'])
        obj.model = state['model']
        obj.scaler = state['scaler']
        obj.stats = state['stats']
        obj.feature_names = state['feature_names']
        obj.is_fitted = True
        print(f"GamePredictor loaded from {pkl_path}")
        return obj


# ═══════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════

def grid_search_cv(raw_df, k=5, seed=42, verbose=True):
    """
    Grid search over model types and regularization strengths.

    Tests:
    - PoissonRegressor with alpha ∈ [0.001, 0.01, 0.1, 1.0, 10.0]
    - Ridge with alpha ∈ [0.01, 0.1, 1.0, 10.0, 100.0]
    - GradientBoosting with lr ∈ [0.01, 0.05, 0.1]

    Returns sorted DataFrame of results.
    """
    configs = []
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        configs.append(('poisson', alpha))
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        configs.append(('ridge', alpha))
    for lr in [0.01, 0.05, 0.1]:
        configs.append(('gbr', lr))

    results = []
    for i, (mt, alpha) in enumerate(configs):
        if verbose:
            print(f"  [{i+1}/{len(configs)}] {mt} alpha={alpha} ...", end='', flush=True)
        pred = GamePredictor(model_type=mt, alpha=alpha)
        cv = pred.cross_validate(raw_df, k=k, seed=seed)
        results.append({
            'model_type': mt,
            'alpha': alpha,
            'cv_rmse': cv['cv_rmse'],
            'cv_mae': cv['cv_mae'],
            'cv_win_accuracy': cv['cv_win_accuracy'],
            'cv_home_r2': cv['cv_home_r2'],
            'cv_away_r2': cv['cv_away_r2'],
        })
        if verbose:
            print(f" RMSE={cv['cv_rmse']:.4f}  WA={cv['cv_win_accuracy']:.1%}")

    results_df = pd.DataFrame(results).sort_values('cv_rmse').reset_index(drop=True)
    return results_df


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE WITH ELO
# ═══════════════════════════════════════════════════════════════

def ensemble_with_elo(raw_df, predictor, elo_model, k=5, seed=42,
                      weights=None):
    """
    Find optimal blend weight between GamePredictor and EnhancedEloModel
    using K-fold CV.

    Tests weights from 0.0 (pure GamePredictor) to 1.0 (pure Elo).
    Returns best weight and blended CV metrics.
    """
    if weights is None:
        weights = [round(w * 0.1, 1) for w in range(11)]  # 0.0 to 1.0

    games_df = aggregate_games(raw_df)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    # Collect CV predictions from both models
    gp_hp, gp_ap = [], []  # GamePredictor preds
    elo_hp, elo_ap = [], []  # Elo preds
    actual_h, actual_a = [], []

    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]
        train_ids = set(train_g['game_id'])
        train_raw = raw_df[raw_df['game_id'].isin(train_ids)]

        # GamePredictor
        fold_stats = compute_stats(train_raw, train_g)
        X_train, y_train = build_dataset(train_g, fold_stats)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        gp_model = predictor._make_model()
        gp_model.fit(X_tr_sc, y_train)

        # Elo
        from utils.enhanced_elo_model import EnhancedEloModel
        elo = EnhancedEloModel(elo_model.params)
        elo.fit(train_g)

        for _, game in val_g.iterrows():
            # GP prediction
            fh = build_features(game, fold_stats, 'home')
            fa = build_features(game, fold_stats, 'away')
            X_val = pd.DataFrame([fh, fa], columns=FEATURE_NAMES)
            X_val_sc = scaler.transform(X_val)
            gp_preds = gp_model.predict(X_val_sc)
            gp_hp.append(max(0.3, gp_preds[0]))
            gp_ap.append(max(0.3, gp_preds[1]))

            # Elo prediction
            eh, ea = elo.predict_goals(game)
            elo_hp.append(eh)
            elo_ap.append(ea)

            actual_h.append(game['home_goals'])
            actual_a.append(game['away_goals'])

    # Try different blend weights
    best_w, best_rmse = 0.0, 999
    weight_results = []
    for w in weights:
        blend_hp = [w * e + (1-w) * g for e, g in zip(elo_hp, gp_hp)]
        blend_ap = [w * e + (1-w) * g for e, g in zip(elo_ap, gp_ap)]
        all_act = actual_h + actual_a
        all_pred = blend_hp + blend_ap
        rmse = float(np.sqrt(mean_squared_error(all_act, all_pred)))
        correct = sum(1 for h,a,ah,aa in zip(blend_hp, blend_ap, actual_h, actual_a)
                       if (h>a) == (ah>aa))
        wa = correct / len(actual_h)
        weight_results.append({'elo_weight': w, 'rmse': rmse, 'win_accuracy': wa})
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w

    return {
        'best_elo_weight': best_w,
        'best_blend_rmse': best_rmse,
        'weight_results': weight_results,
        'gp_only_rmse': weight_results[0]['rmse'],    # w=0
        'elo_only_rmse': weight_results[-1]['rmse'],   # w=1
    }
