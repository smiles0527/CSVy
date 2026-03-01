"""
Focused Model Optimization v2
==============================
KEY FINDING: Competitor simulation showed Ridge(6 simple features) = 1.7268
beats our Poisson(21 features) + Elo blend = 1.7364.

ROOT CAUSE: The competitor Ridge used INLINE team stats (GF/GA/xGF/xGA/SF/SA),
not our game_predictor's complex features. Our features add noise for Ridge.
BUT: With fold isolation (correct CV), Ridge on our features = 2.0+ (bad).

HYPOTHESIS: The competitor Ridge was potentially leaking (stats computed on
full data, not just training fold). Let's verify and find the true best.
"""
import io, sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
_script = Path(__file__).resolve()
_python_dir = _script.parent
while True:
    if (_python_dir / 'utils').is_dir():
        break
    parent = _python_dir.parent
    if parent == _python_dir:
        raise RuntimeError('Cannot locate python/')
    _python_dir = parent
os.chdir(_python_dir)
sys.path.insert(0, str(_python_dir))

from utils.game_predictor import (
    aggregate_games, compute_stats, build_features, build_dataset, FEATURE_NAMES,
)
from utils.enhanced_elo_model import EnhancedEloModel

raw = pd.read_csv('data/whl_2025.csv')
games_df = aggregate_games(raw)

print("=" * 70)
print("  FOCUSED MODEL OPTIMIZATION v2")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# TEST 1: Verify the "simple Ridge" is legit with proper fold isolation
# ═══════════════════════════════════════════════════════════════
print(f"\n--- TEST 1: Simple Ridge with PROPER fold isolation ---\n")

def simple_ridge_cv(alpha=10, k=5, seed=42):
    """The exact same approach as competitor Tier 2, with strict fold isolation."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []

    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        # Compute SIMPLE team stats from training data ONLY
        teams = sorted(set(train_g['home_team']) | set(train_g['away_team']))
        team_map = {}
        for team in teams:
            h = train_g[train_g['home_team'] == team]
            a = train_g[train_g['away_team'] == team]
            n = len(h) + len(a)
            if n == 0:
                team_map[team] = [3.0]*6
                continue
            gf = (h['home_goals'].sum() + a['away_goals'].sum()) / n
            ga = (h['away_goals'].sum() + a['home_goals'].sum()) / n
            xgf = (h['home_xg'].sum() + a['away_xg'].sum()) / n
            xga = (h['away_xg'].sum() + a['home_xg'].sum()) / n
            sf = (h['home_shots'].sum() + a['away_shots'].sum()) / n
            sa = (h['away_shots'].sum() + a['home_shots'].sum()) / n
            team_map[team] = [gf, ga, xgf, xga, sf, sa]

        X_train, y_train = [], []
        for _, g in train_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            X_train.append([ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1])
            y_train.append(g['home_goals'])
            X_train.append([at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0])
            y_train.append(g['away_goals'])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_train)
        model = Ridge(alpha=alpha)
        model.fit(X_sc, y_train)

        for _, g in val_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            X_h = scaler.transform([[ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1]])
            X_a = scaler.transform([[at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0]])
            hp = max(0.3, model.predict(X_h)[0])
            ap = max(0.3, model.predict(X_a)[0])
            all_pred.extend([hp, ap])
            all_act.extend([g['home_goals'], g['away_goals']])

    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


print(f"  Simple Ridge (6 features, fold isolation):")
for alpha in [1, 5, 10, 20, 50, 100, 200, 500]:
    rmse = simple_ridge_cv(alpha=alpha)
    print(f"    alpha={alpha:4d}  RMSE={rmse:.4f}")


# ═══════════════════════════════════════════════════════════════
# TEST 2: Simple POISSON with same 6 features
# ═══════════════════════════════════════════════════════════════
print(f"\n--- TEST 2: Simple Poisson with 6 features ---\n")

def simple_poisson_cv(alpha=10, k=5, seed=42):
    """Poisson with simple 6-feature team stats (fold isolated)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []

    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        teams = sorted(set(train_g['home_team']) | set(train_g['away_team']))
        team_map = {}
        for team in teams:
            h = train_g[train_g['home_team'] == team]
            a = train_g[train_g['away_team'] == team]
            n = len(h) + len(a)
            if n == 0:
                team_map[team] = [3.0]*6
                continue
            gf = (h['home_goals'].sum() + a['away_goals'].sum()) / n
            ga = (h['away_goals'].sum() + a['home_goals'].sum()) / n
            xgf = (h['home_xg'].sum() + a['away_xg'].sum()) / n
            xga = (h['away_xg'].sum() + a['home_xg'].sum()) / n
            sf = (h['home_shots'].sum() + a['away_shots'].sum()) / n
            sa = (h['away_shots'].sum() + a['home_shots'].sum()) / n
            team_map[team] = [gf, ga, xgf, xga, sf, sa]

        X_train, y_train = [], []
        for _, g in train_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            X_train.append([ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1])
            y_train.append(g['home_goals'])
            X_train.append([at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0])
            y_train.append(g['away_goals'])

        X_train = np.array(X_train)
        y_train = np.array(y_train, dtype=float)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_train)
        # Poisson needs positive targets — they already are (goals >= 0)
        model = PoissonRegressor(alpha=alpha, max_iter=3000)
        model.fit(X_sc, y_train)

        for _, g in val_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            X_h = scaler.transform([[ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1]])
            X_a = scaler.transform([[at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0]])
            hp = max(0.3, model.predict(X_h)[0])
            ap = max(0.3, model.predict(X_a)[0])
            all_pred.extend([hp, ap])
            all_act.extend([g['home_goals'], g['away_goals']])

    return float(np.sqrt(mean_squared_error(all_act, all_pred)))

print(f"  Simple Poisson (6 features, fold isolation):")
for alpha in [0.1, 0.5, 1, 5, 10, 20, 50]:
    rmse = simple_poisson_cv(alpha=alpha)
    print(f"    alpha={alpha:5.1f}  RMSE={rmse:.4f}")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Our GamePredictor features (21) — Poisson grid search
# ═══════════════════════════════════════════════════════════════
print(f"\n--- TEST 3: GP features (21) — Poisson alpha sweep ---\n")

def gp_poisson_cv(alpha=20, feature_cols=None, k=5, seed=42):
    """Poisson on our GP features with proper fold isolation."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []

    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        train_ids = set(train_g['game_id'])
        train_raw = raw[raw['game_id'].isin(train_ids)]
        stats = compute_stats(train_raw, train_g)

        X_train, y_train = build_dataset(train_g, stats)
        if feature_cols:
            X_train = X_train[feature_cols]

        sc = StandardScaler()
        X_t = sc.fit_transform(X_train)
        model = PoissonRegressor(alpha=alpha, max_iter=3000)
        model.fit(X_t, y_train)

        for _, g in val_g.iterrows():
            fh = build_features(g, stats, 'home')
            fa = build_features(g, stats, 'away')
            Xv = pd.DataFrame([fh, fa], columns=FEATURE_NAMES)
            if feature_cols:
                Xv = Xv[feature_cols]
            Xv_s = sc.transform(Xv)
            p = model.predict(Xv_s)
            all_pred.extend([max(0.3, p[0]), max(0.3, p[1])])
            all_act.extend([g['home_goals'], g['away_goals']])

    return float(np.sqrt(mean_squared_error(all_act, all_pred)))

print(f"  GP Poisson (all 21 features):")
for alpha in [5, 10, 15, 20, 25, 30, 40, 50]:
    rmse = gp_poisson_cv(alpha=alpha)
    print(f"    alpha={alpha:3d}  RMSE={rmse:.4f}")


# Now test removing H2H features (they seemed harmful)
print(f"\n  GP Poisson WITHOUT H2H:")
no_h2h = [f for f in FEATURE_NAMES if f not in ['h2h_goals', 'h2h_n']]
for alpha in [10, 15, 20, 25, 30]:
    rmse = gp_poisson_cv(alpha=alpha, feature_cols=no_h2h)
    print(f"    alpha={alpha:3d}  RMSE={rmse:.4f}")

# Without H2H and without Max_xG (low correlation)
print(f"\n  GP Poisson WITHOUT H2H, max_xg:")
reduced = [f for f in FEATURE_NAMES if f not in ['h2h_goals', 'h2h_n', 'att_max_xg']]
for alpha in [10, 15, 20, 25]:
    rmse = gp_poisson_cv(alpha=alpha, feature_cols=reduced)
    print(f"    alpha={alpha:3d}  RMSE={rmse:.4f}")


# ═══════════════════════════════════════════════════════════════
# TEST 4: Blend optimization
# ═══════════════════════════════════════════════════════════════
print(f"\n--- TEST 4: Blend weight optimization ---\n")

def blend_cv(gp_alpha, elo_weight, feature_cols=None, k=5, seed=42):
    """GP + Elo blend."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []

    with open('output/predictions/elo/elo_pipeline_summary.json') as f:
        params = json.load(f)['best_params']

    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        # GP
        train_ids = set(train_g['game_id'])
        train_raw = raw[raw['game_id'].isin(train_ids)]
        stats = compute_stats(train_raw, train_g)
        X_train, y_train = build_dataset(train_g, stats)
        if feature_cols:
            X_train = X_train[feature_cols]
        sc = StandardScaler()
        X_t = sc.fit_transform(X_train)
        gp_model = PoissonRegressor(alpha=gp_alpha, max_iter=3000)
        gp_model.fit(X_t, y_train)

        # Elo
        elo = EnhancedEloModel(params)
        elo.fit(train_g)

        for _, g in val_g.iterrows():
            fh = build_features(g, stats, 'home')
            fa = build_features(g, stats, 'away')
            Xv = pd.DataFrame([fh, fa], columns=FEATURE_NAMES)
            if feature_cols:
                Xv = Xv[feature_cols]
            Xv_s = sc.transform(Xv)
            gp_p = gp_model.predict(Xv_s)
            elo_h, elo_a = elo.predict_goals(g)

            gp_w = 1.0 - elo_weight
            bh = gp_w * max(0.3, gp_p[0]) + elo_weight * elo_h
            ba = gp_w * max(0.3, gp_p[1]) + elo_weight * elo_a
            all_pred.extend([bh, ba])
            all_act.extend([g['home_goals'], g['away_goals']])

    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


print(f"  GP(20) + Elo blend weights (all 21 features):")
for ew in np.arange(0.0, 0.65, 0.05):
    rmse = blend_cv(20, ew)
    marker = " <<<" if abs(ew - 0.5) < 0.01 else ""
    print(f"    Elo={ew:.2f}  GP={1-ew:.2f}  RMSE={rmse:.4f}{marker}")

print(f"\n  GP(20) + Elo blend (NO H2H features):")
for ew in np.arange(0.0, 0.65, 0.1):
    rmse = blend_cv(20, ew, feature_cols=no_h2h)
    print(f"    Elo={ew:.2f}  GP={1-ew:.2f}  RMSE={rmse:.4f}")


# ═══════════════════════════════════════════════════════════════
# TEST 5: Simple features + Elo blend
# ═══════════════════════════════════════════════════════════════
print(f"\n--- TEST 5: Simple Ridge/Poisson + Elo blend ---\n")

def simple_elo_blend(model_type='ridge', alpha=10, elo_weight=0.3, k=5, seed=42):
    """Simple features model + Elo blend."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []

    with open('output/predictions/elo/elo_pipeline_summary.json') as f:
        params = json.load(f)['best_params']

    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        # Simple team stats
        teams = sorted(set(train_g['home_team']) | set(train_g['away_team']))
        team_map = {}
        for team in teams:
            h = train_g[train_g['home_team'] == team]
            a = train_g[train_g['away_team'] == team]
            n = len(h) + len(a)
            if n == 0:
                team_map[team] = [3.0]*6
                continue
            gf = (h['home_goals'].sum() + a['away_goals'].sum()) / n
            ga = (h['away_goals'].sum() + a['home_goals'].sum()) / n
            xgf = (h['home_xg'].sum() + a['away_xg'].sum()) / n
            xga = (h['away_xg'].sum() + a['home_xg'].sum()) / n
            sf = (h['home_shots'].sum() + a['away_shots'].sum()) / n
            sa = (h['away_shots'].sum() + a['home_shots'].sum()) / n
            team_map[team] = [gf, ga, xgf, xga, sf, sa]

        X_train, y_train = [], []
        for _, g in train_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            X_train.append([ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1])
            y_train.append(g['home_goals'])
            X_train.append([at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0])
            y_train.append(g['away_goals'])

        X_train = np.array(X_train)
        y_train = np.array(y_train, dtype=float)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_train)

        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        else:
            model = PoissonRegressor(alpha=alpha, max_iter=3000)
        model.fit(X_sc, y_train)

        # Elo
        elo = EnhancedEloModel(params)
        elo.fit(train_g)

        for _, g in val_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            X_h = scaler.transform([[ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1]])
            X_a = scaler.transform([[at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0]])
            hp = max(0.3, model.predict(X_h)[0])
            ap = max(0.3, model.predict(X_a)[0])

            elo_h, elo_a = elo.predict_goals(g)
            gp_w = 1.0 - elo_weight
            bh = gp_w * hp + elo_weight * elo_h
            ba = gp_w * ap + elo_weight * elo_a
            all_pred.extend([bh, ba])
            all_act.extend([g['home_goals'], g['away_goals']])

    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


print(f"  Simple Ridge(10) + Elo blend:")
for ew in np.arange(0.0, 0.65, 0.1):
    rmse = simple_elo_blend('ridge', 10, ew)
    print(f"    Elo={ew:.1f}  RMSE={rmse:.4f}")

print(f"\n  Simple Poisson(5) + Elo blend:")
for ew in np.arange(0.0, 0.65, 0.1):
    rmse = simple_elo_blend('poisson', 5, ew)
    print(f"    Elo={ew:.1f}  RMSE={rmse:.4f}")


# ═══════════════════════════════════════════════════════════════
# TEST 6: THREE-way blend (Simple + GP + Elo)
# ═══════════════════════════════════════════════════════════════
print(f"\n--- TEST 6: THREE-way blend (Simple + GP + Elo) ---\n")

def three_way_blend(gp_alpha, simple_alpha, w_gp, w_elo, k=5, seed=42):
    """Three-way: simple Ridge + GP Poisson + Elo."""
    w_simple = 1.0 - w_gp - w_elo
    if w_simple < -0.01:
        return 99.0

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []

    with open('output/predictions/elo/elo_pipeline_summary.json') as f:
        params = json.load(f)['best_params']

    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        # --- SIMPLE model ---
        teams = sorted(set(train_g['home_team']) | set(train_g['away_team']))
        team_map = {}
        for team in teams:
            h = train_g[train_g['home_team'] == team]
            a = train_g[train_g['away_team'] == team]
            n = len(h) + len(a)
            if n == 0:
                team_map[team] = [3.0]*6
                continue
            gf = (h['home_goals'].sum() + a['away_goals'].sum()) / n
            ga = (h['away_goals'].sum() + a['home_goals'].sum()) / n
            xgf = (h['home_xg'].sum() + a['away_xg'].sum()) / n
            xga = (h['away_xg'].sum() + a['home_xg'].sum()) / n
            sf = (h['home_shots'].sum() + a['away_shots'].sum()) / n
            sa = (h['away_shots'].sum() + a['home_shots'].sum()) / n
            team_map[team] = [gf, ga, xgf, xga, sf, sa]

        Xs_train, ys_train = [], []
        for _, g in train_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            Xs_train.append([ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1])
            ys_train.append(g['home_goals'])
            Xs_train.append([at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0])
            ys_train.append(g['away_goals'])
        sc_simple = StandardScaler()
        Xs_sc = sc_simple.fit_transform(Xs_train)
        simple_model = Ridge(alpha=simple_alpha)
        simple_model.fit(Xs_sc, ys_train)

        # --- GP model ---
        train_ids = set(train_g['game_id'])
        train_raw = raw[raw['game_id'].isin(train_ids)]
        stats = compute_stats(train_raw, train_g)
        X_gp_train, y_gp_train = build_dataset(train_g, stats)
        no_h2h_cols = [f for f in FEATURE_NAMES if f not in ['h2h_goals', 'h2h_n']]
        X_gp_train = X_gp_train[no_h2h_cols]
        sc_gp = StandardScaler()
        Xgp_sc = sc_gp.fit_transform(X_gp_train)
        gp_model = PoissonRegressor(alpha=gp_alpha, max_iter=3000)
        gp_model.fit(Xgp_sc, y_gp_train)

        # --- Elo ---
        elo = EnhancedEloModel(params)
        elo.fit(train_g)

        for _, g in val_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            Xh_s = sc_simple.transform([[ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1]])
            Xa_s = sc_simple.transform([[at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0]])
            sh = max(0.3, simple_model.predict(Xh_s)[0])
            sa = max(0.3, simple_model.predict(Xa_s)[0])

            fh = build_features(g, stats, 'home')
            fa = build_features(g, stats, 'away')
            Xv = pd.DataFrame([fh, fa], columns=FEATURE_NAMES)[no_h2h_cols]
            Xv_s = sc_gp.transform(Xv)
            gp_p = gp_model.predict(Xv_s)
            gh, ga_ = max(0.3, gp_p[0]), max(0.3, gp_p[1])

            elo_h, elo_a = elo.predict_goals(g)

            bh = w_simple * sh + w_gp * gh + w_elo * elo_h
            ba = w_simple * sa + w_gp * ga_ + w_elo * elo_a
            all_pred.extend([bh, ba])
            all_act.extend([g['home_goals'], g['away_goals']])

    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


# Grid search over 3-way weights
print(f"  Three-way blend: Simple Ridge(10) + GP Poisson(20) + Elo")
print(f"  {'w_simple':>9s}  {'w_gp':>6s}  {'w_elo':>6s}  {'RMSE':>8s}")
print(f"  {'-'*35}")

three_results = []
for w_gp in np.arange(0.0, 0.8, 0.1):
    for w_elo in np.arange(0.0, 0.8 - w_gp, 0.1):
        w_s = round(1.0 - w_gp - w_elo, 2)
        if w_s < -0.01:
            continue
        rmse = three_way_blend(20, 10, w_gp, w_elo)
        three_results.append((w_s, w_gp, w_elo, rmse))

three_results.sort(key=lambda x: x[3])
for w_s, w_gp, w_elo, rmse in three_results[:15]:
    print(f"  {w_s:9.1f}  {w_gp:6.1f}  {w_elo:6.1f}  {rmse:8.4f}")

print(f"\n  Best: Simple={three_results[0][0]:.1f}, GP={three_results[0][1]:.1f}, "
      f"Elo={three_results[0][2]:.1f}  RMSE={three_results[0][3]:.4f}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  FINAL COMPARISON")
print(f"{'='*70}")

# Collect all results
all_results = []

# Constant
kf = KFold(n_splits=5, shuffle=True, random_state=42)
c_pred, c_act = [], []
for train_idx, val_idx in kf.split(games_df):
    train_g = games_df.iloc[train_idx]
    val_g = games_df.iloc[val_idx]
    hm, am = train_g['home_goals'].mean(), train_g['away_goals'].mean()
    for _, g in val_g.iterrows():
        c_pred.extend([hm, am])
        c_act.extend([g['home_goals'], g['away_goals']])
c_rmse = float(np.sqrt(mean_squared_error(c_act, c_pred)))
all_results.append(("Constant (naive)", c_rmse))

# Simple Ridge
sr_rmse = simple_ridge_cv(10)
all_results.append(("Simple Ridge(10)", sr_rmse))

# Simple Poisson
sp_rmse = simple_poisson_cv(5)
all_results.append(("Simple Poisson(5)", sp_rmse))

# GP Poisson
gp_rmse = gp_poisson_cv(20)
all_results.append(("GP Poisson(20) 21 feat", gp_rmse))

# GP Poisson no H2H
gp_nh_rmse = gp_poisson_cv(20, feature_cols=no_h2h)
all_results.append(("GP Poisson(20) no H2H", gp_nh_rmse))

# GP+Elo 50/50
gp_elo_rmse = blend_cv(20, 0.5)
all_results.append(("GP(20)+Elo 50/50", gp_elo_rmse))

# Best blend
best_tw = three_results[0]
all_results.append((f"Simple+GP+Elo ({best_tw[0]:.0f}/{best_tw[1]:.0f}/{best_tw[2]:.0f})", best_tw[3]))

print(f"\n  {'Approach':<40s}  {'RMSE':>8s}  {'vs Best':>9s}")
print(f"  {'-'*60}")
all_results.sort(key=lambda x: x[1])
for name, rmse in all_results:
    diff = rmse - all_results[0][1]
    marker = " <<<" if diff == 0 else ""
    print(f"  {name:<40s}  {rmse:8.4f}  {diff:+9.4f}{marker}")

print(f"\n  OVERALL BEST: {all_results[0][0]} = {all_results[0][1]:.4f}")
if all_results[0][1] < 1.7364:
    print(f"  IMPROVEMENT over previous: {1.7364 - all_results[0][1]:+.4f}")
else:
    print(f"  No improvement over previous: {all_results[0][1] - 1.7364:+.4f}")
print(f"{'='*70}")
