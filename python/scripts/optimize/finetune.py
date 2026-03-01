"""
Fine-tune the optimal model configuration.
Key findings so far:
  - Simple Ridge(500, 6 features) = 1.7211  <- possibly best alone
  - Three-way (0.6S + 0.3GP + 0.1E) = 1.7218
  - Simple Poisson(1) = 1.7219

Now: fine-tune Ridge alpha, test higher alphas, test finer blend grid,
test Simple Ridge(high alpha) + small Elo.
"""
import io, sys, os, json
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
print("  FINE-TUNING OPTIMAL MODEL")
print("=" * 70)


def simple_ridge_cv(alpha=10, k=5, seed=42):
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


# ── Fine-tune Ridge alpha ──
print(f"\n  Ridge alpha fine-tune:")
for alpha in [100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 3000, 5000, 10000]:
    rmses = [simple_ridge_cv(alpha, seed=s) for s in range(5)]
    mean_r = np.mean(rmses)
    std_r = np.std(rmses)
    print(f"    alpha={alpha:6d}  mean={mean_r:.4f}  std={std_r:.4f}")


# ── Extended features for Ridge ──
print(f"\n  Ridge with EXTENDED features (add goalie, xg efficiency):")

def extended_ridge_cv(alpha=500, k=5, seed=42, use_goalie=True, use_xg_eff=True,
                       use_pp=True, use_save_rate=True):
    """Ridge with more features extracted simply from game-level data."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []
    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        # Build team stats
        team_stats = {}
        for team in set(train_g['home_team']) | set(train_g['away_team']):
            h = train_g[train_g['home_team'] == team]
            a = train_g[train_g['away_team'] == team]
            n = len(h) + len(a)
            if n == 0:
                team_stats[team] = {'gf': 3, 'ga': 3, 'xgf': 3, 'xga': 3, 'sf': 25, 'sa': 25,
                                    'xg_eff': 1.0, 'save_rate': 0.9, 'ppg': 0.5, 'pkg': 0.5}
                continue
            gf = (h['home_goals'].sum() + a['away_goals'].sum()) / n
            ga = (h['away_goals'].sum() + a['home_goals'].sum()) / n
            xgf = (h['home_xg'].sum() + a['away_xg'].sum()) / n
            xga = (h['away_xg'].sum() + a['home_xg'].sum()) / n
            sf = (h['home_shots'].sum() + a['away_shots'].sum()) / n
            sa = (h['away_shots'].sum() + a['home_shots'].sum()) / n

            # xG efficiency = actual goals / expected goals
            xg_eff = gf / max(xgf, 0.01)

            # Save rate = 1 - (GA / shots against)
            save_rate = 1 - ga / max(sa, 0.01) if sa > 0 else 0.9

            # PP/PK efficiency from game-level data
            ppg = (h['home_ppg'].sum() + a['away_ppg'].sum()) / n if 'home_ppg' in train_g.columns else 0.5
            pkg = (h['away_ppg'].sum() + a['home_ppg'].sum()) / n if 'away_ppg' in train_g.columns else 0.5

            team_stats[team] = {'gf': gf, 'ga': ga, 'xgf': xgf, 'xga': xga,
                                'sf': sf, 'sa': sa, 'xg_eff': xg_eff,
                                'save_rate': save_rate, 'ppg': ppg, 'pkg': pkg}

        def make_features(att_team, def_team, is_home):
            att = team_stats.get(att_team, team_stats.get(list(team_stats.keys())[0]))
            def_ = team_stats.get(def_team, team_stats.get(list(team_stats.keys())[0]))
            feats = [att['gf'], att['xgf'], att['sf'], def_['ga'], def_['xga'], is_home]
            if use_xg_eff:
                feats.append(att['xg_eff'])
            if use_save_rate:
                feats.append(def_['save_rate'])
            if use_pp:
                feats.extend([att['ppg'], def_['pkg']])
            return feats

        X_train, y_train = [], []
        for _, g in train_g.iterrows():
            X_train.append(make_features(g['home_team'], g['away_team'], 1))
            y_train.append(g['home_goals'])
            X_train.append(make_features(g['away_team'], g['home_team'], 0))
            y_train.append(g['away_goals'])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_train)
        model = Ridge(alpha=alpha)
        model.fit(X_sc, y_train)

        for _, g in val_g.iterrows():
            fh = make_features(g['home_team'], g['away_team'], 1)
            fa = make_features(g['away_team'], g['home_team'], 0)
            X_v = scaler.transform([fh, fa])
            p = model.predict(X_v)
            all_pred.extend([max(0.3, p[0]), max(0.3, p[1])])
            all_act.extend([g['home_goals'], g['away_goals']])
    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


feature_configs = [
    ("6 basic",         dict(use_goalie=False, use_xg_eff=False, use_pp=False, use_save_rate=False)),
    ("+ xg_eff",        dict(use_goalie=False, use_xg_eff=True,  use_pp=False, use_save_rate=False)),
    ("+ save_rate",     dict(use_goalie=False, use_xg_eff=False, use_pp=False, use_save_rate=True)),
    ("+ pp/pk",         dict(use_goalie=False, use_xg_eff=False, use_pp=True,  use_save_rate=False)),
    ("+ xg_eff + sr",   dict(use_goalie=False, use_xg_eff=True,  use_pp=False, use_save_rate=True)),
    ("+ all extras",    dict(use_goalie=False, use_xg_eff=True,  use_pp=True,  use_save_rate=True)),
]

print(f"  {'Config':<20s}  ", end="")
for alpha in [200, 500, 1000, 2000]:
    print(f"  a={alpha:4d}", end="")
print()
print(f"  {'-'*20}  " + "  -------" * 4)

for config_name, config in feature_configs:
    print(f"  {config_name:<20s}  ", end="")
    for alpha in [200, 500, 1000, 2000]:
        rmse = extended_ridge_cv(alpha=alpha, **config)
        print(f"  {rmse:.4f}", end="")
    print()


# ── Ridge + Elo blend with high alpha ──
print(f"\n  Simple Ridge + Elo blend (fine-grained):")

def ridge_elo_blend_cv(ridge_alpha, elo_weight, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []
    with open('output/predictions/elo/elo_pipeline_summary.json') as f:
        params = json.load(f)['best_params']
    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]
        team_map = {}
        for team in set(train_g['home_team']) | set(train_g['away_team']):
            h = train_g[train_g['home_team'] == team]
            a = train_g[train_g['away_team'] == team]
            n = len(h) + len(a)
            if n == 0:
                team_map[team] = [3.0]*6
                continue
            team_map[team] = [
                (h['home_goals'].sum() + a['away_goals'].sum()) / n,
                (h['away_goals'].sum() + a['home_goals'].sum()) / n,
                (h['home_xg'].sum() + a['away_xg'].sum()) / n,
                (h['away_xg'].sum() + a['home_xg'].sum()) / n,
                (h['home_shots'].sum() + a['away_shots'].sum()) / n,
                (h['away_shots'].sum() + a['home_shots'].sum()) / n,
            ]
        X_train, y_train = [], []
        for _, g in train_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            X_train.append([ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1])
            y_train.append(g['home_goals'])
            X_train.append([at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0])
            y_train.append(g['away_goals'])
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_train)
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_sc, y_train)
        elo = EnhancedEloModel(params)
        elo.fit(train_g)
        for _, g in val_g.iterrows():
            ht_s = team_map.get(g['home_team'], [3]*6)
            at_s = team_map.get(g['away_team'], [3]*6)
            rh = max(0.3, model.predict(sc.transform([[ht_s[0], ht_s[2], ht_s[4], at_s[1], at_s[3], 1]]))[0])
            ra = max(0.3, model.predict(sc.transform([[at_s[0], at_s[2], at_s[4], ht_s[1], ht_s[3], 0]]))[0])
            elo_h, elo_a = elo.predict_goals(g)
            bh = (1 - elo_weight) * rh + elo_weight * elo_h
            ba = (1 - elo_weight) * ra + elo_weight * elo_a
            all_pred.extend([bh, ba])
            all_act.extend([g['home_goals'], g['away_goals']])
    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


print(f"  {'Config':<20s}  ", end="")
for ew in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    print(f"  elo={ew:.2f}", end="")
print()
print(f"  {'-'*20}  " + "  --------" * 7)

for ra in [200, 500, 1000, 2000]:
    print(f"  Ridge(alpha={ra:<5d}) ", end="")
    for ew in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        rmse = ridge_elo_blend_cv(ra, ew)
        print(f"  {rmse:.4f}", end="")
    print()


# ── Three-way fine-grained ──
print(f"\n  Three-way blend fine-grained (around 0.6/0.3/0.1):")

def three_way_cv(w_simple, w_gp, w_elo, ridge_alpha=500, poisson_alpha=10, k=5, seed=42):
    """Three-way blend with configurable weights and alphas."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []
    with open('output/predictions/elo/elo_pipeline_summary.json') as f:
        params = json.load(f)['best_params']
    no_h2h = [f for f in FEATURE_NAMES if f not in ['h2h_goals', 'h2h_n']]

    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        # Simple Ridge
        team_map = {}
        for team in set(train_g['home_team']) | set(train_g['away_team']):
            h = train_g[train_g['home_team'] == team]
            a = train_g[train_g['away_team'] == team]
            n = len(h) + len(a)
            if n == 0:
                team_map[team] = [3.0]*6
                continue
            team_map[team] = [
                (h['home_goals'].sum() + a['away_goals'].sum()) / n,
                (h['away_goals'].sum() + a['home_goals'].sum()) / n,
                (h['home_xg'].sum() + a['away_xg'].sum()) / n,
                (h['away_xg'].sum() + a['home_xg'].sum()) / n,
                (h['home_shots'].sum() + a['away_shots'].sum()) / n,
                (h['away_shots'].sum() + a['home_shots'].sum()) / n,
            ]
        Xs, ys = [], []
        for _, g in train_g.iterrows():
            h_s = team_map.get(g['home_team'], [3]*6)
            a_s = team_map.get(g['away_team'], [3]*6)
            Xs.append([h_s[0], h_s[2], h_s[4], a_s[1], a_s[3], 1])
            ys.append(g['home_goals'])
            Xs.append([a_s[0], a_s[2], a_s[4], h_s[1], h_s[3], 0])
            ys.append(g['away_goals'])
        sc_s = StandardScaler()
        Xs_sc = sc_s.fit_transform(Xs)
        simple_model = Ridge(alpha=ridge_alpha)
        simple_model.fit(Xs_sc, ys)

        # GP Poisson
        train_ids = set(train_g['game_id'])
        train_raw = raw[raw['game_id'].isin(train_ids)]
        stats = compute_stats(train_raw, train_g)
        X_gp, y_gp = build_dataset(train_g, stats)
        X_gp = X_gp[no_h2h]
        sc_gp = StandardScaler()
        Xgp_sc = sc_gp.fit_transform(X_gp)
        gp_model = PoissonRegressor(alpha=poisson_alpha, max_iter=3000)
        gp_model.fit(Xgp_sc, y_gp)

        # Elo
        elo = EnhancedEloModel(params)
        elo.fit(train_g)

        for _, g in val_g.iterrows():
            h_s = team_map.get(g['home_team'], [3]*6)
            a_s = team_map.get(g['away_team'], [3]*6)
            sh = max(0.3, simple_model.predict(sc_s.transform([[h_s[0], h_s[2], h_s[4], a_s[1], a_s[3], 1]]))[0])
            sa = max(0.3, simple_model.predict(sc_s.transform([[a_s[0], a_s[2], a_s[4], h_s[1], h_s[3], 0]]))[0])

            fh = build_features(g, stats, 'home')
            fa = build_features(g, stats, 'away')
            Xv = pd.DataFrame([fh, fa], columns=FEATURE_NAMES)[no_h2h]
            gp_p = gp_model.predict(sc_gp.transform(Xv))
            gh, ga_ = max(0.3, gp_p[0]), max(0.3, gp_p[1])

            elo_h, elo_a = elo.predict_goals(g)

            bh = w_simple * sh + w_gp * gh + w_elo * elo_h
            ba = w_simple * sa + w_gp * ga_ + w_elo * elo_a
            all_pred.extend([bh, ba])
            all_act.extend([g['home_goals'], g['away_goals']])
    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


# Fine grid around 0.6/0.3/0.1
print(f"  {'w_s':>5s} {'w_gp':>5s} {'w_elo':>5s}  {'RMSE':>8s}")
print(f"  {'-'*30}")
fine_results = []
for w_s in np.arange(0.40, 0.85, 0.05):
    for w_gp in np.arange(0.10, 0.50, 0.05):
        w_e = round(1.0 - w_s - w_gp, 2)
        if w_e < -0.01 or w_e > 0.30:
            continue
        rmse = three_way_cv(w_s, w_gp, max(0, w_e))
        fine_results.append((w_s, w_gp, w_e, rmse))

fine_results.sort(key=lambda x: x[3])
for ws, wg, we, rmse in fine_results[:20]:
    print(f"  {ws:5.2f} {wg:5.2f} {we:5.2f}  {rmse:8.4f}")


# ── Robust estimate of best configs ──
print(f"\n  ROBUST estimates (avg over 10 seeds):")
configs = [
    ("Ridge(2000) alone",       lambda: simple_ridge_cv(2000)),
    ("Ridge(1000) alone",       lambda: simple_ridge_cv(1000)),
    ("Ridge(500) alone",        lambda: simple_ridge_cv(500)),
    ("Ridge(1000)+Elo(0.1)",    lambda: ridge_elo_blend_cv(1000, 0.1)),
    ("Ridge(1000)+Elo(0.15)",   lambda: ridge_elo_blend_cv(1000, 0.15)),
    ("Ridge(1000)+Elo(0.2)",    lambda: ridge_elo_blend_cv(1000, 0.2)),
]

# Add top 3 three-way configs
for i, (ws, wg, we, _) in enumerate(fine_results[:3]):
    configs.append(
        (f"3way({ws:.2f}/{wg:.2f}/{we:.2f})",
         lambda ws=ws, wg=wg, we=we: three_way_cv(ws, wg, we))
    )

print(f"  {'Config':<30s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
print(f"  {'-'*70}")

best_config_name = None
best_config_mean = 99.0
for name, fn in configs:
    rmses = [fn() if i == 0 else simple_ridge_cv(2000, seed=i) for i in range(5)]
    # Actually run properly
    if 'Ridge(2000) alone' in name:
        rmses = [simple_ridge_cv(2000, seed=i) for i in range(10)]
    elif 'Ridge(1000) alone' in name:
        rmses = [simple_ridge_cv(1000, seed=i) for i in range(10)]
    elif 'Ridge(500) alone' in name:
        rmses = [simple_ridge_cv(500, seed=i) for i in range(10)]
    elif 'Elo(0.1)' in name and '3way' not in name:
        rmses = [ridge_elo_blend_cv(1000, 0.1, seed=i) for i in range(10)]
    elif 'Elo(0.15)' in name:
        rmses = [ridge_elo_blend_cv(1000, 0.15, seed=i) for i in range(10)]
    elif 'Elo(0.2)' in name and '3way' not in name:
        rmses = [ridge_elo_blend_cv(1000, 0.2, seed=i) for i in range(10)]
    elif '3way' in name:
        ws, wg, we = [float(x) for x in name.split('(')[1].split(')')[0].split('/')]
        rmses = [three_way_cv(ws, wg, we, seed=i) for i in range(5)]

    mean_r = np.mean(rmses)
    std_r = np.std(rmses)
    print(f"  {name:<30s}  {mean_r:8.4f}  {std_r:8.4f}  {min(rmses):8.4f}  {max(rmses):8.4f}")
    if mean_r < best_config_mean:
        best_config_mean = mean_r
        best_config_name = name


print(f"\n{'='*70}")
print(f"  BEST CONFIG: {best_config_name}  mean RMSE = {best_config_mean:.4f}")
print(f"  Previous best: 1.7364 (GP Poisson(20) + Elo 50/50)")
print(f"  Improvement: {1.7364 - best_config_mean:+.4f}")
print(f"{'='*70}")
