"""
Model Optimization Based on Competitive Intel
==============================================
Findings from scripts/analysis/competitive_analysis.py:
  - Ridge(6 features) = 1.7268 BEATS our GP+Elo = 1.7364
  - H2H features HURT (removing them improves by 0.0095)
  - Our Elo component (1.7464) drags the blend UP
  - Simple team averages (1.7398) beat GP alone (1.7498)

This script systematically finds the BEST model by:
1. Testing all reasonable model families
2. Finding optimal feature subsets
3. Optimizing blend weights
4. Finding the true optimal submission model
"""
import io, sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge, LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
N = len(games_df)

print("=" * 70)
print("  MODEL OPTIMIZATION PHASE")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# STEP 1: Diagnose WHY Ridge beats Poisson
# ═══════════════════════════════════════════════════════════════
print(f"\n--- STEP 1: Why does Ridge beat Poisson? ---\n")

def cv_eval(model_factory, X, y, games, k=5, seed=42, clip=True):
    """CV with pre-built features. model_factory() -> model with .fit/.predict"""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []
    for train_idx, val_idx in kf.split(games):
        X_train, y_train = X.iloc[train_idx*2:(train_idx[-1]+1)*2], y[train_idx[0]*2:(train_idx[-1]+1)*2]
        X_val = X.iloc[val_idx[0]*2:(val_idx[-1]+1)*2]
        y_val = y[val_idx[0]*2:(val_idx[-1]+1)*2]

        sc = StandardScaler()
        X_t = sc.fit_transform(X_train)
        X_v = sc.transform(X_val)

        mdl = model_factory()
        mdl.fit(X_t, y_train)
        p = mdl.predict(X_v)
        if clip:
            p = np.clip(p, 0.3, 10)
        all_pred.extend(p.tolist())
        all_act.extend(y_val.tolist())
    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


# Build FULL feature dataset once
full_stats = compute_stats(raw, games_df)
X_full, y_full = build_dataset(games_df, full_stats)
print(f"  Full feature set: {list(X_full.columns)}")
print(f"  X shape: {X_full.shape}, y shape: {len(y_full)}")

# Direct comparison: same features, different models
def full_cv(model_fn, feature_cols=None, k=5, seed=42):
    """Full CV with fold isolation (recompute stats per fold)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_pred, all_act = [], []
    for train_idx, val_idx in kf.split(games_df):
        train_g = games_df.iloc[train_idx]
        val_g = games_df.iloc[val_idx]

        train_ids = set(train_g['game_id'])
        train_raw = raw[raw['game_id'].isin(train_ids)]
        stats = compute_stats(train_raw, train_g)

        X_train, y_train = build_dataset(train_g, stats)
        if feature_cols is not None:
            X_train = X_train[feature_cols]

        sc = StandardScaler()
        X_t = sc.fit_transform(X_train)
        model = model_fn()
        model.fit(X_t, y_train)

        for _, g in val_g.iterrows():
            fh = build_features(g, stats, 'home')
            fa = build_features(g, stats, 'away')
            Xv = pd.DataFrame([fh, fa], columns=FEATURE_NAMES)
            if feature_cols is not None:
                Xv = Xv[feature_cols]
            Xv_s = sc.transform(Xv)
            p = model.predict(Xv_s)
            all_pred.extend([max(0.3, p[0]), max(0.3, p[1])])
            all_act.extend([g['home_goals'], g['away_goals']])

    return float(np.sqrt(mean_squared_error(all_act, all_pred)))


# Test multiple model types on ALL 21 features
print(f"\n  Model comparison on all {len(FEATURE_NAMES)} features:")
print(f"  {'Model':<40s}  {'RMSE':>8s}")
print(f"  {'-'*50}")

models = [
    ("Poisson(alpha=1)",    lambda: PoissonRegressor(alpha=1, max_iter=3000)),
    ("Poisson(alpha=5)",    lambda: PoissonRegressor(alpha=5, max_iter=3000)),
    ("Poisson(alpha=10)",   lambda: PoissonRegressor(alpha=10, max_iter=3000)),
    ("Poisson(alpha=20)",   lambda: PoissonRegressor(alpha=20, max_iter=3000)),
    ("Poisson(alpha=50)",   lambda: PoissonRegressor(alpha=50, max_iter=3000)),
    ("Ridge(alpha=1)",      lambda: Ridge(alpha=1)),
    ("Ridge(alpha=5)",      lambda: Ridge(alpha=5)),
    ("Ridge(alpha=10)",     lambda: Ridge(alpha=10)),
    ("Ridge(alpha=50)",     lambda: Ridge(alpha=50)),
    ("Ridge(alpha=100)",    lambda: Ridge(alpha=100)),
    ("Lasso(alpha=0.01)",   lambda: Lasso(alpha=0.01, max_iter=5000)),
    ("Lasso(alpha=0.1)",    lambda: Lasso(alpha=0.1, max_iter=5000)),
    ("LinearRegression",    lambda: LinearRegression()),
    ("RF(100, d5)",         lambda: RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)),
    ("GBR(100, d3)",        lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42, learning_rate=0.05)),
]

results = []
for name, fn in models:
    rmse = full_cv(fn)
    results.append((name, rmse))
    print(f"  {name:<40s}  {rmse:8.4f}")

# Sort by RMSE
results.sort(key=lambda x: x[1])
print(f"\n  BEST on full features: {results[0][0]} = {results[0][1]:.4f}")
best_model_name, best_model_rmse = results[0]


# ═══════════════════════════════════════════════════════════════
# STEP 2: Feature subset search
# ═══════════════════════════════════════════════════════════════
print(f"\n--- STEP 2: Feature subset optimization ---\n")

# Start with the best model type, try removing features one at a time
best_model_fn = dict(models)[best_model_name] if best_model_name in dict(models) else None
if best_model_fn is None:
    # fallback: pick the best from results
    for name, fn in models:
        if name == best_model_name:
            best_model_fn = fn
            break

# Forward feature selection: start empty, add most helpful feature each round
print(f"  Forward feature selection (greedy, using {best_model_name})...\n")
remaining = list(FEATURE_NAMES)
selected = []
best_so_far = 99.0

for step in range(len(FEATURE_NAMES)):
    best_feat = None
    best_rmse = 99.0
    for feat in remaining:
        cols = selected + [feat]
        try:
            rmse = full_cv(best_model_fn, feature_cols=cols, seed=42)
            if rmse < best_rmse:
                best_rmse = rmse
                best_feat = feat
        except Exception:
            pass

    if best_feat is None:
        break

    if best_rmse < best_so_far - 0.0001:  # Must improve by at least 0.0001
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_so_far = best_rmse
        marker = " *NEW BEST*" if step == 0 or best_rmse < best_so_far + 0.001 else ""
        print(f"    +{best_feat:<20s}  -> RMSE={best_rmse:.4f}  ({len(selected)} features){marker}")
    else:
        print(f"    No improvement from adding {best_feat} ({best_rmse:.4f} vs {best_so_far:.4f}). Stopping.")
        break

print(f"\n  Optimal feature set ({len(selected)} features):")
print(f"    {selected}")
print(f"    RMSE: {best_so_far:.4f}")


# ═══════════════════════════════════════════════════════════════
# STEP 3: Blend optimization with best GP model
# ═══════════════════════════════════════════════════════════════
print(f"\n--- STEP 3: Optimal blend with Elo ---\n")

def blend_cv(gp_fn, feature_cols, elo_weight, k=5, seed=42):
    """CV with GP+Elo blend."""
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
        if feature_cols is not None:
            X_train = X_train[feature_cols]
        sc = StandardScaler()
        X_t = sc.fit_transform(X_train)
        gp_model = gp_fn()
        gp_model.fit(X_t, y_train)

        # Elo
        elo = EnhancedEloModel(params)
        elo.fit(train_g)

        for _, g in val_g.iterrows():
            fh = build_features(g, stats, 'home')
            fa = build_features(g, stats, 'away')
            Xv = pd.DataFrame([fh, fa], columns=FEATURE_NAMES)
            if feature_cols is not None:
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


# Grid search over blend weights
print(f"  Elo weight sweep (using optimal features + {best_model_name}):")
print(f"  {'Elo Weight':>12s}  {'RMSE':>8s}")
print(f"  {'-'*25}")

blend_results = []
for ew in np.arange(0.0, 0.65, 0.05):
    rmse = blend_cv(best_model_fn, selected if len(selected) > 0 else None, ew)
    blend_results.append((ew, rmse))
    marker = ""
    if ew == 0.0:
        marker = "  (GP only)"
    elif ew == 0.5:
        marker = "  (50/50 — current)"
    print(f"  {ew:12.2f}  {rmse:8.4f}{marker}")

# Also test GP-only (no blend) with optimal features
gp_only_rmse = full_cv(best_model_fn, feature_cols=selected if selected else None)
print(f"\n  GP only (optimal features): {gp_only_rmse:.4f}")

# Find best blend
blend_results.sort(key=lambda x: x[1])
best_elo_w, best_blend_rmse = blend_results[0]
print(f"  Best blend: Elo={best_elo_w:.2f}, GP={1-best_elo_w:.2f}  RMSE={best_blend_rmse:.4f}")


# ═══════════════════════════════════════════════════════════════
# STEP 4: Additional model families with optimal features
# ═══════════════════════════════════════════════════════════════
print(f"\n--- STEP 4: Additional models with optimal features ---\n")

extra_models = [
    ("Ridge(1)",   lambda: Ridge(alpha=1)),
    ("Ridge(5)",   lambda: Ridge(alpha=5)),
    ("Ridge(10)",  lambda: Ridge(alpha=10)),
    ("Ridge(20)",  lambda: Ridge(alpha=20)),
    ("Poisson(1)", lambda: PoissonRegressor(alpha=1, max_iter=3000)),
    ("Poisson(5)", lambda: PoissonRegressor(alpha=5, max_iter=3000)),
    ("Poisson(10)",lambda: PoissonRegressor(alpha=10, max_iter=3000)),
    ("GBR(100)",   lambda: GradientBoostingRegressor(100, max_depth=3, min_samples_leaf=10, random_state=42, learning_rate=0.05)),
    ("RF(200,d5)", lambda: RandomForestRegressor(200, max_depth=5, min_samples_leaf=10, random_state=42)),
]

print(f"  {'Model':<25s}  {'Opt Feats':>10s}  {'+ Elo@{:.0f}%':>12s}".format(best_elo_w*100))
print(f"  {'-'*55}")

for name, fn in extra_models:
    rmse_alone = full_cv(fn, feature_cols=selected if selected else None)
    rmse_blend = blend_cv(fn, selected if selected else None, best_elo_w)
    print(f"  {name:<25s}  {rmse_alone:10.4f}  {rmse_blend:12.4f}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  OPTIMIZATION RESULTS SUMMARY")
print(f"{'='*70}")
print(f"  Previous best:     1.7364 (Poisson(20) + Elo 50/50, 21 features)")
print(f"  Best model type:   {best_model_name}")
print(f"  Optimal features:  {len(selected)} features")
print(f"  Feature names:     {selected}")
print(f"  GP-only RMSE:      {gp_only_rmse:.4f}")
print(f"  Best Elo weight:   {best_elo_w:.2f}")
print(f"  New best RMSE:     {best_blend_rmse:.4f}")
improvement = 1.7364 - best_blend_rmse
print(f"  Improvement:       {improvement:+.4f}")
print(f"{'='*70}")
