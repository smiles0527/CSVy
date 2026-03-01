"""
Optimized Game Predictor Pipeline v2
=====================================
Based on competitive intelligence findings:
  - Simple Ridge(700, 6 features) outperforms complex 21-feature Poisson
  - H2H features hurt performance 
  - Three-way blend (75% Simple + 10% GP + 15% Elo) achieves RMSE 1.7212
  - Previous best was 1.7364 (Poisson(20) + Elo 50/50)

Architecture:
  Model A: Simple Ridge (alpha=700) — 6 basic team features, 75% weight
  Model B: GP Poisson (alpha=10) — 19 features (no H2H), 10% weight
  Model C: Enhanced Elo — historical rating system, 15% weight
"""
import io, sys, os, json, pickle, time
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
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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

print("=" * 65)
print("  OPTIMIZED GAME PREDICTOR PIPELINE v2")
print("  Three-way blend: Simple Ridge + GP Poisson + Elo")
print("=" * 65)

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
RIDGE_ALPHA = 700
POISSON_ALPHA = 10
W_SIMPLE = 0.75
W_GP = 0.10
W_ELO = 0.15
GP_FEATURES = [f for f in FEATURE_NAMES if f not in ['h2h_goals', 'h2h_n']]  # 19 features

print(f"\n  Config:")
print(f"    Ridge alpha: {RIDGE_ALPHA}")
print(f"    Poisson alpha: {POISSON_ALPHA}")
print(f"    GP features: {len(GP_FEATURES)} (no H2H)")
print(f"    Blend weights: Simple={W_SIMPLE}, GP={W_GP}, Elo={W_ELO}")

# ═══════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════
print("\n[1/7] Loading data ...")
raw = pd.read_csv('data/whl_2025.csv')
games_df = aggregate_games(raw)
N = len(games_df)
print(f"  Shifts: {len(raw):,} -> Games: {N:,}")
print(f"  Teams: {games_df['home_team'].nunique()}")
print(f"  Home goals: {games_df['home_goals'].mean():.2f} +/- {games_df['home_goals'].std():.2f}")
print(f"  Away goals: {games_df['away_goals'].mean():.2f} +/- {games_df['away_goals'].std():.2f}")

# ═══════════════════════════════════════════════════
# 2. CROSS-VALIDATION (proper fold isolation)
# ═══════════════════════════════════════════════════
print("\n[2/7] 5-fold cross-validation (each model independently) ...")


def compute_simple_team_stats(train_g):
    """Compute 6 basic team stats from training games only."""
    team_map = {}
    for team in set(train_g['home_team']) | set(train_g['away_team']):
        h = train_g[train_g['home_team'] == team]
        a = train_g[train_g['away_team'] == team]
        n = len(h) + len(a)
        if n == 0:
            team_map[team] = [3.0] * 6
            continue
        team_map[team] = [
            (h['home_goals'].sum() + a['away_goals'].sum()) / n,   # GF/g
            (h['away_goals'].sum() + a['home_goals'].sum()) / n,   # GA/g
            (h['home_xg'].sum() + a['away_xg'].sum()) / n,        # xGF/g
            (h['away_xg'].sum() + a['home_xg'].sum()) / n,        # xGA/g
            (h['home_shots'].sum() + a['away_shots'].sum()) / n,   # SF/g
            (h['away_shots'].sum() + a['home_shots'].sum()) / n,   # SA/g
        ]
    return team_map


def make_simple_features(att_team, def_team, is_home, team_map):
    """6-feature vector: att_GF, att_xGF, att_SF, def_GA, def_xGA, is_home."""
    att = team_map.get(att_team, [3.0] * 6)
    def_ = team_map.get(def_team, [3.0] * 6)
    return [att[0], att[2], att[4], def_[1], def_[3], is_home]


# Run full 3-model CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_blend_pred, cv_blend_act = [], []
cv_simple_pred, cv_gp_pred, cv_elo_pred = [], [], []
cv_actual_h, cv_actual_a = [], []

# Also load Elo params
elo_summary_path = 'output/predictions/elo/elo_pipeline_summary.json'
with open(elo_summary_path) as f:
    elo_params = json.load(f)['best_params']

for fold, (train_idx, val_idx) in enumerate(kf.split(games_df)):
    train_g = games_df.iloc[train_idx]
    val_g = games_df.iloc[val_idx]
    print(f"  Fold {fold+1}/5 ({len(train_g)} train, {len(val_g)} val) ...", end=" ")

    # --- Model A: Simple Ridge ---
    team_map = compute_simple_team_stats(train_g)
    X_s, y_s = [], []
    for _, g in train_g.iterrows():
        X_s.append(make_simple_features(g['home_team'], g['away_team'], 1, team_map))
        y_s.append(g['home_goals'])
        X_s.append(make_simple_features(g['away_team'], g['home_team'], 0, team_map))
        y_s.append(g['away_goals'])
    sc_s = StandardScaler()
    X_s = sc_s.fit_transform(X_s)
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_s, y_s)

    # --- Model B: GP Poisson ---
    train_ids = set(train_g['game_id'])
    train_raw = raw[raw['game_id'].isin(train_ids)]
    stats = compute_stats(train_raw, train_g)
    X_gp, y_gp = build_dataset(train_g, stats)
    X_gp = X_gp[GP_FEATURES]
    sc_gp = StandardScaler()
    Xgp = sc_gp.fit_transform(X_gp)
    poisson = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=3000)
    poisson.fit(Xgp, y_gp)

    # --- Model C: Elo ---
    elo = EnhancedEloModel(elo_params)
    elo.fit(train_g)

    # --- Predict on validation ---
    for _, g in val_g.iterrows():
        # Simple
        fh_s = make_simple_features(g['home_team'], g['away_team'], 1, team_map)
        fa_s = make_simple_features(g['away_team'], g['home_team'], 0, team_map)
        sh = max(0.3, ridge.predict(sc_s.transform([fh_s]))[0])
        sa = max(0.3, ridge.predict(sc_s.transform([fa_s]))[0])

        # GP
        fh_gp = build_features(g, stats, 'home')
        fa_gp = build_features(g, stats, 'away')
        Xv = pd.DataFrame([fh_gp, fa_gp], columns=FEATURE_NAMES)[GP_FEATURES]
        gp_p = poisson.predict(sc_gp.transform(Xv))
        gh, ga = max(0.3, gp_p[0]), max(0.3, gp_p[1])

        # Elo
        elo_h, elo_a = elo.predict_goals(g)

        # Blend
        bh = W_SIMPLE * sh + W_GP * gh + W_ELO * elo_h
        ba = W_SIMPLE * sa + W_GP * ga + W_ELO * elo_a

        cv_blend_pred.extend([bh, ba])
        cv_blend_act.extend([g['home_goals'], g['away_goals']])
        cv_simple_pred.extend([sh, sa])
        cv_gp_pred.extend([gh, ga])
        cv_elo_pred.extend([elo_h, elo_a])
        cv_actual_h.append(g['home_goals'])
        cv_actual_a.append(g['away_goals'])

    print("done")

# Evaluate
rmse_blend = float(np.sqrt(mean_squared_error(cv_blend_act, cv_blend_pred)))
rmse_simple = float(np.sqrt(mean_squared_error(cv_blend_act, cv_simple_pred)))
rmse_gp = float(np.sqrt(mean_squared_error(cv_blend_act, cv_gp_pred)))
rmse_elo = float(np.sqrt(mean_squared_error(cv_blend_act, cv_elo_pred)))

# Win accuracy
n_games_val = len(cv_actual_h)
blend_correct = sum(1 for i in range(n_games_val)
                    if (cv_blend_pred[i*2] > cv_blend_pred[i*2+1]) ==
                       (cv_actual_h[i] > cv_actual_a[i]))
blend_wa = blend_correct / n_games_val

print(f"\n  CV Results:")
print(f"    Simple Ridge alone:  RMSE={rmse_simple:.4f}")
print(f"    GP Poisson alone:    RMSE={rmse_gp:.4f}")
print(f"    Elo alone:           RMSE={rmse_elo:.4f}")
print(f"    THREE-WAY BLEND:     RMSE={rmse_blend:.4f}  WA={blend_wa:.1%}")
print(f"    Previous best:       RMSE=1.7364 (old GP+Elo 50/50)")
print(f"    Improvement:         {1.7364 - rmse_blend:+.4f}")


# ═══════════════════════════════════════════════════
# 3. WEIGHT SWEEP (verify optimal blend)
# ═══════════════════════════════════════════════════
print("\n[3/7] Weight sweep verification ...")
# Quick sweep around our chosen weights
best_ws, best_wg, best_we, best_r = W_SIMPLE, W_GP, W_ELO, rmse_blend
for dws in [-0.05, 0, 0.05]:
    for dwe in [-0.05, 0, 0.05]:
        ws = W_SIMPLE + dws
        we = W_ELO + dwe
        wg = round(1.0 - ws - we, 2)
        if wg < 0 or ws < 0 or we < 0:
            continue
        # Recompute blend RMSE from stored predictions
        new_pred = []
        for i in range(0, len(cv_simple_pred), 2):
            sh_, sa_ = cv_simple_pred[i], cv_simple_pred[i+1]
            gh_, ga_ = cv_gp_pred[i], cv_gp_pred[i+1]
            eh_, ea_ = cv_elo_pred[i], cv_elo_pred[i+1]
            new_pred.append(ws * sh_ + wg * gh_ + we * eh_)
            new_pred.append(ws * sa_ + wg * ga_ + we * ea_)
        r = float(np.sqrt(mean_squared_error(cv_blend_act, new_pred)))
        if r < best_r:
            best_ws, best_wg, best_we, best_r = ws, wg, we, r

print(f"  Optimal weights: Simple={best_ws:.2f}, GP={best_wg:.2f}, Elo={best_we:.2f}")
print(f"  Optimal RMSE: {best_r:.4f}")

# Update if better
if best_r < rmse_blend - 0.0001:
    W_SIMPLE, W_GP, W_ELO = best_ws, best_wg, best_we
    print(f"  Updated blend weights to Simple={W_SIMPLE}, GP={W_GP}, Elo={W_ELO}")


# ═══════════════════════════════════════════════════
# 4. TRAIN FINAL MODELS ON ALL DATA
# ═══════════════════════════════════════════════════
print(f"\n[4/7] Training final models on all {N} games ...")

# Model A: Simple Ridge
team_map_full = compute_simple_team_stats(games_df)
X_s_all, y_s_all = [], []
for _, g in games_df.iterrows():
    X_s_all.append(make_simple_features(g['home_team'], g['away_team'], 1, team_map_full))
    y_s_all.append(g['home_goals'])
    X_s_all.append(make_simple_features(g['away_team'], g['home_team'], 0, team_map_full))
    y_s_all.append(g['away_goals'])
sc_s_final = StandardScaler()
X_s_all = sc_s_final.fit_transform(X_s_all)
ridge_final = Ridge(alpha=RIDGE_ALPHA)
ridge_final.fit(X_s_all, y_s_all)
print(f"  Model A: Ridge(alpha={RIDGE_ALPHA}), 6 features")

# Model B: GP Poisson
stats_full = compute_stats(raw, games_df)
X_gp_all, y_gp_all = build_dataset(games_df, stats_full)
X_gp_all = X_gp_all[GP_FEATURES]
sc_gp_final = StandardScaler()
Xgp_all = sc_gp_final.fit_transform(X_gp_all)
poisson_final = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=3000)
poisson_final.fit(Xgp_all, y_gp_all)
print(f"  Model B: Poisson(alpha={POISSON_ALPHA}), {len(GP_FEATURES)} features")

# Model C: Elo
elo_final = EnhancedEloModel(elo_params)
elo_final.fit(games_df)
print(f"  Model C: Enhanced Elo")

# Goalie mapping
team_goalies = stats_full['goalies']
goalie_stats = stats_full['goalie']
print(f"  Goalies mapped: {len(team_goalies)} teams")


# ═══════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════
print(f"\n[5/7] Feature importance:")

# Ridge coefficients
simple_feat_names = ['att_GF', 'att_xGF', 'att_SF', 'def_GA', 'def_xGA', 'is_home']
print(f"\n  Model A (Ridge) coefficients:")
for name, coef in sorted(zip(simple_feat_names, ridge_final.coef_), key=lambda x: -abs(x[1])):
    print(f"    {name:12s}  {coef:+.4f}")

# Poisson coefficients  
print(f"\n  Model B (Poisson) top coefficients:")
gp_imp = sorted(zip(GP_FEATURES, poisson_final.coef_), key=lambda x: -abs(x[1]))
for name, coef in gp_imp[:10]:
    print(f"    {name:18s}  {coef:+.4f}")


# ═══════════════════════════════════════════════════
# 6. ROUND 1 PREDICTIONS
# ═══════════════════════════════════════════════════
print(f"\n[6/7] Round 1 predictions ...")
matchups = pd.read_excel('data/WHSDSC_Rnd1_matchups.xlsx')
home_col = [c for c in matchups.columns if 'home' in c.lower()][0]
away_col = [c for c in matchups.columns if 'away' in c.lower()][0]
id_col = 'game_id' if 'game_id' in matchups.columns else None

preds = []
print(f"\n  {'ID':>6s}  {'Home':>20s}  {'Away':>20s}  {'Ridge':>6s}  {'GP':>6s}  {'Elo':>6s}  {'BLEND':>7s}  {'':>7s}  {'Winner':>20s}")
print(f"  {'-'*6}  {'-'*20}  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*20}")

for _, row in matchups.iterrows():
    ht, at = row[home_col], row[away_col]
    gid = row[id_col] if id_col else ''

    # Model A: Simple Ridge
    fh_s = make_simple_features(ht, at, 1, team_map_full)
    fa_s = make_simple_features(at, ht, 0, team_map_full)
    r_h = max(0.3, ridge_final.predict(sc_s_final.transform([fh_s]))[0])
    r_a = max(0.3, ridge_final.predict(sc_s_final.transform([fa_s]))[0])

    # Model B: GP Poisson
    game = {
        'home_team': ht, 'away_team': at,
        'home_goalie': team_goalies.get(ht, ''),
        'away_goalie': team_goalies.get(at, ''),
    }
    fh_gp = build_features(game, stats_full, 'home')
    fa_gp = build_features(game, stats_full, 'away')
    Xv = pd.DataFrame([fh_gp, fa_gp], columns=FEATURE_NAMES)[GP_FEATURES]
    gp_p = poisson_final.predict(sc_gp_final.transform(Xv))
    g_h, g_a = max(0.3, gp_p[0]), max(0.3, gp_p[1])

    # Model C: Elo
    e_h, e_a = elo_final.predict_goals(game)

    # Blend
    bh = W_SIMPLE * r_h + W_GP * g_h + W_ELO * e_h
    ba = W_SIMPLE * r_a + W_GP * g_a + W_ELO * e_a
    winner = ht if bh > ba else at

    preds.append({
        'game_id': gid,
        'home_team': ht,
        'away_team': at,
        'ridge_home': round(r_h, 3),
        'ridge_away': round(r_a, 3),
        'gp_home': round(g_h, 3),
        'gp_away': round(g_a, 3),
        'elo_home': round(e_h, 3),
        'elo_away': round(e_a, 3),
        'predicted_score_home': round(bh, 2),
        'predicted_score_away': round(ba, 2),
        'predicted_winner': winner,
        'home_goalie': team_goalies.get(ht, ''),
        'away_goalie': team_goalies.get(at, ''),
    })
    print(f"  {gid:>6s}  {ht:>20s}  {at:>20s}  "
          f"{r_h:6.2f}  {g_h:6.2f}  {e_h:6.2f}  "
          f"{bh:7.2f}  {ba:7.2f}  {winner}")

pred_df = pd.DataFrame(preds)
totals = pred_df['predicted_score_home'] + pred_df['predicted_score_away']
home_wins = (pred_df['predicted_score_home'] > pred_df['predicted_score_away']).sum()
print(f"\n  Avg total: {totals.mean():.2f}  Range: {totals.min():.2f}-{totals.max():.2f}")
print(f"  Home wins: {home_wins}/16 ({home_wins/16:.0%})")


# ═══════════════════════════════════════════════════
# 7. SAVE OUTPUTS
# ═══════════════════════════════════════════════════
print(f"\n[7/7] Saving outputs ...")
os.makedirs('output/predictions/game_predictor', exist_ok=True)
os.makedirs('output/models/game_predictor', exist_ok=True)

# Submission CSV
submission = pred_df[['game_id', 'predicted_score_home', 'predicted_score_away']].copy()
submission.to_csv('output/predictions/game_predictor/submission.csv', index=False)
print(f"  [OK] submission.csv ({len(submission)} rows)")

# Full predictions
pred_df.to_csv('output/predictions/game_predictor/round1_final_predictions.csv', index=False)
print(f"  [OK] round1_final_predictions.csv")

# Save models
model_bundle = {
    'ridge': ridge_final,
    'ridge_scaler': sc_s_final,
    'poisson': poisson_final,
    'poisson_scaler': sc_gp_final,
    'elo': elo_final,
    'team_map': team_map_full,
    'stats': stats_full,
    'gp_features': GP_FEATURES,
    'weights': {'simple': W_SIMPLE, 'gp': W_GP, 'elo': W_ELO},
    'config': {
        'ridge_alpha': RIDGE_ALPHA,
        'poisson_alpha': POISSON_ALPHA,
    }
}
with open('output/models/game_predictor/optimized_model_v2.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)
print(f"  [OK] optimized_model_v2.pkl")

# Summary JSON
summary = {
    'version': 'v2_optimized',
    'architecture': 'Three-way blend: Simple Ridge + GP Poisson + Enhanced Elo',
    'weights': {'simple_ridge': W_SIMPLE, 'gp_poisson': W_GP, 'elo': W_ELO},
    'config': {
        'ridge_alpha': RIDGE_ALPHA,
        'poisson_alpha': POISSON_ALPHA,
        'gp_features': len(GP_FEATURES),
        'simple_features': 6,
    },
    'cv_results': {
        'blend_rmse': round(rmse_blend, 4),
        'blend_win_accuracy': round(blend_wa, 4),
        'simple_rmse': round(rmse_simple, 4),
        'gp_rmse': round(rmse_gp, 4),
        'elo_rmse': round(rmse_elo, 4),
    },
    'improvement': {
        'vs_constant': round(1.7707 - rmse_blend, 4),
        'vs_old_blend': round(1.7364 - rmse_blend, 4),
        'vs_baseline_ensemble': round(1.8071 - rmse_blend, 4),
    },
    'n_games': N,
    'n_teams': games_df['home_team'].nunique(),
    'predictions': preds,
    'goalie_mapping': {t: g for t, g in team_goalies.items()},
    'goalie_stats': {g: {k: round(float(v), 4) for k, v in s.items() if isinstance(v, (int, float))}
                     for g, s in goalie_stats.items()},
}
with open('output/predictions/game_predictor/pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  [OK] pipeline_summary.json")


# ═══════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  FINAL RESULTS — Optimized Pipeline v2")
print(f"{'='*65}")
print(f"  Architecture:  {W_SIMPLE:.0%} Ridge + {W_GP:.0%} GP Poisson + {W_ELO:.0%} Elo")
print(f"  Ridge:         alpha={RIDGE_ALPHA}, 6 features")
print(f"  GP Poisson:    alpha={POISSON_ALPHA}, {len(GP_FEATURES)} features (no H2H)")
print(f"  Elo:           Enhanced v3")
print(f"  -----------------------------------------")
print(f"  BLEND CV RMSE:    {rmse_blend:.4f}")
print(f"  Blend Win Acc:    {blend_wa:.1%}")
print(f"  -----------------------------------------")
print(f"  vs Baseline:      {1.8071 - rmse_blend:+.4f} improvement")
print(f"  vs Old blend:     {1.7364 - rmse_blend:+.4f} improvement")
print(f"  vs Constant:      {1.7707 - rmse_blend:+.4f} improvement")
print(f"{'='*65}")
