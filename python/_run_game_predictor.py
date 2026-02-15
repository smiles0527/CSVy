"""
Full Game Predictor Pipeline
=============================
1. Load shift-level data
2. Grid search: Poisson / Ridge / GBR × alpha
3. Train final model on all 1312 games
4. Feature importance
5. Round 1 predictions
6. Elo ensemble (find optimal blend weight)
7. Save outputs + comparison
"""
import io, sys, os, json, pickle
import numpy as np
import pandas as pd

# ── Windows UTF-8 fix ──
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

from utils.game_predictor import (
    GamePredictor, aggregate_games, compute_stats,
    build_features, build_dataset, grid_search_cv, ensemble_with_elo,
    FEATURE_NAMES,
)
from utils.enhanced_elo_model import EnhancedEloModel

print("=" * 65)
print("  GAME PREDICTOR PIPELINE — Full Feature Model")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("\n[1/7] Loading data ...")
raw = pd.read_csv('data/whl_2025.csv')
games_df = aggregate_games(raw)
print(f"  Shifts: {len(raw):,}  →  Games: {len(games_df):,}")
print(f"  Teams: {games_df['home_team'].nunique()}")
print(f"  Goalies: {len(set(games_df['home_goalie']) | set(games_df['away_goalie']))}")
print(f"  Home goals: {games_df['home_goals'].mean():.2f} ± {games_df['home_goals'].std():.2f}")
print(f"  Away goals: {games_df['away_goals'].mean():.2f} ± {games_df['away_goals'].std():.2f}")

# ═══════════════════════════════════════════════════════════════
# 2. GRID SEARCH CV
# ═══════════════════════════════════════════════════════════════
print("\n[2/7] Grid search cross-validation (5-fold) ...")
print(f"  Testing 13 configs: 5 Poisson + 5 Ridge + 3 GBR\n")
gs_results = grid_search_cv(raw, k=5, seed=42, verbose=True)

print(f"\n  Top 5 configurations:")
print(gs_results.head(5).to_string(index=False))

best_row = gs_results.iloc[0]
best_type = best_row['model_type']
best_alpha = best_row['alpha']
best_cv_rmse = best_row['cv_rmse']
best_cv_wa = best_row['cv_win_accuracy']
print(f"\n  >>> BEST: {best_type} alpha={best_alpha}  "
      f"RMSE={best_cv_rmse:.4f}  WA={best_cv_wa:.1%}")

# ═══════════════════════════════════════════════════════════════
# 3. TRAIN FINAL MODEL
# ═══════════════════════════════════════════════════════════════
print("\n[3/7] Training final model on all data ...")
final_model = GamePredictor(model_type=best_type, alpha=best_alpha)
final_model.fit(raw, games_df)
print(f"  Model type: {best_type}")
print(f"  Alpha: {best_alpha}")
print(f"  Features: {len(FEATURE_NAMES)}")

# Sanity: evaluate on training data (should overfit, but shows predictions work)
train_eval = final_model.evaluate(games_df)
print(f"  Train RMSE (sanity): {train_eval['combined_rmse']:.4f}")
print(f"  Train WA (sanity):   {train_eval['win_accuracy']:.1%}")

# ═══════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
print("\n[4/7] Feature importance:")
imp = final_model.feature_importance()
for rank, (feat, val) in enumerate(imp[:10], 1):
    bar = "+" * int(abs(val) * 20) if abs(val) < 5 else "+" * 15
    print(f"  {rank:2d}. {feat:18s}  {val:+.4f}  {bar}")
if len(imp) > 10:
    print(f"  ... ({len(imp) - 10} more features)")

# ═══════════════════════════════════════════════════════════════
# 5. ROUND 1 PREDICTIONS
# ═══════════════════════════════════════════════════════════════
print("\n[5/7] Round 1 predictions ...")
matchups = pd.read_excel('data/WHSDSC_Rnd1_matchups.xlsx')
home_col = [c for c in matchups.columns if 'home' in c.lower()][0]
away_col = [c for c in matchups.columns if 'away' in c.lower()][0]
id_col = [c for c in matchups.columns if 'game' in c.lower() or 'id' in c.lower()]
id_col = id_col[0] if id_col else None

# Map teams to their primary goalie for info
team_goalies = final_model.stats['goalies']
goalie_stats = final_model.stats['goalie']

preds = []
print(f"\n  {'Game':>4s}  {'Home':>20s}  {'Away':>20s}  {'H':>5s}  {'A':>5s}  {'Winner':>20s}")
print(f"  {'-'*4}  {'-'*20}  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*20}")
for i, row in matchups.iterrows():
    game = {
        'home_team': row[home_col],
        'away_team': row[away_col],
        'home_goalie': team_goalies.get(row[home_col], ''),
        'away_goalie': team_goalies.get(row[away_col], ''),
    }
    hp, ap = final_model.predict(game)
    winner = row[home_col] if hp > ap else row[away_col]
    gid = row[id_col] if id_col else i + 1

    preds.append({
        'game_id': gid,
        'home_team': row[home_col],
        'away_team': row[away_col],
        'predicted_score_home': round(hp, 2),
        'predicted_score_away': round(ap, 2),
        'predicted_winner': winner,
        'home_goalie': game['home_goalie'],
        'away_goalie': game['away_goalie'],
    })
    print(f"  {gid:>4}  {row[home_col]:>20s}  {row[away_col]:>20s}  "
          f"{hp:5.2f}  {ap:5.2f}  {winner}")

pred_df = pd.DataFrame(preds)
totals = pred_df['predicted_score_home'] + pred_df['predicted_score_away']
print(f"\n  Avg total: {totals.mean():.2f}  Range: {totals.min():.2f}-{totals.max():.2f}")
print(f"  Home wins: {(pred_df['predicted_score_home'] > pred_df['predicted_score_away']).sum()}/16")

# ═══════════════════════════════════════════════════════════════
# 6. ELO ENSEMBLE
# ═══════════════════════════════════════════════════════════════
print("\n[6/7] Elo ensemble blend ...")

# Load best Elo params (from previous pipeline)
elo_summary_path = 'output/predictions/elo/elo_pipeline_summary.json'
if os.path.exists(elo_summary_path):
    with open(elo_summary_path) as f:
        elo_summary = json.load(f)
    elo_params = elo_summary['best_params']
    print(f"  Loaded Elo params from {elo_summary_path}")
else:
    # Fallback: best known params from Phase 9
    elo_params = {
        'k_factor': 5, 'home_advantage': 180, 'mov_multiplier': 0.0,
        'xg_weight': 0.7, 'k_decay': 0.01, 'k_min': 8,
        'shot_share_weight': 0, 'penalty_weight': 0,
        'scoring_baseline': 'team', 'xg_pred_weight': 0.5,
        'elo_shift_scale': 0.7, 'rolling_window': 20, 'initial_rating': 1500,
    }
    print("  Using fallback Elo params (Phase 9 best)")

elo_model = EnhancedEloModel(elo_params)
elo_model.fit(games_df)

# Get standalone Elo CV for comparison
print("  Running Elo 5-fold CV ...")
elo_cv_hp, elo_cv_ap, elo_cv_ha, elo_cv_aa = [], [], [], []
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(games_df):
    train_g = games_df.iloc[train_idx]
    val_g = games_df.iloc[val_idx]
    fold_elo = EnhancedEloModel(elo_params)
    fold_elo.fit(train_g)
    for _, g in val_g.iterrows():
        eh, ea = fold_elo.predict_goals(g)
        elo_cv_hp.append(eh); elo_cv_ap.append(ea)
        elo_cv_ha.append(g['home_goals']); elo_cv_aa.append(g['away_goals'])

elo_cv_all_act = elo_cv_ha + elo_cv_aa
elo_cv_all_pred = elo_cv_hp + elo_cv_ap
from sklearn.metrics import mean_squared_error
elo_cv_rmse = float(np.sqrt(mean_squared_error(elo_cv_all_act, elo_cv_all_pred)))
elo_cv_correct = sum(1 for h,a,ah,aa in zip(elo_cv_hp,elo_cv_ap,elo_cv_ha,elo_cv_aa)
                     if (h>a)==(ah>aa))
elo_cv_wa = elo_cv_correct / len(elo_cv_ha)
print(f"  Elo standalone CV:  RMSE={elo_cv_rmse:.4f}  WA={elo_cv_wa:.1%}")

# Find optimal blend
print("  Finding optimal ensemble weight ...")
blend_results = ensemble_with_elo(raw, final_model, elo_model, k=5, seed=42)

print(f"\n  Blend weight results:")
for wr in blend_results['weight_results']:
    marker = " <<<" if wr['elo_weight'] == blend_results['best_elo_weight'] else ""
    print(f"    Elo={wr['elo_weight']:.1f}  GP={1-wr['elo_weight']:.1f}  "
          f"RMSE={wr['rmse']:.4f}  WA={wr['win_accuracy']:.1%}{marker}")

best_w = blend_results['best_elo_weight']
print(f"\n  >>> Best blend: Elo={best_w:.1f}, GamePredictor={1-best_w:.1f}")
print(f"      Blend RMSE: {blend_results['best_blend_rmse']:.4f}")
print(f"      GP-only:    {blend_results['gp_only_rmse']:.4f}")
print(f"      Elo-only:   {blend_results['elo_only_rmse']:.4f}")

# Make blended Round 1 predictions
blend_preds = []
for p in preds:
    game = {'home_team': p['home_team'], 'away_team': p['away_team'],
            'home_goalie': p['home_goalie'], 'away_goalie': p['away_goalie']}
    gp_h, gp_a = final_model.predict(game)
    elo_h, elo_a = elo_model.predict_goals(game)
    bh = best_w * elo_h + (1 - best_w) * gp_h
    ba = best_w * elo_a + (1 - best_w) * gp_a
    winner = p['home_team'] if bh > ba else p['away_team']
    blend_preds.append({
        'game_id': p['game_id'],
        'home_team': p['home_team'],
        'away_team': p['away_team'],
        'predicted_score_home': round(bh, 2),
        'predicted_score_away': round(ba, 2),
        'gp_home': round(gp_h, 2), 'gp_away': round(gp_a, 2),
        'elo_home': round(elo_h, 2), 'elo_away': round(elo_a, 2),
        'predicted_winner': winner,
    })
blend_df = pd.DataFrame(blend_preds)

print(f"\n  Blended Round 1 predictions:")
print(f"  {'Game':>4s}  {'Home':>20s}  {'Away':>20s}  {'Blend_H':>7s}  {'Blend_A':>7s}  "
      f"{'GP_H':>5s}  {'GP_A':>5s}  {'Elo_H':>5s}  {'Elo_A':>5s}")
for _, r in blend_df.iterrows():
    print(f"  {r['game_id']:>4}  {r['home_team']:>20s}  {r['away_team']:>20s}  "
          f"{r['predicted_score_home']:7.2f}  {r['predicted_score_away']:7.2f}  "
          f"{r['gp_home']:5.2f}  {r['gp_away']:5.2f}  "
          f"{r['elo_home']:5.2f}  {r['elo_away']:5.2f}")

# ═══════════════════════════════════════════════════════════════
# 7. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════
print("\n[7/7] Saving outputs ...")
os.makedirs('output/predictions/game_predictor', exist_ok=True)
os.makedirs('output/models/game_predictor', exist_ok=True)

# Grid search results
gs_results.to_csv('output/predictions/game_predictor/grid_search_cv.csv', index=False)
print("  [OK] grid_search_cv.csv")

# GP-only round 1
pred_df.to_csv('output/predictions/game_predictor/round1_gp_predictions.csv', index=False)
print("  [OK] round1_gp_predictions.csv")

# Blended round 1
blend_df.to_csv('output/predictions/game_predictor/round1_blend_predictions.csv', index=False)
print("  [OK] round1_blend_predictions.csv")

# Competition submission format
submission = blend_df[['game_id', 'predicted_score_home', 'predicted_score_away']].copy()
submission.to_csv('output/predictions/game_predictor/submission.csv', index=False)
print("  [OK] submission.csv")

# Save model
final_model.save('output/models/game_predictor/best_game_predictor')

# Full summary
summary = {
    'model': f'GamePredictor ({best_type}, alpha={best_alpha})',
    'cv_rmse': round(best_cv_rmse, 4),
    'cv_win_accuracy': round(best_cv_wa, 4),
    'features': FEATURE_NAMES,
    'n_features': len(FEATURE_NAMES),
    'n_games': len(games_df),
    'grid_search': gs_results.to_dict('records'),
    'feature_importance': [(f, round(float(v), 4)) for f, v in imp],
    'elo_cv_rmse': round(elo_cv_rmse, 4),
    'elo_cv_win_accuracy': round(elo_cv_wa, 4),
    'ensemble': {
        'best_elo_weight': best_w,
        'blend_cv_rmse': round(blend_results['best_blend_rmse'], 4),
        'weight_sweep': blend_results['weight_results'],
    },
    'comparisons': {
        'naive_rmse': round(float(np.sqrt(np.mean(
            (games_df['home_goals'] - games_df['home_goals'].mean())**2 +
            (games_df['away_goals'] - games_df['away_goals'].mean())**2
        ) / 2)), 4),
        'gp_cv_rmse': round(best_cv_rmse, 4),
        'elo_cv_rmse': round(elo_cv_rmse, 4),
        'blend_cv_rmse': round(blend_results['best_blend_rmse'], 4),
        'baseline_ensemble_rmse': 1.8071,
    },
    'predictions_gp': preds,
    'predictions_blend': blend_preds,
    'goalie_mapping': {t: g for t, g in team_goalies.items()},
    'goalie_stats': {g: {k: round(float(v), 4) for k, v in stats.items() if isinstance(v, (int, float))}
                     for g, stats in goalie_stats.items()},
}

with open('output/predictions/game_predictor/pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("  [OK] pipeline_summary.json")

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  FINAL RESULTS — Game Predictor Pipeline")
print(f"{'='*65}")
print(f"  Model:             {best_type} (alpha={best_alpha})")
print(f"  Features:          {len(FEATURE_NAMES)}")
print(f"  CV RMSE:           {best_cv_rmse:.4f}")
print(f"  CV Win Accuracy:   {best_cv_wa:.1%}")
print(f"  Elo CV RMSE:       {elo_cv_rmse:.4f}")
print(f"  Elo CV WA:         {elo_cv_wa:.1%}")
print(f"  Blend weight:      Elo={best_w:.1f} + GP={1-best_w:.1f}")
print(f"  Blend CV RMSE:     {blend_results['best_blend_rmse']:.4f}")
print(f"  -----------------------------------------")
print(f"  Baseline ensemble: 1.8071")
print(f"  Elo v3 (80/20):    1.8131")
print(f"  vs Baseline:       {best_cv_rmse - 1.8071:+.4f}")
print(f"  vs Elo:            {best_cv_rmse - elo_cv_rmse:+.4f}")
print(f"{'='*65}")
