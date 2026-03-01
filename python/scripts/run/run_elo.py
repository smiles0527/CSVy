#!/usr/bin/env python3
"""
Enhanced Elo Model — Full Pipeline
====================================
1. Load WHL data & aggregate to games
2. Grid-search Enhanced Elo hyperparameters (K-fold CV)
3. Select best config
4. Train on ALL data
5. Generate Round 1 predictions with correct game_id format
"""
import sys, os, json
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

import pandas as pd
import numpy as np
from itertools import product
from utils.hockey_features import aggregate_to_games
from utils.enhanced_elo_model import EnhancedEloModel

OUTPUT = _python_dir / 'output' / 'predictions' / 'elo'
MODEL_DIR = _python_dir / 'output' / 'models'
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────
raw = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'whl_2025.csv'))
if raw['game_id'].dtype == object:
    raw['game_id'] = raw['game_id'].str.replace('game_', '', regex=False).astype(int)

games = aggregate_to_games(raw).sort_values('game_id').reset_index(drop=True)

print("=" * 75)
print("ENHANCED ELO MODEL PIPELINE")
print("=" * 75)
print(f"Games: {len(games)}, Teams: {games['home_team'].nunique()}")
print(f"Home goals mean: {games['home_goals'].mean():.3f}, Away: {games['away_goals'].mean():.3f}")
print()

# ── PHASE 1: Grid search (5-fold CV) ─────────────────────────────
print("[1] Grid search over Enhanced Elo hyperparameters (5-fold CV)...")

param_grid = {
    'k_factor':       [5, 10, 20],
    'home_advantage': [100, 150, 180, 200],
    'mov_multiplier': [0.0, 0.3],
    'xg_weight':      [0.3, 0.7],
    'k_decay':        [0.0, 0.01],
    'xg_pred_weight': [0.0, 0.5],
    'elo_shift_scale':[0.5, 0.7, 1.0],
    'rolling_window': [15, 20],
}

# Fixed params
fixed = {
    'initial_rating': 1500,
    'k_min': 8,
    'shot_share_weight': 0.0,
    'penalty_weight': 0.0,
    'scoring_baseline': 'team',
}

keys = list(param_grid.keys())
combos = list(product(*[param_grid[k] for k in keys]))
print(f"  Total configs: {len(combos)}")

n_folds = 5
fold_size = len(games) // n_folds
results = []

for i, vals in enumerate(combos):
    params = dict(zip(keys, vals))
    params.update(fixed)

    fold_rmses = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_df = games.iloc[val_start:val_end]
        train_df = pd.concat([games.iloc[:val_start], games.iloc[val_end:]])

        model = EnhancedEloModel(params)
        model.fit(train_df)
        metrics = model.evaluate(val_df)
        fold_rmses.append(metrics['combined_rmse'])

    mean_rmse = np.mean(fold_rmses)
    results.append({**params, 'mean_cv_rmse': mean_rmse, 'std_rmse': np.std(fold_rmses)})

    if (i + 1) % 100 == 0:
        print(f"  ... {i+1}/{len(combos)} configs evaluated")

results_df = pd.DataFrame(results).sort_values('mean_cv_rmse')
best_row = results_df.iloc[0]
best_params = {k: best_row[k] for k in keys}
best_params.update(fixed)

print(f"\n  Best CV RMSE: {best_row['mean_cv_rmse']:.4f} ± {best_row['std_rmse']:.4f}")
print(f"  Best params: { {k: best_params[k] for k in keys} }")

# Save comparison
results_df.to_csv(OUTPUT / 'elo_comparison.csv', index=False)
print(f"\n  Comparison saved: {OUTPUT / 'elo_comparison.csv'}")

# ── PHASE 2: Fine sweep around best ──────────────────────────────
print("\n[2] Fine sweep around best config...")

fine_grid = {
    'k_factor':       [max(1, best_params['k_factor'] - 2), best_params['k_factor'], best_params['k_factor'] + 2],
    'home_advantage': [best_params['home_advantage'] - 20, best_params['home_advantage'], best_params['home_advantage'] + 20],
    'elo_shift_scale':[max(0.1, best_params['elo_shift_scale'] - 0.2), best_params['elo_shift_scale'], best_params['elo_shift_scale'] + 0.2],
    'xg_pred_weight': [max(0.0, best_params['xg_pred_weight'] - 0.1), best_params['xg_pred_weight'], min(1.0, best_params['xg_pred_weight'] + 0.1)],
    'rolling_window': [max(5, best_params['rolling_window'] - 5), best_params['rolling_window'], best_params['rolling_window'] + 5],
}

fine_combos = list(product(*[fine_grid[k] for k in fine_grid.keys()]))
fine_keys = list(fine_grid.keys())
print(f"  Fine sweep configs: {len(fine_combos)}")

fine_results = []
for vals in fine_combos:
    params = dict(best_params)  # start from coarse best
    params.update(dict(zip(fine_keys, vals)))

    fold_rmses = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_df = games.iloc[val_start:val_end]
        train_df = pd.concat([games.iloc[:val_start], games.iloc[val_end:]])

        model = EnhancedEloModel(params)
        model.fit(train_df)
        metrics = model.evaluate(val_df)
        fold_rmses.append(metrics['combined_rmse'])

    mean_rmse = np.mean(fold_rmses)
    fine_results.append({**params, 'mean_cv_rmse': mean_rmse, 'std_rmse': np.std(fold_rmses)})

fine_df = pd.DataFrame(fine_results).sort_values('mean_cv_rmse')
final_best_row = fine_df.iloc[0]
final_best_params = {k: final_best_row[k] for k in list(param_grid.keys()) + list(fixed.keys())}

print(f"  Final best CV RMSE: {final_best_row['mean_cv_rmse']:.4f} ± {final_best_row['std_rmse']:.4f}")

# ── PHASE 3: Train on ALL data ───────────────────────────────────
print("\n[3] Training Enhanced Elo on ALL data...")
final_model = EnhancedEloModel(final_best_params)
final_model.fit(games)

# Quick 80/20 eval for reporting
split = int(len(games) * 0.8)
eval_model = EnhancedEloModel(final_best_params)
eval_model.fit(games.iloc[:split])
test_metrics = eval_model.evaluate(games.iloc[split:])
print(f"  80/20 test RMSE: {test_metrics['combined_rmse']:.4f}, WinAcc: {test_metrics['win_accuracy']:.1%}")

# ── PHASE 4: Round 1 predictions ─────────────────────────────────
print("\n[4] Generating Round 1 predictions...")
matchups = pd.read_excel(str(_python_dir / 'data' / 'WHSDSC_Rnd1_matchups.xlsx'))
print(f"  Matchups loaded: {len(matchups)} games")
print()

predictions = []
print(f"{'Game':<6}{'Home':<16}{'Away':<17}{'PredH':>6}{'PredA':>6} {'Winner':<18}{'Conf':>6}")
print("-" * 75)

for _, row in matchups.iterrows():
    game_num = int(row['game'])
    game_id = str(row['game_id'])
    home = row['home_team']
    away = row['away_team']

    hp, ap = final_model.predict_goals({'home_team': home, 'away_team': away})
    winner, conf = final_model.predict_winner({'home_team': home, 'away_team': away})

    hp = round(hp, 2)
    ap = round(ap, 2)

    predictions.append({
        'game': game_num,
        'game_id': game_id,
        'home_team': home,
        'away_team': away,
        'pred_home_goals': hp,
        'pred_away_goals': ap,
        'predicted_winner': winner,
        'confidence': round(conf, 3),
    })

    print(f"{game_num:<6}{home:<16}{away:<17}{hp:>6.2f}{ap:>6.2f} {winner:<18}{conf:>5.1%}")

home_picks = sum(1 for p in predictions if p['predicted_winner'] == p['home_team'])
away_picks = len(predictions) - home_picks
avg_conf = np.mean([p['confidence'] for p in predictions])
print(f"\nHome picks: {home_picks}, Away picks: {away_picks}")
print(f"Avg confidence: {avg_conf:.1%}")

# ── Save everything ───────────────────────────────────────────────
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(OUTPUT / 'round1_elo_predictions.csv', index=False)

# Rankings
rankings = final_model.get_rankings()
off_rankings = final_model.get_off_rankings()

# Save model
final_model.save_model(str(MODEL_DIR / 'enhanced_elo'))

# Summary JSON
summary = {
    'model': 'Enhanced Elo v3 (venue-scaled additive + independent off/def)',
    'best_params': {k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)
                    for k, v in final_best_params.items()},
    'grid_search_configs': len(combos),
    'fine_sweep_configs': len(fine_combos),
    'test_metrics': {k: round(float(v), 6) if isinstance(v, float) else v
                     for k, v in test_metrics.items()},
    'comparison': {
        'enhanced_elo_rmse': round(test_metrics['combined_rmse'], 4),
    },
    'overall_rankings': {t: round(float(r), 1) for t, r in rankings},
    'offensive_rankings': {t: round(float(r), 1) for t, r in off_rankings},
    'predictions': predictions,
}

with open(OUTPUT / 'elo_pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 75)
print("ENHANCED ELO PIPELINE COMPLETE")
print("=" * 75)
print(f"  Comparison:  {OUTPUT / 'elo_comparison.csv'}")
print(f"  Predictions: {OUTPUT / 'round1_elo_predictions.csv'}")
print(f"  Summary:     {OUTPUT / 'elo_pipeline_summary.json'}")
print(f"  Model:       {MODEL_DIR / 'enhanced_elo.pkl'}")
