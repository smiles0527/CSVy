#!/usr/bin/env python3
"""
Baseline Model — Full Pipeline
===============================
1. Compare ALL baselines (original 6 + new 3) on WHL data
2. Tune hyperparameters (window, decay, prior_weight, Dixon-Coles decay)
3. Build ensemble of top baselines
4. Generate Round 1 predictions from best model
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
from utils.hockey_features import aggregate_to_games
from utils.baseline_model import (
    GlobalMeanBaseline, TeamMeanBaseline, HomeAwayBaseline,
    MovingAverageBaseline, WeightedHistoryBaseline, PoissonBaseline,
    DixonColesBaseline, BayesianTeamBaseline, EnsembleBaseline,
)

OUTPUT = _python_dir / 'output' / 'predictions'
MODEL_DIR = _python_dir / 'output' / 'models'
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────
raw = pd.read_csv(str(_python_dir / 'data' / 'whl_2025.csv'))
if raw['game_id'].dtype == object:
    raw['game_id'] = raw['game_id'].str.replace('game_', '', regex=False).astype(int)

games = aggregate_to_games(raw).sort_values('game_id').reset_index(drop=True)
split = int(len(games) * 0.8)
train = games.iloc[:split]
test  = games.iloc[split:]

print("=" * 75)
print("BASELINE MODEL PIPELINE")
print("=" * 75)
print(f"Games: {len(games)}, Train: {len(train)}, Test: {len(test)}, Teams: {games['home_team'].nunique()}")
print(f"Home goals mean: {games['home_goals'].mean():.3f}, Away: {games['away_goals'].mean():.3f}")
print()

# ── PHASE 1: Compare all baselines ───────────────────────────────
print("[1] Comparing ALL baselines...")
print()

models = [
    ("GlobalMean",           GlobalMeanBaseline()),
    ("TeamMean",             TeamMeanBaseline()),
    ("HomeAway",             HomeAwayBaseline()),
    ("MovingAvg(5)",         MovingAverageBaseline({'window': 5})),
    ("MovingAvg(10)",        MovingAverageBaseline({'window': 10})),
    ("MovingAvg(20)",        MovingAverageBaseline({'window': 20})),
    ("WeightedHist(0.9)",    WeightedHistoryBaseline({'decay': 0.9})),
    ("WeightedHist(0.95)",   WeightedHistoryBaseline({'decay': 0.95})),
    ("WeightedHist(0.99)",   WeightedHistoryBaseline({'decay': 0.99})),
    ("Poisson",              PoissonBaseline()),
    ("DixonColes(1.0)",      DixonColesBaseline()),
    ("DixonColes(0.99)",     DixonColesBaseline({'decay': 0.99})),
    ("DixonColes(0.98)",     DixonColesBaseline({'decay': 0.98})),
    ("DixonColes(0.95)",     DixonColesBaseline({'decay': 0.95})),
    ("BayesianTeam(3)",      BayesianTeamBaseline({'prior_weight': 3})),
    ("BayesianTeam(5)",      BayesianTeamBaseline({'prior_weight': 5})),
    ("BayesianTeam(10)",     BayesianTeamBaseline({'prior_weight': 10})),
]

results = []
print(f"{'Model':<25} {'HomeRMSE':>9} {'AwayRMSE':>9} {'CombRMSE':>9} {'CombMAE':>8} {'WinAcc':>8}")
print("-" * 75)

for name, model in models:
    model.fit(train)
    m = model.evaluate(test)
    results.append({'name': name, 'model': model, **m})
    print(f"{name:<25} {m['rmse']:>9.4f} {m['away_rmse']:>9.4f} {m['combined_rmse']:>9.4f} {m['combined_mae']:>8.4f} {m['win_accuracy']:>7.1%}")

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])
results_df = results_df.sort_values('combined_rmse')
results_df.to_csv(str(OUTPUT / 'baseline_comparison.csv'), index=False)

print()
best_by_rmse = results_df.iloc[0]
best_by_acc = results_df.sort_values('win_accuracy', ascending=False).iloc[0]
print(f"Best by RMSE:    {best_by_rmse['name']} ({best_by_rmse['combined_rmse']:.4f})")
print(f"Best by Win Acc: {best_by_acc['name']} ({best_by_acc['win_accuracy']:.1%})")

# ── PHASE 2: Fine-tune Dixon-Coles & Bayesian via 5-fold CV ──────
print()
print("[2] Fine-tuning Dixon-Coles decay (5-fold time-based CV)...")
n_folds = 5
fold_size = len(games) // n_folds

def cv_rmse(model_factory, param):
    """5-fold block holdout CV (same structure as Elo pipeline)."""
    fold_rmses = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_df = games.iloc[val_start:val_end]
        train_df = pd.concat([games.iloc[:val_start], games.iloc[val_end:]])
        m = model_factory(param)
        m.fit(train_df)
        fold_rmses.append(m.evaluate(val_df)['combined_rmse'])
    return np.mean(fold_rmses), np.std(fold_rmses)

best_dc_decay, best_dc_rmse, best_dc_std = 1.0, 999, 0.0
for d_pct in range(90, 101):
    d = d_pct / 100.0
    mean_r, std_r = cv_rmse(lambda decay: DixonColesBaseline({'decay': decay}), d)
    if mean_r < best_dc_rmse:
        best_dc_decay, best_dc_rmse, best_dc_std = d, mean_r, std_r
print(f"  Best Dixon-Coles decay = {best_dc_decay}, CV RMSE = {best_dc_rmse:.4f} ± {best_dc_std:.4f}")

# Fine-tune BayesianTeam prior_weight
print("\n[3] Fine-tuning Bayesian prior weight (5-fold CV)...")
best_pw, best_pw_rmse, best_pw_std = 5, 999, 0.0
for pw in range(1, 21):
    mean_r, std_r = cv_rmse(lambda pw_val: BayesianTeamBaseline({'prior_weight': pw_val}), pw)
    if mean_r < best_pw_rmse:
        best_pw, best_pw_rmse, best_pw_std = pw, mean_r, std_r
print(f"  Best prior_weight = {best_pw}, CV RMSE = {best_pw_rmse:.4f} ± {best_pw_std:.4f}")

# ── PHASE 3: Build ensemble of top baselines ─────────────────────
print()
print("[4] Building ensemble baseline...")
top3 = [
    HomeAwayBaseline(),
    PoissonBaseline(),
    DixonColesBaseline({'decay': best_dc_decay}),
    BayesianTeamBaseline({'prior_weight': best_pw}),
]
ensemble = EnsembleBaseline({'models': top3, 'method': 'inverse_rmse'})
ensemble.fit(train)
ens_metrics = ensemble.evaluate(test)

print(f"  Ensemble RMSE:    {ens_metrics['combined_rmse']:.4f}")
print(f"  Ensemble Win Acc: {ens_metrics['win_accuracy']:.1%}")
print(f"  Sub-model weights: {ensemble.get_summary()['weights']}")

# ── PHASE 4: Identify best overall model ─────────────────────────
# Compare ensemble to singles
all_candidates = [
    ('DixonColes-tuned', DixonColesBaseline({'decay': best_dc_decay})),
    ('Bayesian-tuned',   BayesianTeamBaseline({'prior_weight': best_pw})),
    ('Ensemble',         ensemble),
]

print()
print("[5] Final model selection...")
print(f"{'Model':<25} {'CombRMSE':>9} {'WinAcc':>8}")
print("-" * 45)

best_final_model = None
best_final_rmse = 999
best_name = ''

for name, model in all_candidates:
    if name != 'Ensemble':
        model.fit(train)
    m = model.evaluate(test)
    print(f"{name:<25} {m['combined_rmse']:>9.4f} {m['win_accuracy']:>7.1%}")
    if m['combined_rmse'] < best_final_rmse:
        best_final_rmse = m['combined_rmse']
        best_final_model = model
        best_name = name

# Also include the original best
for r in results:
    if r['combined_rmse'] < best_final_rmse:
        best_final_rmse = r['combined_rmse']
        best_final_model = r['model']
        best_name = r['name']

print(f"\n>>> BEST BASELINE: {best_name} (RMSE={best_final_rmse:.4f})")

# ── PHASE 5: Train best on ALL data & predict Round 1 ────────────
print()
print("[6] Training best baseline on ALL data + predicting Round 1...")

# Re-fit on full dataset
best_final_model.fit(games)
best_final_model.save_model(str(MODEL_DIR / 'best_baseline.pkl'))

# Load matchups
matchups = pd.read_excel(str(_python_dir / 'data' / 'WHSDSC_Rnd1_matchups.xlsx'))
print(f"  Matchups loaded: {len(matchups)} games")
print()

predictions = []
print(f"{'Game':<5} {'Home':<15} {'Away':<15} {'PredH':>6} {'PredA':>6} {'Winner':<15} {'Conf':>6}")
print("-" * 75)

for _, matchup in matchups.iterrows():
    ht = matchup['home_team']
    at = matchup['away_team']
    h_goals, a_goals = best_final_model.predict_goals(matchup)
    winner, prob = best_final_model.predict_winner(matchup)
    confidence = max(prob, 1 - prob)

    predictions.append({
        'game': matchup['game'],
        'game_id': matchup['game_id'],
        'home_team': ht,
        'away_team': at,
        'pred_home_goals': round(h_goals, 2),
        'pred_away_goals': round(a_goals, 2),
        'predicted_winner': winner,
        'confidence': round(confidence, 3),
    })
    print(f"{matchup['game']:<5} {ht:<15} {at:<15} {h_goals:>6.2f} {a_goals:>6.2f} {winner:<15} {confidence:>5.1%}")

pred_df = pd.DataFrame(predictions)
pred_path = OUTPUT / 'round1_baseline_predictions.csv'
pred_df.to_csv(str(pred_path), index=False)

home_picks = (pred_df['predicted_winner'] == pred_df['home_team']).sum()
away_picks = len(pred_df) - home_picks
print(f"\nHome picks: {home_picks}, Away picks: {away_picks}")
print(f"Avg confidence: {pred_df['confidence'].mean():.1%}")

# ── PHASE 6: Save summary ────────────────────────────────────────
summary = {
    'best_model': best_name,
    'best_rmse': best_final_rmse,
    'tuned_params': {
        'dixon_coles_decay': best_dc_decay,
        'bayesian_prior_weight': best_pw,
        'cv_rmse_dc': round(best_dc_rmse, 4),
        'cv_rmse_bayesian': round(best_pw_rmse, 4),
        'cv_n_folds': n_folds,
    },
    'test_metrics': {k: v for k, v in ens_metrics.items()},
    'predictions': predictions,
}

with open(str(OUTPUT / 'baseline_pipeline_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print()
print("=" * 75)
print("BASELINE PIPELINE COMPLETE")
print("=" * 75)
print(f"  Comparison:  {OUTPUT / 'baseline_comparison.csv'}")
print(f"  Predictions: {pred_path}")
print(f"  Model:       {MODEL_DIR / 'best_baseline.pkl'}")
print(f"  Summary:     {OUTPUT / 'baseline_pipeline_summary.json'}")
print()
print("  For multi-run robustness + uncertainty (std):")
print("    python scripts/run/run_recursive.py --baseline-only --mode expand")
print("    python scripts/run/run_recursive.py --baseline-only --mode hyperparam")
