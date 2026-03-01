#!/usr/bin/env python3
"""Run baseline_elo pipeline (no matplotlib display)."""
import pandas as pd
import numpy as np
import yaml
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, sys, pathlib

_script = pathlib.Path(__file__).resolve()
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
from utils.baseline_elo import BaselineEloModel

# Load config from YAML (proj_root has config/)
proj_root = _python_dir if (_python_dir / 'config').is_dir() else _python_dir.parent
config_path = proj_root / 'config' / 'hyperparams' / 'model_baseline_elo.yaml'
config = None
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

data_cfg = config.get('data', {}) if config else {}
val_cfg = config.get('validation', {}) if config else {}
hp = config.get('hyperparameters', {}) if config else {}
out_cfg = config.get('output', {}) if config else {}

def _expand_param(val, default_list):
    """Support list or {min, max, step} dict for hyperparams. Handles float steps via np.arange."""
    if val is None:
        return default_list
    if isinstance(val, dict) and all(k in val for k in ('min', 'max', 'step')):
        mn, mx, st = float(val['min']), float(val['max']), float(val['step'])
        arr = np.arange(mn, mx + st / 2, st)
        return [round(float(x), 10) for x in arr]
    if isinstance(val, list):
        return val
    return default_list

# Grid keys (list/range) vs formula constants (single values)
GRID_KEYS = ['k_factor', 'initial_rating']
FORMULA_KEYS = ['elo_scale', 'league_avg_goals', 'goal_diff_half_range']

csv_path = data_cfg.get('csv_path', 'data/whl_2025.csv')
matchups_path = data_cfg.get('matchups_path', 'data/WHSDSC_Rnd1_matchups.xlsx')
n_folds = val_cfg.get('n_runs', 3)
train_ratio = val_cfg.get('train_ratio', 0.7)
param_grid = dict(
    k_factor=_expand_param(hp.get('k_factor'), list(range(5, 101, 5))),
    initial_rating=_expand_param(hp.get('initial_rating'), [1200]),
)
_formula_defaults = {'elo_scale': 400, 'league_avg_goals': 3.0, 'goal_diff_half_range': 6.0}
formula_constants = {k: hp.get(k, _formula_defaults[k]) for k in FORMULA_KEYS}
out_dir = pathlib.Path(out_cfg.get('out_dir', 'output/predictions/baseline_elo/goals'))
comp_csv = out_cfg.get('comparison_csv', str(out_dir / 'comparison.csv'))
k_csv = out_cfg.get('k_metrics_csv', str(out_dir / 'k_metrics.csv'))
r1_csv = out_cfg.get('round1_csv', str(out_dir / 'round1_predictions.csv'))
summary_json = out_cfg.get('summary_json', str(out_dir / 'pipeline_summary.json'))

raw = pd.read_csv(csv_path)
games_df = raw.groupby('game_id').agg(
    home_team=('home_team','first'), away_team=('away_team','first'),
    home_goals=('home_goals','sum'), away_goals=('away_goals','sum'),
    went_ot=('went_ot','max')).reset_index()
extracted = games_df['game_id'].astype(str).str.extract(r'(\d+)')
games_df['game_num'] = pd.to_numeric(extracted[0], errors='coerce').fillna(0).astype(int)
games_df = games_df.sort_values('game_num').reset_index(drop=True)

n = len(games_df)
test_size = int(n * (1 - train_ratio))  # 30% test
def get_fold_splits(games_df, fold):
    """Block holdout: train ~70%, test ~30%. 3 folds = 3 different block positions."""
    # Fold 0: train first 70%, test last 30%
    # Fold 1: train first 35% + last 35%, test middle 30%
    # Fold 2: train last 70%, test first 30%
    if fold == 0:
        train = games_df.iloc[: n - test_size].copy()
        test = games_df.iloc[n - test_size :].copy()
    elif fold == 1:
        half = (n - test_size) // 2
        train = pd.concat([games_df.iloc[:half], games_df.iloc[half + test_size :]], ignore_index=True)
        test = games_df.iloc[half : half + test_size].copy()
    else:
        train = games_df.iloc[test_size:].copy()
        test = games_df.iloc[:test_size].copy()
    return train, test

keys = list(param_grid.keys())
combos = list(product(*[param_grid[k] for k in keys]))
train_df, test_df = get_fold_splits(games_df, 0)

results = []
for vals in combos:
    params = dict(zip(keys, vals))
    params.update(formula_constants)
    m = BaselineEloModel(params)
    m.fit(train_df)
    met = m.evaluate(test_df)
    brier_loss, log_loss = BaselineEloModel.compute_brier_logloss(m, test_df)
    results.append({'config': f"BaselineElo(k={params['k_factor']},init=1200)", **params, **met, 'brier_loss': brier_loss, 'log_loss': log_loss})
results_df = pd.DataFrame(results).sort_values('combined_rmse')
best_params = {'k_factor': int(results_df.iloc[0]['k_factor']), 'initial_rating': 1200, **formula_constants}
k_metrics_df = results_df[['k_factor', 'win_accuracy', 'brier_loss', 'log_loss']].rename(
    columns={'k_factor': 'k', 'win_accuracy': 'accuracy'}
)

out_dir.mkdir(parents=True, exist_ok=True)
results_df.to_csv(comp_csv, index=False)
print(f'[OK] {comp_csv}')

k_metrics_df.to_csv(k_csv, index=False)
print(f'[OK] {k_csv}')

print(f'CWD: {os.getcwd()}')
print(f'Raw: {len(raw)} rows, Games: {len(games_df)}, Teams: {games_df["home_team"].nunique()}')
print(f'Split: {int(100*train_ratio)}% train / {int(100*(1-train_ratio))}% test, {n_folds} folds, ~{len(train_df)} train / ~{len(test_df)} test')
print(f'\n[OK] {len(results_df)} configs. Best: k={best_params["k_factor"]}, init=1200')

per_run = []
for fold in range(n_folds):
    train_f, test_f = get_fold_splits(games_df, fold)
    m = BaselineEloModel(best_params)  # best_params includes formula_constants
    m.fit(train_f)
    met = m.evaluate(test_f)
    per_run.append(met)
    print(f'Run {fold+1}: combined_rmse={met["combined_rmse"]:.4f}, win_accuracy={met["win_accuracy"]:.1%}')

rmse_m, rmse_s = np.mean([x['combined_rmse'] for x in per_run]), np.std([x['combined_rmse'] for x in per_run])
acc_m, acc_s = np.mean([x['win_accuracy'] for x in per_run]), np.std([x['win_accuracy'] for x in per_run])
print(f'\nSummary (mean +/- std over {n_folds} runs):')
print(f'  Combined RMSE:  {rmse_m:.4f} +/- {rmse_s:.4f}')
print(f'  Win Accuracy:   {acc_m:.1%} +/- {acc_s:.1%}')

final = BaselineEloModel(best_params)
final.fit(games_df)
print(f'\nTeam Rankings (top 10):')
for r, (t, elo) in enumerate(final.get_rankings(10), 1):
    print(f'  {r:2d}. {t:20s}  {elo:.1f}')

# Round 1
matchups = pd.read_excel(matchups_path)
hc = [c for c in matchups.columns if 'home' in c.lower()][0]
ac = [c for c in matchups.columns if 'away' in c.lower()][0]
print(f'\nRound 1 ({len(matchups)} matchups):')
preds = []
for i, row in matchups.iterrows():
    g = {'home_team': row[hc], 'away_team': row[ac]}
    h, a = final.predict_goals(g)
    w, c = final.predict_winner(g)
    preds.append({'game': i+1, 'home_team': row[hc], 'away_team': row[ac], 'pred_home_goals': round(h, 2),
                  'pred_away_goals': round(a, 2), 'predicted_winner': w, 'confidence': round(c, 4)})
    print(f'  {i+1:2d}. {row[hc]:15s} vs {row[ac]:15s} -> {w} ({c:.1%}) pred {h:.1f}-{a:.1f}')

pred_df = pd.DataFrame(preds)
pred_df.to_csv(r1_csv, index=False)
print(f'\n[OK] {r1_csv}')

summary = {
    'model': 'BaselineElo',
    'train_ratio': val_cfg.get('train_ratio', 0.7),
    'n_runs': n_folds,
    'multi_run_metrics': {
        'combined_rmse_mean': round(rmse_m, 6),
        'combined_rmse_std': round(rmse_s, 6),
        'win_accuracy_mean': round(acc_m, 6),
        'win_accuracy_std': round(acc_s, 6),
    },
    'per_run_metrics': [{'run': i+1, 'combined_rmse': round(x['combined_rmse'], 6), 'win_accuracy': round(x['win_accuracy'], 6)} for i, x in enumerate(per_run)],
    'best_params': best_params,
    'team_rankings': {t: round(r, 1) for t, r in final.get_rankings()},
    'predictions': preds,
}
with open(summary_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[OK] {summary_json}')

print(f'\nSUMMARY: RMSE {rmse_m:.4f} +/- {rmse_s:.4f}, Win Acc {acc_m:.1%} +/- {acc_s:.1%}')
