#!/usr/bin/env python3
"""Run baseline Elo xG sweep with k-value grid, validation, and plots (same output structure as baseline_elo)."""
import os
import pandas as pd
import numpy as np
import yaml
import json
import sys
import pathlib
from scipy import stats

_cwd = pathlib.Path(os.path.abspath('')).resolve()
if (_cwd / 'python').is_dir():
    _python_dir = _cwd / 'python'
elif _cwd.name == 'baseline_elo' and (_cwd.parent.parent / 'data').is_dir():
    _python_dir = _cwd.parent.parent
elif _cwd.name == 'training' and (_cwd.parent / 'data').is_dir():
    _python_dir = _cwd.parent
elif (_cwd / 'data').is_dir():
    _python_dir = _cwd
else:
    raise RuntimeError('Cannot locate python/')

os.chdir(_python_dir)
sys.path.insert(0, str(_python_dir))

# Load config early for live_dashboard check
proj_root = _python_dir.parent if (_python_dir.parent / 'config').is_dir() else _python_dir
config_path = proj_root / 'config' / 'hyperparams' / 'model_baseline_elo_xg_sweep.yaml'
_config_pre = {}
if config_path.exists():
    with open(config_path, 'r') as f:
        _config_pre = yaml.safe_load(f) or {}

# Backend: use display for live dashboard, else headless
_use_live = '--live' in sys.argv or _config_pre.get('live_dashboard', False)
if _use_live:
    import matplotlib
    for backend in ('TkAgg', 'Qt5Agg', 'GTK4Agg', 'WXAgg', 'MacOSX'):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
    else:
        _use_live = False
if not _use_live:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.baseline_elo_xg import BaselineEloXGModel


def _expand_param(val, default_list):
    if val is None:
        return default_list
    if isinstance(val, dict) and all(k in val for k in ('min', 'max', 'step')):
        mn, mx, st = float(val['min']), float(val['max']), float(val['step'])
        arr = np.arange(mn, mx + st / 2, st)
        return [round(float(x), 10) for x in arr]
    if isinstance(val, list):
        return val
    return default_list


config = _config_pre

data_cfg = config.get('data', {})
val_cfg = config.get('validation', {})
hp = config.get('hyperparameters', {})
out_cfg = config.get('output', {})

csv_path = data_cfg.get('csv_path', 'data/whl_2025.csv')
matchups_path = data_cfg.get('matchups_path', 'data/WHSDSC_Rnd1_matchups.xlsx')
train_ratio = val_cfg.get('train_ratio', 0.7)
formula_constants = {
    'initial_rating': hp.get('initial_rating', 1200),
    'elo_scale': hp.get('elo_scale', 400),
    'league_avg_goals': hp.get('league_avg_goals', 3.0),
    'goal_diff_half_range': hp.get('goal_diff_half_range', 6.0),
}

k_range = _expand_param(hp.get('k_factor'), list(range(5, 101, 5)))
if config.get('quick_test'):
    k_range = list(range(5, 25, 5))

out_dir = pathlib.Path(out_cfg.get('out_dir', 'output/predictions/baseline_elo_xg'))
sweep_dir = pathlib.Path(out_cfg.get('sweep_dir', str(out_dir / 'sweep')))
validation_dir = pathlib.Path(out_cfg.get('validation_dir', str(out_dir / 'validation')))
plots_dir = pathlib.Path(out_cfg.get('plots_dir', str(out_dir / 'plots')))
for d in (out_dir, sweep_dir, validation_dir, plots_dir):
    d.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(csv_path)
games_df = raw.groupby('game_id').agg(
    home_team=('home_team', 'first'),
    away_team=('away_team', 'first'),
    home_goals=('home_goals', 'sum'),
    away_goals=('away_goals', 'sum'),
    home_xg=('home_xg', 'sum'),
    away_xg=('away_xg', 'sum'),
    went_ot=('went_ot', 'max'),
).reset_index()
extracted = games_df['game_id'].astype(str).str.extract(r'(\d+)')
games_df['game_num'] = pd.to_numeric(extracted[0], errors='coerce').fillna(0).astype(int)
games_df = games_df.sort_values('game_num').reset_index(drop=True)

n = len(games_df)
test_size = int(n * (1 - train_ratio))
train_df = games_df.iloc[: n - test_size].copy()
test_df = games_df.iloc[n - test_size :].copy()

results = []
print(f'[Sweep] xG Elo: {len(k_range)} k-values')
if _use_live:
    try:
        from utils.live_dashboard import LivePlotter
        _plotter = LivePlotter(metrics=['win_accuracy', 'brier_loss', 'combined_rmse'], update_interval=1)
        _plotter.start(title='xG Elo K-Sweep')
    except Exception as e:
        print(f'[Live] Dashboard unavailable: {e}')
        _use_live = False
        _plotter = None
else:
    _plotter = None

for step_idx, k in enumerate(k_range):
    params = {'k_factor': k, **formula_constants}
    m = BaselineEloXGModel(params)
    m.fit(train_df)
    met = m.evaluate(test_df)
    brier_loss, log_loss = BaselineEloXGModel.compute_brier_logloss(m, test_df)
    results.append({'k': k, 'config': f"xG(k={k})", **met, 'brier_loss': brier_loss, 'log_loss': log_loss})
    if _use_live and _plotter is not None:
        _plotter.update({
            'win_accuracy': met.get('win_accuracy', 0),
            'brier_loss': brier_loss,
            'combined_rmse': met.get('combined_rmse', 0),
        }, epoch=step_idx)

if _use_live and _plotter is not None:
    try:
        _plotter.finalize()
    except Exception:
        pass

results_df = pd.DataFrame(results).sort_values('combined_rmse')
k_metrics_df = results_df[['k', 'win_accuracy', 'brier_loss', 'log_loss', 'combined_rmse']].rename(
    columns={'win_accuracy': 'accuracy'}
)

comp_csv = sweep_dir / 'comparison.csv'
k_csv = sweep_dir / 'k_metrics.csv'
results_df.to_csv(comp_csv, index=False)
k_metrics_df.to_csv(k_csv, index=False)
print(f'[OK] {comp_csv}')
print(f'[OK] {k_csv}')

best_row = results_df.iloc[0]
best_k = best_row['k']
print(f'\nBest k={best_k}')

# K vs metrics plots
metrics_to_plot = ['accuracy', 'brier_loss', 'log_loss', 'combined_rmse']
for metric in metrics_to_plot:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = k_metrics_df['k'].values
    y = k_metrics_df[metric].values
    ax.scatter(x, y, alpha=0.5, s=15)
    if len(x) > 2:
        coef = np.polyfit(x, y, min(3, len(x) - 1))
        x_smooth = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_smooth, np.polyval(coef, x_smooth), '--', alpha=0.8, label='best-fit')
    ax.set_xlabel('k')
    ax.set_ylabel(metric)
    ax.set_title(f'xG Elo: k vs {metric}')
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / f'k_vs_{metric}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[OK] {plots_dir / f"k_vs_{metric}.png"}')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, metric in enumerate(metrics_to_plot):
    ax = axes.flat[idx]
    ax.scatter(k_metrics_df['k'], k_metrics_df[metric], alpha=0.6, s=10)
    if len(k_metrics_df) > 2:
        coef = np.polyfit(k_metrics_df['k'], k_metrics_df[metric], min(2, len(k_metrics_df) - 1))
        x_s = np.linspace(k_metrics_df['k'].min(), k_metrics_df['k'].max(), 100)
        ax.plot(x_s, np.polyval(coef, x_s), '--', alpha=0.7)
    ax.set_xlabel('k')
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(True, alpha=0.3)
fig.suptitle('xG Elo: K vs Metrics Panel')
fig.tight_layout()
fig.savefig(plots_dir / 'k_vs_metrics_panel.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'[OK] {plots_dir / "k_vs_metrics_panel.png"}')

# Validation
n_test = len(test_df)
const_50_brier = 0.25
const_50_log = np.log(2)
league_avg_brier = 0.25
league_avg_log = np.log(2)
pd.DataFrame([
    {'baseline': 'constant_50', 'win_accuracy': 0.5, 'brier_loss': const_50_brier, 'log_loss': const_50_log},
    {'baseline': 'league_avg', 'win_accuracy': 0.5, 'brier_loss': league_avg_brier, 'log_loss': league_avg_log},
]).to_csv(validation_dir / 'baselines.csv', index=False)
print(f'[OK] {validation_dir / "baselines.csv"}')

acc = best_row['win_accuracy']
n_correct = int(round(acc * n_test))
p_val = 1.0 - stats.binom.cdf(n_correct - 1, n_test, 0.5) if acc > 0.5 else stats.binom.cdf(n_correct, n_test, 0.5)
pd.DataFrame([{'win_accuracy': acc, 'n_test': n_test, 'p_value': p_val}]).to_csv(validation_dir / 'significance.csv', index=False)
pd.DataFrame([{
    'win_accuracy': acc,
    'brier_loss': best_row['brier_loss'],
    'log_loss': best_row['log_loss'],
    'beats_constant_50_acc': acc > 0.5,
    'beats_brier': best_row['brier_loss'] < const_50_brier,
    'beats_log': best_row['log_loss'] < const_50_log,
}]).to_csv(validation_dir / 'comparison.csv', index=False)
print(f'[OK] {validation_dir / "significance.csv"}')
print(f'[OK] {validation_dir / "comparison.csv"}')

# Calibration (xG outcome)
final_model = BaselineEloXGModel({'k_factor': best_k, **formula_constants})
final_model.fit(train_df)
pred_probs = []
actual_wins = []
for _, game in test_df.iterrows():
    _, conf = final_model.predict_winner(game)
    pred_probs.append(conf)
    hg, ag = game.get('home_xg', 0), game.get('away_xg', 0)
    actual_wins.append(1 if hg > ag else 0)
pred_probs = np.array(pred_probs)
actual_wins = np.array(actual_wins)

bins = np.linspace(0, 1, 11)
cal_rows = []
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    mask = (pred_probs >= lo) & (pred_probs < hi) if hi < 1 else (pred_probs >= lo) & (pred_probs <= hi)
    if mask.sum() > 0:
        cal_rows.append({
            'bin_lo': lo, 'bin_hi': hi, 'pred_avg': pred_probs[mask].mean(),
            'actual_rate': actual_wins[mask].mean(), 'count': int(mask.sum()),
        })
pd.DataFrame(cal_rows).to_csv(validation_dir / 'calibration_stats.csv', index=False)
print(f'[OK] {validation_dir / "calibration_stats.csv"}')

fig, ax = plt.subplots(figsize=(6, 5))
if cal_rows:
    cal_df = pd.DataFrame(cal_rows)
    ax.scatter(cal_df['pred_avg'], cal_df['actual_rate'], s=cal_df['count'], alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.set_xlabel('Predicted home win probability (bin avg)')
ax.set_ylabel('Actual home win rate (xG)')
ax.set_title('xG Elo: Calibration')
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.savefig(plots_dir / 'calibration_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'[OK] {plots_dir / "calibration_plot.png"}')

# Elo vs standings (xG-based standings)
standings = {}
for _, g in games_df.iterrows():
    ht, at = g['home_team'], g['away_team']
    hxg, axg = g['home_xg'], g['away_xg']
    for t, w in [(ht, hxg > axg), (at, axg > hxg)]:
        standings[t] = standings.get(t, 0) + (1 if w else 0)
standings_rank = {t: r for r, (t, _) in enumerate(sorted(standings.items(), key=lambda x: -x[1]), 1)}
elo_rank = {t: r for r, (t, _) in enumerate(final_model.get_rankings(), 1)}
common = [t for t in elo_rank if t in standings_rank]
if len(common) >= 3:
    r, p = stats.spearmanr([elo_rank[t] for t in common], [standings_rank[t] for t in common])
else:
    r, p = np.nan, np.nan
pd.DataFrame([{'spearman_r': r, 'p_value': p}]).to_csv(validation_dir / 'elo_vs_standings.csv', index=False)
print(f'[OK] {validation_dir / "elo_vs_standings.csv"}')

summary_lines = [
    '# xG Elo Validation Summary',
    '',
    '## Naive Baselines',
    f'- Constant 50%: Brier~{const_50_brier}, Log~{const_50_log:.4f}',
    '',
    f'## Best (k={best_k})',
    f'- acc={acc:.1%}, brier={best_row["brier_loss"]:.4f}, log={best_row["log_loss"]:.4f}',
    f'- Beats constant 50%: {acc > 0.5}',
    f'- Significance p={p_val:.4f}',
    f'- Elo vs xG-standings: r={r:.4f}, p={p:.4f}',
]
with open(validation_dir / 'summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))
print(f'[OK] {validation_dir / "summary.md"}')

# Round 1 predictions
matchups = pd.read_excel(matchups_path)
hc = [c for c in matchups.columns if 'home' in c.lower()][0]
ac = [c for c in matchups.columns if 'away' in c.lower()][0]
final_model.fit(games_df)
preds = []
for i, row in matchups.iterrows():
    g = {'home_team': row[hc], 'away_team': row[ac]}
    h, a = final_model.predict_goals(g)
    w, c = final_model.predict_winner(g)
    preds.append({
        'game': i + 1, 'home_team': row[hc], 'away_team': row[ac],
        'pred_home': round(h, 2), 'pred_away': round(a, 2),
        'predicted_winner': w, 'confidence': round(c, 4),
    })
pred_df = pd.DataFrame(preds)
pred_df.to_csv(sweep_dir / 'round1_predictions.csv', index=False)
print(f'[OK] {sweep_dir / "round1_predictions.csv"}')

summary = {
    'model': 'BaselineEloXGSweep',
    'best_k': float(best_k),
    'train_ratio': train_ratio,
    'n_games': len(games_df),
    'n_train': len(train_df),
    'n_test': len(test_df),
    'best_params': {'k_factor': float(best_k), **formula_constants},
    'team_rankings': {t: round(r, 1) for t, r in final_model.get_rankings()},
    'predictions': preds,
}
with open(sweep_dir / 'pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[OK] {sweep_dir / "pipeline_summary.json"}')

print(f'\nDone. Best k={best_k} | acc={acc:.1%} brier={best_row["brier_loss"]:.4f}')
