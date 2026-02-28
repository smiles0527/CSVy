#!/usr/bin/env python3
"""Run baseline Elo sweep across iterations 1.0, 1.1, 2.0 with k-value grid and validation."""
import pandas as pd
import numpy as np
import yaml
import json
import os
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

# Load config early for live_dashboard
proj_root = _python_dir.parent if (_python_dir.parent / 'config').is_dir() else _python_dir
config_path = proj_root / 'config' / 'hyperparams' / 'model_baseline_elo_sweep.yaml'
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

from utils.baseline_elo import BaselineEloModel
from utils.baseline_elo_xg import BaselineEloXGModel
from utils.baseline_elo_offdef import BaselineEloOffDefModel


def _expand_param(val, default_list):
    """Support list or {min, max, step} dict. Handles float steps via np.arange."""
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
_raw_iter = config.get('iterations', {})
iterations_cfg = {str(k): v for k, v in _raw_iter.items()}
if config.get('quick_test'):
    iterations_cfg = {'1.0': {'min': 5, 'max': 20, 'step': 5}, '1.1': {'min': 5, 'max': 20, 'step': 5}, '2.0': {'min': 5, 'max': 30, 'step': 5}}
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

out_dir = pathlib.Path(out_cfg.get('out_dir', 'output/predictions/baseline_elo'))
sweep_dir = pathlib.Path(out_cfg.get('sweep_dir', str(out_dir / 'sweep')))
validation_dir = pathlib.Path(out_cfg.get('validation_dir', str(out_dir / 'validation')))
plots_dir = pathlib.Path(out_cfg.get('plots_dir', str(out_dir / 'plots')))
for d in (out_dir, sweep_dir, validation_dir, plots_dir):
    d.mkdir(parents=True, exist_ok=True)

# Load and aggregate data
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


def get_fold_splits(games_df, fold):
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


train_df, test_df = get_fold_splits(games_df, 0)

# Model classes and data columns per iteration
ITERATION_CONFIG = {
    '1.0': {
        'model_class': BaselineEloModel,
        'train_cols': ['home_team', 'away_team', 'home_goals', 'away_goals'],
        'eval_outcome': 'goals',
    },
    '1.1': {
        'model_class': BaselineEloXGModel,
        'train_cols': ['home_team', 'away_team', 'home_xg', 'away_xg'],
        'eval_outcome': 'xg',
    },
    '2.0': {
        'model_class': BaselineEloOffDefModel,
        'train_cols': ['home_team', 'away_team', 'home_xg', 'away_xg'],
        'eval_outcome': 'xg',
    },
}

results = []
if _use_live:
    try:
        from utils.live_dashboard import LivePlotter
        _plotter = LivePlotter(metrics=['win_accuracy', 'brier_loss', 'combined_rmse'], update_interval=1)
        _plotter.start(title='Baseline Elo K-Sweep')
    except Exception as e:
        print(f'[Live] Dashboard unavailable: {e}')
        _use_live = False
        _plotter = None
else:
    _plotter = None

# For 2.0 (Off/Def): use shift-level data when home_off_line, away_off_line, toi exist
raw_has_lines = all(c in raw.columns for c in ['home_off_line', 'away_off_line', 'toi'])
train_raw = raw[raw['game_id'].isin(train_df['game_id'])] if raw_has_lines else None
test_raw = raw[raw['game_id'].isin(test_df['game_id'])] if raw_has_lines else None

_step_idx = 0
for model_iteration, cfg in ITERATION_CONFIG.items():
    k_range = _expand_param(iterations_cfg.get(model_iteration), [5, 10, 15, 20])
    model_class = cfg['model_class']
    print(f'[Sweep] Iteration {model_iteration}: {len(k_range)} k-values')
    for k in k_range:
        params = {'k_factor': k, **formula_constants}
        m = model_class(params)
        if model_iteration == '2.0' and train_raw is not None:
            m.fit(train_raw)
        else:
            m.fit(train_df)
        met = m.evaluate(test_df)
        brier_loss, log_loss = model_class.compute_brier_logloss(m, test_df)
        results.append({
            'model_iteration': model_iteration,
            'k': k,
            'config': f"{model_iteration}(k={k})",
            **met,
            'brier_loss': brier_loss,
            'log_loss': log_loss,
        })
        if _use_live and _plotter is not None:
            _plotter.update({
                'win_accuracy': met.get('win_accuracy', 0),
                'brier_loss': brier_loss,
                'combined_rmse': met.get('combined_rmse', 0),
            }, epoch=_step_idx)
        _step_idx += 1

if _use_live and _plotter is not None:
    try:
        _plotter.finalize()
    except Exception:
        pass

results_df = pd.DataFrame(results).sort_values(['model_iteration', 'combined_rmse'])
k_metrics_df = results_df[['model_iteration', 'k', 'win_accuracy', 'brier_loss', 'log_loss', 'combined_rmse']].rename(
    columns={'win_accuracy': 'accuracy'}
)

# Write CSVs to sweep/
comp_csv = pathlib.Path(out_cfg.get('comparison_csv', str(sweep_dir / 'comparison.csv')))
k_csv = pathlib.Path(out_cfg.get('k_metrics_csv', str(sweep_dir / 'k_metrics.csv')))
results_df.to_csv(comp_csv, index=False)
k_metrics_df.to_csv(k_csv, index=False)
print(f'[OK] {comp_csv}')
print(f'[OK] {k_csv}')

# Best params per iteration and overall
best_overall = results_df.iloc[0]
best_iteration = best_overall['model_iteration']
best_k = best_overall['k']
print(f'\nBest overall: iteration {best_iteration}, k={best_k}')

# K vs metrics plots with line of best fit
metrics_to_plot = ['accuracy', 'brier_loss', 'log_loss', 'combined_rmse']
for metric in metrics_to_plot:
    fig, ax = plt.subplots(figsize=(8, 5))
    for it in ['1.0', '1.1', '2.0']:
        sub = k_metrics_df[k_metrics_df['model_iteration'] == it]
        if len(sub) == 0:
            continue
        x = sub['k'].values
        y = sub[metric].values
        ax.scatter(x, y, alpha=0.5, s=10, label=f'Iteration {it}')
        if len(x) > 2:
            coef = np.polyfit(x, y, min(3, len(x) - 1))
            x_smooth = np.linspace(x.min(), x.max(), 200)
            y_fit = np.polyval(coef, x_smooth)
            ax.plot(x_smooth, y_fit, '--', alpha=0.8, label=f'{it} best-fit')
    ax.set_xlabel('k')
    ax.set_ylabel(metric)
    ax.set_title(f'k vs {metric} by model iteration')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / f'k_vs_{metric}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[OK] {plots_dir / f"k_vs_{metric}.png"}')

# 4-panel overview
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, metric in enumerate(metrics_to_plot):
    ax = axes.flat[idx]
    for it in ['1.0', '1.1', '2.0']:
        sub = k_metrics_df[k_metrics_df['model_iteration'] == it]
        if len(sub) == 0:
            continue
        ax.scatter(sub['k'], sub[metric], alpha=0.4, s=6, label=it)
        if len(sub) > 2:
            coef = np.polyfit(sub['k'], sub[metric], min(2, len(sub) - 1))
            x_s = np.linspace(sub['k'].min(), sub['k'].max(), 100)
            ax.plot(x_s, np.polyval(coef, x_s), '--', alpha=0.7)
    ax.set_xlabel('k')
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle('K vs Metrics Panel')
fig.tight_layout()
fig.savefig(plots_dir / 'k_vs_metrics_panel.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'[OK] {plots_dir / "k_vs_metrics_panel.png"}')

# Validation: naive baselines
n_test = len(test_df)
const_50_acc = (test_df['home_goals'] > test_df['away_goals']).mean() if n_test else 0.5
const_50_brier = 0.25
const_50_log = np.log(2)
league_avg_h = (test_df['home_goals'].mean() + test_df['away_goals'].mean()) / 2
league_avg_acc = 0.5  # tie-break random
league_avg_brier = 0.25
league_avg_log = np.log(2)
baseline_rows = [
    {'baseline': 'constant_50', 'win_accuracy': 0.5, 'brier_loss': const_50_brier, 'log_loss': const_50_log},
    {'baseline': 'league_avg', 'win_accuracy': league_avg_acc, 'brier_loss': league_avg_brier, 'log_loss': league_avg_log},
]
pd.DataFrame(baseline_rows).to_csv(pathlib.Path(out_cfg.get('validation_baselines_csv', str(validation_dir / 'baselines.csv'))), index=False)
print(f'[OK] {validation_dir / "baselines.csv"}')

# Validation: significance and comparison
best_per_iter = results_df.groupby('model_iteration').first().reset_index()
sig_rows = []
comp_rows = []
for _, row in best_per_iter.iterrows():
    it = row['model_iteration']
    acc = row['win_accuracy']
    n_correct = int(round(acc * n_test))
    p_val = 1.0 - stats.binom.cdf(n_correct - 1, n_test, 0.5) if acc > 0.5 else stats.binom.cdf(n_correct, n_test, 0.5)
    sig_rows.append({'model_iteration': it, 'win_accuracy': acc, 'n_test': n_test, 'p_value': p_val})
    comp_rows.append({
        'model_iteration': it,
        'win_accuracy': acc,
        'brier_loss': row['brier_loss'],
        'log_loss': row['log_loss'],
        'beats_constant_50_acc': acc > 0.5,
        'beats_brier': row['brier_loss'] < const_50_brier,
        'beats_log': row['log_loss'] < const_50_log,
    })
pd.DataFrame(sig_rows).to_csv(pathlib.Path(out_cfg.get('validation_significance_csv', str(validation_dir / 'significance.csv'))), index=False)
pd.DataFrame(comp_rows).to_csv(pathlib.Path(out_cfg.get('validation_comparison_csv', str(validation_dir / 'comparison.csv'))), index=False)
print(f'[OK] {validation_dir / "significance.csv"}')
print(f'[OK] {validation_dir / "comparison.csv"}')

# Calibration
final_model_class = ITERATION_CONFIG[best_iteration]['model_class']
final_params = {'k_factor': best_k, **formula_constants}
final_model = final_model_class(final_params)
final_model.fit(train_raw if best_iteration == '2.0' and train_raw is not None else train_df)

pred_probs = []
actual_wins = []
use_goals = best_iteration == '1.0'
for _, game in test_df.iterrows():
    _, conf = final_model.predict_winner(game)
    pred_probs.append(conf)
    if use_goals:
        hg, ag = game.get('home_goals', 0), game.get('away_goals', 0)
    else:
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
pd.DataFrame(cal_rows).to_csv(pathlib.Path(out_cfg.get('calibration_stats_csv', str(validation_dir / 'calibration_stats.csv'))), index=False)
print(f'[OK] {validation_dir / "calibration_stats.csv"}')

# Calibration plot (reliability diagram)
fig, ax = plt.subplots(figsize=(6, 5))
if cal_rows:
    cal_df = pd.DataFrame(cal_rows)
    ax.scatter(cal_df['pred_avg'], cal_df['actual_rate'], s=cal_df['count'], alpha=0.7, label='Bins')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.set_xlabel('Predicted home win probability (bin avg)')
ax.set_ylabel('Actual home win rate')
ax.set_title('Calibration (reliability diagram)')
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.savefig(plots_dir / 'calibration_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'[OK] {plots_dir / "calibration_plot.png"}')

# Elo vs standings correlation
standings = {}
for _, g in games_df.iterrows():
    ht, at = g['home_team'], g['away_team']
    hg, ag = g['home_goals'], g['away_goals']
    for t, w in [(ht, hg > ag), (at, ag > hg)]:
        standings[t] = standings.get(t, 0) + (1 if w else 0)
standings_rank = {t: r for r, (t, _) in enumerate(sorted(standings.items(), key=lambda x: -x[1]), 1)}

corr_rows = []
for it in ['1.0', '1.1', '2.0']:
    best_row = results_df[results_df['model_iteration'] == it].iloc[0]
    params = {'k_factor': best_row['k'], **formula_constants}
    m = ITERATION_CONFIG[it]['model_class'](params)
    m.fit(train_raw if it == '2.0' and train_raw is not None else train_df)
    elo_rank = {t: r for r, (t, _) in enumerate(m.get_rankings(), 1)}
    common = [t for t in elo_rank if t in standings_rank]
    if len(common) >= 3:
        r, p = stats.spearmanr([elo_rank[t] for t in common], [standings_rank[t] for t in common])
    else:
        r, p = np.nan, np.nan
    corr_rows.append({'model_iteration': it, 'spearman_r': r, 'p_value': p})
pd.DataFrame(corr_rows).to_csv(pathlib.Path(out_cfg.get('elo_vs_standings_csv', str(validation_dir / 'elo_vs_standings.csv'))), index=False)
print(f'[OK] {validation_dir / "elo_vs_standings.csv"}')

# Validation summary
summary_lines = [
    '# Validation Summary',
    '',
    '## Naive Baselines',
    f'- Constant 50%: Brier~{const_50_brier}, Log~{const_50_log:.4f}',
    f'- League avg: Brier~{league_avg_brier}, Log~{league_avg_log:.4f}',
    '',
    '## Best per iteration vs baselines',
]
for _, row in best_per_iter.iterrows():
    summary_lines.append(f"- **{row['model_iteration']}** (k={row['k']}): acc={row['win_accuracy']:.1%}, brier={row['brier_loss']:.4f}, log={row['log_loss']:.4f}")
    summary_lines.append(f"  - Beats constant 50% acc: {row['win_accuracy'] > 0.5}")
    summary_lines.append(f"  - Beats Brier baseline: {row['brier_loss'] < const_50_brier}")
summary_lines.extend([
    '',
    '## Statistical significance (win accuracy > 50%)',
])
for r in sig_rows:
    summary_lines.append(f"- {r['model_iteration']}: p={r['p_value']:.4f} {'(significant)' if r['p_value'] < 0.05 else '(not significant)'}")
summary_lines.extend([
    '',
    '## Elo vs standings (Spearman)',
])
for r in corr_rows:
    summary_lines.append(f"- {r['model_iteration']}: r={r['spearman_r']:.4f}, p={r['p_value']:.4f}")

with open(pathlib.Path(out_cfg.get('validation_summary_md', str(validation_dir / 'summary.md'))), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))
print(f'[OK] {validation_dir / "summary.md"}')

# Round 1 predictions
matchups = pd.read_excel(matchups_path)
hc = [c for c in matchups.columns if 'home' in c.lower()][0]
ac = [c for c in matchups.columns if 'away' in c.lower()][0]
final_model.fit(raw if best_iteration == '2.0' and raw_has_lines else games_df)
preds = []
for i, row in matchups.iterrows():
    g = {'home_team': row[hc], 'away_team': row[ac]}
    h, a = final_model.predict_goals(g)
    w, c = final_model.predict_winner(g)
    pred_type = 'goals' if best_iteration == '1.0' else 'xg'
    preds.append({
        'game': i + 1, 'home_team': row[hc], 'away_team': row[ac],
        'pred_home': round(h, 2), 'pred_away': round(a, 2),
        'predicted_winner': w, 'confidence': round(c, 4),
        'model_iteration': best_iteration, 'pred_type': pred_type,
    })
pred_df = pd.DataFrame(preds)
r1_csv = pathlib.Path(out_cfg.get('round1_csv', str(sweep_dir / 'round1_predictions.csv')))
pred_df.to_csv(r1_csv, index=False)
print(f'[OK] {r1_csv}')

# Per-iteration team rankings (so dashboard shows correct data for 1.0/1.1/2.0)
team_rankings_by_iteration = {}
for it in ['1.0', '1.1', '2.0']:
    row = best_per_iter[best_per_iter['model_iteration'] == it].iloc[0]
    params = {'k_factor': row['k'], **formula_constants}
    m = ITERATION_CONFIG[it]['model_class'](params)
    m.fit(train_raw if it == '2.0' and train_raw is not None else train_df)
    team_rankings_by_iteration[it] = {t: round(r, 1) for t, r in m.get_rankings()}

# Rating histograms: one per iteration (1.0, 1.1, 2.0)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
labels = {'1.0': 'Goals', '1.1': 'xG', '2.0': 'Off/Def'}
colors = {'1.0': '#1f77b4', '1.1': '#ff7f0e', '2.0': '#2ca02c'}
for idx, it in enumerate(['1.0', '1.1', '2.0']):
    ratings = list(team_rankings_by_iteration[it].values())
    axes[idx].hist(ratings, bins=15, edgecolor='black', alpha=0.7, color=colors[it])
    axes[idx].set_xlabel('Rating')
    axes[idx].set_ylabel('Count (teams)' if idx == 0 else '')
    axes[idx].set_title(f'{labels[it]} (iteration {it})')
    axes[idx].axvline(1200, color='gray', linestyle='--', alpha=0.7, label='Base 1200')
    axes[idx].legend()
fig.suptitle('Team rating distributions by iteration')
fig.tight_layout()
fig.savefig(plots_dir / 'rating_histograms.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'[OK] {plots_dir / "rating_histograms.png"}')

# Pipeline summary
summary = {
    'model': 'BaselineEloSweep',
    'iterations': ['1.0', '1.1', '2.0'],
    'best_iteration': best_iteration,
    'best_k': float(best_k),
    'train_ratio': train_ratio,
    'n_games': len(games_df),
    'n_train': len(train_df),
    'n_test': len(test_df),
    'best_params': {'k_factor': float(best_k), **formula_constants},
    'team_rankings': {t: round(r, 1) for t, r in final_model.get_rankings()},
    'team_rankings_by_iteration': team_rankings_by_iteration,
    'predictions': preds,
}
summary_path = pathlib.Path(out_cfg.get('summary_json', str(sweep_dir / 'pipeline_summary.json')))
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[OK] {summary_path}')

print(f'\nDone. Best: {best_iteration} k={best_k} | acc={best_overall["win_accuracy"]:.1%} brier={best_overall["brier_loss"]:.4f}')
