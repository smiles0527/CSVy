#!/usr/bin/env python3
"""
Recursive Multi-Run Runner
==========================
Runs Baseline (and optionally Elo) N times with different strategies.
Saves: run_01.csv ... run_N.csv, average.csv, average_clean.csv, summary.json.

Modes:
  shuffle   - Random shuffles (default; baseline is stable, little variation)
  expand    - Expanding window: train on 40%, 55%, 70%, 85%, 100% of season
  bootstrap - Resample games with replacement per run
  hyperparam- Ensemble over prior_weight=10,15,20,25 and decay=0.98,1.0

Usage:
    python _run_recursive.py                    # 10 runs, both models
    python _run_recursive.py --baseline-only    # baseline only
    python _run_recursive.py --mode expand      # expanding-window (temporal)
    python _run_recursive.py --mode hyperparam  # hyperparameter ensemble
    python _run_recursive.py 20 --baseline-only --mode bootstrap
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
from pathlib import Path
from collections import Counter
from utils.hockey_features import aggregate_to_games
from utils.enhanced_elo_model import EnhancedEloModel


def safe_to_csv(df, path, retries=5, delay=1):
    """Write CSV with retry logic for Windows file locks."""
    for attempt in range(retries):
        try:
            df.to_csv(path, index=False)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                # Last resort: write to alternate filename
                alt = path.with_suffix(f'.tmp{int(time.time())}.csv')
                df.to_csv(alt, index=False)
                print(f"  WARNING: {path.name} locked, saved to {alt.name}")
from utils.baseline_model import (
    DixonColesBaseline, BayesianTeamBaseline,
    HomeAwayBaseline, PoissonBaseline, EnsembleBaseline,
    poisson_win_confidence,
)

def parse_args():
    p = argparse.ArgumentParser(description='Recursive multi-run baseline/elo predictor')
    p.add_argument('n_runs', nargs='?', type=int, default=10, help='Number of runs (default 10)')
    p.add_argument('--baseline-only', action='store_true', help='Run baseline only, skip Elo')
    p.add_argument('--mode', choices=['shuffle', 'expand', 'bootstrap', 'hyperparam'],
                   default='shuffle',
                   help='shuffle=random order; expand=40%%..100%%; bootstrap=resample; hyperparam=grid')
    return p.parse_args()

args = parse_args()
N_RUNS = args.n_runs
BASELINE_ONLY = args.baseline_only
MODE = args.mode

BASE_DIR   = Path(os.path.dirname(__file__))
OUTPUT_BL  = BASE_DIR / 'output' / 'predictions' / 'baseline' / 'recursive'
OUTPUT_ELO = BASE_DIR / 'output' / 'predictions' / 'elo' / 'recursive'
OUTPUT_BL.mkdir(parents=True, exist_ok=True)
OUTPUT_ELO.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────
raw = pd.read_csv(str(BASE_DIR / 'data' / 'whl_2025.csv'))
if raw['game_id'].dtype == object:
    raw['game_id'] = raw['game_id'].str.replace('game_', '', regex=False).astype(int)

games = aggregate_to_games(raw).sort_values('game_id').reset_index(drop=True)
matchups = pd.read_excel(str(BASE_DIR / 'data' / 'WHSDSC_Rnd1_matchups.xlsx'))

# Mode-specific run count
if MODE == 'expand':
    pcts = [0.40, 0.55, 0.70, 0.85, 1.0]
    N_RUNS = len(pcts)
elif MODE == 'hyperparam':
    HYPERPARAM_GRID = [
        {'prior_weight': pw, 'dc_decay': dc}
        for pw in [10, 15, 20, 25] for dc in [0.98, 1.0]
    ]
    N_RUNS = len(HYPERPARAM_GRID)

print("=" * 75)
print(f"RECURSIVE MULTI-RUN  mode={MODE}  runs={N_RUNS}  baseline_only={BASELINE_ONLY}")
print("=" * 75)
print(f"Games: {len(games)}, Teams: {games['home_team'].nunique()}, Matchups: {len(matchups)}")
print()

# ── Elo best params ──────────────────────────────────────────────
elo_params = {
    'k_factor': 5, 'home_advantage': 180, 'mov_multiplier': 0.0,
    'xg_weight': 0.7, 'k_decay': 0.01, 'k_min': 8,
    'xg_pred_weight': 0.5, 'elo_shift_scale': 0.7, 'rolling_window': 20,
    'initial_rating': 1500, 'shot_share_weight': 0.0, 'penalty_weight': 0.0,
    'scoring_baseline': 'team',
}

# ── Helper: predict matchups ─────────────────────────────────────
def predict_matchups_baseline(model, matchups_df):
    rows = []
    for _, row in matchups_df.iterrows():
        hp, ap = model.predict_goals(row)
        winner, prob = model.predict_winner(row)
        rows.append({
            'game': int(row['game']),
            'game_id': str(row['game_id']),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'pred_home_goals': round(hp, 4),
            'pred_away_goals': round(ap, 4),
            'predicted_winner': winner,
            'confidence': round(max(prob, 1 - prob), 4),
        })
    return pd.DataFrame(rows)


def predict_matchups_elo(model, matchups_df):
    rows = []
    for _, row in matchups_df.iterrows():
        hp, ap = model.predict_goals({'home_team': row['home_team'], 'away_team': row['away_team']})
        winner, conf = model.predict_winner({'home_team': row['home_team'], 'away_team': row['away_team']})
        rows.append({
            'game': int(row['game']),
            'game_id': str(row['game_id']),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'pred_home_goals': round(hp, 4),
            'pred_away_goals': round(ap, 4),
            'predicted_winner': winner,
            'confidence': round(conf, 4),
        })
    return pd.DataFrame(rows)


# ── Helper: build baseline ensemble ──────────────────────────────
def build_baseline(train_df, dc_decay=1.0, prior_weight=20):
    """Build the tuned ensemble baseline on given training data."""
    sub_models = [
        DixonColesBaseline({'decay': dc_decay}),
        BayesianTeamBaseline({'prior_weight': prior_weight}),
        HomeAwayBaseline(),
        PoissonBaseline(),
    ]
    ensemble = EnsembleBaseline({
        'models': sub_models,
        'method': 'inverse_rmse',
    })
    ensemble.fit(train_df)
    return ensemble


# ── Run loop ─────────────────────────────────────────────────────
all_bl_preds  = []
all_elo_preds = []
bl_rmses  = []
elo_rmses = []

header = f"{'Run':<5} {'Baseline RMSE':>14}"
if not BASELINE_ONLY:
    header += f" {'Elo RMSE':>14} {'BL WinAcc':>10} {'Elo WinAcc':>10}"
else:
    header += f" {'BL WinAcc':>10}"
print(header)
print("-" * (58 if not BASELINE_ONLY else 35))

for run in range(1, N_RUNS + 1):
    rng = np.random.RandomState(run * 42)

    if MODE == 'shuffle':
        train_df = games.sample(frac=1.0, random_state=rng).reset_index(drop=True)
        split = int(len(train_df) * 0.8)
        eval_train_df = train_df.iloc[:split]
        eval_val_df = train_df.iloc[split:]
        dc_decay, prior_weight = 1.0, 20
    elif MODE == 'expand':
        pct = pcts[run - 1]
        n = int(len(games) * pct)
        train_df = games.iloc[:n]
        split = int(len(train_df) * 0.8)
        eval_train_df = train_df.iloc[:split]
        eval_val_df = train_df.iloc[split:] if split < len(train_df) else train_df.iloc[:0]
        dc_decay, prior_weight = 1.0, 20
    elif MODE == 'bootstrap':
        idx = rng.choice(len(games), size=len(games), replace=True)
        train_df = games.iloc[idx].reset_index(drop=True)
        split = int(len(train_df) * 0.8)
        eval_train_df = train_df.iloc[:split]
        eval_val_df = train_df.iloc[split:]
        dc_decay, prior_weight = 1.0, 20
    else:  # hyperparam
        hp = HYPERPARAM_GRID[run - 1]
        train_df = games
        split = int(len(games) * 0.8)
        eval_train_df = games.iloc[:split]
        eval_val_df = games.iloc[split:]
        dc_decay = hp['dc_decay']
        prior_weight = hp['prior_weight']

    # ── Baseline ──
    bl_model = build_baseline(train_df, dc_decay=dc_decay, prior_weight=prior_weight)
    bl_eval = build_baseline(eval_train_df, dc_decay=dc_decay, prior_weight=prior_weight)
    bl_metrics = bl_eval.evaluate(eval_val_df) if len(eval_val_df) > 0 else {'combined_rmse': 0.0, 'win_accuracy': 0.0}
    bl_rmse = bl_metrics['combined_rmse']
    bl_wa = bl_metrics['win_accuracy']

    bl_df = predict_matchups_baseline(bl_model, matchups)
    safe_to_csv(bl_df, OUTPUT_BL / f'run_{run:02d}.csv')
    all_bl_preds.append(bl_df)
    bl_rmses.append(bl_rmse)

    # ── Elo ──
    if not BASELINE_ONLY:
        elo_model = EnhancedEloModel(elo_params)
        elo_model.fit(train_df)
        elo_eval = EnhancedEloModel(elo_params)
        elo_eval.fit(eval_train_df)
        elo_metrics = elo_eval.evaluate(eval_val_df) if len(eval_val_df) > 0 else {'combined_rmse': 0.0, 'win_accuracy': 0.0}
        elo_rmse = elo_metrics['combined_rmse']
        elo_wa = elo_metrics['win_accuracy']
        elo_df = predict_matchups_elo(elo_model, matchups)
        safe_to_csv(elo_df, OUTPUT_ELO / f'run_{run:02d}.csv')
        all_elo_preds.append(elo_df)
        elo_rmses.append(elo_rmse)
        print(f"{run:<5} {bl_rmse:>14.4f} {elo_rmse:>14.4f} {bl_wa:>9.1%} {elo_wa:>9.1%}")
    else:
        print(f"{run:<5} {bl_rmse:>14.4f} {bl_wa:>9.1%}")

# ── Compute averages ─────────────────────────────────────────────
print()
print("=" * 75)
print("AVERAGING ACROSS RUNS")
print("=" * 75)


def average_predictions(pred_list, output_dir, label):
    """Stack all runs, compute mean of goals, majority vote on winner, Poisson confidence from averaged goals."""
    home_goals = np.array([df['pred_home_goals'].values for df in pred_list])
    away_goals = np.array([df['pred_away_goals'].values for df in pred_list])

    avg_home = np.mean(home_goals, axis=0)
    avg_away = np.mean(away_goals, axis=0)
    std_home = np.std(home_goals, axis=0)
    std_away = np.std(away_goals, axis=0)

    # Recompute confidence from averaged goals (Poisson) — consistent with final prediction
    confidences = np.array([poisson_win_confidence(h, a) for h, a in zip(avg_home, avg_away)])

    # Majority vote for winner
    ref = pred_list[0]
    winners = []
    for g in range(len(ref)):
        votes = [df.iloc[g]['predicted_winner'] for df in pred_list]
        winner = Counter(votes).most_common(1)[0][0]
        winners.append(winner)

    avg_df = pd.DataFrame({
        'game':              ref['game'],
        'game_id':           ref['game_id'],
        'home_team':         ref['home_team'],
        'away_team':         ref['away_team'],
        'pred_home_goals':   np.round(avg_home, 2),
        'pred_away_goals':   np.round(avg_away, 2),
        'predicted_winner':  winners,
        'confidence':        np.round(confidences, 3),
        'std_home_goals':    np.round(std_home, 4),
        'std_away_goals':    np.round(std_away, 4),
    })

    safe_to_csv(avg_df, output_dir / 'average.csv')

    # Also save a clean version (no std cols) matching submission format
    clean = avg_df[['game', 'game_id', 'home_team', 'away_team',
                     'pred_home_goals', 'pred_away_goals',
                     'predicted_winner', 'confidence']].copy()
    safe_to_csv(clean, output_dir / 'average_clean.csv')

    print(f"\n{label} — Average predictions ({N_RUNS} runs):")
    print(f"{'Game':<6}{'Home':<16}{'Away':<17}{'AvgH':>6}{'AvgA':>6}{'StdH':>7}{'StdA':>7} {'Winner':<16}")
    print("-" * 82)
    for _, r in avg_df.iterrows():
        print(f"{int(r['game']):<6}{r['home_team']:<16}{r['away_team']:<17}"
              f"{r['pred_home_goals']:>6.2f}{r['pred_away_goals']:>6.2f}"
              f"{r['std_home_goals']:>7.4f}{r['std_away_goals']:>7.4f} {r['predicted_winner']:<16}")

    return avg_df


bl_avg = average_predictions(all_bl_preds, OUTPUT_BL, "BASELINE")
if not BASELINE_ONLY:
    elo_avg = average_predictions(all_elo_preds, OUTPUT_ELO, "ELO")

# ── Summary stats ────────────────────────────────────────────────
summary = {
    'n_runs': N_RUNS,
    'mode': MODE,
    'baseline_only': BASELINE_ONLY,
    'baseline': {
        'rmse_mean': round(float(np.mean(bl_rmses)), 4),
        'rmse_std': round(float(np.std(bl_rmses)), 4),
        'rmse_min': round(float(np.min(bl_rmses)), 4),
        'rmse_max': round(float(np.max(bl_rmses)), 4),
        'all_rmses': [round(float(r), 4) for r in bl_rmses],
    },
}
if not BASELINE_ONLY:
    summary['elo'] = {
        'rmse_mean': round(float(np.mean(elo_rmses)), 4),
        'rmse_std': round(float(np.std(elo_rmses)), 4),
        'rmse_min': round(float(np.min(elo_rmses)), 4),
        'rmse_max': round(float(np.max(elo_rmses)), 4),
        'all_rmses': [round(float(r), 4) for r in elo_rmses],
    }

with open(OUTPUT_BL.parent / 'recursive_summary.json', 'w') as f:
    json.dump({**summary, 'model': 'baseline'}, f, indent=2)
if not BASELINE_ONLY:
    with open(OUTPUT_ELO.parent / 'recursive_summary.json', 'w') as f:
        json.dump({**summary, 'model': 'elo'}, f, indent=2)

print()
print("=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"  Baseline RMSE: {np.mean(bl_rmses):.4f} ± {np.std(bl_rmses):.4f}  (range: {np.min(bl_rmses):.4f} – {np.max(bl_rmses):.4f})")
if not BASELINE_ONLY:
    print(f"  Elo RMSE:      {np.mean(elo_rmses):.4f} ± {np.std(elo_rmses):.4f}  (range: {np.min(elo_rmses):.4f} – {np.max(elo_rmses):.4f})")
print()
print("  Baseline files:")
print(f"    Individual: {OUTPUT_BL / 'run_01.csv'} ... run_{N_RUNS:02d}.csv")
print(f"    Average:    {OUTPUT_BL / 'average.csv'} (with std_home_goals, std_away_goals)")
print(f"    Clean avg:  {OUTPUT_BL / 'average_clean.csv'}")
if not BASELINE_ONLY:
    print()
    print("  Elo files:")
    print(f"    Individual: {OUTPUT_ELO / 'run_01.csv'} ... run_{N_RUNS:02d}.csv")
    print(f"    Average:    {OUTPUT_ELO / 'average.csv'}")
    print(f"    Clean avg:  {OUTPUT_ELO / 'average_clean.csv'}")
print()
print(f"  Summary:  {OUTPUT_BL.parent / 'recursive_summary.json'}")
