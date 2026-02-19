#!/usr/bin/env python3
"""
Quick Elo regeneration — uses known best params, trains on all data,
outputs correctly-formatted Round 1 predictions with game_id column.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
from utils.hockey_features import aggregate_to_games
from utils.enhanced_elo_model import EnhancedEloModel

OUTPUT = Path(os.path.dirname(__file__)) / 'output' / 'predictions' / 'elo'
MODEL_DIR = Path(os.path.dirname(__file__)) / 'output' / 'models'
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────
raw = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'whl_2025.csv'))
if raw['game_id'].dtype == object:
    raw['game_id'] = raw['game_id'].str.replace('game_', '', regex=False).astype(int)

games = aggregate_to_games(raw).sort_values('game_id').reset_index(drop=True)

print("=" * 75)
print("ENHANCED ELO — REGENERATE OUTPUT (known best params)")
print("=" * 75)
print(f"Games: {len(games)}, Teams: {games['home_team'].nunique()}")
print()

# ── Best params from previous grid search ─────────────────────────
best_params = {
    'k_factor': 5,
    'home_advantage': 180,
    'mov_multiplier': 0.0,
    'xg_weight': 0.7,
    'k_decay': 0.01,
    'k_min': 8,
    'xg_pred_weight': 0.5,
    'elo_shift_scale': 0.7,
    'rolling_window': 20,
    'initial_rating': 1500,
    'shot_share_weight': 0.0,
    'penalty_weight': 0.0,
    'scoring_baseline': 'team',
}

# ── Quick 80/20 eval for reporting ────────────────────────────────
print("[1] 80/20 evaluation...")
split = int(len(games) * 0.8)
eval_model = EnhancedEloModel(best_params)
eval_model.fit(games.iloc[:split])
test_metrics = eval_model.evaluate(games.iloc[split:])
print(f"  Test RMSE: {test_metrics['combined_rmse']:.4f}, WinAcc: {test_metrics['win_accuracy']:.1%}")
print()

# ── Train on ALL data ─────────────────────────────────────────────
print("[2] Training Enhanced Elo on ALL data...")
final_model = EnhancedEloModel(best_params)
final_model.fit(games)
print(f"  Trained on {len(games)} games, {len(final_model.ratings)} teams")
print()

# ── Round 1 predictions ──────────────────────────────────────────
print("[3] Generating Round 1 predictions...")
matchups = pd.read_excel(os.path.join(os.path.dirname(__file__), 'data', 'WHSDSC_Rnd1_matchups.xlsx'))
print(f"  Matchups loaded: {len(matchups)} games")
print()

predictions = []
print(f"{'Game':<6}{'GameID':<10}{'Home':<16}{'Away':<17}{'PredH':>6}{'PredA':>6} {'Winner':<18}{'Conf':>6}")
print("-" * 85)

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

    print(f"{game_num:<6}{game_id:<10}{home:<16}{away:<17}{hp:>6.2f}{ap:>6.2f} {winner:<18}{conf:>5.1%}")

home_picks = sum(1 for p in predictions if p['predicted_winner'] == p['home_team'])
away_picks = len(predictions) - home_picks
avg_conf = np.mean([p['confidence'] for p in predictions])
print(f"\nHome picks: {home_picks}, Away picks: {away_picks}")
print(f"Avg confidence: {avg_conf:.1%}")

# ── Save everything ───────────────────────────────────────────────
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(OUTPUT / 'round1_elo_predictions.csv', index=False)
print(f"\nSaved: {OUTPUT / 'round1_elo_predictions.csv'}")

# Rankings
rankings = final_model.get_rankings()
off_rankings = final_model.get_off_rankings()

# Save model
final_model.save_model(str(MODEL_DIR / 'enhanced_elo'))

# Summary JSON
summary = {
    'model': 'Enhanced Elo v3 (venue-scaled additive + independent off/def)',
    'best_params': {k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)
                    for k, v in best_params.items()},
    'test_metrics': {k: round(float(v), 6) if isinstance(v, float) else v
                     for k, v in test_metrics.items()},
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
print(f"  Predictions: {OUTPUT / 'round1_elo_predictions.csv'}")
print(f"  Summary:     {OUTPUT / 'elo_pipeline_summary.json'}")
print(f"  Model:       {MODEL_DIR / 'enhanced_elo.pkl'}")
