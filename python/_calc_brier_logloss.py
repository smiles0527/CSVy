#!/usr/bin/env python3
"""Calculate Brier score and Log loss from baseline_elo predictions."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.baseline_elo import BaselineEloModel, get_value

# Load data
raw = pd.read_csv('data/whl_2025.csv')
games_df = raw.groupby('game_id').agg(
    home_team=('home_team','first'), away_team=('away_team','first'),
    home_goals=('home_goals','sum'), away_goals=('away_goals','sum'),
).reset_index()
extracted = games_df['game_id'].astype(str).str.extract(r'(\d+)')
games_df['game_num'] = pd.to_numeric(extracted[0], errors='coerce').fillna(0).astype(int)
games_df = games_df.sort_values('game_num').reset_index(drop=True)

# 70/30 split, fold 1
n_folds, fold_size = 3, len(games_df) // 3
val_start, val_end = 0, fold_size
train_df = pd.concat([games_df.iloc[:val_start], games_df.iloc[val_end:]], ignore_index=True)
test_df = games_df.iloc[val_start:val_end].copy()

# Fit model (k=16, init=1200)
model = BaselineEloModel({'k_factor': 16, 'initial_rating': 1200})
model.fit(train_df)

# For each test game: EA = P(home wins), OA = 1 if home wins else 0
brier_sum = 0.0
logloss_sum = 0.0
eps = 1e-10  # avoid log(0)

for _, game in test_df.iterrows():
    home_team = get_value(game, 'home_team')
    away_team = get_value(game, 'away_team')
    home_goals = get_value(game, 'home_goals', 0)
    away_goals = get_value(game, 'away_goals', 0)

    # EA = expected score for home (win prob)
    winner, conf = model.predict_winner(game)
    EA = conf if winner == home_team else (1 - conf)
    EB = 1 - EA

    # OA = actual outcome for home (1 win, 0 loss)
    OA = 1 if home_goals > away_goals else 0
    OB = 1 - OA

    # Clip for log
    EA_c = np.clip(EA, eps, 1 - eps)
    EB_c = np.clip(EB, eps, 1 - eps)

    brier_sum += (EA - OA) ** 2
    logloss_sum += -(OA * np.log(EA_c) + OB * np.log(EB_c))

n = len(test_df)
print(f"Test games: {n}")
print(f"Sum (EA - OA)^2 (Brier sum):     {brier_sum:.6f}")
print(f"Mean (EA - OA)^2 (Brier score):  {brier_sum/n:.6f}")
print(f"Sum -(OA*log(EA) + OB*log(EB)):  {logloss_sum:.6f}")
print(f"Mean -(OA*log(EA) + OB*log(EB)) (Log loss): {logloss_sum/n:.6f}")
