"""Quick audit of the new optimized submission."""
import io, sys, os
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))

pass_count, fail_count = 0, 0

def check(name, condition, detail=""):
    global pass_count, fail_count
    if condition:
        pass_count += 1
        print(f"  PASS  {name}")
    else:
        fail_count += 1
        print(f"  FAIL  {name}: {detail}")

print("=" * 50)
print("  QUICK AUDIT — Optimized v2 Submission")
print("=" * 50)

# Load files
sub = pd.read_csv('output/predictions/game_predictor/submission.csv')
matchups = pd.read_excel('data/WHSDSC_Rnd1_matchups.xlsx')

# Format checks
check("Has 16 rows", len(sub) == 16, f"got {len(sub)}")
check("Columns correct", list(sub.columns) == ['game_id', 'predicted_score_home', 'predicted_score_away'],
      f"got {list(sub.columns)}")
check("game_id are strings", all(isinstance(x, str) for x in sub['game_id']),
      f"types: {sub['game_id'].dtype}")
check("game_id format", all(x.startswith('game_') for x in sub['game_id']))
check("All 16 game_ids present",
      set(sub['game_id']) == {f'game_{i}' for i in range(1, 17)},
      f"missing: {set(f'game_{i}' for i in range(1, 17)) - set(sub['game_id'])}")

# Game ID alignment
if 'game_id' in matchups.columns:
    check("game_ids match matchups", set(sub['game_id']) == set(matchups['game_id']))

# Prediction sanity
check("Home preds all positive", (sub['predicted_score_home'] > 0).all())
check("Away preds all positive", (sub['predicted_score_away'] > 0).all())
check("Home preds in [0.5, 8]",
      (sub['predicted_score_home'] >= 0.5).all() and (sub['predicted_score_home'] <= 8).all(),
      f"range: {sub['predicted_score_home'].min():.2f}-{sub['predicted_score_home'].max():.2f}")
check("Away preds in [0.5, 8]",
      (sub['predicted_score_away'] >= 0.5).all() and (sub['predicted_score_away'] <= 8).all(),
      f"range: {sub['predicted_score_away'].min():.2f}-{sub['predicted_score_away'].max():.2f}")
check("Home mean near 3.09",
      abs(sub['predicted_score_home'].mean() - 3.09) < 0.5,
      f"got {sub['predicted_score_home'].mean():.2f}")
check("Away mean near 2.67",
      abs(sub['predicted_score_away'].mean() - 2.67) < 0.5,
      f"got {sub['predicted_score_away'].mean():.2f}")
check("Not all same prediction",
      sub['predicted_score_home'].std() > 0.1)
check("Total goals reasonable",
      4.5 < (sub['predicted_score_home'] + sub['predicted_score_away']).mean() < 7.0,
      f"avg total: {(sub['predicted_score_home'] + sub['predicted_score_away']).mean():.2f}")
check("No NaN values", sub.isna().sum().sum() == 0)
check("No duplicate game_ids", sub['game_id'].duplicated().sum() == 0)

print(f"\n  Result: {pass_count} PASS, {fail_count} FAIL")
if fail_count == 0:
    print("  ALL CHECKS PASSED")
print("=" * 50)
