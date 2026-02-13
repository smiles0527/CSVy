"""
Full proxy validation pipeline:
  Step 1: Verify game_id ordering is chronological
  Step 2: Build all 4 proxies (game_gap, road_trip, lineup_deviation, schedule_density)
  Step 3: Sanity checks (groupby analysis)
  Step 4: Correlation matrix vs outcomes
  Step 5: Ablation test (baseline vs +proxy RMSE)
  Step 6: Report which proxies help
"""
import sys, os
sys.path.insert(0, 'python')

import pandas as pd
import numpy as np
from scipy import stats
from utils.hockey_features import aggregate_to_games, engineer_features, get_model_features

np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 140)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)
raw = pd.read_csv('python/data/whl_2025.csv')

# Convert game_id from "game_123" strings to integers for ordering
if raw['game_id'].dtype == object and raw['game_id'].str.startswith('game_').any():
    raw['game_id'] = raw['game_id'].str.replace('game_', '', regex=False).astype(int)

games = aggregate_to_games(raw)
print(f"Raw: {len(raw):,} shifts -> {len(games):,} games, {games['home_team'].nunique()} teams")
print(f"Game ID range: {games['game_id'].min()} - {games['game_id'].max()} (dtype: {games['game_id'].dtype})")

# ─────────────────────────────────────────────────────────────
# STEP 1: VERIFY GAME_ID ORDERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1: VERIFY GAME_ID IS CHRONOLOGICAL")
print("=" * 70)

# Check: do teams appear roughly evenly across game_ids?
# If chronological, each team should appear throughout the full range
team_appearances = []
for team in games['home_team'].unique():
    home_ids = games[games['home_team'] == team]['game_id'].values
    away_ids = games[games['away_team'] == team]['game_id'].values
    all_ids = sorted(np.concatenate([home_ids, away_ids]))
    team_appearances.append({
        'team': team,
        'first_game_id': all_ids[0],
        'last_game_id': all_ids[-1],
        'n_games': len(all_ids),
        'avg_gap': np.mean(np.diff(all_ids)) if len(all_ids) > 1 else 0,
        'min_gap': np.min(np.diff(all_ids)) if len(all_ids) > 1 else 0,
        'max_gap': np.max(np.diff(all_ids)) if len(all_ids) > 1 else 0,
    })

ta = pd.DataFrame(team_appearances).sort_values('first_game_id')

print(f"\nGame ID range: {games['game_id'].min()} - {games['game_id'].max()}")
print(f"All teams start near game 1: min first_game = {ta['first_game_id'].min()}, max first_game = {ta['first_game_id'].max()}")
print(f"All teams end near last game: min last_game = {ta['last_game_id'].min()}, max last_game = {ta['last_game_id'].max()}")
print(f"Average gap between appearances: {ta['avg_gap'].mean():.1f} game_ids (expect ~16 for 32 teams)")
print(f"Min gap ever: {ta['min_gap'].min()} (1 = back-to-back possible)")
print(f"Max gap ever: {ta['max_gap'].max()}")

# Check if teams interleave (chronological) vs cluster (by division)
# Chronological: each game_id has exactly 2 teams (home + away)
games_per_id = raw.groupby('game_id')[['home_team', 'away_team']].first()
unique_check = games_per_id.nunique(axis=0)
print(f"\nEach game_id has 1 home + 1 away team: {(games_per_id.nunique() == 1).all()}")

# Check: consecutive game_ids should feature different teams (interleaved schedule)
consecutive_same = 0
sorted_games = games.sort_values('game_id')
for i in range(1, len(sorted_games)):
    prev = sorted_games.iloc[i-1]
    curr = sorted_games.iloc[i]
    prev_teams = {prev['home_team'], prev['away_team']}
    curr_teams = {curr['home_team'], curr['away_team']}
    if prev_teams & curr_teams:  # shared team
        gap = curr['game_id'] - prev['game_id']
        if gap == 1:
            consecutive_same += 1

print(f"Back-to-back appearances (gap=1): {consecutive_same} out of {len(sorted_games)-1} consecutive pairs")
pct_b2b = consecutive_same / (len(sorted_games) - 1) * 100
print(f"  = {pct_b2b:.1f}% (expected ~10-20% for realistic hockey schedule)")

chronological = ta['avg_gap'].mean() > 5 and ta['first_game_id'].max() < 50
print(f"\n{'✓' if chronological else '✗'} game_id appears {'CHRONOLOGICAL' if chronological else 'NOT CLEARLY CHRONOLOGICAL'}")
print(f"  -> {'Safe to use as time proxy' if chronological else 'Use with caution'}")


# ─────────────────────────────────────────────────────────────
# STEP 2: BUILD ALL 4 PROXY FEATURES 
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: BUILD PROXY FEATURES")
print("=" * 70)

# -- Proxy 1: Game Gap (rest proxy) --
print("\n[Proxy 1] Game Gap (rest)...")
sorted_games = games.sort_values('game_id').copy()

# Build team game log with game_id ordering
team_game_ids = {}  # team -> [list of game_ids in order]
for _, g in sorted_games.iterrows():
    for team_col in ['home_team', 'away_team']:
        team = g[team_col]
        if team not in team_game_ids:
            team_game_ids[team] = []
        team_game_ids[team].append(g['game_id'])

# For each game, compute gap since team's previous game
home_gaps, away_gaps = [], []
team_prev = {}  # team -> last game_id
for _, g in sorted_games.iterrows():
    gid = g['game_id']
    ht, at = g['home_team'], g['away_team']
    
    h_gap = gid - team_prev.get(ht, gid - 16)  # default ~1 round gap
    a_gap = gid - team_prev.get(at, gid - 16)
    
    home_gaps.append(h_gap)
    away_gaps.append(a_gap)
    
    team_prev[ht] = gid
    team_prev[at] = gid

sorted_games['home_game_gap'] = home_gaps
sorted_games['away_game_gap'] = away_gaps
sorted_games['rest_diff'] = sorted_games['home_game_gap'] - sorted_games['away_game_gap']
sorted_games['home_b2b'] = (sorted_games['home_game_gap'] <= 1).astype(int)
sorted_games['away_b2b'] = (sorted_games['away_game_gap'] <= 1).astype(int)
print(f"  home_game_gap: mean={sorted_games['home_game_gap'].mean():.1f}, min={sorted_games['home_game_gap'].min()}, max={sorted_games['home_game_gap'].max()}")
print(f"  B2B rate: home={sorted_games['home_b2b'].mean():.1%}, away={sorted_games['away_b2b'].mean():.1%}")

# -- Proxy 2: Road Trip Length (travel proxy) --
print("\n[Proxy 2] Road Trip Length (travel)...")
team_home_away_seq = {}  # team -> [(game_id, is_home), ...]
for _, g in sorted_games.iterrows():
    gid = g['game_id']
    ht, at = g['home_team'], g['away_team']
    team_home_away_seq.setdefault(ht, []).append((gid, True))
    team_home_away_seq.setdefault(at, []).append((gid, False))

# Compute consecutive away/home streaks
team_road_trip = {}  # (team, game_id) -> road_trip_length
team_home_stand = {}  # (team, game_id) -> home_stand_length
for team, seq in team_home_away_seq.items():
    road_len = 0
    home_len = 0
    for gid, is_home in seq:
        if is_home:
            road_len = 0
            home_len += 1
        else:
            road_len += 1
            home_len = 0
        team_road_trip[(team, gid)] = road_len
        team_home_stand[(team, gid)] = home_len

sorted_games['away_road_trip_len'] = sorted_games.apply(
    lambda r: team_road_trip.get((r['away_team'], r['game_id']), 0), axis=1)
sorted_games['home_home_stand_len'] = sorted_games.apply(
    lambda r: team_home_stand.get((r['home_team'], r['game_id']), 0), axis=1)

print(f"  away_road_trip_len: mean={sorted_games['away_road_trip_len'].mean():.1f}, max={sorted_games['away_road_trip_len'].max()}")
print(f"  home_home_stand_len: mean={sorted_games['home_home_stand_len'].mean():.1f}, max={sorted_games['home_home_stand_len'].max()}")

# -- Proxy 3: Lineup Deviation (injury proxy) --
print("\n[Proxy 3] Lineup Deviation (injury)...")

# Work from raw shift data to get goalie + line info per game per team
game_lineup_info = raw.groupby('game_id').agg(
    home_goalie_primary=('home_goalie', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None),
    away_goalie_primary=('away_goalie', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None),
    home_n_off_lines=('home_off_line', 'nunique'),
    away_n_off_lines=('away_off_line', 'nunique'),
    home_n_def_pairs=('home_def_pairing', 'nunique'),
    away_n_def_pairs=('away_def_pairing', 'nunique'),
).reset_index()

# Build rolling "expected" goalie per team (most common in last 10 games)
sorted_games = sorted_games.merge(game_lineup_info, on='game_id', how='left')

# Track each team's recent goalies
team_goalie_history = {}  # team -> deque of last 10 goalies
from collections import deque, Counter

for _, g in sorted_games.iterrows():
    gid = g['game_id']
    
    for prefix, team_col in [('home', 'home_team'), ('away', 'away_team')]:
        team = g[team_col]
        goalie = g.get(f'{prefix}_goalie_primary')
        
        if team not in team_goalie_history:
            team_goalie_history[team] = deque(maxlen=10)
        
        history = team_goalie_history[team]
        if len(history) >= 3:
            # Most common goalie in recent history
            primary = Counter(history).most_common(1)[0][0]
            is_backup = 1 if goalie != primary else 0
        else:
            is_backup = 0
        
        sorted_games.loc[sorted_games['game_id'] == gid, f'{prefix}_backup_goalie'] = is_backup
        
        if goalie is not None:
            history.append(goalie)

# Lineup novelty: compare line combos to recent norms
# Use n_off_lines and n_def_pairs as proxies (more combos = more disruption/juggling)
team_avg_lines = {}
team_line_history = {}
for _, g in sorted_games.iterrows():
    gid = g['game_id']
    for prefix, team_col in [('home', 'home_team'), ('away', 'away_team')]:
        team = g[team_col]
        n_lines = g.get(f'{prefix}_n_off_lines', 0)
        n_dpairs = g.get(f'{prefix}_n_def_pairs', 0)
        
        if team not in team_line_history:
            team_line_history[team] = deque(maxlen=10)
        
        history = team_line_history[team]
        if len(history) >= 3:
            avg_lines = np.mean([h[0] for h in history])
            avg_dpairs = np.mean([h[1] for h in history])
            # Deviation from norm
            line_dev = abs(n_lines - avg_lines) / (avg_lines + 1)
            dpair_dev = abs(n_dpairs - avg_dpairs) / (avg_dpairs + 1)
            novelty = (line_dev + dpair_dev) / 2
        else:
            novelty = 0.0
        
        sorted_games.loc[sorted_games['game_id'] == gid, f'{prefix}_lineup_novelty'] = novelty
        history.append((n_lines, n_dpairs))

print(f"  Backup goalie starts: home={sorted_games['home_backup_goalie'].sum():.0f}, away={sorted_games['away_backup_goalie'].sum():.0f}")
print(f"  Lineup novelty: home mean={sorted_games['home_lineup_novelty'].mean():.3f}, away mean={sorted_games['away_lineup_novelty'].mean():.3f}")

# -- Proxy 4: Schedule Density --
print("\n[Proxy 4] Schedule Density...")

window_size = 20  # look back 20 game_ids
home_density, away_density = [], []
for _, g in sorted_games.iterrows():
    gid = g['game_id']
    ht, at = g['home_team'], g['away_team']
    
    # Count team's games in [gid-window, gid)
    h_games = [tid for tid in team_game_ids.get(ht, []) if gid - window_size <= tid < gid]
    a_games = [tid for tid in team_game_ids.get(at, []) if gid - window_size <= tid < gid]
    
    home_density.append(len(h_games))
    away_density.append(len(a_games))

sorted_games['home_schedule_density_20'] = home_density
sorted_games['away_schedule_density_20'] = away_density
sorted_games['schedule_density_diff'] = sorted_games['home_schedule_density_20'] - sorted_games['away_schedule_density_20']

print(f"  home_schedule_density_20: mean={sorted_games['home_schedule_density_20'].mean():.1f}, max={sorted_games['home_schedule_density_20'].max()}")
print(f"  away_schedule_density_20: mean={sorted_games['away_schedule_density_20'].mean():.1f}, max={sorted_games['away_schedule_density_20'].max()}")

proxy_cols = [
    'home_game_gap', 'away_game_gap', 'rest_diff', 'home_b2b', 'away_b2b',
    'away_road_trip_len', 'home_home_stand_len',
    'home_backup_goalie', 'away_backup_goalie', 'home_lineup_novelty', 'away_lineup_novelty',
    'home_schedule_density_20', 'away_schedule_density_20', 'schedule_density_diff',
]
print(f"\nTotal proxy features: {len(proxy_cols)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: SANITY CHECKS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: SANITY CHECKS (does it behave like reality?)")
print("=" * 70)

# Sanity 1: B2B teams should score fewer goals
print("\n--- Back-to-Back Impact ---")
home_b2b_goals = sorted_games.groupby('home_b2b')['home_goals'].mean()
away_b2b_goals = sorted_games.groupby('away_b2b')['away_goals'].mean()
home_b2b_wins = sorted_games.groupby('home_b2b')['home_win'].mean()

print(f"  Home goals when B2B=0: {home_b2b_goals.get(0, 'N/A'):.2f}, B2B=1: {home_b2b_goals.get(1, 'N/A'):.2f}")
print(f"  Away goals when B2B=0: {away_b2b_goals.get(0, 'N/A'):.2f}, B2B=1: {away_b2b_goals.get(1, 'N/A'):.2f}")
print(f"  Home win% when B2B=0: {home_b2b_wins.get(0, 'N/A'):.1%}, B2B=1: {home_b2b_wins.get(1, 'N/A'):.1%}")
b2b_makes_sense = home_b2b_goals.get(1, 99) <= home_b2b_goals.get(0, 0)
print(f"  {'✓' if b2b_makes_sense else '✗'} B2B teams score {'FEWER' if b2b_makes_sense else 'MORE'} goals (expected: fewer)")

# Sanity 2: Road trip fatigue
print("\n--- Road Trip Impact ---")
sorted_games['road_trip_bucket'] = pd.cut(sorted_games['away_road_trip_len'], 
                                          bins=[-1, 0, 1, 2, 5, 99], 
                                          labels=['home', '1st_road', '2nd_road', '3-5_road', '6+_road'])
road_impact = sorted_games.groupby('road_trip_bucket', observed=True).agg(
    away_goals=('away_goals', 'mean'),
    home_win_pct=('home_win', 'mean'),
    n_games=('game_id', 'count'),
).round(3)
print(road_impact.to_string())

# Sanity 3: Backup goalie impact
print("\n--- Backup Goalie Impact ---")
for prefix in ['home', 'away']:
    col = f'{prefix}_backup_goalie'
    goal_col = f'{prefix}_goals' if prefix == 'home' else f'{prefix}_goals'
    ga_col = 'away_goals' if prefix == 'home' else 'home_goals'
    
    if col in sorted_games.columns:
        grp = sorted_games.groupby(col).agg(
            goals_for=(f'{prefix}_goals', 'mean'),
            goals_against=(ga_col, 'mean'),
            n=('game_id', 'count'),
        ).round(3)
        print(f"  {prefix} team:")
        print(f"    Starter: GF={grp.loc[0.0, 'goals_for']:.2f}, GA={grp.loc[0.0, 'goals_against']:.2f} (n={grp.loc[0.0, 'n']:.0f})")
        if 1.0 in grp.index:
            print(f"    Backup:  GF={grp.loc[1.0, 'goals_for']:.2f}, GA={grp.loc[1.0, 'goals_against']:.2f} (n={grp.loc[1.0, 'n']:.0f})")

# Sanity 4: Rest diff impact
print("\n--- Rest Differential Impact ---")
sorted_games['rest_bucket'] = pd.cut(sorted_games['rest_diff'], 
                                     bins=[-99, -3, -1, 1, 3, 99],
                                     labels=['away_much_more', 'away_slightly', 'even', 'home_slightly', 'home_much_more'])
rest_impact = sorted_games.groupby('rest_bucket', observed=True).agg(
    home_win_pct=('home_win', 'mean'),
    avg_goal_diff=('goal_diff', 'mean'),
    n_games=('game_id', 'count'),
).round(3)
print(rest_impact.to_string())

# ─────────────────────────────────────────────────────────────
# STEP 4: CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: CORRELATION ANALYSIS")
print("=" * 70)

outcomes = ['home_goals', 'away_goals', 'home_win', 'goal_diff']
available_proxy = [c for c in proxy_cols if c in sorted_games.columns]

print(f"\nPearson correlations (proxy vs outcome):")
print(f"{'Proxy':<30} {'home_goals':>10} {'away_goals':>10} {'home_win':>10} {'goal_diff':>10} {'p-value':>10}")
print("-" * 90)

significant_proxies = []
for proxy in available_proxy:
    vals = sorted_games[proxy].dropna()
    if vals.std() == 0:
        print(f"{proxy:<30} {'(no variance)':>10}")
        continue
    
    corrs = []
    min_p = 1.0
    for outcome in outcomes:
        mask = sorted_games[proxy].notna() & sorted_games[outcome].notna()
        r, p = stats.pearsonr(sorted_games.loc[mask, proxy], sorted_games.loc[mask, outcome])
        corrs.append(f"{r:>10.4f}")
        min_p = min(min_p, p)
    
    sig = "**" if min_p < 0.01 else "*" if min_p < 0.05 else ""
    print(f"{proxy:<30} {'  '.join(corrs)}  {min_p:>8.4f} {sig}")
    
    if min_p < 0.10:  # Relaxed threshold for inclusion
        significant_proxies.append(proxy)

print(f"\nSignificant proxies (p<0.10): {significant_proxies}")

# ─────────────────────────────────────────────────────────────
# STEP 5: ABLATION TEST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: ABLATION TEST (does adding proxies improve prediction?)")
print("=" * 70)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Use the existing hockey features as baseline
game_features, _ = engineer_features(raw, rolling_window=10, verbose=False)

# Merge proxy features onto game_features
proxy_df = sorted_games[['game_id'] + available_proxy].copy()
game_features = game_features.merge(proxy_df, on='game_id', how='left')

# Define feature sets for ablation
base_features = [c for c in game_features.columns if c not in 
                 ['game_id', 'home_team', 'away_team', 'home_goals', 'away_goals',
                  'home_win', 'goal_diff', 'total_goals', 'went_ot',
                  'home_assists', 'home_shots', 'home_xg', 'home_max_xg',
                  'away_assists', 'away_shots', 'away_xg', 'away_max_xg',
                  'home_penalties_committed', 'home_penalty_minutes',
                  'away_penalties_committed', 'away_penalty_minutes',
                  'toi', 'n_shifts'] + available_proxy
                 and game_features[c].dtype in ['float64', 'int64', 'float32', 'int32']]

rest_proxy = ['home_game_gap', 'away_game_gap', 'rest_diff', 'home_b2b', 'away_b2b']
travel_proxy = ['away_road_trip_len', 'home_home_stand_len']
lineup_proxy = ['home_backup_goalie', 'away_backup_goalie', 'home_lineup_novelty', 'away_lineup_novelty']
density_proxy = ['home_schedule_density_20', 'away_schedule_density_20', 'schedule_density_diff']

rest_proxy = [c for c in rest_proxy if c in game_features.columns]
travel_proxy = [c for c in travel_proxy if c in game_features.columns]
lineup_proxy = [c for c in lineup_proxy if c in game_features.columns]
density_proxy = [c for c in density_proxy if c in game_features.columns]

configs = {
    'Baseline (84 features)': base_features,
    '+ Rest proxy': base_features + rest_proxy,
    '+ Travel proxy': base_features + travel_proxy,
    '+ Lineup proxy': base_features + lineup_proxy,
    '+ Schedule density': base_features + density_proxy,
    '+ ALL proxies': base_features + available_proxy,
}

# Use time-series split (expanding window)
tscv = TimeSeriesSplit(n_splits=5)
sorted_gf = game_features.sort_values('game_id').reset_index(drop=True)

results = {}
for config_name, feature_list in configs.items():
    actual_features = [f for f in feature_list if f in sorted_gf.columns]
    X = sorted_gf[actual_features].fillna(sorted_gf[actual_features].median())
    
    home_rmses, away_rmses, combined_rmses, win_accs = [], [], [], []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_home_train = sorted_gf.iloc[train_idx]['home_goals']
        y_home_test = sorted_gf.iloc[test_idx]['home_goals']
        y_away_train = sorted_gf.iloc[train_idx]['away_goals']
        y_away_test = sorted_gf.iloc[test_idx]['away_goals']
        
        # Train home goal model
        model_h = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42,
                                            learning_rate=0.1, subsample=0.8)
        model_h.fit(X_train, y_home_train)
        h_pred = model_h.predict(X_test)
        
        # Train away goal model
        model_a = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42,
                                            learning_rate=0.1, subsample=0.8)
        model_a.fit(X_train, y_away_train)
        a_pred = model_a.predict(X_test)
        
        h_rmse = np.sqrt(mean_squared_error(y_home_test, h_pred))
        a_rmse = np.sqrt(mean_squared_error(y_away_test, a_pred))
        c_rmse = np.sqrt(mean_squared_error(
            list(y_home_test) + list(y_away_test),
            list(h_pred) + list(a_pred)
        ))
        
        # Win accuracy
        pred_home_win = h_pred > a_pred
        actual_home_win = y_home_test.values > y_away_test.values
        win_acc = (pred_home_win == actual_home_win).mean()
        
        home_rmses.append(h_rmse)
        away_rmses.append(a_rmse)
        combined_rmses.append(c_rmse)
        win_accs.append(win_acc)
    
    results[config_name] = {
        'home_rmse': np.mean(home_rmses),
        'away_rmse': np.mean(away_rmses),
        'combined_rmse': np.mean(combined_rmses),
        'win_accuracy': np.mean(win_accs),
        'n_features': len(actual_features),
    }

print(f"\n{'Config':<25} {'N_feat':>6} {'Home_RMSE':>10} {'Away_RMSE':>10} {'Comb_RMSE':>10} {'Win_Acc':>8}")
print("-" * 75)

baseline_rmse = results['Baseline (84 features)']['combined_rmse']
for name, r in results.items():
    delta = r['combined_rmse'] - baseline_rmse
    delta_str = f"({delta:+.4f})" if name != 'Baseline (84 features)' else ""
    print(f"{name:<25} {r['n_features']:>6} {r['home_rmse']:>10.4f} {r['away_rmse']:>10.4f} {r['combined_rmse']:>10.4f} {r['win_accuracy']:>7.1%} {delta_str}")

# ─────────────────────────────────────────────────────────────
# STEP 6: RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: RECOMMENDATIONS")
print("=" * 70)

# Which proxies improved combined RMSE?
for name, r in results.items():
    if name == 'Baseline (84 features)':
        continue
    delta = r['combined_rmse'] - baseline_rmse
    win_delta = r['win_accuracy'] - results['Baseline (84 features)']['win_accuracy']
    if delta < -0.001 or win_delta > 0.005:
        print(f"  ✓ KEEP {name}: RMSE {delta:+.4f}, Win Acc {win_delta:+.1%}")
    elif abs(delta) < 0.001:
        print(f"  ~ NEUTRAL {name}: RMSE {delta:+.4f}, Win Acc {win_delta:+.1%} (marginal)")
    else:
        print(f"  ✗ DROP {name}: RMSE {delta:+.4f}, Win Acc {win_delta:+.1%} (hurts)")

all_delta = results['+ ALL proxies']['combined_rmse'] - baseline_rmse
all_win_delta = results['+ ALL proxies']['win_accuracy'] - results['Baseline (84 features)']['win_accuracy']
print(f"\n  NET IMPACT of all proxies: RMSE {all_delta:+.4f}, Win Acc {all_win_delta:+.1%}")

print("\nDone!")
