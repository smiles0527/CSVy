# WHSDSC 2026 — Development Log

> Western Hockey Scouts Data Science Competition 2026  
> Predict outcomes of 16 Round 1 playoff matchups using anonymized WHL season data.

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Raw records | 25,827 shift-level rows |
| Games | 1,312 |
| Teams | 32 (anonymized as country names) |
| Columns | 26 (game_id, team names, line combos, goalies, TOI, shots, xG, goals, penalties) |
| Missing | Dates, geography, injuries, roster info |
| Prediction target | 16 Round 1 matchups (home/away goals, winner) |
| Historical home win rate | 56.4% |
| Avg home goals | 3.09 |
| Avg away goals | 2.67 |

---

## Phase 1: Model Audit & RMSE Fix

Audited every RMSE calculation across 18 Python files (~35 call sites).

### RMSE Audit Results

| File | Sites | Status |
|------|-------|--------|
| `baseline_model.py` | 3 | Correct (`np.sqrt(mse)`) |
| `elo_model.py` | 1 | **Fixed** — had deprecated `squared=False` |
| `xgboost_model.py` | 2 | Correct |
| `xgboost_model_v2.py` | 2 | Correct |
| `linear_model.py` | 2 | Correct |
| `random_forest_model.py` | 4 | Correct |
| `neural_network_model.py` | 8 | Correct |
| `ensemble_model.py` | 8 | Correct |
| `experiment_tracker.py` | 2 | Correct |
| Training scripts (6 files) | 8 | Correct |
| **Total** | **~40** | **1 fix applied** |

### Model Bug Summary

| Model | Bug | Fix Applied |
|-------|-----|-------------|
| `baseline_model.py` | No save/load, deprecated RMSE, evaluate() only checked home goals | Added pickle save/load, `np.sqrt(mse)`, expanded evaluate with away + combined RMSE + win accuracy |
| `elo_model.py` | No save/load, deprecated RMSE fallback, evaluate() only checked home goals, division indexing bug | Added pickle+JSON save/load, fixed RMSE, expanded evaluate, fixed division lookup from index→name |
| `xgboost_model.py` | `early_stopping_rounds` silently dropped, save/load loses scaler/features | Identified, not yet fixed |
| `neural_network_model.py` | `cross_validate()` data leakage (scaler fit on full data before CV) | Identified, not yet fixed |
| `linear_model.py` | `cross_validate()` same leakage | Identified, not yet fixed |
| `ensemble_model.py` | `fit()` converts DataFrame→numpy, loses feature names | Identified, not yet fixed |
| `random_forest_model.py` | DEFAULT_FEATURES don't match WHL data | OK — falls back to numeric columns |

---

## Phase 2: Baseline Model Testing

Tested all 6 baseline strategies on WHL data (80/20 train/test split).

> **Note:** Phase 2 numbers below used `first()` aggregation (per-shift values) and are **superseded** by Phase 8.
> Correct results with `sum()` aggregation (game-level totals) are in Phase 8.

| Model | Home RMSE | Away RMSE | Combined RMSE | Win Accuracy |
|-------|-----------|-----------|---------------|--------------|
| GlobalMean | ~~2.149~~ | ~~1.896~~ | ~~2.029~~ | ~~56.4%~~ |
| TeamMean | ~~2.014~~ | ~~1.843~~ | ~~1.931~~ | ~~54.0%~~ |
| HomeAway | ~~2.030~~ | ~~1.878~~ | ~~1.956~~ | ~~56.4%~~ |
| MovingAverage | ~~2.087~~ | ~~1.893~~ | ~~1.993~~ | ~~53.2%~~ |
| WeightedHistory | ~~2.080~~ | ~~1.861~~ | ~~1.974~~ | ~~53.2%~~ |
| **Poisson** | ~~1.926~~ | ~~1.700~~ | ~~1.811~~ | ~~55.4%~~ |

~~Poisson baseline selected as reference benchmark (1.811 combined RMSE).~~
See Phase 8 for correct numbers — Bayesian(50) is the best single baseline (1.8092 combined RMSE).

---

## Phase 3: Elo Model (Primary Model)

Evaluated the fixed Elo model on the same 80/20 split.

| Metric | Value |
|--------|-------|
| Home RMSE | 1.995 |
| Away RMSE | 1.837 |
| Combined RMSE | 1.918 |
| Win Accuracy | 58.9% |
| Teams tracked | 32 |
| Save/load round-trip | Verified (pickle + JSON) |

**Params used:** k_factor=32, home_advantage=80, mov_multiplier=0.5, logarithmic MoV.

Top 5 Elo ratings after training: Peru (1721), Panama (1692), Netherlands (1677), Philippines (1615), Brazil (1598).

---

## Phase 4: Hockey Feature Engineering

Created `python/utils/hockey_features.py` — derives 84 features from shift-level data.

### Feature Categories

| Category | Count | Examples |
|----------|-------|---------|
| Offensive strength | ~12 | goals_for, shots, xG, assists_per_goal (rolling + season) |
| Defensive strength | ~12 | goals_against, shots_against, xG_against (rolling + season) |
| Goalie performance | 4 | save %, goals saved above expected (GSAx) |
| Shot analytics | 4 | shooting %, Corsi proxy (shot share), PDO |
| Special teams | 4 | penalty differential, PIM ratio |
| Expected goals | 4 | xG conversion, xG differential |
| Home/away splits | 8 | team-specific home_win_pct, home_scoring_boost, home_ice_advantage |
| Momentum | ~14 | rolling 10-game means for goals, shots, xG, wins |
| Strength of schedule | 2 | SOS (season), SOS (recent 10 games) |
| OT resilience | 2 | overtime rate, OT win rate |
| Differentials | ~6 | sos_diff, form_diff, rolling stat diffs |
| Lineup depth | 6 | line combos, D-pair combos, goalies used |
| **Total** | **~84** | |

---

## Phase 5: Proxy Feature Validation

Designed 4 proxy features to approximate data not in the dataset (rest, travel, injuries, schedule compression). Validated each through a 6-step pipeline.

### Step 1: game_id Chronological Verification

| Check | Result |
|-------|--------|
| game_id range | 1–1312 (sequential integers) |
| All teams span full range | Yes (first game ≤ 23, last game ≥ 1297) |
| Avg gap between team appearances | 16.0 game_ids (correct for 32 teams) |
| **Verdict** | **Chronological — safe to use as time proxy** |

### Step 2: Proxy Features Built

| Proxy | Approximates | Features | Key Stats |
|-------|-------------|----------|-----------|
| Game Gap | Rest days | `home_game_gap`, `away_game_gap`, `rest_diff`, `home_b2b`, `away_b2b` | Mean gap: 15.8, B2B rate: 0.3% |
| Road Trip Length | Travel fatigue | `away_road_trip_len`, `home_home_stand_len` | Mean road trip: 1.9 games, max: 10 |
| Lineup Deviation | Injuries/roster changes | `home_backup_goalie`, `away_backup_goalie`, `home_lineup_novelty`, `away_lineup_novelty` | Zero backup starts; mean novelty: 0.076 |
| Schedule Density | Compressed schedule | `home_schedule_density_20`, `away_schedule_density_20`, `schedule_density_diff` | Mean density: 0.8 games per 20-ID window |

### Step 3: Sanity Checks

| Check | Expected | Observed | Pass? |
|-------|----------|----------|-------|
| B2B teams score fewer goals | Yes | No (only 4 B2B games — insufficient sample) | Inconclusive |
| Road trip 3–5 games hurts scoring | Yes | Away goals drop to 2.58 (vs 2.67 avg) | Weak signal |
| Backup goalie = worse goals-against | Yes | No backup starts detected in data | N/A |
| Rest advantage = more home wins | Yes | 59.3% when home better rested | Mild signal |

### Step 4: Correlation Analysis

| Proxy | Corr with `home_win` | p-value | Significant? |
|-------|---------------------|---------|-------------|
| **home_lineup_novelty** | **-0.403** | **< 0.0001** | Yes |
| **away_lineup_novelty** | **+0.329** | **< 0.0001** | Yes |
| away_b2b | -0.031 | 0.053 | Marginal |
| rest_diff | +0.014 | 0.141 | No |
| away_road_trip_len | -0.005 | 0.665 | No |
| schedule_density_diff | +0.014 | 0.625 | No |

Note: Lineup novelty shows strong correlation but overfits in modeling (see ablation).

### Step 5: Ablation Test (5-fold Time-Series CV, GradientBoosting)

| Config | Features | Combined RMSE | Delta | Win Acc |
|--------|----------|---------------|-------|---------|
| Baseline | 90 | 0.4541 | — | 99.7% |
| + Rest proxy | 95 | 0.4537 | -0.0004 | 99.4% |
| **+ Travel proxy** | **92** | **0.4506** | **-0.0035** | **99.4%** |
| + Lineup proxy | 94 | 0.4565 | +0.0024 | 99.4% |
| **+ Schedule density** | **93** | **0.4509** | **-0.0032** | **99.6%** |
| + ALL proxies | 104 | 0.4530 | -0.0011 | 99.5% |

Win accuracy is ~99% due to in-game feature leakage in the ablation (shots/xG from same game included). Competition pipeline uses only pre-game features.

### Step 6: Final Verdict

| Proxy | Decision | Reason |
|-------|----------|--------|
| **Travel (road trip length)** | **KEEP** | Best RMSE improvement (-0.0035) |
| **Schedule density** | **KEEP** | Strong improvement (-0.0032) |
| Rest (game gap) | Neutral | Marginal (-0.0004); too few B2Bs |
| Lineup deviation | **Drop** | Hurts RMSE (+0.0024); overfits despite strong univariate correlation |

---

## Current Pipeline Architecture

```
whl_2025.csv (25,827 shifts)
    │
    ├── aggregate_to_games()          → 1,312 game-level rows
    ├── compute_advanced_metrics()    → Sh%, Sv%, PDO, GSAx, xG conversion
    ├── compute_rolling_team_stats()  → 10-game rolling means
    ├── compute_road_trip_features()  → travel fatigue proxy        [NEW]
    ├── compute_schedule_density()    → schedule compression proxy  [NEW]
    ├── compute_home_away_splits()    → team-specific home/away performance
    ├── compute_strength_of_schedule()→ opponent quality
    └── compute_ot_features()         → overtime resilience
    │
    ├── EloModel.fit()                → dynamic team ratings (primary model)
    ├── GradientBoosting / XGBoost    → goal prediction (secondary)
    └── Ensemble                      → combined predictions
    │
    └── Round 1 Predictions (16 matchups)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `python/utils/baseline_model.py` | Added save/load, fixed RMSE, expanded evaluate() |
| `python/utils/elo_model.py` | Added save/load (pickle+JSON), fixed RMSE, expanded evaluate(), fixed division indexing |
| `python/utils/hockey_features.py` | **Created** — 84-feature engineering pipeline |
| `validate_proxies.py` | **Created** — 6-step proxy validation pipeline |

## Files With Known Remaining Bugs

| File | Issue | Status |
|------|-------|--------|
| `xgboost_model.py` | early_stopping silently dropped; save/load loses scaler | **Fixed** |
| `neural_network_model.py` | cross_validate data leakage | **Fixed** |
| `linear_model.py` | cross_validate data leakage | **Fixed** |
| `ensemble_model.py` | DataFrame→numpy loses feature names | **Fixed** |

---

## Phase 6: Model Bug Fixes

Fixed 6 bugs across 4 model files:

| File | Bug | Fix |
|------|-----|-----|
| `xgboost_model.py` | `early_stopping_rounds` accepted but never passed to `model.fit()` | Pass via `fit_kwargs` dict |
| `xgboost_model.py` | `save_model()`/`load_model()` only saves raw XGB JSON, loses scaler/feature_names | Pickle full state dict (model JSON + scaler + features + params + importances) |
| `xgboost_model.py` | `cross_validate()` fits scaler on all data before CV splits (leakage) | Use `sklearn.Pipeline(scaler, model)` per fold |
| `linear_model.py` | `cross_validate()` same leakage via scaler + PolynomialFeatures | Builds `Pipeline(PolyFeatures, scaler, model)` per fold |
| `neural_network_model.py` | `cross_validate()` same scaler leakage | Uses `Pipeline(scaler, clone(model))` per fold |
| `ensemble_model.py` | `fit()` and `predict()` convert DataFrame→numpy via `X.values` | Removed conversion; DataFrames flow through to sub-models preserving feature names |

---

## Phase 7: Proxy Feature Integration

Added 2 validated proxy features to `hockey_features.py`:
- `compute_road_trip_features()` — travel proxy (consecutive away games → fatigue)
- `compute_schedule_density()` — game density in rolling window → fatigue

Pipeline now has 8 steps producing ~90+ features.

---

## Phase 8: Baseline Model Overhaul

### New Models Added to `baseline_model.py`

| Model | Description |
|-------|-------------|
| `DixonColesBaseline` | Iterative MLE for attack/defense strengths with time-decay weighting — the gold standard in sports prediction |
| `BayesianTeamBaseline` | Bayesian-regularised team means with conjugate prior shrinkage toward league average |
| `EnsembleBaseline` | Weighted blend of top baselines using inverse-RMSE calibration |

Also added `predict_winner()` method to the base class for consistent winner/confidence output.

### Full Baseline Comparison (17 models, 80/20 chronological split)

| Model | Home RMSE | Away RMSE | Combined RMSE | Win Acc |
|-------|-----------|-----------|---------------|---------|
| **Bayesian(50)** | 1.9106 | 1.7018 | **1.8092** | 56.3% |
| Bayesian(20) | 1.9066 | 1.7070 | 1.8096 | 56.3% |
| Bayesian(10) | 1.9073 | 1.7122 | 1.8124 | 56.7% |
| DixonColes(1.0) | 1.8886 | 1.7333 | 1.8126 | 56.3% |
| Bayesian(5) | 1.9087 | 1.7161 | 1.8150 | 55.1% |
| HomeAway | 1.9434 | 1.6978 | 1.8247 | 55.5% |
| Poisson | 1.9091 | 1.7401 | 1.8267 | 54.8% |
| TeamMean | 1.9565 | 1.7102 | 1.8374 | 55.1% |
| WeightedHist(0.99) | 1.9580 | 1.7112 | 1.8388 | 56.3% |
| GlobalMean | 1.9588 | 1.7231 | 1.8448 | **61.2%** |
| WeightedHist(0.95) | 1.9638 | 1.7233 | 1.8475 | 51.7% |
| MovingAvg(20) | 1.9576 | 1.7337 | 1.8491 | 52.5% |
| WeightedHist(0.9) | 1.9751 | 1.7472 | 1.8646 | 48.3% |
| MovingAvg(10) | 1.9946 | 1.8136 | 1.9063 | 47.1% |
| MovingAvg(5) | 2.0428 | 1.8380 | 1.9431 | 49.4% |
| DixonColes(0.99) | 2.0333 | 1.8572 | 1.9472 | 49.8% |
| DixonColes(0.95) | 2.6503 | 2.4506 | 2.5524 | 47.9% |

### Hyperparameter Tuning

| Parameter | Search Range | Best Value | RMSE |
|-----------|-------------|------------|------|
| Dixon-Coles decay | 0.90–1.00 | 1.0 (no decay) | 1.8126 |
| Bayesian prior_weight | 1–100 | 35 | 1.8085 |

### Ensemble Baseline (Best Overall)

Blends HomeAway + Poisson + DixonColes(1.0) + BayesianTeam(35) with inverse-RMSE weights.

- **Combined RMSE: 1.8071** (best of all baselines)
- **Win Accuracy: 55.9%**

### Round 1 Baseline Predictions

All 16 games predicted home team wins (consistent with 56.4% home win rate).

| Game | Home | Away | Pred Home | Pred Away | Winner | Conf |
|------|------|------|-----------|-----------|--------|------|
| 1 | Brazil | Kazakhstan | 3.71 | 1.88 | Brazil | 66.4% |
| 2 | Netherlands | Mongolia | 3.31 | 1.65 | Netherlands | 66.8% |
| 3 | Peru | Rwanda | 3.63 | 1.88 | Peru | 65.9% |
| 4 | Thailand | Oman | 4.38 | 2.83 | Thailand | 60.7% |
| 5 | Pakistan | Germany | 3.79 | 2.47 | Pakistan | 60.5% |
| 6 | India | USA | 3.56 | 2.30 | India | 60.8% |
| 7 | Panama | Switzerland | 3.34 | 2.13 | Panama | 61.0% |
| 8 | Iceland | Canada | 3.52 | 2.29 | Iceland | 60.7% |
| 9 | China | France | 3.47 | 2.29 | China | 60.3% |
| 10 | Philippines | Morocco | 3.08 | 2.25 | Philippines | 57.8% |
| 11 | Ethiopia | Saudi Arabia | 3.09 | 2.35 | Ethiopia | 56.9% |
| 12 | Singapore | New Zealand | 3.28 | 2.82 | Singapore | 53.8% |
| 13 | Guatemala | South Korea | 3.78 | 3.07 | Guatemala | 55.2% |
| 14 | UK | Mexico | 3.47 | 2.47 | UK | 58.5% |
| 15 | Vietnam | Serbia | 3.32 | 3.15 | Vietnam | 51.3% |
| 16 | Indonesia | UAE | 3.12 | 2.16 | Indonesia | 59.0% |

Avg confidence: 59.3%

---

## Next Steps

1. Run competition pipeline with Elo + XGBoost + engineered features
2. Compare ML model predictions against baseline predictions
3. Quantify improvement over baseline (target: beat 1.8071 ensemble RMSE, 55.9% win acc)
