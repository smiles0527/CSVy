# Model Values Index

Quick reference: **which value does what** across all prediction models.

---

## Baseline Elo Family (1.0, 1.1, 2.0)

| Value | Default | Models | Purpose |
|-------|---------|--------|---------|
| `k_factor` | 32 | all | How much ratings change per game. Higher = more volatile. Sweep tunes over 5–30 or 0.1–100. |
| `initial_rating` | 1200 | all | Starting Elo for new teams. Baseline Elo uses 1200 (Enhanced Elo uses 1500). |
| `elo_scale` | 400 | all | Denominator in win-prob: P = 1/(1+10^(-diff/scale)). 400 = standard Elo; 200-point diff ≈ 76% win prob. |
| `league_avg_goals` | 3.0 | 1.0, 1.1 | Fallback when converting win prob → predicted goals. |
| `goal_diff_half_range` | 6.0 | 1.0, 1.1 | Max goals swing from win prob: goals = league_avg ± (range × (win_prob − 0.5)). |
| `league_avg_xg` | (computed) | 2.0 | xG per hour 5v5 from training data. Used for expected xG in Off/Def updates. |
| `time_factor` | 1.0 | 2.0 | Multiplier for game-level expected xG (60 min game = 1.0). |
| `LN10` | ln(10) ≈ 2.3026 | 2.0 | Scales delta: k×(obs−exp)/LN10. Matches log-based Elo convention. |

### Off/Def (2.0) specifics

| Value | Purpose |
|-------|---------|
| `O` | Offense rating per team, per line (first_off, second_off). Higher = scores more. |
| `D` | Defense rating per team, per line. Lower = allows fewer goals. |
| `Net (O−D)` | Team strength = avg(O−D). Can be negative (league-wide D > O is normal). Higher = stronger. |
| `multi` | 10^((O−D)/400). Multiplies league_avg_xg for expected xG. |
| `delta` | k×(obs−exp)/LN10. Added to O, subtracted from D. |

---

## EloModel (full-featured Elo)

| Value | Default | Purpose |
|-------|---------|---------|
| `initial_rating` | 1500 | Base Elo; D1/D2/D3 get +100/0/−100. |
| `k_factor` | 32 | Base rating change; scaled by margin of victory. |
| `home_advantage` | — | Elo points added for home team. |
| `mov_multiplier` | 0 | 0 = ignore margin; >0 = blowouts move ratings more. |
| `mov_method` | 'logarithmic' | 'linear' or 'logarithmic' for MOV scaling. |
| `ot_win_multiplier` | 0.75 | OT win counts as partial win. |
| `rest_advantage_per_day` | — | Rating boost per day of rest. |
| `b2b_penalty` | — | Penalty for back-to-back games. |
| 400 | hardcoded | Elo scale in calculate_expected_score. |

---

## Enhanced Elo (Off/Def + extras)

| Value | Default | Purpose |
|-------|---------|---------|
| `initial_rating` | 1500 | Base for O, D, and overall Elo. |
| `k_factor` | 32 | Base K; can decay with games played. |
| `k_decay` | 0 | How much K shrinks per game (dynamic K). |
| `k_min` | 8 | Floor for K after decay. |
| `home_advantage` | 100 | Home ice Elo boost. |
| `mov_multiplier` | 1.0 | Margin-of-victory weight. |
| `xg_weight` | 0 | Blend of xG vs goals for MOV calc. |
| `rolling_window` | 15 | Games for rolling averages. |
| `shot_share_weight` | 0 | Corsi-style adjustment. |
| `penalty_weight` | 0 | Penalty differential adjustment. |
| `scoring_baseline` | 'team' | 'league' or 'team' for venue scaling. |
| `xg_pred_weight` | 0 | Blend xG into goal predictions. |
| `elo_shift_scale` | 2.0 | How Elo diff maps to goal predictions. |
| 400 | hardcoded | Elo scale in expected_score. |

---

## Baseline Model Family

| Value | Model | Purpose |
|-------|-------|---------|
| `window` | MovingAverage | Number of recent games for moving avg. |
| `decay` | WeightedHistory | Exponential decay factor (0.9 = recent games matter more). |
| `max_iter` | DixonColes | Max EM iterations for Poisson params. |
| `tol` | DixonColes | Convergence tolerance. |
| `home_adv` | DixonColes | Home advantage multiplier (~1.15). |
| `prior_weight` | BayesianTeam | Strength of prior toward league average. |
| `models` | EnsembleBaseline | Sub-models to ensemble. |
| `weights` | EnsembleBaseline | Optional fixed weights. |
| `method` | EnsembleBaseline | 'inverse_rmse' or similar. |

---

## Output / Pipeline Values

| Value | Source | Purpose |
|-------|--------|---------|
| `team_rankings` | get_rankings() | 1.0/1.1: Elo (1100–1300). 2.0: Net O−D (−120 to −65). |
| `confidence` | predict_winner() | Win probability 0–1. |
| `P` (dashboard) | rating − mean | P(team beats avg) = 1/(1+10^(-diff/400)). |
| `train_ratio` | config | 0.7 = 70% train, 30% test. |
| `const_50_brier` | validation | 0.25 = Brier for random 50/50. |
| `const_50_log` | validation | ln(2) ≈ 0.693 = log loss for 50/50. |
| `league_avg_goals_per_game` | pipeline | Total goals / games. |
| `league_avg_xg_per_game` | pipeline | Total xG / games. |
| `league_avg_xg_per_hour_5v5` | 2.0 model | xG per hour at 5v5 (shift-level). |

---

## Formula Quick Reference

**Win probability (all Elo models):**
```
P(A beats B) = 1 / (1 + 10^((rating_B - rating_A) / 400))
```

**Off/Def expected xG:**
```
multi = 10^((O_attacker - D_defender) / 400)
expected_xG = league_avg_xg × multi × time_hours
```

**Off/Def update:**
```
delta = k × (observed_xG - expected_xG) / ln(10)
O_attacker += delta
D_defender -= delta
```

**Goals from win prob (1.0, 1.1):**
```
home_goals = league_avg + goal_diff_half_range × (win_prob - 0.5)
away_goals = league_avg - goal_diff_half_range × (win_prob - 0.5)
```
