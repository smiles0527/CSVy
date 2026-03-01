# Baseline Elo Models — Design Document

A design specification for the Baseline Elo family: **1.0 (Goals)**, **1.1 (xG)**, and **2.0 (Off/Def)**. All share minimalism (no home advantage, MOV, rest, etc.) and a common API.

---

## 1. Purpose & Scope

### 1.1 Goals

- **Benchmark**: Simple, interpretable baselines for comparing complex models.
- **Minimalism**: No home advantage, margin-of-victory, rest, travel, or overtime differentiation.
- **Interface compatibility**: Same `fit` / `predict_goals` / `predict_winner` / `evaluate` API across iterations.

### 1.2 Iteration Overview

| Iteration | Outcome | Data | Rating |
|-----------|---------|------|--------|
| **1.0** | Actual goals | Game-level | Single rating per team |
| **1.1** | Expected goals (xG) | Game-level | Single rating per team |
| **2.0** | xG per shift | Shift-level | O and D per team, per line (L1/L2) |

### 1.3 Out of Scope (intentionally)

| Feature | Status |
|---------|--------|
| Home ice advantage | Not included |
| Margin of victory | Not included |
| Overtime differentiation | Treated as binary win/loss |
| Rest, travel, injuries | Not included |
| Season carryover | Single-season only |
| Division/tier seeding | All teams start equal |

---

## 2. Iteration 1.0 — Goals

### 2.1 Data

- **Input**: Game-level DataFrame with `home_team`, `away_team`, `home_goals`, `away_goals`.
- **Shift-level CSV** → aggregated by `game_id` (sum goals, first home/away).
- **Order**: Chronological by `game_id` / `game_num`.

### 2.2 Outcome Encoding

- \(hg > ag\) → home wins → \(O_{\mathrm{home}} = 1\), \(O_{\mathrm{away}} = 0\)
- \(hg \leq ag\) (including ties) → home loses → \(O_{\mathrm{home}} = 0\), \(O_{\mathrm{away}} = 1\)

### 2.3 Expected Score (Win Probability)

For team A (rating \(r_{a}\)) vs team B (rating \(r_{b}\)):

$$
E_{a} = \frac{1}{1 + 10^{(r_{b} - r_{a}) / s}}
$$

with \(s\) = `elo_scale` (default 400). \(E_{a} + E_{b} = 1\).

### 2.4 Rating Update (Zero-Sum)

$$
\Delta_{a} = k \cdot (O_{a} - E_{a}), \quad \Delta_{b} = k \cdot (O_{b} - E_{b})
$$

\(O_{a}, O_{b} \in \{0, 1\}\), \(O_{a} + O_{b} = 1\). \(\Delta_{a} + \Delta_{b} = 0\).

### 2.5 Goal Prediction (Derived)

Elo yields win probability; we map to goals:

$$
\mathrm{adj} = g_{\mathrm{half}} \cdot (p_{\mathrm{home}} - 0.5)
$$

$$
\mathrm{homeGoals} = \max(0, \mu + \mathrm{adj}), \quad \mathrm{awayGoals} = \max(0, \mu - \mathrm{adj})
$$

- \(\mu\) = `league_avg_goals` (default 3.0)
- \(g_{\mathrm{half}}\) = `goal_diff_half_range` (default 6)

---

## 3. Iteration 1.1 — xG

### 3.1 Data

- **Input**: Game-level with `home_team`, `away_team`, `home_xg`, `away_xg`.
- **Aggregation**: Shift-level → sum `home_xg`, `away_xg` per game.
- **Order**: Chronological.

### 3.2 Outcome Encoding

- home_xg > away_xg → home wins → \(O_{\mathrm{home}} = 1\), \(O_{\mathrm{away}} = 0\)
- Otherwise → home loses → \(O_{\mathrm{home}} = 0\), \(O_{\mathrm{away}} = 1\)

### 3.3 Expected Score & Rating Update

Same as 1.0: \(E_{a} = 1/(1 + 10^{(r_{b} - r_{a})/s})\), \(\Delta_{a} = k(O_{a} - E_{a})\).

### 3.4 xG Prediction (Derived)

Same formula as 1.0 goals, but output is *predicted xG*:

$$
\mathrm{adj} = g_{\mathrm{half}} \cdot (p_{\mathrm{home}} - 0.5)
$$

$$
\mathrm{predHomeXg} = \max(0, \mu + \mathrm{adj}), \quad \mathrm{predAwayXg} = \max(0, \mu - \mathrm{adj})
$$

- **Evaluation**: RMSE, win accuracy, Brier, log loss — all on xG (not goals).

---

## 4. Iteration 2.0 — Off/Def

### 4.1 Concept

- **Separate O and D** ratings per team, per line (L1 = first_off, L2 = second_off).
- **Updates** driven by observed xG vs expected xG (not win/loss binary).
- **Time-aware**: Uses TOI delta (change in time) per shift for expected xG scaling.
- **xG rate**: League baseline = xG per hour from training data.

### 4.2 Data

- **Shift-level (preferred)**: `home_team`, `away_team`, `home_xg`, `away_xg`, `home_off_line`, `away_off_line`, `toi`, `game_id`.
- **Lines**: Only `first_off` and `second_off` (L1, L2).
- **TOI delta**: \(t_{\Delta} = \mathrm{toi}_{\mathrm{current}} - \mathrm{toi}_{\mathrm{previous}}\) within each game (assumes cumulative toi or ordered shifts). First row per game: \(t_{\Delta} = \mathrm{toi}\). If \(t_{\Delta} \leq 0\), fallback to raw toi.
- **Game-level fallback**: If no line/TOI columns, treat entire game as line 1.

### 4.3 League Average xG Rate

$$
\mathrm{leagueAvgXg} = \frac{\sum \mathrm{xG}}{(\sum t_{\Delta}) / 3600}
$$

xG per hour (5v5). Used as baseline for expected xG.

### 4.4 Expected xG (Per Shift)

For a matchup: home line \(h\) vs away line \(a\), and away line \(a\) vs home line \(h\):

$$
\mathrm{multi}_{h} = 10^{(O_{h} - D_{a}) / s}, \quad \mathrm{multi}_{a} = 10^{(O_{a} - D_{h}) / s}
$$

$$
\mathrm{expHomeXg} = \mathrm{leagueAvgXg} \cdot \mathrm{multi}_{h} \cdot \frac{t_{\Delta}}{3600}
$$

$$
\mathrm{expAwayXg} = \mathrm{leagueAvgXg} \cdot \mathrm{multi}_{a} \cdot \frac{t_{\Delta}}{3600}
$$

- \(t_{\Delta}\) in seconds; divided by 3600 for hours.
- Minimum \(t_{\Delta}/3600 = 0.001\) to avoid zero.

### 4.5 O/D Rating Update

$$
\delta_{h} = k \cdot \frac{o_{h} - e_{h}}{\ln(10)}
$$

$$
\delta_{a} = k \cdot \frac{o_{a} - e_{a}}{\ln(10)}
$$

where \(o_{h}\) = observed home xG, \(e_{h}\) = expected home xG (and similarly for away).

- \(O_{h} \leftarrow O_{h} + \delta_{h}\), \(D_{a} \leftarrow D_{a} - \delta_{h}\) (home offense vs away defense)
- \(O_{a} \leftarrow O_{a} + \delta_{a}\), \(D_{h} \leftarrow D_{h} - \delta_{a}\) (away offense vs home defense)

### 4.6 Net Rating & Win Probability

- **Net (O−D)**: Per team, average of (O−D) over L1 and L2. Can be negative; higher = stronger.
- **Predict winner**: Uses Elo formula on rating diff:
  \[
  d = (O_{h} - D_{a}) - (O_{a} - D_{h})
  \]
  \[
  P_{\mathrm{home}} = \frac{1}{1 + 10^{-d/s}}
  \]
  where \(P_{\mathrm{home}}\) = probability home wins
- **Predict xG**: Uses average line ratings; expected xG = league_avg_xg × multi × time_factor (game-level = 1.0 hour equivalent).

### 4.7 Game-Level Fallback

When no shift/line data: one O and D per team (line 1 only). `league_avg_xg` = total xG / (2 × n_games). `time_factor` = 1.0 (one game unit).

---

## 5. Shared Parameters

| Param | Default | 1.0 | 1.1 | 2.0 | Role |
|-------|---------|-----|-----|-----|------|
| `k_factor` | 32 | ✓ | ✓ | ✓ | Rating change magnitude (swept 0.1–100 for 2.0) |
| `initial_rating` | 1200 | ✓ | ✓ | ✓ | Starting O, D per team/line |
| `elo_scale` | 400 | ✓ | ✓ | ✓ | Divisor in win-prob formula |
| `league_avg_goals` | 3.0 | ✓ | ✓ | — | Baseline for goal/xG prediction (1.0, 1.1) |
| `goal_diff_half_range` | 6.0 | ✓ | ✓ | — | Win-prob → goal spread |
| `league_avg_xg` | (computed) | — | — | ✓ | xG per hour from data |
| `time_factor` | 1.0 | — | — | ✓ | Game-level xG time scaling |

---

## 6. Data Flow Summary

| Iteration | Input | Outcome | Output |
|-----------|-------|---------|--------|
| 1.0 | Games: goals | \(O \in \{0,1\}\) from goals | Rankings, predict_goals, predict_winner |
| 1.1 | Games: xG | \(O \in \{0,1\}\) from xG | Rankings, predict_goals (xG), predict_winner |
| 2.0 | Shifts: xG, line, TOI | obs vs exp xG per shift | O/D per line, predict_goals (xG), predict_winner |

---

## 7. Edge Cases & Robustness

| Case | 1.0 / 1.1 | 2.0 |
|------|-----------|-----|
| Ties | Treated as home loss | N/A (continuous xG) |
| Self-play | Row skipped | Row skipped |
| NaN team names | Coerced to Unknown | Coerced to Unknown |
| Unknown team at predict | Uses initial_rating | Uses initial_rating |
| No line/TOI columns | N/A | Game-level fallback |
| \(t_{\Delta} \leq 0\) | N/A | Fallback to raw toi |
| Empty DataFrame | Zero metrics | Zero metrics |

---

## 8. Validation & Evaluation

- **Train/test**: Configurable split (e.g. 70/30 block CV, 3 folds).
- **K-sweep**: `k_factor` grid (e.g. 0.1–100 step 0.1 for 2.0).
- **Selection**: Best by `combined_rmse` or by accuracy.
- **Metrics**: RMSE, MAE, R², win_accuracy, Brier loss, log loss.

---

## 9. Implementation References

| Iteration | Module | Config |
|-----------|--------|--------|
| 1.0 | `python/utils/baseline_elo.py` | `model_baseline_elo.yaml` |
| 1.1 | `python/utils/baseline_elo_xg.py` | `model_baseline_elo_xg.yaml` |
| 2.0 | `python/utils/baseline_elo_offdef.py` | `model_baseline_elo_sweep.yaml` |

**Unified sweep**: `python/_run_baseline_elo_sweep.py` — runs 1.0, 1.1, 2.0 or `--2.0-only`.

---

## 10. References

- Elo, A. (1978). *The Rating of Chessplayers, Past and Present*.
- `python/MODEL_VALUES_INDEX.md` — value reference across models.
- `docs/baseline_elo_xg_calculations.md` — detailed 1.1 formulas.
