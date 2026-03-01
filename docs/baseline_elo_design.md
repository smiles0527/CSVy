# Baseline Elo Models — Design Document

A design specification for the Baseline Elo family: **1.0 (Goals)**, **1.1 (xG)**, and **2.0 (Off/Def)**. All share minimalism (no home advantage, MOV, rest, etc.) and a common API.

---

## 1. Purpose & Scope

### 1.1 Goals

- **Benchmark**: Simple, interpretable baselines for comparing complex models.
- **Minimalism**: No home advantage, margin-of-victory, rest, travel, or overtime differentiation.
- **Interface compatibility**: Same `fit` / `predict_goals` / `predict_winner` / `evaluate` API across iterations.

### 1.2 Design Principles

- **Evaluation targets**: 1.0 evaluated on actual goals (RMSE, win accuracy on goals); 1.1 and 2.0 evaluated on xG (RMSE, win accuracy on xG).
- **Brier and log loss**: All models output win probabilities; outcome for 1.0 = goals win (home_goals > away_goals); 1.1 and 2.0 = xG win (home_xg > away_xg).
- **Common API**: `fit(df)`, `predict_goals(game)`, `predict_winner(game)`, `evaluate(games_df)`, `get_rankings(top_n)`.

### 1.3 Iteration Overview

| Iteration | Outcome | Data | Rating |
|-----------|---------|------|--------|
| **1.0** | Actual goals | Game-level | Single Elo per team |
| **1.1** | Expected goals (xG) | Game-level | Single Elo per team |
| **2.0** | xG per shift | Shift-level | O and D per team, per line (L1/L2) |

### 1.4 Out of Scope (intentionally)

| Feature | Status |
|---------|--------|
| Home ice advantage | Not included |
| Margin of victory | Not included |
| Overtime differentiation | Treated as binary win/loss |
| Rest, travel, injuries | Not included |
| Season carryover | Single-season only |
| Division/tier seeding | All teams start equal |

---

## 2. Complete Parameter & Value Reference

### 2.1 Shared Parameters

| Param | Default | 1.0 | 1.1 | 2.0 | Role |
|-------|---------|-----|-----|-----|------|
| `k_factor` | 32 | ✓ | ✓ | ✓ | Rating change magnitude per game/shift. Swept 0.1–500 for 2.0 (or 0.1–100 with `--to-100`). |
| `initial_rating` | 1200 | ✓ | ✓ | ✓ | Starting Elo (1.0, 1.1) or O/D (2.0) for new teams. |
| `elo_scale` | 400 | ✓ | ✓ | ✓ | Divisor in expected score: \(E = 1/(1 + 10^{(r_b - r_a)/s})\). 200-point diff ≈ 76% win prob; 400-point ≈ 91%. |
| `league_avg_goals` | 3.0 | ✓ | ✓ | — | Baseline \(\mu\) for goal/xG prediction: goals = \(\mu \pm \mathrm{adj}\). |
| `goal_diff_half_range` | 6.0 | ✓ | ✓ | — | Max swing: \(\mathrm{adj} = g_{\mathrm{half}} \cdot (p - 0.5)\). At p=1: 9 goals; p=0: 0 goals. |
| `league_avg_xg` | (computed) | — | — | ✓ | xG per hour from training data. 2.0: \(\sum \mathrm{xG} / (\sum t_{\Delta}/3600)\). |
| `time_factor` | 1.0 | — | — | ✓ | Game-level xG scaling. 1.0 = one game (60 min equivalent). |
| `LN10` | \(\ln(10) \approx 2.3026\) | — | — | 2.0 | Part of gradient scaling: \(\delta = k \cdot (\mathrm{LN10}/s) \cdot (o - e)\). |

### 2.2 Literal Constants (hardcoded)

| Value | Where | Purpose |
|-------|-------|---------|
| 3600 | 2.0 | Seconds per hour; TOI in seconds ÷ 3600 = hours. |
| 0.001 | 2.0 | Minimum \(t_{\Delta}/3600\) (time_frac) to avoid division by zero. |
| 0.01 | 2.0 | Minimum \(\sum t_{\Delta}/3600\) when computing league_avg_xg. |
| \(10^{-10}\) | Brier/log | Epsilon for clipping probabilities to (eps, 1−eps) in log-loss. |

### 2.3 Column Aliases

All models support flexible column names via aliases:

**1.0, 1.1, 2.0**: `home_team`, `away_team` → `home`, `team_home`, `away`, `team_away`, `visitor`, etc.
**1.0**: `home_goals`, `away_goals` → `home_score`, `h_goals`, `goals_home`, etc.
**1.1, 2.0**: `home_xg`, `away_xg` → `home_xG`, `xg_home`, `h_xg`, etc.
**2.0**: `home_off_line`, `away_off_line`, `toi` (or `time_on_ice`).

---

## 3. Iteration 1.0 — Goals

### 3.1 Data

- **Input**: Game-level DataFrame with `home_team`, `away_team`, `home_goals`, `away_goals`.
- **Aggregation** (when source is shift-level): group by `game_id`; sum `home_goals`, `away_goals`; first `home_team`, `away_team`.
- **Order**: Chronological by `game_id` / `game_num` for block cross-validation.
- **Missing values**: NaN team names → `Unknown_Home` / `Unknown_Away`; NaN goals → 0.

### 3.2 Outcome Encoding

- \(hg > ag\) → home wins → \(O_{\mathrm{home}} = 1\), \(O_{\mathrm{away}} = 0\)
- \(hg \leq ag\) (including ties) → home loses → \(O_{\mathrm{home}} = 0\), \(O_{\mathrm{away}} = 1\)
- **Self-play**: Rows where \(ht = at\) are skipped (no update).
- **Zero-sum**: \(O_a + O_b = 1\) always.

### 3.3 Expected Score (Win Probability)

For team A (rating \(r_a\)) vs team B (rating \(r_b\)):

$$
E_a = \frac{1}{1 + 10^{(r_b - r_a) / s}}
$$

with \(s\) = `elo_scale` (default 400). \(E_a + E_b = 1\).

**Properties**:
- 200-point rating gap → stronger team ≈ 76%
- 400-point gap → stronger team ≈ 91%
- **Fallback**: If \(s \leq 0\), \(E_a = E_b = 0.5\).

### 3.4 Rating Update (Zero-Sum)

$$
\Delta_a = k \cdot (O_a - E_a), \quad \Delta_b = k \cdot (O_b - E_b)
$$

\(O_a, O_b \in \{0, 1\}\), \(O_a + O_b = 1\). Hence \(\Delta_a + \Delta_b = 0\): one team gains exactly what the other loses.

$$
r_a^{\mathrm{new}} = r_a + \Delta_a, \quad r_b^{\mathrm{new}} = r_b + \Delta_b
$$

### 3.5 Goal Prediction (Derived)

Elo yields win probability; we map to goals:

$$
\mathrm{adj} = g_{\mathrm{half}} \cdot (p_{\mathrm{home}} - 0.5)
$$

$$
\mathrm{homeGoals} = \max(0, \mu + \mathrm{adj}), \quad \mathrm{awayGoals} = \max(0, \mu - \mathrm{adj})
$$

- \(\mu\) = `league_avg_goals` (default 3.0)
- \(g_{\mathrm{half}}\) = `goal_diff_half_range` (default 6)

**Interpretation**:
- \(p_{\mathrm{home}} = 0.5\) → 3–3
- \(p_{\mathrm{home}} = 1\) → 9–0
- \(p_{\mathrm{home}} = 0\) → 0–9
- \(\mathrm{adj}\) is the signed swing from even; \(\mathrm{adj} \in [-3, 3]\) when \(p \in [0, 1]\).

### 3.6 Ranking (1.0)

- **get_rankings()**: Returns list of `(team, rating)` sorted by rating descending.
- **Rating**: Single Elo per team. Higher = stronger. Typical range ~1100–1300.
- **P (dashboard)**: \(P(\text{team beats league average}) = 1/(1 + 10^{-(r - \bar{r})/s})\) where \(\bar{r}\) = mean rating.

---

## 4. Iteration 1.1 — xG

### 4.1 Data

- **Input**: Game-level with `home_team`, `away_team`, `home_xg`, `away_xg`.
- **Aggregation** (when source is shift-level): group by `game_id`; sum `home_xg`, `away_xg`; first teams.
- **Order**: Chronological.

### 4.2 Outcome Encoding

- home_xg > away_xg → home wins → \(O_{\mathrm{home}} = 1\), \(O_{\mathrm{away}} = 0\)
- Otherwise (including ties) → home loses → \(O_{\mathrm{home}} = 0\), \(O_{\mathrm{away}} = 1\)
- **Ties**: Treated as home loss.
- **Self-play**: Row skipped.

### 4.3 Expected Score & Rating Update

Same as 1.0:

$$
E_a = \frac{1}{1 + 10^{(r_b - r_a)/s}}, \quad \Delta_a = k(O_a - E_a)
$$

### 4.4 xG Prediction (Derived)

Same formula as 1.0 goals, but output is *predicted xG*:

$$
\mathrm{adj} = g_{\mathrm{half}} \cdot (p_{\mathrm{home}} - 0.5)
$$

$$
\mathrm{predHomeXg} = \max(0, \mu + \mathrm{adj}), \quad \mathrm{predAwayXg} = \max(0, \mu - \mathrm{adj})
$$

- **Evaluation**: RMSE, MAE, R², win_accuracy, Brier, log loss — all on xG (not goals).
- **Win accuracy**: Correct when (pred_home_xg > pred_away_xg) iff (actual_home_xg > actual_away_xg).
- **Brier/Log loss**: Outcome = 1 if home_xg > away_xg else 0.

### 4.5 Ranking (1.1)

- Same as 1.0: single Elo per team; `get_rankings` sorted descending.
- P(beats avg) = \(1/(1 + 10^{-(r - \bar{r})/s})\).

---

## 5. Iteration 2.0 — Off/Def

### 5.1 Concept

- **Separate O and D** ratings per team, per line (L1 = first_off, L2 = second_off).
- **Updates** driven by observed xG vs expected xG (not win/loss binary).
- **Time-aware**: Uses TOI delta per shift for expected xG scaling.
- **xG rate**: League baseline = xG per hour from training data.
- **Lines**: Only `first_off` and `second_off`; other line values ignored or fallback to L1.

### 5.2 Data

- **Shift-level (preferred)**: `home_team`, `away_team`, `home_xg`, `away_xg`, `home_off_line`, `away_off_line`, `toi`, `game_id`.
- **Line parsing**: `first_off` / `first` → L1; `second_off` / `second` → L2; else → L1.
- **TOI delta**: \(t_{\Delta} = \mathrm{toi}_{\mathrm{current}} - \mathrm{toi}_{\mathrm{previous}}\) within each game (cumulative toi assumed). First row per game: \(t_{\Delta} = \mathrm{toi}\). If \(t_{\Delta} \leq 0\), fallback to raw toi.
- **Game-level fallback**: If no `home_off_line`, `away_off_line`, or `toi`, treat entire game as line 1. `league_avg_xg` = total xG / (2 × n_games). `time_factor` = 1.0.

### 5.3 League Average xG Rate

$$
\mathrm{leagueAvgXg} = \frac{\sum \mathrm{xG}}{(\sum t_{\Delta}) / 3600}
$$

xG per hour (5v5). Minimum \((\sum t_{\Delta})/3600 = 0.01\) to avoid division by zero.

### 5.4 Expected xG (Per Shift)

For matchup: home line \(h\) vs away line \(a\):

$$
\mathrm{multi}_h = 10^{(O_h - D_a) / s}, \quad \mathrm{multi}_a = 10^{(O_a - D_h) / s}
$$

$$
\mathrm{expHomeXg} = \mathrm{leagueAvgXg} \cdot \mathrm{multi}_h \cdot \frac{t_{\Delta}}{3600}
$$

$$
\mathrm{expAwayXg} = \mathrm{leagueAvgXg} \cdot \mathrm{multi}_a \cdot \frac{t_{\Delta}}{3600}
$$

- \(t_{\Delta}\) in seconds; divided by 3600 for hours.
- Minimum \(t_{\Delta}/3600 = 0.001\) to avoid zero.
- **multi**: Multiplier on league rate; >1 when attacker stronger than defender.

### 5.5 O/D Rating Update

$$
\delta_h = k \cdot \frac{\ln(10)}{s} \cdot (o_h - e_h)
$$

$$
\delta_a = k \cdot \frac{\ln(10)}{s} \cdot (o_a - e_a)
$$

where \(o_h\) = observed home xG, \(e_h\) = expected home xG, \(s\) = `elo_scale`.

- \(O_h \leftarrow O_h + \delta_h\), \(D_a \leftarrow D_a - \delta_h\) (home offense vs away defense)
- \(O_a \leftarrow O_a + \delta_a\), \(D_h \leftarrow D_h - \delta_a\) (away offense vs home defense)

**Derivation**: Poisson log-likelihood gradient; see `docs/baseline_elo_offdef_calculations.md`.

### 5.6 Net Rating & Win Probability

**Team net (for ranking)**:

$$
\mathrm{Net} = w_1 (O_{L1} - D_{L1}) + w_2 (O_{L2} - D_{L2})
$$

with \(w_1 + w_2 = 1\). Weights \(w_1, w_2\) come from empirical TOI shares during fit (L1 TOI / total TOI, L2 TOI / total TOI). When no shift/TOI data (game-level fallback), defaults to \(w_1 = w_2 = 0.5\).

- Can be negative (league-wide D > O is normal). Higher = stronger.

**Predict winner**:

$$
d = (O_h - D_a) - (O_a - D_h)
$$

where \(O_h, D_h, O_a, D_a\) are *TOI-weighted* over L1 and L2 for game-level prediction.

$$
P_{\mathrm{home}} = \frac{1}{1 + 10^{-d/s}}
$$

**Predict xG (game-level)**:

- Uses TOI-weighted O, D over lines for each team.
- \(\mathrm{predHomeXg} = \max(0, \mathrm{league\_avg\_xg} \cdot \mathrm{multi}_h \cdot \mathrm{time\_factor})\)
- \(\mathrm{time\_factor} = 1.0\) = one game unit (60 min equivalent).
- \(\mathrm{league\_avg\_xg}\) fallback = 3.0 if not computed.

### 5.7 O/D Identifiability

Predictions depend only on differences \((O_h - D_a)\) and \((O_a - D_h)\). Adding constant \(c\) to all O and D: \((O+c) - (D+c) = O - D\). So O and D are identified only up to an additive constant. Raw O/D tables are not absolutely interpretable; only differences and nets matter.

### 5.8 Ranking (2.0)

- **get_rankings()**: Returns `(team, net)` sorted by net descending.
- **Net**: TOI-weighted average of (O−D) over L1 and L2: \(w_1(O_{L1}-D_{L1}) + w_2(O_{L2}-D_{L2})\). Can be negative; typical range ~−120 to −65 (or positive with gradient formula).
- **P (dashboard)**: \(P(\text{beats avg}) = 1/(1 + 10^{-(\mathrm{net} - \bar{\mathrm{net}})/s})\).

---

## 6. Ranking Systems — Comparison

| Model | Rating type | get_rankings output | Sort | P formula | Typical range |
|-------|-------------|---------------------|------|-----------|---------------|
| **1.0** | Single Elo \(r\) | (team, r) | desc by r | \(1/(1+10^{-(r-\bar{r})/s})\) | 1100–1300 |
| **1.1** | Single Elo \(r\) | (team, r) | desc by r | same | 1100–1300 |
| **2.0** | Net = TOI-weighted avg(O−D) | (team, net) | desc by net | same, net as "rating" | −120 to −65 or positive |

**Dashboard P**: For any model, P = probability team beats league-average opposition. \(\mathrm{diff} = \mathrm{rating} - \mathrm{mean}(\mathrm{ratings})\); \(P = 1/(1 + 10^{-\mathrm{diff}/s})\).

---

## 7. Data Flow Summary

| Iteration | Input | Outcome | Output |
|-----------|-------|---------|--------|
| 1.0 | Games: goals | \(O \in \{0,1\}\) from goals | Rankings, predict_goals, predict_winner |
| 1.1 | Games: xG | \(O \in \{0,1\}\) from xG | Rankings, predict_goals (xG), predict_winner |
| 2.0 | Shifts: xG, line, TOI | obs vs exp xG per shift | O/D per line, predict_goals (xG), predict_winner |

---

## 8. Edge Cases & Robustness

| Case | 1.0 / 1.1 | 2.0 |
|------|-----------|-----|
| Ties | Treated as home loss | N/A (continuous xG) |
| Self-play (ht=at) | Row skipped | Row skipped |
| NaN team names | Coerced to Unknown_Home / Unknown_Away | Same |
| Unknown team at predict | Uses initial_rating | Same |
| No line/TOI columns | N/A | Game-level fallback |
| Empty shifts (L1/L2 filter) | N/A | Falls back to game-level fit |
| \(t_{\Delta} \leq 0\) | N/A | Fallback to raw toi |
| Empty DataFrame | Zero metrics | Zero metrics |
| Brier/Log clipping | \(P \in [10^{-10}, 1-10^{-10}]\) | Same |

---

## 9. Validation & Evaluation

- **Train/test**: Configurable split (e.g. 70/30 block CV, 3 folds). Chronological order preserved.
- **K-sweep**: `k_factor` grid. 1.0: 0.1–35 step 0.1; 1.1: 0.1–100; 2.0: 0.1–500 step 0.1 (or `--2.0-only --to-100` for 0.1–100).
- **Selection**: Best by `combined_rmse` (or configurable). Per-iteration best = first row after sort by combined_rmse within that iteration.
- **Metrics**: RMSE, MAE, R², win_accuracy, Brier loss, log loss.
- **Outcome for Brier/Log**: 1.0 = (home_goals > away_goals); 1.1, 2.0 = (home_xg > away_xg).

---

## 10. Implementation References

| Iteration | Module | Config |
|-----------|--------|--------|
| 1.0 | `python/utils/baseline_elo.py` | `model_baseline_elo.yaml` |
| 1.1 | `python/utils/baseline_elo_xg.py` | `model_baseline_elo_xg.yaml` |
| 2.0 | `python/utils/baseline_elo_offdef.py` | `model_baseline_elo_sweep.yaml` |

**Unified sweep**: `python/scripts/run/run_baseline_elo_sweep.py` — runs 1.0, 1.1, 2.0 or `--2.0-only`. Use `--to-100` with `--2.0-only` for k ∈ [0.1, 100] step 0.1.

---

## 11. References

- Elo, A. (1978). *The Rating of Chessplayers, Past and Present*.
- `python/MODEL_VALUES_INDEX.md` — value reference across models.
- `docs/baseline_elo_xg_calculations.md` — detailed 1.1 formulas.
- `docs/baseline_elo_offdef_calculations.md` — 2.0 derivation (Poisson gradient).
- `docs/baseline_elo_2_0_CHANGES.md` — change log for 2.0 gradient fix.
- `docs/baseline_elo_2_0_TOI_WEIGHTS.md` — TOI-weighted ranking and prediction.
