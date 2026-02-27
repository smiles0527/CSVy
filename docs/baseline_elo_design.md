# Baseline Elo Model — Design Document

A thorough design specification for the minimal Elo rating system used as a hockey prediction baseline.

---

## 1. Purpose & Scope

### 1.1 Goals

- **Benchmark**: Provide a simple, interpretable baseline for comparing more complex models (full Elo, XGBoost, ensemble).
- **Minimalism**: No home advantage, margin-of-victory, rest, travel, or overtime differentiation.
- **Interface compatibility**: Same `fit` / `predict_goals` / `predict_winner` / `evaluate` API as other models for fair comparison.

### 1.2 Out of Scope (intentionally)

| Feature | Status | Rationale |
|--------|--------|-----------|
| Home ice advantage | Not included | Simplification; full Elo has it |
| Margin of victory (MOV) | Not included | Classic Elo is win/loss only |
| Overtime vs regulation | Not differentiated | Treated as binary win/loss |
| Rest, travel, injuries | Not included | No contextual features |
| Season carryover | N/A | Single-season model |
| Division/tier seeding | Not included | All teams start equal |

---

## 2. Mathematical Formulation

### 2.1 Expected Score (Win Probability)

For team A (rating \(r_a\)) vs team B (rating \(r_b\)):

$$
E_a = \frac{1}{1 + 10^{(r_b - r_a) / s}}
$$

with \(s = 400\) (configurable `elo_scale`).

- \(E_a + E_b = 1\)
- 200-point gap ≈ 76% for stronger team
- 400-point gap ≈ 91%

**Example**: Rating diff +100 → E(stronger) ≈ 64%; +200 → 76%; +400 → 91%.

### 2.2 Rating Update (Zero-Sum)

After each match:

$$
\Delta_a = k \cdot (O_a - E_a)
$$

$$
\Delta_b = k \cdot (O_b - E_b)
$$

where \(O_a, O_b \in \{0, 1\}\) with \(O_a + O_b = 1\) (binary win/loss).

**Conservation**: \(\Delta_a + \Delta_b = k(O_a + O_b - E_a - E_b) = k(1 - 1) = 0\). One team gains exactly what the other loses.

### 2.3 Goal Prediction (Derived)

Elo yields win probability, not goals. We map win prob to goals:

$$
\text{adj} = g_{half} \cdot (p_{home} - 0.5)
$$

$$
\text{home\_goals} = \max(0, \mu + \text{adj})
$$

$$
\text{away\_goals} = \max(0, \mu - \text{adj})
$$

- \(\mu\) = `league_avg_goals` (default 3.0)
- \(g_{half}\) = `goal_diff_half_range` (default 6)
- \(p_{home}\) = 0.5 gives 3–3; \(p_{home} = 1\) gives 9–0; \(p_{home} = 0\) gives 0–9
- \(\max(0, \cdot)\) prevents negative goals

---

## 3. Parameters

| Param | Default | Role | Grid? |
|-------|---------|------|-------|
| `k_factor` | 32 | Rating volatility per game | Yes (5–100) |
| `initial_rating` | 1200 | Starting rating for new teams | Yes (fixed 1200) |
| `elo_scale` | 400 | Divisor in expected-score | No |
| `league_avg_goals` | 3.0 | Baseline for goal prediction | No |
| `goal_diff_half_range` | 6.0 | Win-prob → goal spread | No |

---

## 4. Data Flow

### 4.1 Input

- **Shift-level CSV** → aggregated to game level (sum goals, first home/away).
- **Required columns** (or aliases): `home_team`, `away_team`, `home_goals`, `away_goals`.
- **Order**: Games processed chronologically (by `game_id` / `game_num`).

### 4.2 Outcome Encoding

- \(hg > ag\) → home wins → \(O_{home} = 1, O_{away} = 0\)
- \(hg \leq ag\) (includes ties) → home loses → \(O_{home} = 0, O_{away} = 1\)
- Overtime wins treated identically to regulation wins.

### 4.3 Outputs

- **Rankings**: `(team, rating)` sorted descending.
- **predict_winner**: `(team, confidence)` where confidence = win probability.
- **predict_goals**: `(home_goals, away_goals)` — derived from win prob.
- **Metrics**: RMSE, MAE, R², win_accuracy, Brier loss, log loss.

---

## 5. Edge Cases & Robustness

| Case | Handling |
|------|----------|
| Ties (\(hg = ag\)) | Treated as home loss (\(O=0\)) |
| Self-play (home=away) | Row skipped |
| NaN team names | Coerced to `Unknown_Home` / `Unknown_Away` |
| NaN goals | Treated as 0 |
| Unknown team at predict | Uses `initial_rating` |
| \(elo\_scale \leq 0\) | Fallback: expected score = 0.5 |
| Negative goal prediction | Clipped to 0 |
| Empty DataFrame (evaluate) | Returns zero metrics, no crash |

---

## 6. Validation & Evaluation

- **Train/test**: Configurable split (default 70/30 block CV, 3 folds).
- **Grid search**: Over `k_factor` (and optionally `initial_rating`).
- **Selection**: Best by `combined_rmse` on first fold.
- **Metrics**: RMSE (goals), win accuracy, Brier loss, log loss.

---

## 7. Implementation Notes

- **Core**: `EloSystem` dataclass (rating logic) + `BaselineEloModel` (DataFrame API).
- **Config**: YAML (`config/hyperparams/model_baseline_elo.yaml`) for all hyperparameters and paths.
- **Pipeline**: `_run_baseline_elo.py` (CLI) and `train_baseline_elo.ipynb` (interactive).

---

## 8. Comparison to Full Elo Model

| Aspect | Baseline Elo | Full Elo (model3) |
|--------|--------------|-------------------|
| Home advantage | None | Yes (configurable) |
| MOV | None | Yes (linear/log) |
| Overtime | Same as regulation | Discounted (e.g. 0.75) |
| Rest / travel | None | Yes |
| Division seeding | None | D1/D2/D3 |
| Params | 5 | 10+ |

---

## 9. Possible Extensions

- **OT multiplier**: Add `ot_win_multiplier` and use `went_ot` from data.
- **Home advantage**: Fixed rating boost for home team.
- **MOV**: Scale K by goal differential (would break strict zero-sum per match if not careful).

---

## 10. References

- Elo, A. (1978). *The Rating of Chessplayers, Past and Present*.
- FIFA/CONCACAF-style Elo for football.
- `config/hyperparams/model_baseline_elo.yaml` — runtime config.
- `python/utils/baseline_elo.py` — implementation.
