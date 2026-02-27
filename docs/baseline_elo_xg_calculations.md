# Baseline Elo xG Model — All Calculations

A complete specification of every formula and calculation used in the Baseline Elo xG model. This model is identical to Baseline Elo except it uses **expected goals (xG)** instead of actual goals for outcomes, aggregation, and evaluation.

---

## 1. Data Aggregation (Shift → Game Level)

Raw shift-level CSV is aggregated to one row per game:

| Aggregation | Formula | Description |
|-------------|---------|-------------|
| `home_xg` | \(\sum \text{home\_xg}\) | Sum of home xG over all shifts in the game |
| `away_xg` | \(\sum \text{away\_xg}\) | Sum of away xG over all shifts in the game |
| `home_team` | \(\text{first}(\text{home\_team})\) | First (constant) value per game |
| `away_team` | \(\text{first}(\text{away\_team})\) | First (constant) value per game |

**Code**: `raw.groupby('game_id').agg(home_xg=('home_xg','sum'), away_xg=('away_xg','sum'), ...)`

Games are sorted chronologically by `game_id` / `game_num` for block cross-validation.

---

## 2. Outcome Encoding (xG-Based)

For each game, the outcome is binary: did the home team "win" on xG?

\[
O_{\text{home}} = \begin{cases}
1 & \text{if } \text{home\_xg} > \text{away\_xg} \\
0 & \text{otherwise (tie or away wins)}
\end{cases}
\]

\[
O_{\text{away}} = 1 - O_{\text{home}}
\]

- **Ties** (home_xg = away_xg): Treated as home loss (\(O_{\text{home}} = 0\))
- **Overtime**: Not differentiated; OT wins treated same as regulation

---

## 3. Expected Score (Elo Win Probability)

For team A (rating \(r_a\)) vs team B (rating \(r_b\)), the expected score (win probability) for A:

\[
E_a = \frac{1}{1 + 10^{(r_b - r_a) / s}}
\]

where \(s\) = `elo_scale` (default 400).

\[
E_b = 1 - E_a
\]

**Properties**:
- \(E_a + E_b = 1\)
- 200-point rating gap → stronger team ≈ 76%
- 400-point gap → stronger team ≈ 91%

**Fallback**: If \(s \leq 0\), then \(E_a = E_b = 0.5\).

---

## 4. Rating Update (Zero-Sum)

After each match, ratings are updated:

\[
\Delta_a = k \cdot (O_a - E_a)
\]

\[
\Delta_b = k \cdot (O_b - E_b)
\]

\[
r_a^{\text{new}} = r_a + \Delta_a, \quad r_b^{\text{new}} = r_b + \Delta_b
\]

where \(k\) = `k_factor` (grid-searched 5–100, step 5).

**Zero-sum**: \(\Delta_a + \Delta_b = k(O_a + O_b - E_a - E_b) = k(1 - 1) = 0\). One team gains exactly what the other loses.

---

## 5. Predicted xG (From Win Probability)

Elo yields win probability, not xG. We map win probability to expected xG:

\[
\text{adj} = g_{\text{half}} \cdot (p_{\text{home}} - 0.5)
\]

\[
\text{pred\_home\_xg} = \max(0, \mu + \text{adj})
\]

\[
\text{pred\_away\_xg} = \max(0, \mu - \text{adj})
\]

**Parameters**:
- \(\mu\) = `league_avg_goals` (default 3.0) — baseline xG per team
- \(g_{\text{half}}\) = `goal_diff_half_range` (default 6.0) — xG spread per unit of win prob above 0.5
- \(p_{\text{home}}\) = home win probability from Elo

**Examples**:
- \(p_{\text{home}} = 0.5\) → 3.0–3.0
- \(p_{\text{home}} = 1\) → 9.0–0.0
- \(p_{\text{home}} = 0\) → 0.0–9.0

---

## 6. Evaluation Metrics

### 6.1 RMSE (Root Mean Squared Error)

\[
\text{RMSE}_{\text{home}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i^{\text{pred}} - y_i^{\text{actual}})^2}
\]

\[
\text{RMSE}_{\text{combined}} = \sqrt{\frac{1}{2n}\sum_{i=1}^{n} \left[ (h_i^{\text{pred}} - h_i^{\text{actual}})^2 + (a_i^{\text{pred}} - a_i^{\text{actual}})^2 \right]}
\]

where \(h\) = home xG, \(a\) = away xG.

### 6.2 MAE (Mean Absolute Error)

\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i^{\text{pred}} - y_i^{\text{actual}}|
\]

### 6.3 R² (Coefficient of Determination)

\[
R^2 = 1 - \frac{\sum (y_i^{\text{actual}} - y_i^{\text{pred}})^2}{\sum (y_i^{\text{actual}} - \bar{y})^2}
\]

If all actual values are identical, \(R^2 = 0\) (avoids division issues).

### 6.4 Win Accuracy

\[
\text{Win Accuracy} = \frac{\text{\# games where pred winner matches actual winner}}{n}
\]

Actual winner = team with higher xG. Predicted winner = team with higher predicted xG.

### 6.5 Brier Loss

For each game, let:
- \(E_A\) = predicted probability that home wins (from Elo)
- \(O_A\) = 1 if home wins (home_xg > away_xg), else 0
- \(O_B = 1 - O_A\)

\[
\text{Brier}_{\text{game}} = (E_A - O_A)^2
\]

\[
\text{Brier Loss} = \frac{1}{n}\sum_{\text{games}} (E_A - O_A)^2
\]

### 6.6 Log Loss (Cross-Entropy)

\[
\text{LogLoss}_{\text{game}} = -\left[ O_A \log(E_A) + O_B \log(E_B) \right]
\]

where \(E_A, E_B\) are clipped to \([\epsilon, 1-\epsilon]\) (e.g. \(\epsilon = 10^{-10}\)) to avoid \(\log(0)\).

\[
\text{Log Loss} = \frac{1}{n}\sum_{\text{games}} \text{LogLoss}_{\text{game}}
\]

---

## 7. Parameters Summary

| Parameter | Symbol | Default | Role | Grid? |
|-----------|--------|---------|------|-------|
| `k_factor` | \(k\) | 32 | Rating volatility per game | Yes (5–100, step 5) |
| `initial_rating` | \(r_0\) | 1200 | Starting rating for new teams | Yes (fixed 1200) |
| `elo_scale` | \(s\) | 400 | Divisor in expected-score formula | No |
| `league_avg_goals` | \(\mu\) | 3.0 | Baseline xG per team | No |
| `goal_diff_half_range` | \(g_{\text{half}}\) | 6.0 | Win-prob → xG spread | No |

---

## 8. Implemented Locations

- **Core model**: `python/utils/baseline_elo_xg.py`
- **Pipeline**: `python/_run_baseline_elo_xg.py`
- **Config**: `config/hyperparams/model_baseline_elo_xg.yaml`
- **Training**: `python/training/baseline_elo_xg/train_baseline_elo_xg.ipynb`
