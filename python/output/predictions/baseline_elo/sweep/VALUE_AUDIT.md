# Baseline Elo Sweep — Value Audit

Comprehensive scan of every numeric value, constant, and formula in the baseline Elo pipeline.

---

## 1. Model Constants (config-driven)

| Source | Key | Default | Used For |
|--------|-----|---------|----------|
| `model_baseline_elo_sweep.yaml` | `initial_rating` | 1200 | O, D, Elo starting value |
| `model_baseline_elo_sweep.yaml` | `elo_scale` | 400 | P = 1/(1+10^(-diff/scale)) |
| `model_baseline_elo_sweep.yaml` | `league_avg_goals` | 3.0 | Goals/xG prediction fallback |
| `model_baseline_elo_sweep.yaml` | `goal_diff_half_range` | 6.0 | Goals spread (baseline_elo, baseline_elo_xg) |
| `model_baseline_elo_sweep.yaml` | `train_ratio` | 0.7 | Train/test split |

---

## 2. Model Defaults (fallbacks when not in params)

| File | Param | Default | Line |
|------|-------|---------|------|
| baseline_elo_offdef.py | k_factor | 32 | 81 |
| baseline_elo_offdef.py | initial_rating | 1200 | 83 |
| baseline_elo_offdef.py | elo_scale | 400 | 84 |
| baseline_elo_offdef.py | goal_diff_half_range | 6.0 | 85 |
| baseline_elo_offdef.py | time_factor | 1.0 | 86 |
| baseline_elo.py | k_factor | 32 | 190 |
| baseline_elo.py | initial_rating | 1200 | 191 |
| baseline_elo.py | elo_scale | 400 | 192 |
| baseline_elo_xg.py | k_factor | 32 | 171 |
| baseline_elo_xg.py | initial_rating | 1200 | 172 |
| baseline_elo_xg.py | elo_scale | 400 | 173 |

---

## 3. Off/Def Model (baseline_elo_offdef.py)

| Value | Location | Purpose |
|-------|----------|---------|
| LN10 = np.log(10) ≈ 2.3026 | 26 | Gradient scaling: delta = k*(LN10/elo_scale)*(obs-exp) |
| 10.0 ** ((O-D)/elo_scale) | 153-154, 207-208, 239-240, 257 | Multiplier for expected xG |
| 3600.0 | 184, 195 | toi seconds → hours |
| 0.001 | 195 | min time_frac (avoid div by zero) |
| 0.01 | 184 | min total_toi/3600 for league_avg |
| 3.0 | 243 | league_avg_xg fallback in predict_goals |
| 1e-10 | 330 | eps for Brier/Log loss clipping |

**Update formulas (shift-level):**
- exp_h = league_avg_xg * 10^((O_h-D_a)/400) * time_frac
- delta_h = k * LN10 / elo_scale * (obs_home - exp_h)
- O[ht][hl] += delta_h, D[at][al] -= delta_h

**P formula (predict_winner):**
- diff = (O_h - D_a) - (O_a - D_h)
- P = 1/(1+10^(-diff/elo_scale))

---

## 4. Classic Elo (baseline_elo.py, baseline_elo_xg.py)

| Value | Purpose |
|-------|---------|
| expected_score: 1/(1+10^((rb-ra)/scale)) | P(team_a beats team_b) when ra=team_a, rb=team_b |
| delta = k * (outcome - expected) | Elo update |
| goals = league_avg ± (goal_diff_half_range * (win_prob - 0.5)) | predict_goals |

---

## 5. Sweep (scripts/run/run_baseline_elo_sweep.py)

| Value | Location | Purpose |
|-------|----------|---------|
| quick_test k ranges | 77 | 1.0: 5,10,15,20; 1.1: 5–20 step 5; 2.0: 5–100 step 5 |
| Full config k ranges | 31-33 | 1.0: 0.1–35 step 0.1; 1.1: 0.1–100; 2.0: 0.1–500 step 0.1 |
| const_50_brier | 282 | 0.25 |
| const_50_log | 283 | np.log(2) ≈ 0.693 |
| league_avg_acc | 284 | 0.5 |
| league_avg_brier | 285 | 0.25 |
| league_avg_log | 286 | np.log(2) |
| bins | 335 | np.linspace(0, 1, 11) for calibration |
| 1200 | 459 | Base line in rating histogram |

**Round 1 prediction flow:** Uses `final_model` (best iteration). For 2.0, fits on `raw` (shift-level) when available.

---

## 6. Pipeline Output (pipeline_summary.json)

| Field | Sample Values |
|-------|---------------|
| best_iteration | "1.0" |
| best_k | 5 |
| train_ratio | 0.7 |
| n_games | 1312 |
| n_train | 919 |
| n_test | 393 |
| team_rankings (1.0) | brazil 1290.3 … mongolia 1126 (positive Elo) |
| team_rankings (2.0) | serbia -69 … netherlands -114.6 (Net O-D) |
| league_avg_goals_per_game | 5.758 |
| league_avg_xg_per_game | 5.940 |
| league_avg_xg_per_hour_5v5 | 4.4991 |
| predictions confidence | 0.5016–0.7054 (0–1) |

---

## 7. Dashboard (baseline_results_dashboard.py)

| Value | Purpose |
|-------|---------|
| elo_scale from best_params | default 400 |
| P = 1/(1+10^(-diff/elo_scale)) | diff = rating - mean_rating |
| mean_rating | mean of team ratings in current view |
| 999 | sentinel for missing rank |
| "—" | display for missing/NaN |

---

## 8. Cross-Checks

### Elo P formula consistency
- baseline_elo.py: `1/(1+10^((rb-ra)/scale))` — P(ra beats rb)
- baseline_elo_offdef.py: `1/(1+10^(-diff/scale))` with diff = (O_h-D_a)-(O_a-D_h) — same structure
- Dashboard: `1/(1+10^(-diff/scale))` with diff = rating - mean — consistent

### 2.0 negative ratings
- team_rankings_by_iteration["2.0"] = net = O−D
- O, D start at 1200; fitted values: O ~1133–1182, D ~1225–1271
- All net negative; serbia -69 (best), netherlands -114.6 (worst)
- Ranking order: higher net = stronger ✓

### Round 1 predictions
- best_iteration = 1.0 → uses BaselineEloModel (goals)
- predictions use confidence from predict_winner (0–1) ✓

---

## 9. Potential Issues

1. **scripts/analysis/calc_brier_logloss.py** uses k=16, 70/30 fold — different from sweep (k=5, 70/30 fold 0). Standalone script, not part of sweep.
2. **Dashboard Round 1** column "Rating" for 2.0 shows raw net (e.g. -69) — consider labeling "Net (O−D)" for clarity.
3. **Config quick_test** overrides full k ranges — pipeline_summary reflects quick_test run (k 5,10,15,20 for 1.0).
