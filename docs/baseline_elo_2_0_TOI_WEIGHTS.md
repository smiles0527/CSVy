# Baseline Elo 2.0 — TOI-Weighted Ranking Changes

## Summary

2.0 now uses TOI-weighted line aggregation for ranking, `predict_goals`, and `predict_winner` instead of equal-weight (0.5/0.5).

---

## Changes

### Net (ranking)

**Before:** \(\mathrm{Net} = \frac{1}{2}[(O_{L1}-D_{L1}) + (O_{L2}-D_{L2})]\)

**After:** \(\mathrm{Net} = w_1 (O_{L1}-D_{L1}) + w_2 (O_{L2}-D_{L2})\) with \(w_1 + w_2 = 1\)

Weights \(w_1, w_2\) = empirical TOI shares from training shifts (L1 TOI / total, L2 TOI / total). Fallback: \(w_1 = w_2 = 0.5\) when no shift/TOI data.

---

### predict_goals / predict_winner

**Before:** \(O_h, D_h, O_a, D_a\) = arithmetic mean over L1 and L2.

**After:** \(O_h, D_h, O_a, D_a\) = TOI-weighted sum over L1 and L2 using per-team weights.

---

### Implementation

- `baseline_elo_offdef.py`: `team_toi_weights` dict; `_get_line_weights(team)`; TOI pre-pass in `_fit_shifts`; `_team_net`, `predict_goals`, `predict_winner` use weights.
- Game-level fit: `team_toi_weights` remains empty; `_get_line_weights` returns (0.5, 0.5).
