# Baseline Elo 2.0 changes

---

## Code

### `python/utils/baseline_elo_offdef.py`

**Line ~157–158** (`_fit_games`):

```
- delta_h = self.k * (obs_home - exp_h) / LN10
- delta_a = self.k * (obs_away - exp_a) / LN10
+ delta_h = self.k * LN10 / self.elo_scale * (obs_home - exp_h)
+ delta_a = self.k * LN10 / self.elo_scale * (obs_away - exp_a)
```

**Line ~230–231** (`_fit_shifts`):

```
- delta_h = self.k * (obs_home - exp_h) / LN10
- delta_a = self.k * (obs_away - exp_a) / LN10
+ delta_h = self.k * LN10 / self.elo_scale * (obs_home - exp_h)
+ delta_a = self.k * LN10 / self.elo_scale * (obs_away - exp_a)
```

**Module docstring** (lines 9–13):

```
- O += k * (observed_xG - expected_xG) / ln(10)
- D -= k * (observed_xG - expected_xG) / ln(10)
+ O += k * ln(10) / elo_scale * (observed_xG - expected_xG)
+ D -= k * ln(10) / elo_scale * (observed_xG - expected_xG)
```

---

## Config

### `config/hyperparams/model_baseline_elo_sweep.yaml`

**2.0 k range** (extend max for new formula):

```
# iterations.2.0:
- max: 100
+ max: 500   # or adjust as needed; new formula yields smaller effective updates
```

**quick_test 2.0 range** (in `python/scripts/run/run_baseline_elo_sweep.py`):

```
- 2.0: { min: 5, max: 30, step: 5 }
+ 2.0: { min: 5, max: 100, step: 5 }
```

**--2.0-only** (same script):

```
- max: 100
+ max: 500
```

---

## Documentation

### `docs/baseline_elo_design.md`

**Section 4.5** — replace formulas:

```
- δₕ = k · (oₕ − eₕ) / ln(10)
- δₐ = k · (oₐ − eₐ) / ln(10)
+ δₕ = k · (ln(10) / s) · (oₕ − eₕ)
+ δₐ = k · (ln(10) / s) · (oₐ − eₐ)
```

Add: *Derivation: Poisson log-likelihood gradient; see `docs/baseline_elo_offdef_calculations.md`.*

**Section 10 References** — add:

```
- docs/baseline_elo_offdef_calculations.md — 2.0 derivation
```

---

### `python/MODEL_VALUES_INDEX.md`

**LN10 row**:

```
- | LN10 | ln(10) ≈ 2.3026 | 2.0 | Scales delta: k×(obs−exp)/LN10. Matches log-based Elo convention. |
+ | LN10 | ln(10) ≈ 2.3026 | 2.0 | Part of gradient scaling: delta = k·(LN10/elo_scale)·(obs−exp). |
```

**2.0 specifics delta row**:

```
- | delta | k×(obs−exp)/LN10. Added to O, subtracted from D. |
+ | delta | k·(LN10/elo_scale)·(obs−exp). Added to O, subtracted from D. |
```

**Formula Quick Reference — Off/Def update**:

```
- delta = k × (observed_xG - expected_xG) / ln(10)
+ delta = k × (ln(10) / elo_scale) × (observed_xG - expected_xG)
```

---

### `python/output/predictions/baseline_elo/sweep/VALUE_AUDIT.md`

**Section 3 Off/Def — update formulas**:

```
- delta_h = k * (obs_home - exp_h) / LN10
- delta_a = k * (obs_away - exp_a) / LN10
+ delta_h = k * LN10 / elo_scale * (obs_home - exp_h)
+ delta_a = k * LN10 / elo_scale * (obs_away - exp_a)
```

(Regenerate sample values after re-running sweep if needed.)

---

### New: `docs/baseline_elo_offdef_calculations.md`

Create file with:

- Expected xG formula (4.4)
- Poisson log-likelihood
- Gradient ∂ℓ/∂Oₕ, ∂ℓ/∂Dₐ
- Update rule δ = k·(ln10/s)·(o−e)
- Note on O/D identifiability

---

### `docs/baseline_elo_xg_calculations.md`

**References section** — add (if 2.0 is mentioned):

```
- 2.0 derivation: docs/baseline_elo_offdef_calculations.md
```
