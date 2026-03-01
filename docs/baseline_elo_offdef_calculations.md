# Baseline Elo Off/Def (2.0) — Calculations & Derivation

Complete derivation of the Off/Def model formulas, including the gradient-consistent O/D update rule.

---

## 1. Expected xG (Per Shift)

For a matchup: home line \(h\) (offense \(O_h\), defense \(D_h\)) vs away line \(a\) (offense \(O_a\), defense \(D_a\)):

\[
\mathrm{multi}_h = 10^{(O_h - D_a) / s}
\]

\[
\mathrm{multi}_a = 10^{(O_a - D_h) / s}
\]

\[
e_h = \mathrm{leagueAvgXg} \cdot \mathrm{multi}_h \cdot \frac{t_{\Delta}}{3600}
\]

\[
e_a = \mathrm{leagueAvgXg} \cdot \mathrm{multi}_a \cdot \frac{t_{\Delta}}{3600}
\]

- \(s\) = `elo_scale` (400)
- \(t_{\Delta}\) = TOI delta in seconds
- \(\mathrm{leagueAvgXg}\) = xG per hour from training data

---

## 2. Poisson Log-Likelihood

Treat observed xG \(o_h\) as a Poisson outcome with mean \(e_h\):

\[
\ell = o_h \ln(e_h) - e_h \quad \text{(plus constant terms)}
\]

---

## 3. Gradient w.r.t. Offense

\[
\ln(e_h) = \ln(C) + \frac{O_h - D_a}{s} \ln(10)
\]

where \(C\) = leagueAvgXg \(\cdot\) (time factor) (constant in \(O_h\)).

\[
\frac{\partial e_h}{\partial O_h} = e_h \cdot \frac{\ln(10)}{s}
\]

\[
\frac{\partial \ell}{\partial O_h} = \left( \frac{o_h}{e_h} - 1 \right) \cdot e_h \cdot \frac{\ln(10)}{s} = (o_h - e_h) \cdot \frac{\ln(10)}{s}
\]

For gradient ascent (maximize log-likelihood):

\[
\delta = \eta \cdot \frac{\ln(10)}{s} \cdot (o - e)
\]

---

## 4. Update Rule (Implementation)

Using learning rate \(\eta \equiv k\):

\[
\delta_h = k \cdot \frac{\ln(10)}{s} \cdot (o_h - e_h)
\]

\[
\delta_a = k \cdot \frac{\ln(10)}{s} \cdot (o_a - e_a)
\]

- \(O_h \leftarrow O_h + \delta_h\), \(D_a \leftarrow D_a - \delta_h\)
- \(O_a \leftarrow O_a + \delta_a\), \(D_h \leftarrow D_h - \delta_a\)

**Code** (Python):

```python
LN10 = np.log(10)
delta_h = k * LN10 / elo_scale * (obs_home - exp_h)
delta_a = k * LN10 / elo_scale * (obs_away - exp_a)
```

---

## 5. O/D Identifiability

Predictions depend only on differences \((O_h - D_a)\) and \((O_a - D_h)\). Adding a constant \(c\) to all O and D:

\[
(O_h + c) - (D_a + c) = O_h - D_a
\]

So O and D are identified only up to an additive constant. Raw O/D tables are not absolutely interpretable; only differences and nets matter.

---

## 6. Net Rating & Win Probability

**Team net (ranking):**

\[
\mathrm{Net} = \frac{1}{2}\bigl[ (O_{L1} - D_{L1}) + (O_{L2} - D_{L2}) \bigr]
\]

**Win probability:**

\[
d = (O_h - D_a) - (O_a - D_h)
\]

\[
P_{\mathrm{home}} = \frac{1}{1 + 10^{-d/s}}
\]

---

## References

- `docs/baseline_elo_design.md` — design specification
- `python/utils/baseline_elo_offdef.py` — implementation
- `docs/baseline_elo_xg_calculations.md` — 1.1 (game-level xG) formulas
