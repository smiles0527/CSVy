# Scripts — which command runs what

Run from `python/` directory.

---

## run/

| Command | What it does |
|---------|--------------|
| `python scripts/run/run_baseline_elo.py` | Baseline Elo (goals): k-grid, validation, Round 1 predictions |
| `python scripts/run/run_baseline_elo_xg.py` | Baseline Elo (xG): same as above, xG-based outcomes |
| `python scripts/run/run_baseline_elo_sweep.py` | Sweep 1.0, 1.1, 2.0: k-grid, validation, plots |
| `python scripts/run/run_baseline_elo_sweep.py --2.0-only` | Sweep 2.0 only (1000 k values: 0.1–100) |
| `python scripts/run/run_baseline_elo_sweep.py --live` | Sweep with live plot |
| `python scripts/run/run_baseline_elo_xg_sweep.py` | xG Elo sweep: k-grid, validation, plots |
| `python scripts/run/run_baseline_dashboard.py` | Build HTML dashboards (goals, xG, 2.0) |
| `python scripts/run/run_baselines.py` | All 9 baselines: compare, tune, ensemble, Round 1 |
| `python scripts/run/run_elo.py` | Enhanced Elo: grid search, fine sweep, Round 1 |
| `python scripts/run/run_game_predictor.py` | Full GP pipeline: Poisson/Ridge/GBR, Elo blend |
| `python scripts/run/run_optimized_v2.py` | Three-way blend: Ridge + GP + Elo |
| `python scripts/run/run_regen_elo.py` | Regenerate Elo Round 1 with known best params |
| `python scripts/run/run_recursive.py` | Multi-run: shuffle/expand/bootstrap/hyperparam |

---

## optimize/

| Command | What it does |
|---------|--------------|
| `python scripts/optimize/optimize_model.py` | Find best model: families, features, blend |
| `python scripts/optimize/optimize_v2.py` | Focused optimization, fold isolation checks |
| `python scripts/optimize/finetune.py` | Fine-tune Ridge alpha, blend grid |

---

## analysis/

| Command | What it does |
|---------|--------------|
| `python scripts/analysis/competitive_analysis.py` | Simulate competitors, measure edge |
| `python scripts/analysis/calc_brier_logloss.py` | Brier & log loss from baseline_elo |
| `python scripts/analysis/quick_audit.py` | Audit submission format (rows, columns, ranges) |

---

## tools/

| Command | What it does |
|---------|--------------|
| `python scripts/tools/gen_k_sweep_html.py` | Generate docs/k_sweep_2_0.html from sweep CSV |
