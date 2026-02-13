# WHSDSC 2026 - Hockey Prediction

Predict outcomes of 16 Round 1 playoff matchups for the Western Hockey Scouts Data Science Competition 2026.

## Quick Start

1. Open VS Code in this folder
2. Open `python/training/train_baseline.ipynb`
3. Select the `.venv` Python interpreter (bottom-right)
4. Run all cells with **Shift+Enter**

## Project Structure

```
python/
  utils/                  # Model source code (imports go here)
    baseline_model.py     # 9 baseline models (GlobalMean, Dixon-Coles, Bayesian, etc.)
    elo_model.py          # Dynamic Elo ratings
    hockey_features.py    # Feature engineering from shift data
    ...                   # linear, xgboost, random_forest, neural_network, ensemble
  training/               # Training notebooks (one per model)
    train_baseline.ipynb  # <-- START HERE
  validation/             # Validation test suites (one per model)
  tutorials/              # Walkthrough tutorials (one per model)
  data/                   # WHL dataset + matchups
  output/                 # Results, predictions, saved models
  _run_baselines.py       # Standalone baseline pipeline script
docs/
  DEVELOPMENT_LOG.md      # Progress log with all results
  DATASET_FEATURES.md     # Raw feature reference
TODO.md                   # Task tracker
```

## Current Results (Baseline)

| Model | RMSE | Win Acc |
|-------|:----:|:-------:|
| Ensemble | 1.8078 | 55.5% |
| Bayesian (pw=20) | 1.8096 | 56.3% |
| Dixon-Coles (d=1.0) | 1.8126 | 56.3% |
| HomeAway | 1.8247 | 55.5% |

## Data

- **Training**: `python/data/whl_2025.csv` (25,827 shifts, 1,312 games, 32 teams)
- **Predictions**: `python/data/WHSDSC_Rnd1_matchups.xlsx` (16 matchups)
