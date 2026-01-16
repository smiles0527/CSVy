# Model Index

Quick reference for which files belong to which model.

---

## Model 1: Baseline Models

Simple reference baselines for benchmarking.

| File | Purpose |
|------|---------|
| `models/baseline_model.ipynb` | Core implementation notebook |
| `utils/baseline_model.py` | Reusable module with 6 baseline classes |
| `training/train_baseline.ipynb` | Train & compare all baselines |
| `validation/validate_baseline.ipynb` | 8 validation tests |
| `baseline_tutorial.ipynb` | Tutorial explaining each baseline |

**Classes:** `GlobalMeanBaseline`, `TeamMeanBaseline`, `HomeAwayBaseline`, `MovingAverageBaseline`, `WeightedHistoryBaseline`, `PoissonBaseline`

---

## Model 2: ELO Ratings

Dynamic rating system that updates after each game.

| File | Purpose |
|------|---------|
| `models/elo_model.ipynb` | Core implementation notebook |
| `utils/elo_model.py` | Reusable EloModel class |
| `training/train_elo.ipynb` | Grid search hyperparameter training |
| `validation/validate_elo.ipynb` | Validation tests |
| `elo_tutorial.ipynb` | Tutorial with examples |

**Classes:** `EloModel`

**Key Hyperparameters:** `k_factor`, `home_advantage`, `initial_rating`

---

## Model 3: Linear Regression

*(Not yet implemented)*

| File | Purpose |
|------|---------|
| `utils/linear_model.py` | Core module |
| `training/train_linear.ipynb` | Training notebook |
| `validation/validate_linear.ipynb` | Validation tests |

---

## Model 4: XGBoost / Random Forest

*(Not yet implemented)*

| File | Purpose |
|------|---------|
| `utils/tree_model.py` | XGBoost/RF implementations |
| `training/train_xgboost.ipynb` | XGBoost training |
| `training/train_random_forest.ipynb` | Random Forest training |
| `validation/validate_tree.ipynb` | Validation tests |

---

## Model 5: Ensemble

*(Not yet implemented)*

| File | Purpose |
|------|---------|
| `utils/ensemble_model.py` | Ensemble combining multiple models |
| `training/train_ensemble.ipynb` | Ensemble training |
| `validation/validate_ensemble.ipynb` | Validation tests |

---

## Shared Utilities

| File | Purpose |
|------|---------|
| `utils/__init__.py` | Exports all model classes |
| `utils/data_loader.py` | Data loading utilities *(planned)* |
| `utils/feature_engineering.py` | Feature creation *(planned)* |

---

## Directory Structure

```
python/
├── models/                     # Core implementation notebooks
│   ├── baseline_model.ipynb    # Model 1
│   └── elo_model.ipynb         # Model 2
│
├── utils/                      # Reusable model modules
│   ├── __init__.py
│   ├── baseline_model.py       # Model 1
│   └── elo_model.py            # Model 2
│
├── training/                   # Training notebooks
│   ├── train_baseline.ipynb    # Model 1
│   └── train_elo.ipynb         # Model 2
│
├── validation/                 # Validation notebooks
│   ├── validate_baseline.ipynb # Model 1
│   └── validate_elo.ipynb      # Model 2
│
├── baseline_tutorial.ipynb     # Model 1 tutorial
├── elo_tutorial.ipynb          # Model 2 tutorial
│
└── MODEL_INDEX.md              # This file
```

---

## Status Summary

| Model | Module | Training | Validation | Tutorial | Status |
|-------|--------|----------|------------|----------|--------|
| 1. Baseline | ✅ | ✅ | ✅ | ✅ | **Complete** |
| 2. ELO | ✅ | ✅ | ✅ | ✅ | **Complete** |
| 3. Linear | ❌ | ❌ | ❌ | ❌ | Not Started |
| 4. XGBoost/RF | ❌ | ❌ | ❌ | ❌ | Not Started |
| 5. Ensemble | ❌ | ❌ | ❌ | ❌ | Not Started |

---

*Last updated: January 2026*
