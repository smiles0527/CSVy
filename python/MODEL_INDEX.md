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

Linear regression with ElasticNet regularization (L1/L2), polynomial features, and feature scaling.

| File | Purpose |
|------|---------|
| `models/linear_model.ipynb` | Core implementation notebook |
| `utils/linear_model.py` | Reusable module with LinearRegressionModel class |
| `training/train_linear.ipynb` | Training with hyperparameter tuning |
| `tutorials/linear_tutorial.ipynb` | Tutorial explaining regularization |
| `validation/validate_linear.ipynb` | Validation tests |

**Classes:** `LinearRegressionModel`, `LinearGoalPredictor`

**Key Hyperparameters:** `alpha` (regularization strength), `l1_ratio` (0=Ridge, 1=Lasso, 0.5=ElasticNet), `poly_degree`, `scaling`

**Functions:** `grid_search_linear`, `random_search_linear`, `compare_regularization`

---

## Model 4: XGBoost / Random Forest

Gradient boosting and tree-based models for high accuracy.

| File | Purpose |
|------|---------|
| `models/xgboost_model.ipynb` | Core implementation notebook |
| `utils/xgboost_model.py` | Reusable XGBoost module |
| `training/train_xgboost.ipynb` | Training with hyperparameter tuning |
| `training/train_random_forest.ipynb` | Random Forest training *(planned)* |
| `validation/validate_xgboost.ipynb` | Validation tests *(planned)* |

**Classes:** `XGBoostModel`, `XGBoostGoalPredictor`

**Key Hyperparameters:** `learning_rate`, `n_estimators`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`

**Functions:** `grid_search_xgboost`, `random_search_xgboost`

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
│   ├── elo_model.ipynb         # Model 2
│   ├── linear_model.ipynb      # Model 3
│   └── xgboost_model.ipynb     # Model 4
│
├── utils/                      # Reusable model modules
│   ├── __init__.py
│   ├── baseline_model.py       # Model 1
│   ├── elo_model.py            # Model 2
│   ├── linear_model.py         # Model 3
│   ├── xgboost_model.py        # Model 4
│   └── xgboost_model_v2.py     # Model 4 (production version)
│
├── training/                   # Training notebooks
│   ├── train_baseline.ipynb    # Model 1
│   ├── train_elo.ipynb         # Model 2
│   ├── train_linear.ipynb      # Model 3
│   └── train_xgboost.ipynb     # Model 4
│
├── tutorials/                  # Tutorial notebooks
│   ├── baseline_tutorial.ipynb # Model 1
│   ├── elo_tutorial.ipynb      # Model 2
│   └── linear_tutorial.ipynb   # Model 3
│
├── validation/                 # Validation notebooks
│   ├── validate_baseline.ipynb # Model 1
│   ├── validate_elo.ipynb      # Model 2
│   └── validate_linear.ipynb   # Model 3
│
└── MODEL_INDEX.md              # This file
```

---

## Status Summary

| Model | Module | Training | Validation | Tutorial | Status |
|-------|--------|----------|------------|----------|--------|
| 1. Baseline | ✅ | ✅ | ✅ | ✅ | **Complete** |
| 2. ELO | ✅ | ✅ | ✅ | ✅ | **Complete** |
| 3. Linear | ✅ | ✅ | ✅ | ✅ | **Complete** |
| 4. XGBoost | ✅ | ✅ | ❌ | ❌ | **In Progress** |
| 5. Ensemble | ❌ | ❌ | ❌ | ❌ | Not Started |

---

*Last updated: January 2026*
