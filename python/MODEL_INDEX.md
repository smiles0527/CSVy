# Model Index

Quick reference: which files belong to which model.

All model code lives in `utils/`. Notebooks import from there.

**See also:** [MODEL_VALUES_INDEX.md](MODEL_VALUES_INDEX.md) — explains which value does what (params, constants, formulas).

---

## Model 1: Baseline (9 models)

| File | Purpose |
|------|---------|
| `utils/baseline_model.py` | All 9 baseline classes + compare/save/load |
| `training/train_baseline.ipynb` | Train all 17 configs, tune, predict Round 1 |
| `validation/validate_baseline.ipynb` | 9 validation tests |
| `scripts/run/run_baselines.py` | Standalone pipeline script |

**Classes:** `GlobalMeanBaseline`, `TeamMeanBaseline`, `HomeAwayBaseline`, `MovingAverageBaseline`, `WeightedHistoryBaseline`, `PoissonBaseline`, `DixonColesBaseline`, `BayesianTeamBaseline`, `EnsembleBaseline`

---

## Baseline Elo (minimal Elo benchmark)

| File | Purpose |
|------|---------|
| `utils/baseline_elo.py` | BaselineEloModel class (classic Elo, no home/MOV) |
| `training/baseline_elo/train_baseline_elo.ipynb` | Train + k-grid search + Round 1 predictions |
| `validation/validate_baseline_elo.ipynb` | Validation tests |
| `scripts/run/run_baseline_elo.py` | Standalone pipeline script |
| `config/hyperparams/model_baseline_elo.yaml` | YAML config |

---

## Model 2: Elo Ratings

| File | Purpose |
|------|---------|
| `utils/elo_model.py` | EloModel class |
| `training/train_elo.ipynb` | Train + hyperparameter search |
| `validation/validate_elo.ipynb` | Validation tests |

---

## Model 3: Linear Regression

| File | Purpose |
|------|---------|
| `utils/linear_model.py` | LinearRegressionModel + LinearGoalPredictor |
| `training/train_linear.ipynb` | Train notebook |
| `training/linear_hyperparam_search.py` | Hyperparameter search |
| `validation/validate_linear.ipynb` | Validation tests |

---

## Model 4a: XGBoost

| File | Purpose |
|------|---------|
| `utils/xgboost_model.py` | XGBoostModel |
| `training/train_xgboost.ipynb` | Train notebook |
| `training/xgboost_hyperparam_search.py` | Hyperparameter search |
| `validation/validate_xgboost.ipynb` | Validation tests |

---

## Model 4b: Random Forest

| File | Purpose |
|------|---------|
| `utils/random_forest_model.py` | RandomForestModel + RandomForestGoalPredictor |
| `training/train_random_forest.ipynb` | Train notebook |
| `training/random_forest_hyperparam_search.py` | Hyperparameter search |
| `validation/validate_random_forest.ipynb` | Validation tests |

---

## Model 5: Ensemble

| File | Purpose |
|------|---------|
| `utils/ensemble_model.py` | EnsembleModel + StackedEnsemble |
| `training/train_ensemble.ipynb` | Train notebook |
| `training/ensemble_hyperparam_search.py` | Hyperparameter search |
| `validation/validate_ensemble.ipynb` | Validation tests |

---

## Model 6: Neural Network

| File | Purpose |
|------|---------|
| `utils/neural_network_model.py` | NeuralNetworkModel (sklearn MLP) |
| `training/train_neural_network.ipynb` | Train notebook |
| `training/neural_network_hyperparam_search.py` | Hyperparameter search |
| `validation/validate_neural_network.ipynb` | Validation tests |

---

## Shared Utilities

| File | Purpose |
|------|---------|
| `utils/hockey_features.py` | Feature engineering from WHL shift data |
| `utils/experiment_tracker.py` | MLflow experiment tracking |
| `utils/training_callbacks.py` | Progress bars, early stopping |
| `utils/live_dashboard.py` | Real-time training visualization |
