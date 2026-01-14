# 🚀 QUICK START - Win in 5 Steps

## Step 1: Preprocess Data (5 minutes)
```bash
ruby cli.rb competitive-pipeline data/your_nhl_data.csv -o data/processed
```
**Output**: `competitive_features.csv` with 14+ advanced features  
**Features added**: team_strength_index, pythagorean_wins, momentum_score, clutch_factor, rest_days, home_away_diff, consistency_score, time_weight, and more

---

## Step 2: Generate Hyperparameters (10 minutes)
```bash
# Smart search for expensive models
ruby cli.rb hyperparam-bayesian config/hyperparams/model4_xgboost.yaml --iterations 50

# Genetic algorithm for large spaces
ruby cli.rb hyperparam-genetic config/hyperparams/model3_elo.yaml --population 50 --generations 20

# Grid search for quick models
ruby cli.rb hyperparam-grid config/hyperparams/model2_linear_regression.yaml
ruby cli.rb hyperparam-grid config/hyperparams/model4_random_forest.yaml
ruby cli.rb hyperparam-grid config/hyperparams/model5_ensemble.yaml
```
**Output**: CSV files with 2,812 total hyperparameter combinations to test

---

## Step 3: Train Models (DeepNote/Python - 4 hours)
```python
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load hyperparameters
configs = pd.read_csv('model4_xgboost_bayesian_optimization.csv')

# Load preprocessed data
X_train = pd.read_csv('data/processed/train.csv')
y_train = X_train['target']
X_train = X_train.drop('target', axis=1)

# Train each config
for idx, row in configs.iterrows():
    params = row.to_dict()
    exp_id = params.pop('experiment_id')
    
    # Remove tracking columns
    for col in ['rmse', 'mae', 'r2', 'notes', 'timestamp']:
        params.pop(col, None)
    
    # Train
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    configs.loc[idx, 'rmse'] = mean_squared_error(y_test, y_pred, squared=False)
    configs.loc[idx, 'r2'] = r2_score(y_test, y_pred)
    
    # Save best predictions
    if configs.loc[idx, 'rmse'] < 2.0:
        pd.DataFrame({'actual': y_test, 'predicted': y_pred}).to_csv(
            f'predictions/xgb_exp{exp_id}.csv', index=False
        )

# Save results
configs.to_csv('model4_xgboost_bayesian_optimization.csv', index=False)
```

---

## Step 4: Find Best Hyperparameters (2 minutes)
```bash
# Pull results from DeepNote
git pull

# Find optimal params for each model
ruby cli.rb best-params experiments/model4_xgboost_bayesian_optimization.csv --metric rmse
ruby cli.rb best-params experiments/model3_elo_genetic_algorithm.csv --metric rmse
ruby cli.rb best-params experiments/model2_linear_regression_grid_search.csv --metric rmse

# Compare top experiments
ruby cli.rb compare-experiments experiments/model4_xgboost_bayesian_optimization.csv 5,12,23,41
```
**Output**: Optimal hyperparameters for each model

---

## Step 5: Optimize Ensemble (5 minutes)
```bash
# After training all models with best params, save predictions to predictions/
# predictions/baseline_preds.csv
# predictions/linear_preds.csv
# predictions/elo_preds.csv
# predictions/xgboost_preds.csv
# predictions/rf_preds.csv

# Find optimal weights
ruby cli.rb ensemble-optimize predictions/ --actuals data/processed/test.csv -o weights.csv

# Check diversity (must be > 0.5)
ruby cli.rb diversity-analysis predictions/ data/processed/test.csv

# Validate final ensemble
ruby cli.rb validate-model predictions/final_ensemble.csv --bootstrap --calibration
```
**Output**: 
- Optimal ensemble weights
- Diversity score (higher = better)
- Bootstrap confidence intervals
- Calibration metrics

---

## 🎯 Expected Results

| Model | RMSE | Improvement |
|-------|------|-------------|
| Baseline | 3.5 | - |
| Linear | 2.7 | 23% |
| ELO | 2.4 | 31% |
| XGBoost | 2.0 | 43% |
| Random Forest | 2.2 | 37% |
| **Ensemble** | **1.7** | **51%** 🏆 |

---

## 🔥 Power Commands

### See All Available Commands
```bash
ruby cli.rb help
```

### Add Results from Training
```bash
ruby cli.rb add-result experiments/xgb_bayesian.csv 42 --rmse 1.95 --mae 1.52 --r2 0.84
```

### Export Best Params to Python
```bash
ruby cli.rb export-params config/hyperparams/model4_xgboost.yaml -f python -o best_params.py
```

### Quick Data Quality Check
```bash
ruby cli.rb diagnose data/your_data.csv
```

### Generate HTML Report
```bash
ruby cli.rb report data/your_data.csv
```

---

## 📊 What the Tools Do

### `competitive-pipeline`
**Does:** Full preprocessing + 14 advanced features + train/test split  
**Time:** 30 seconds - 2 minutes  
**Output:** Ready-to-train dataset with competition-winning features

### `hyperparam-bayesian`
**Does:** Smart hyperparameter search using Gaussian Process  
**Time:** 2-10 minutes (depends on iterations)  
**Output:** CSV with tested configs, sorted by performance  
**Best for:** Expensive models (XGBoost, Neural Networks)

### `hyperparam-genetic`
**Does:** Evolution-based search with crossover/mutation  
**Time:** 5-15 minutes  
**Output:** CSV with evolved configurations  
**Best for:** Large search spaces (1000+ combinations)

### `ensemble-optimize`
**Does:** Finds optimal weights to combine model predictions  
**Time:** 10-30 seconds  
**Output:** Optimal weights for each model  
**Best for:** Final ensemble after training all models

### `diversity-analysis`
**Does:** Checks if models make different mistakes (good for ensemble)  
**Time:** 5-10 seconds  
**Output:** Error correlation matrix + diversity score  
**Best for:** Deciding if ensemble will help

### `validate-model`
**Does:** Bootstrap CI + calibration analysis  
**Time:** 30 seconds - 2 minutes  
**Output:** Confidence intervals + calibration error  
**Best for:** Final validation before submission

---

## 💡 Pro Tips

1. **Use Bayesian for XGBoost** (it's expensive to train)
   ```bash
   ruby cli.rb hyperparam-bayesian config/hyperparams/model4_xgboost.yaml --iterations 50
   ```

2. **Check diversity BEFORE optimizing ensemble**
   - If diversity < 0.3: Models too similar, won't help
   - If diversity > 0.5: Great! Ensemble will boost performance

3. **Time series data = time series validation**
   - Use expanding window CV (built into `competitive-pipeline`)
   - Never use random splits (causes data leakage)

4. **Monitor for overfitting**
   - Train RMSE - Test RMSE should be < 0.3
   - If gap > 0.5: Reduce complexity or add regularization

5. **Bootstrap gives you confidence**
   ```bash
   ruby cli.rb validate-model preds.csv --bootstrap
   # Tight CI (< 0.2): High confidence
   # Wide CI (> 0.5): Model unstable
   ```

---

## 🚨 Common Issues

### "File not found" errors
```bash
# Create directories automatically
ruby cli.rb hyperparam-random config/hyperparams/model3_elo.yaml 10 -o experiments/elo_random.csv
# CSVy auto-creates experiments/ folder
```

### RMSE > 3.0 (Not competitive)
- Check: Are all 14 features included?
- Run: `ruby cli.rb diagnose data/processed/competitive_features.csv`
- Verify: Time-based train/test split (no data leakage)

### Ensemble doesn't beat individual models
- Check: `ruby cli.rb diversity-analysis predictions/ actuals.csv`
- If diversity < 0.3: Models too similar, try different algorithms
- If diversity > 0.5: Should work, check weight optimization

### Git issues with DeepNote
```bash
# Push local changes
git add . && git commit -m "Update features" && git push

# Pull DeepNote results
git pull origin main
```

---

## 🎓 Learn More

- **Full documentation**: [README.md](README.md)
- **Winning strategy**: [WINNING_STRATEGY.md](WINNING_STRATEGY.md)
- **Hyperparameter configs**: `config/hyperparams/`
- **Sample data**: `data/sample_nhl_standings.csv`

---

## 🏆 You're Ready to Win!

**You now have:**
- ✅ 40+ CLI commands
- ✅ 14+ advanced features
- ✅ 5 optimization algorithms
- ✅ Ensemble methods (stacking, blending, weighted voting)
- ✅ Model validation (bootstrap, calibration, time series CV)
- ✅ 2,812 hyperparameter combinations

**Target**: Ensemble RMSE < 1.7 goals (51% better than baseline)

**Go execute!** 🚀🏒🏆
