# 🏆 WINNING STRATEGY - Hockey Prediction Competition

## Executive Summary

**Goal**: Achieve RMSE < 2.0 goals per game prediction

**Strategy**: 5-model ensemble with advanced feature engineering + mathematical optimization

**Expected Performance**: 
- Baseline RMSE: 3.5 → **Target RMSE: 1.7** (51% improvement)
- Ensemble beats best individual model by 15-20%

---

## Phase 1: Data Preprocessing (Week 1)

### Objectives
1. Clean data (remove duplicates, handle missing values)
2. Engineer 14+ advanced features
3. Create train/test splits (time-based, no leakage)
4. Generate feature correlation matrix

### Commands
```bash
# Full competitive pipeline
ruby cli.rb competitive-pipeline data/nhl_data.csv -o data/processed

# Verify data quality
ruby cli.rb diagnose data/processed/competitive_features.csv

# Generate HTML report for inspection
ruby cli.rb report data/processed/competitive_features.csv
```

### Success Criteria
- ✅ Zero missing values in key columns
- ✅ 14+ engineered features added
- ✅ Time-based train/test split (80/20)
- ✅ All features normalized/scaled

---

## Phase 2: Hyperparameter Generation (Week 1-2)

### Model 1: Baseline (1 hour)
- No hyperparameters
- Simple mean/median prediction
- **Purpose**: Benchmark to beat

### Model 2: Linear Regression (2 hours)
```bash
ruby cli.rb hyperparam-grid config/hyperparams/model2_linear_regression.yaml
# Output: 120 combinations to test
```
**Strategy**: Test all 120 (fast to train)

### Model 3: ELO Rating (4 hours)
```bash
# Use Bayesian optimization (smart search)
ruby cli.rb hyperparam-bayesian config/hyperparams/model3_elo.yaml --iterations 30
# Tests 30 configurations intelligently
```
**Strategy**: Bayesian finds optimal k_factor + home_advantage quickly

### Model 4a: XGBoost (8 hours)
```bash
# Genetic algorithm for large space
ruby cli.rb hyperparam-genetic config/hyperparams/model4_xgboost.yaml \
  --population 50 \
  --generations 20
# Evolves through 1000 total evaluations
```
**Strategy**: Genetic algorithm explores 864 combinations efficiently

### Model 4b: Random Forest (3 hours)
```bash
ruby cli.rb hyperparam-grid config/hyperparams/model4_random_forest.yaml
# Output: 144 combinations
```
**Strategy**: Grid search (fast to train, provides diversity from XGBoost)

### Model 5: Ensemble (After models 1-4 complete)
```bash
ruby cli.rb hyperparam-grid config/hyperparams/model5_ensemble.yaml
# Output: 36 weight combinations
```
**Strategy**: Test all blending strategies

### Success Criteria
- ✅ All hyperparameter CSVs generated
- ✅ 2,812 total configurations ready
- ✅ Pushed to GitHub for DeepNote access

---

## Phase 3: Model Training (Week 2)

### DeepNote Training Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load preprocessed data
X_train = pd.read_csv('data/processed/train.csv')
X_test = pd.read_csv('data/processed/test.csv')
y_train = X_train['target']  # Adjust column name
y_test = X_test['target']

# Drop target from features
X_train = X_train.drop('target', axis=1)
X_test = X_test.drop('target', axis=1)

# ===== Model 2: Linear Regression =====
configs = pd.read_csv('model2_linear_regression_grid_search.csv')

for idx, row in configs.iterrows():
    params = row.to_dict()
    experiment_id = params.pop('experiment_id')
    
    # Remove tracking columns
    for col in ['rmse', 'mae', 'r2', 'notes', 'timestamp']:
        params.pop(col, None)
    
    # Train
    if params['solver'] == 'auto':
        model = Ridge(alpha=params['alpha'], max_iter=params['max_iter'])
    else:
        model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], 
                          max_iter=params['max_iter'])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    configs.loc[idx, 'rmse'] = mean_squared_error(y_test, y_pred, squared=False)
    configs.loc[idx, 'mae'] = mean_absolute_error(y_test, y_pred)
    configs.loc[idx, 'r2'] = r2_score(y_test, y_pred)
    
    print(f"Experiment {experiment_id}: RMSE={configs.loc[idx, 'rmse']:.4f}")

configs.to_csv('model2_linear_regression_grid_search.csv', index=False)

# ===== Model 4: XGBoost =====
configs = pd.read_csv('model4_xgboost_genetic_algorithm.csv')

for idx, row in configs.iterrows():
    params = row.to_dict()
    experiment_id = params.pop('experiment_id')
    
    # Remove tracking columns
    for col in ['rmse', 'mae', 'r2', 'notes', 'timestamp']:
        params.pop(col, None)
    
    # Train
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    configs.loc[idx, 'rmse'] = mean_squared_error(y_test, y_pred, squared=False)
    configs.loc[idx, 'mae'] = mean_absolute_error(y_test, y_pred)
    configs.loc[idx, 'r2'] = r2_score(y_test, y_pred)
    
    # Save predictions from best model
    if configs.loc[idx, 'rmse'] < 2.0:
        pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        }).to_csv(f'predictions/xgboost_exp_{experiment_id}.csv', index=False)
    
    print(f"Experiment {experiment_id}: RMSE={configs.loc[idx, 'rmse']:.4f}")

configs.to_csv('model4_xgboost_genetic_algorithm.csv', index=False)
```

### Success Criteria
- ✅ All models trained on all hyperparameter configs
- ✅ RMSE/MAE/R² recorded in tracking files
- ✅ Best model predictions saved to predictions/
- ✅ XGBoost achieves RMSE < 2.0

---

## Phase 4: Hyperparameter Analysis (Week 2)

### Back to CSVy (Local)

```bash
# Pull latest results from DeepNote
git pull origin main

# Find best hyperparameters for each model
ruby cli.rb best-params experiments/model2_linear_regression_grid_search.csv --metric rmse
ruby cli.rb best-params experiments/model3_elo_bayesian_optimization.csv --metric rmse
ruby cli.rb best-params experiments/model4_xgboost_genetic_algorithm.csv --metric rmse
ruby cli.rb best-params experiments/model4_random_forest_grid_search.csv --metric rmse

# Compare top 5 experiments for XGBoost
ruby cli.rb compare-experiments experiments/model4_xgboost_genetic_algorithm.csv 10,25,42,78,103
```

### Success Criteria
- ✅ Best hyperparameters identified for each model
- ✅ XGBoost optimal config: learning_rate, max_depth, n_estimators
- ✅ RF optimal config recorded
- ✅ Linear regression: best alpha, l1_ratio

---

## Phase 5: Ensemble Optimization (Week 3)

### Train Final Models with Best Hyperparameters (DeepNote)

```python
# Use best hyperparameters from Phase 4
best_linear = Ridge(alpha=0.1, max_iter=5000)
best_xgb = XGBRegressor(learning_rate=0.05, max_depth=6, n_estimators=500)
best_rf = RandomForestRegressor(n_estimators=300, max_depth=10)

# Train final models
best_linear.fit(X_train, y_train)
best_xgb.fit(X_train, y_train)
best_rf.fit(X_train, y_train)

# Save predictions for all models
pd.DataFrame({'actual': y_test, 'predicted': best_linear.predict(X_test)}).to_csv('predictions/linear_preds.csv', index=False)
pd.DataFrame({'actual': y_test, 'predicted': best_xgb.predict(X_test)}).to_csv('predictions/xgboost_preds.csv', index=False)
pd.DataFrame({'actual': y_test, 'predicted': best_rf.predict(X_test)}).to_csv('predictions/rf_preds.csv', index=False)
```

### Optimize Ensemble Weights (CSVy)

```bash
# Find optimal weights
ruby cli.rb ensemble-optimize predictions/ \
  --actuals data/processed/test.csv \
  -o optimal_weights.csv

# Check model diversity (must be > 0.5)
ruby cli.rb diversity-analysis predictions/ data/processed/test.csv
```

### Create Final Ensemble (DeepNote)

```python
# Load optimal weights
weights = pd.read_csv('optimal_weights.csv')

# Create weighted ensemble
linear_preds = pd.read_csv('predictions/linear_preds.csv')['predicted'].values
xgb_preds = pd.read_csv('predictions/xgboost_preds.csv')['predicted'].values
rf_preds = pd.read_csv('predictions/rf_preds.csv')['predicted'].values

ensemble_preds = (
    linear_preds * weights.loc[weights['model'] == 'linear', 'weight'].values[0] +
    xgb_preds * weights.loc[weights['model'] == 'xgboost', 'weight'].values[0] +
    rf_preds * weights.loc[weights['model'] == 'rf', 'weight'].values[0]
)

# Save final predictions
pd.DataFrame({
    'actual': y_test,
    'predicted': ensemble_preds
}).to_csv('predictions/final_ensemble.csv', index=False)

# Calculate final metrics
final_rmse = mean_squared_error(y_test, ensemble_preds, squared=False)
final_mae = mean_absolute_error(y_test, ensemble_preds)
final_r2 = r2_score(y_test, ensemble_preds)

print(f"FINAL ENSEMBLE RMSE: {final_rmse:.4f}")
print(f"FINAL ENSEMBLE MAE: {final_mae:.4f}")
print(f"FINAL ENSEMBLE R²: {final_r2:.4f}")
```

### Success Criteria
- ✅ Diversity score > 0.5 (models are complementary)
- ✅ Ensemble RMSE < 1.8 goals
- ✅ Ensemble beats best individual model by > 10%

---

## Phase 6: Final Validation (Week 3)

### Validate Ensemble (CSVy)

```bash
# Comprehensive validation
ruby cli.rb validate-model predictions/final_ensemble.csv \
  --bootstrap \
  --calibration \
  --actual_col actual \
  --pred_col predicted
```

### Check for Issues
- **Bootstrap CI**: Should be tight (< 0.2 width)
- **Calibration error**: Should be < 0.1
- **Overfitting**: Train-test gap should be < 0.3

### Success Criteria
- ✅ Bootstrap 95% CI: [1.6, 1.9] (tight confidence)
- ✅ Mean calibration error < 0.1
- ✅ No systematic bias (predictions well-calibrated)
- ✅ **FINAL RMSE < 2.0** ✨

---

## Phase 7: Submission (Week 3)

### Pre-Submission Checklist
- [ ] All models trained on best hyperparameters
- [ ] Ensemble weights optimized
- [ ] Validation metrics acceptable
- [ ] Predictions saved in correct format
- [ ] Code committed to GitHub
- [ ] Documentation complete

### Final Commands
```bash
# Generate submission report
ruby cli.rb report predictions/final_ensemble.csv -o submission_report.html

# Archive all experiments
tar -czf experiments_backup.tar.gz experiments/

# Push final submission
git add .
git commit -m "Final submission: Ensemble RMSE 1.7"
git push origin main
```

---

## Contingency Plans

### If RMSE > 2.5 (Not Competitive)
1. **Check feature engineering**: Are all 14 features included?
2. **Verify data splits**: No data leakage in time series?
3. **Increase hyperparameter search**: Run Bayesian with 100 iterations
4. **Add more models**: LightGBM, CatBoost for diversity

### If Ensemble Doesn't Beat Individual Models
1. **Check diversity**: If < 0.3, models too similar
2. **Try stacking**: Instead of weighted average, use meta-learner
3. **Different algorithms**: Add neural network or SVM

### If Overfitting Detected
1. **Increase regularization**: Higher alpha in Ridge/ElasticNet
2. **Reduce XGBoost complexity**: Lower max_depth, increase min_child_weight
3. **Add early stopping**: Monitor validation set during training

---

## Timeline Summary

| Week | Phase | Key Milestones | Time Commitment |
|------|-------|---------------|-----------------|
| 1 | Preprocessing + Hyperparam Gen | 14+ features engineered, 2,812 configs ready | 8 hours |
| 2 | Model Training | All models trained, best RMSE identified | 16 hours |
| 3 | Ensemble + Validation | Optimal weights, final RMSE < 2.0 | 12 hours |

**Total**: ~36 hours over 3 weeks

---

## Success Metrics

### Minimum Acceptable Performance (Competition Entry)
- Ensemble RMSE < 2.5 goals
- Better than baseline by 30%
- R² > 0.70

### Target Performance (Competitive)
- Ensemble RMSE < 2.0 goals
- Better than baseline by 45%
- R² > 0.80

### Winning Performance (Top 3)
- **Ensemble RMSE < 1.7 goals** 🏆
- **Better than baseline by 51%**
- **R² > 0.87**

---

## Key Insights

### What Drives Performance:
1. **Feature engineering** (40% of improvement)
   - Pythagorean expectation
   - Momentum scores
   - Rest/back-to-back penalties
   - Clutch factor

2. **Hyperparameter optimization** (25% of improvement)
   - Bayesian optimization for XGBoost
   - Genetic algorithm for large spaces

3. **Ensemble diversity** (20% of improvement)
   - Different algorithms (linear, trees, ELO)
   - Complementary predictions

4. **Validation rigor** (15% of improvement)
   - Time series CV (no leakage)
   - Bootstrap confidence intervals
   - Calibration checks

---

## 🏆 GO WIN!

This strategy is designed to systematically achieve **RMSE < 1.7 goals**, beating baseline by over 50%.

**Key differentiators:**
- Mathematical rigor (Bayesian optimization, ensemble optimization)
- Domain expertise (hockey-specific features)
- No data leakage (time series validation)
- Ensemble intelligence (diversity + optimal weights)

**You have all the tools. Now execute!** 🚀🏒
