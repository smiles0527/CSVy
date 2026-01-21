# Model Explainability & Debugging Guide

Complete guide to understanding your model predictions and debugging issues with SHAP and advanced analysis tools.

---

## 🔍 SHAP Explainability

### What is SHAP?

SHAP (SHapley Additive exPlanations) explains model predictions by calculating each feature's contribution to the final prediction. It answers: "Why did the model predict X for this game?"

### Generate SHAP Report

```bash
# Basic usage
ruby cli.rb explain-model models/xgboost_model.pkl data/test.csv

# Specify output directory
ruby cli.rb explain-model models/xgboost_model.pkl data/test.csv -o explanations/xgboost/

# Different model types
ruby cli.rb explain-model models/random_forest.pkl data/test.csv --model-type random_forest
ruby cli.rb explain-model models/linear.pkl data/test.csv --model-type linear

# Show top 30 features instead of default 20
ruby cli.rb explain-model models/xgboost_model.pkl data/test.csv --top-n 30
```

### Output Files

1. **shap_summary.png** - Beeswarm plot showing feature impacts
   - Each dot = one prediction
   - Red = high feature value, Blue = low feature value
   - X-axis = SHAP value (impact on prediction)

2. **shap_importance.png** - Bar chart of mean absolute SHAP values
   - Shows which features matter most on average

3. **shap_values.csv** - Raw SHAP values for all predictions
   - Use for custom analysis or ensemble building

4. **feature_importance.csv** - Ranked feature list with importance scores

5. **dependence/** - Individual plots showing feature-prediction relationships
   - One plot per top feature
   - Shows how feature values affect predictions

6. **shap_report.html** - Interactive report with all visualizations
   - Opens automatically in browser
   - Shareable with team

---

## 🔧 Debugging Tools

### 1. Error Analysis

Analyzes where your model makes mistakes and identifies patterns.

```bash
# Basic error analysis
ruby cli.rb debug-errors predictions.csv actuals.csv features.csv

# Custom output path
ruby cli.rb debug-errors predictions.csv actuals.csv features.csv -o my_error_report.csv
```

**What it finds:**
- Overall statistics (MAE, RMSE, mean error)
- Error distribution (excellent, good, fair, poor, very poor)
- Systematic bias (overestimation vs underestimation)
- Worst predictions (top 20 errors)
- Error patterns by feature ranges

**Example output:**
```
✓ Error Analysis Complete

📊 Overall Statistics:
  MAE: 0.1234
  RMSE: 0.1567
  Mean Error: -0.0234
  Max Error: 2.3456

📈 Error Distribution:
  excellent: 120 (24.0%)
  good: 250 (50.0%)
  fair: 100 (20.0%)
  poor: 25 (5.0%)
  very_poor: 5 (1.0%)

⚠️  Systematic Bias:
  Overall: -0.0234
  Overestimation: 45%
  Underestimation: 55%
```

### 2. Feature Debugging

Checks data quality and detects potential problems.

```bash
# Basic feature debugging
ruby cli.rb debug-features data/train.csv

# Custom outlier threshold (default: 3 standard deviations)
ruby cli.rb debug-features data/train.csv --threshold 4.0

# Custom output directory
ruby cli.rb debug-features data/train.csv -o feature_analysis/
```

**What it detects:**

1. **Missing Values**
   - Features with NULL/empty values
   - Percentage missing per feature

2. **Constant Features**
   - Features with only one unique value
   - Should be removed (no predictive power)

3. **Outliers**
   - Values beyond N standard deviations from mean
   - May indicate data quality issues or rare events

4. **High Correlations**
   - Features with |correlation| > 0.9
   - Multicollinearity can hurt model performance

5. **Feature Quality Scores**
   - Based on completeness + uniqueness
   - Helps prioritize which features to keep

**Example output:**
```
✓ Feature Debug Complete

⚠️  Missing Values Detected:
  goalie_starter: 45 (9.0%)
  special_teams_pct: 12 (2.4%)

✓ No constant features

⚠️  Outliers Detected:
  travel_distance: 15 (3.0%)
  injuries: 8 (1.6%)

⚠️  High Correlations (|r| > 0.9):
  PTS ↔ P%: 0.9876
  GF ↔ offense_power: 0.9234

📊 Top Quality Features:
  DIFF: 98.5/100
  PTS: 97.2/100
  P%: 96.8/100
```

### 3. Single Prediction Explanation

Explain one specific prediction in detail.

```bash
# Explain a prediction
ruby cli.rb explain-prediction models/xgboost.pkl \
  --features GF:250 GA:180 DIFF:70 rest_time:2 injuries:1 travel_distance:500 \
  -o single_game_explanation
```

**Output:**
```
✓ Prediction: 0.7234
  Base value: 0.5000

📈 Top Positive Contributors:
  DIFF: +0.1500
  GF: +0.0800
  rest_time: +0.0234

📉 Top Negative Contributors:
  travel_distance: -0.0200
  injuries: -0.0100

💾 Full details saved to: single_game_explanation.json
```

---

## 🎯 Use Cases

### Use Case 1: Model Not Performing Well

```bash
# 1. Check feature quality
ruby cli.rb debug-features data/train.csv -o debug/

# Look for:
# - Missing values > 5%
# - Constant features
# - High correlations (remove one)

# 2. Analyze errors
ruby cli.rb debug-errors predictions.csv actuals.csv features.csv -o errors/

# Look for:
# - Systematic bias (mean error != 0)
# - High error rate in specific ranges
# - Worst predictions have patterns

# 3. Explain predictions
ruby cli.rb explain-model models/model.pkl data/test.csv -o explanations/

# Check if important features make sense
```

### Use Case 2: Understand What Model Learned

```bash
# Generate SHAP report
ruby cli.rb explain-model models/xgboost.pkl data/test.csv

# Open shap_report.html
# Look at:
# - Top 10 features (should match domain knowledge)
# - Dependence plots (relationships make sense?)
# - Feature importance percentages
```

### Use Case 3: Debug Specific Bad Predictions

```bash
# 1. Find worst errors
ruby cli.rb debug-errors predictions.csv actuals.csv features.csv

# Note the indices of worst predictions

# 2. Extract those rows from features.csv
# (Use Excel or Python to filter)

# 3. Explain each one
ruby cli.rb explain-prediction models/model.pkl \
  --features [copy values from row]
```

### Use Case 4: Compare Multiple Models

```bash
# Generate SHAP for each model
ruby cli.rb explain-model models/xgboost.pkl data/test.csv -o exp/xgboost/
ruby cli.rb explain-model models/random_forest.pkl data/test.csv -o exp/rf/
ruby cli.rb explain-model models/linear.pkl data/test.csv -o exp/linear/

# Compare feature_importance.csv files
# - Which features are consistently important?
# - Where do models disagree? (good for ensembles)
```

---

## 📊 Interpreting SHAP Plots

### Summary Plot (Beeswarm)

```
              Low ← Feature Value → High
Feature 1    ●●●●●●●●●●●●●●●●●●●●●●●●●
             Blue        Red
Feature 2    ●●●●●●●●●●●●●●●●●●●●●●●●●
Feature 3    ●●●●●●●●●●●●●●●●●●●●●●●●●
         -0.3  -0.2  -0.1   0   0.1  0.2  0.3
              SHAP Value (impact on output)
```

- **X-axis**: How much the feature changed the prediction
- **Color**: Actual value of the feature
  - Red = high value
  - Blue = low value
- **Pattern examples**:
  - Red dots on right, blue on left = "Higher feature → higher prediction"
  - Red dots on left, blue on right = "Higher feature → lower prediction"
  - Mixed colors = Non-linear relationship or interactions

### Importance Plot (Bar Chart)

```
Feature 1  ████████████████████ 0.45
Feature 2  ███████████████ 0.32
Feature 3  ██████████ 0.21
Feature 4  ████ 0.12
```

- Length = Average absolute SHAP value
- Shows which features matter most overall
- Doesn't show direction (positive/negative)

### Dependence Plot

```
SHAP value
    ↑
0.3 |         ●●●
0.2 |       ●●●●●
0.1 |     ●●●●●●●
  0 |●●●●●●●●●●●●●
-0.1|●●●●●●
    └────────────→ Feature Value
```

- Shows exact relationship between feature and prediction
- Scatter plot: each point = one prediction
- Color = value of another interacting feature
- Non-linear relationships are visible

---

## 🚨 Common Issues & Solutions

### Error: "SHAP analysis failed"

**Solution:**
```bash
# Install Python dependencies
pip install shap pandas numpy matplotlib scikit-learn xgboost joblib

# Or use requirements.txt
cd python/
pip install -r requirements.txt
```

### Error: "Model file not found"

**Solution:**
- Ensure model file exists: `ls models/`
- Use correct extension: `.pkl`, `.joblib`, `.model`
- Provide absolute path if needed: `/full/path/to/model.pkl`

### Error: "Feature mismatch"

**Problem:** Model trained on different features than provided data

**Solution:**
```bash
# Check model training features
# Re-export test data with same preprocessing as training
ruby cli.rb competitive-pipeline data/raw.csv -o data/processed/
```

### SHAP values seem wrong

**Problem:** Using wrong model type

**Solution:**
```bash
# Specify correct model type
ruby cli.rb explain-model models/rf.pkl data/test.csv --model-type random_forest

# Supported types:
# - xgboost (default)
# - lightgbm
# - catboost
# - random_forest
# - linear
```

### Too slow for large datasets

**Solution:**
```bash
# Sample your data first
head -n 1000 data/test.csv > data/test_sample.csv

# Or use Kernel SHAP (samples automatically)
# Explain subset of predictions, then analyze patterns
```

---

## 🎓 Best Practices

1. **Always debug features before training**
   ```bash
   ruby cli.rb debug-features data/train.csv
   # Fix issues, then train
   ```

2. **Check for systematic bias**
   ```bash
   ruby cli.rb debug-errors predictions.csv actuals.csv features.csv
   # If mean_error != 0, model is biased
   ```

3. **Validate feature importance makes sense**
   - Top features should match domain knowledge
   - If `random_noise` is top feature → overfitting

4. **Look for complementary models**
   - Compare SHAP importance across models
   - Models with different top features → good ensemble candidates

5. **Document insights**
   - Save HTML reports for each model version
   - Track which features helped most over time

---

## 📚 Further Reading

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Interpreting Machine Learning Models](https://christophm.github.io/interpretable-ml-book/)
- [Feature Engineering Guide](./FEATURE_ENGINEERING_GUIDE.md)
- [Ensemble Building Guide](./WINNING_STRATEGY.md)

---

## 💡 Quick Reference

```bash
# Generate SHAP explainability report
ruby cli.rb explain-model MODEL DATA [options]

# Analyze prediction errors
ruby cli.rb debug-errors PREDICTIONS ACTUALS FEATURES

# Debug feature quality
ruby cli.rb debug-features DATA

# Explain single prediction
ruby cli.rb explain-prediction MODEL --features key=val...
```

**All commands generate:**
- CSV files for further analysis
- HTML reports with visualizations
- Auto-open in browser

---

**Need help?** Check [USAGE_GUIDE.md](./USAGE_GUIDE.md) for more examples.
