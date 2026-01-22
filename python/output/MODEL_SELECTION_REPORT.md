# Model Selection Report

**Generated:** 2026-01-21 14:44:03

## Summary

**Best Model:** Linear
**Test R2 Score:** 0.1348

## All Models Comparison

| Model   |   Train R2 |   Test R2 |   Train RMSE |   Test RMSE |   Train MAE |   Test MAE |   Overfit Gap |
|:--------|-----------:|----------:|-------------:|------------:|------------:|-----------:|--------------:|
| Linear  |   0.139029 |  0.134763 |      2.07821 |     2.38466 |     1.65289 |    1.83932 |    0.00426611 |

## Rankings

1. **Linear** - R2 = 0.1348

## Production Deployment

 The **Linear** model is ready for production.

 Model file: `models/production_model.pkl`

### Usage Example
```python
import pickle

# Load model
with open('models/production_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_new)
```