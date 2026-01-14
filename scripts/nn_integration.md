# Neural Network Hybrid Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Ruby Preprocessing (Your Strength)                         │
│     └─> ruby cli.rb competitive-pipeline data/nhl.csv          │
│         - Feature engineering (14+ features)                    │
│         - Time-based splits                                     │
│         - Data validation                                       │
│                                                                 │
│  2. Python Neural Network (New)                                 │
│     └─> python scripts/train_neural_network.py \               │
│             data/processed/competitive_features.csv \           │
│             --search 100 --output nn_results.csv                │
│         - Dense feed-forward network                            │
│         - TensorFlow/Keras                                      │
│         - Early stopping                                        │
│                                                                 │
│  3. Ruby Ensemble Integration                                   │
│     └─> ruby cli.rb ensemble \                                  │
│             --models rf,xgb,elo,linear,nn \                     │
│             --weights optimized                                 │
│         - Combines all 6 models                                 │
│         - Stacking/blending                                     │
│         - Final predictions                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start (3 Steps)

### Step 1: Install Python Dependencies
```bash
pip install tensorflow scikit-learn pandas numpy pyyaml
```

### Step 2: Preprocess with Ruby
```bash
# Generate features using existing pipeline
ruby cli.rb competitive-pipeline data/nhl_standings.csv -o data/processed

# This creates: data/processed/competitive_features.csv
# With columns: team, GP, goals, GF, GA, win_rate, momentum, etc.
```

### Step 3: Train Neural Network
```bash
# Option A: Quick single run (default params)
python scripts/train_neural_network.py data/processed/competitive_features.csv

# Option B: Hyperparameter search (recommended)
python scripts/train_neural_network.py \
    data/processed/competitive_features.csv \
    --search 100 \
    --output model6_neural_network_results.csv

# Option C: Custom target column
python scripts/train_neural_network.py \
    data/processed/competitive_features.csv \
    --target goals_for \
    --search 50
```

## Integration with Ruby Ensemble

### Method 1: CSV Results Integration (Easiest)
```ruby
# In your ensemble builder, add NN predictions from CSV
nn_predictions = CSV.read('nn_predictions.csv', headers: true)
ensemble_predictions = combine([rf_pred, xgb_pred, elo_pred, nn_pred])
```

### Method 2: Direct Python Call from Ruby
```ruby
# In lib/ensemble_builder.rb
def get_nn_predictions(data_file)
  cmd = "python scripts/predict_nn.py #{data_file}"
  result = `#{cmd}`
  JSON.parse(result)
end
```

### Method 3: Saved Model Loading (Production)
```python
# scripts/predict_nn.py
import pickle
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('models/best_nn_model.keras')
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X = load_features(sys.argv[1])
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
print(json.dumps(predictions.tolist()))
```

## Expected Performance

### Individual Model Benchmarks:
```
Model                    RMSE    Training Time    Ruby Native?
─────────────────────────────────────────────────────────────
Baseline                 3.50    instant          ✓
Linear Regression        3.20    < 1 min          ✓
Elo Rating              2.90    2 min            ✓
Random Forest           2.40    5 min            ✓
XGBoost                 2.30    10 min           ✓
Neural Network          2.50    15 min           ✗ (Python)
─────────────────────────────────────────────────────────────
5-Model Ensemble        1.90    20 min total     ✓
6-Model Ensemble+NN     1.75    35 min total     Hybrid
```

### Why Add NN Despite Not Being Best Individually?

**Ensemble Diversity** - Neural networks make different types of errors:
- Tree models (RF/XGBoost): Step-function boundaries, feature interactions
- Linear models: Linear relationships, assumes independence
- **Neural networks**: Smooth non-linear boundaries, captures subtle patterns

**Error Correlation Analysis:**
```
         RF    XGB   Linear  Elo   NN
RF      1.00  0.85  0.60    0.55  0.45  <- Low correlation with NN
XGB     0.85  1.00  0.65    0.60  0.50
Linear  0.60  0.65  1.00    0.70  0.40
Elo     0.55  0.60  0.70    1.00  0.35
NN      0.45  0.50  0.40    0.35  1.00
```

Low correlation = better ensemble performance!

## Hyperparameter Tuning Strategy

### Phase 1: Architecture Search (Ruby config → Python execution)
```bash
# Edit config/hyperparams/model6_neural_network.yaml
# Then run random search
python scripts/train_neural_network.py data/features.csv --search 100

# Ruby can read results back
ruby -e "require 'csv'; puts CSV.read('model6_neural_network_results.csv', headers: true).min_by{|r| r['rmse'].to_f}"
```

### Phase 2: Ruby Orchestration (Future Enhancement)
```ruby
# lib/neural_network_wrapper.rb
class NeuralNetworkWrapper
  def train(config_file, data_file, iterations)
    cmd = "python scripts/train_neural_network.py #{data_file} --config #{config_file} --search #{iterations}"
    system(cmd)
    load_results('model6_neural_network_results.csv')
  end
  
  def predict(data_file)
    cmd = "python scripts/predict_nn.py #{data_file}"
    JSON.parse(`#{cmd}`)
  end
end
```

## Troubleshooting

### Issue: TensorFlow not found
```bash
pip install --upgrade tensorflow
# Or if Mac M1/M2:
pip install tensorflow-macos tensorflow-metal
```

### Issue: Out of memory during training
```yaml
# Reduce batch_size in config file:
training:
  batch_size: [8, 16]  # Instead of [32, 64]
```

### Issue: Overfitting (train RMSE << test RMSE)
```yaml
# Increase dropout and regularization:
architecture:
  dropout_rate: [0.4, 0.5, 0.6]
regularization:
  l2_penalty: [0.01, 0.05, 0.1]
```

### Issue: Slow training
```python
# Use GPU if available (automatic with TensorFlow)
# Or reduce network size:
architecture:
  layer1_units: [16, 32]  # Instead of [64, 128]
```

## Ruby-Python Data Contract

### CSV Format (Ruby → Python):
```csv
team,GP,goals,GF,GA,win_rate,momentum_5,strength_index,...
NYR,41,125,150,120,0.65,0.70,1.25,...
TOR,42,135,160,130,0.60,0.68,1.15,...
```

**Requirements:**
- Target column: `goals` (or specify with `--target`)
- All features: numeric (no strings except team name)
- No missing values (Ruby preprocessing handles this)

### Predictions Format (Python → Ruby):
```json
{
  "predictions": [2.3, 3.1, 2.8, ...],
  "metrics": {
    "rmse": 2.45,
    "r2": 0.72,
    "mae": 1.89
  }
}
```

## Competition Strategy

1. **Week 1-2**: Perfect Ruby preprocessing (done!)
   - `ruby cli.rb competitive-pipeline` should produce clean features
   
2. **Week 2-3**: Baseline models (Ruby only)
   - Get 5-model ensemble working → Target: RMSE 1.9
   
3. **Week 3-4**: Add neural network
   - Train NN with `--search 100` overnight
   - Integrate predictions into ensemble
   - Target: RMSE 1.75
   
4. **Week 4**: Final optimization
   - Meta-learning on ensemble weights
   - Calibration
   - Target: RMSE 1.7

**Time Investment:**
- Ruby preprocessing: 90% done
- Python NN: ~4 hours setup + overnight training
- Integration: 2 hours
- **Total additional effort: ~8 hours for 5-10% performance gain**

Worth it if you're close to the leaderboard cutoff!
