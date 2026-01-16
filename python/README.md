# CSVy Python Package

Python modules and notebooks for machine learning workflows.

## Structure

```
python/
├── __init__.py                  # Package initialization
├── advanced_features.py         # Feature engineering module
├── stacked_ensemble.py          # Meta-learning ensemble module
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── notebooks/                   # Jupyter notebooks
    ├── 01_advanced_features_demo.ipynb
    ├── 02_stacked_ensemble_demo.ipynb
    ├── elo_model.ipynb
    └── train_elo.ipynb
```

## Setup

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with conda
conda create -n csvy python=3.9
conda activate csvy
pip install -r requirements.txt
```

### DeepNote/Cloud Setup

1. Upload all files to your notebook environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### As Python Modules

```python
# Feature engineering
from python.advanced_features import AdvancedFeatures
import pandas as pd

df = pd.read_csv('data.csv')
af = AdvancedFeatures(df)
enhanced_df = af.add_all_features()

# Stacked ensemble
from python.stacked_ensemble import StackedEnsemble

stacker = StackedEnsemble(meta_model_type='ridge')
predictions_df, model_names = stacker.load_base_predictions('predictions/')
results = stacker.train(predictions_df, y_true)
```

### Via Ruby CLI

The Ruby CLI automatically calls these modules:

```bash
# Feature engineering
ruby cli.rb add-features data.csv -o enhanced.csv

# Stacked ensemble training
ruby cli.rb train-stacked-ensemble predictions/ actuals.csv
```

## Notebooks

### 01_advanced_features_demo.ipynb
Demonstrates the 9 advanced features for hockey prediction with visualizations.

### 02_stacked_ensemble_demo.ipynb
Shows meta-learning approach combining 6 base models.

### elo_model.ipynb
ELO rating system implementation for hockey.

### train_elo.ipynb  
Full ELO training workflow with hyperparameter search.

## Workflow Integration

### Ruby → Python Flow

1. Ruby CLI (`cli.rb`) handles user interface
2. Ruby modules (`lib/*.rb`) process data
3. Ruby calls Python scripts (`../scripts/*.py`) for ML tasks
4. Python modules (this folder) provide standalone functionality

### Example: Neural Network Training

```bash
# User command
ruby cli.rb train-neural-network data.csv --search 100

# What happens:
# 1. Ruby: lib/neural_network_wrapper.rb prepares data
# 2. Ruby calls: scripts/train_neural_network.py (subprocess)
# 3. Python trains TensorFlow model
# 4. Ruby: collects results and displays to user
```

## Dependencies

See `requirements.txt` for full list:

- **Core**: pandas, numpy
- **ML**: scikit-learn, tensorflow, xgboost
- **Visualization**: matplotlib, seaborn
- **Utilities**: pyyaml

## Development

### Running Tests

```python
# Run standalone module
python advanced_features.py

# Run stacked ensemble demo
python stacked_ensemble.py
```

### Creating New Modules

1. Add `.py` file to `python/` directory
2. Import in `__init__.py` if needed
3. Create demo notebook in `python/notebooks/`
4. Update this README

## DeepNote Workflow (Cloud Development)

### Step 1: Generate Hyperparameters (Ruby - Local)
```bash
# In your CSVy repo
ruby cli.rb hyperparam-grid config/hyperparams/model3_elo.yaml -o output/hyperparams/model3_elo_grid.csv
```

### Step 2: Train ELO Model (Python - DeepNote)

**Open notebooks in order:**

1. **`elo_model.ipynb`** - ELO implementation class
   - Run all cells to define the `EloModel` class
   - No data needed, just the model code

2. **`train_elo.ipynb`** - Hyperparameter grid search
   - Loads 648 configs from Ruby
   - Trains ELO for each config
   - Outputs: `model3_elo_results.csv`
   - **Estimated runtime:** 15-30 minutes for 648 configs

3. **Results:**
   - Best configuration saved
   - Metrics tracked: RMSE, MAE, R²
   - Visualizations generated

### Step 3: Download Results (DeepNote → Local)
- Download `output/hyperparams/model3_elo_results.csv`
- Move to your local CSVy repo

### Step 4: Generate HTML Report (Ruby - Local)
```bash
ruby cli.rb report output/hyperparams/model3_elo_results.csv --open
```

## File Structure

```
python/
├── elo_model.ipynb          # ELO model class definition
├── train_elo.ipynb          # Training loop + grid search
├── requirements.txt         # Python dependencies
└── README.md                # This file

Data flow:
  Ruby (local) → model3_elo_grid.csv → DeepNote
  DeepNote → model3_elo_results.csv → Ruby (local)
  Ruby → HTML report
```

## Expected Performance

With proper hyperparameter tuning and the built-in features (rest_time, travel_distance, injuries):

- **RMSE:** 2.0-2.5 goals per game
- **R²:** 0.70-0.80
- **MAE:** 1.5-2.0 goals

## Key Features Used

- `rest_time` → rest_advantage_per_day, b2b_penalty
- `travel_distance` → travel fatigue (15 pts / 1000 miles)
- `injuries` → injury penalty (25 pts / player)
- `division` → initial ratings (D1=1600, D2=1500, D3=1400)
- `home_advantage` → home ice boost (50-150 pts)

## Tips for DeepNote

1. **Enable GPU** (optional, but speeds up pandas operations)
2. **Use progress bars:** `from tqdm import tqdm` for long loops
3. **Save checkpoints:** Save results every 100 configs in case of crashes
4. **Monitor memory:** 648 configs × full dataset can be memory-intensive

## Troubleshooting

**Problem:** Out of memory  
**Solution:** Process configs in batches of 100, save results incrementally

**Problem:** Taking too long  
**Solution:** Sample 10% of data for quick testing, then run full dataset overnight

**Problem:** Missing columns in data  
**Solution:** Check column names match: `home_team`, `away_team`, `home_goals`, `away_goals`, `home_rest`, `away_rest`, etc.
