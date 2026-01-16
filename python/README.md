# Python ELO Model - DeepNote Workflow

## Folder Structure

```
python/
├── models/           # Model definition notebooks
│   └── elo_model.ipynb
├── training/         # Training loop notebooks
│   └── train_elo.ipynb
├── validation/       # Validation test notebooks
│   └── validate_elo.ipynb
├── utils/            # Reusable Python modules
│   ├── __init__.py
│   └── elo_model.py  # EloModel class for importing
├── data/             # Data files (gitignored except .gitkeep)
├── README.md
└── requirements.txt
```

## Setup

1. **Upload to DeepNote:**
   - Upload entire `python/` folder to DeepNote
   - Maintain folder structure for imports to work

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Upload data:**
   - Upload `output/hyperparams/model3_elo_grid.csv` (from Ruby)
   - Upload your hockey dataset (`data/hockey_data.csv`)

## Workflow

### Step 1: Generate Hyperparameters (Ruby - Local)
```bash

ruby cli.rb hyperparam-grid config/hyperparams/model3_elo.yaml -o output/hyperparams/model3_elo_grid.csv
```

### Step 2: Train ELO Model (Python - DeepNote)

**Open notebooks in order:**

1. **`models/elo_model.ipynb`** - ELO implementation class
   - Run all cells to define the `EloModel` class
   - No data needed, just the model code

2. **`training/train_elo.ipynb`** - Hyperparameter grid search
   - Loads 648 configs from Ruby
   - Trains ELO for each config
   - Outputs: `model3_elo_results.csv`
   - **Estimated runtime:** 15-30 minutes for 648 configs

3. **`validation/validate_elo.ipynb`** - Validation tests
   - Run after training to validate model behavior
   - All 4 tests should pass

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
