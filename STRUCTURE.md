# CSVy - Project Structure

```
CSVy/
├── cli.rb                    # Main CLI entry point
├── README.md                 # Project overview
├── TODO.md                   # Competition tasks
├── KNOWN_ISSUES.md           # Known issues tracker
├── Gemfile                   # Ruby dependencies
├── Rakefile                  # Build tasks
│
├── lib/                      # Ruby modules (core business logic)
│   ├── advanced_features.rb      # Advanced feature engineering
│   ├── stacked_ensemble.rb       # Meta-learning ensemble
│   ├── neural_network_wrapper.rb # TensorFlow/Keras integration
│   ├── csv_cleaner.rb
│   ├── csv_diagnostics.rb
│   ├── csv_io_handler.rb
│   ├── csv_merger.rb
│   ├── csv_processor.rb
│   ├── data_preprocessor.rb
│   ├── data_validator.rb
│   ├── database_manager.rb
│   ├── dataframe_handler.rb
│   ├── ensemble_builder.rb
│   ├── html_reporter.rb
│   ├── hyperparameter_manager.rb
│   ├── model_tracker.rb
│   ├── model_validator.rb
│   └── time_series_features.rb
│
├── python/                   # Python modules and notebooks
│   ├── __init__.py               # Python package init
│   ├── advanced_features.py      # Python feature engineering
│   ├── stacked_ensemble.py       # Python stacking implementation
│   ├── README.md                 # Python setup guide
│   ├── requirements.txt          # Python dependencies
│   └── notebooks/                # Jupyter notebooks
│       ├── 01_advanced_features_demo.ipynb
│       ├── 02_stacked_ensemble_demo.ipynb
│       ├── elo_model.ipynb
│       └── train_elo.ipynb
│
├── scripts/                  # Ruby/Python integration scripts
│   ├── competitive_pipeline.rb   # Full training pipeline
│   ├── train_neural_network.py   # NN training (called by Ruby)
│   ├── predict_nn.py             # NN predictions (called by Ruby)
│   ├── preprocess_hockey.sh      # Data preprocessing
│   └── nn_integration.md         # Integration docs
│
├── config/                   # Configuration files
│   └── hyperparams/          # Model hyperparameter configs
│       ├── model1_baseline.yaml
│       ├── model2_linear_regression.yaml
│       ├── model3_elo.yaml
│       ├── model4_random_forest.yaml
│       ├── model4_xgboost.yaml
│       ├── model5_ensemble.yaml
│       └── model6_neural_network.yaml
│
├── data/                     # Sample datasets
│   ├── sample_advanced.csv
│   ├── sample_employees.csv
│   ├── sample_nhl_standings.csv
│   ├── sample_nhl_standings_report.html
│   ├── sample_products.csv
│   ├── sample_students_dirty.csv
│   ├── sample_weather.csv
│   └── test_fix.csv
│
├── docs/                     # Documentation
│   ├── guides/               # User guides
│   │   ├── QUICK_START.md
│   │   ├── USAGE_GUIDE.md
│   │   ├── FEATURES_GUIDE.md
│   │   ├── NEURAL_NETWORK_GUIDE.md
│   │   ├── CALCULATIONS.md
│   │   ├── FEATURES.md
│   │   ├── QUICK_REFERENCE.md
│   │   ├── WINNING_STRATEGY.md
│   │   └── ELO_IMPLEMENTATION.md
│   ├── api/                  # API documentation (future)
│   ├── ADVANCED_FEATURES_GUIDE.md
│   ├── DATASET_FEATURES.md
│   ├── FEATURE_ENGINEERING_GUIDE.md
│   └── HOCKEY_FEATURES.md
│
├── spec/                     # RSpec tests
│   ├── spec_helper.rb
│   ├── csv_cleaner_spec.rb
│   ├── csv_merger_spec.rb
│   ├── csv_processor_spec.rb
│   ├── data_preprocessor_spec.rb
│   └── data_validator_spec.rb
│
├── experiments/              # Experiment tracking
│   └── *.csv                 # Model comparison results
│
└── .github/                  # GitHub Actions CI/CD
    └── workflows/
```

## Key Directories

### `/lib` - Ruby Core Modules
- Business logic implementation
- Called directly by `cli.rb`
- Pure Ruby, no external process calls

### `/python` - Python Package
- Standalone Python modules
- Jupyter notebooks for exploration
- Can be used independently or via Ruby wrapper

### `/scripts` - Integration Layer
- Scripts that bridge Ruby ↔ Python
- Called by Ruby via subprocess
- Training/prediction pipelines

### `/config` - Configuration
- YAML configs for hyperparameters
- Model settings and search spaces

### `/docs` - Documentation
- `/guides` - User-facing tutorials
- `/api` - API reference (future)
- Root level - Technical specs

## File Organization Rules

### Ruby Files (`.rb`)
- **Core logic**: `/lib/*.rb`
- **Tests**: `/spec/*_spec.rb`
- **Scripts**: `/scripts/*.rb` (if they call Python)

### Python Files (`.py`)
- **Modules**: `/python/*.py` (importable)
- **Integration scripts**: `/scripts/*.py` (called by Ruby CLI)

### Notebooks (`.ipynb`)
- All in `/python/notebooks/`
- Numbered for workflow order

### Documentation (`.md`)
- **Guides**: `/docs/guides/`
- **Technical**: `/docs/`
- **Root**: Only README, STRUCTURE, TODO, KNOWN_ISSUES

## Usage Examples

```bash
# Ruby CLI (uses /lib and calls /scripts/python)
ruby cli.rb clean data/input.csv

# Python modules (importable)
python -c "from python.advanced_features import AdvancedFeatures"

# Notebooks (in /python/notebooks)
jupyter notebook python/notebooks/01_advanced_features_demo.ipynb
```
│   │   ├── QUICK_REFERENCE.md
│   │   ├── USAGE_GUIDE.md
│   │   ├── WINNING_STRATEGY.md
│   │   ├── CALCULATIONS.md
│   │   └── FEATURES.md
│   └── HOCKEY_FEATURES.md    # Hockey-specific features
│
├── output/                   # Generated files (gitignored)
│   ├── hyperparams/          # Generated hyperparameter CSVs
│   ├── reports/              # HTML tracking reports
│   └── predictions/          # Model predictions
│
├── experiments/              # Experiment tracking
│   └── elo_random.csv
│
├── scripts/                  # Utility scripts
│   ├── competitive_pipeline.rb
│   └── preprocess_hockey.sh
│
└── spec/                     # RSpec tests
    ├── spec_helper.rb
    ├── csv_cleaner_spec.rb
    ├── csv_merger_spec.rb
    ├── csv_processor_spec.rb
    ├── data_preprocessor_spec.rb
    └── data_validator_spec.rb
```

## File Organization

### Root Level
- **cli.rb**: Main command-line interface
- **README.md**: Project documentation
- **TODO.md**: Competition task list

### lib/
Core Ruby modules for data processing, feature engineering, and model management.

### config/hyperparams/
YAML files defining hyperparameter search spaces for all 5 models.

### data/
Sample datasets for testing and development.

### docs/
All documentation consolidated here:
- **guides/**: User guides and strategies
- **HOCKEY_FEATURES.md**: Feature engineering documentation

### output/
Generated files organized by type:
- **hyperparams/**: CSV files with hyperparameter configurations
- **reports/**: HTML tracking dashboards
- **predictions/**: Model prediction outputs

### experiments/
Experiment tracking and results.

### scripts/
Automation scripts for data pipelines.

### spec/
RSpec test files.
