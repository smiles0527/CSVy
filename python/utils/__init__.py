"""
CSVy Python Utilities

Reusable modules for hockey prediction models.
"""

from .elo_model import EloModel
from .baseline_model import (
    BaselineModel,
    GlobalMeanBaseline,
    TeamMeanBaseline,
    HomeAwayBaseline,
    MovingAverageBaseline,
    WeightedHistoryBaseline,
    PoissonBaseline,
    compare_baselines
)
from .linear_model import (
    LinearRegressionModel,
    LinearGoalPredictor,
    grid_search_linear,
    random_search_linear,
    compare_regularization,
    create_polynomial_features,
)
from .xgboost_model import (
    XGBoostModel,
    XGBoostGoalPredictor,
    grid_search_xgboost,
    random_search_xgboost,
)

__all__ = [
    # ELO Model (Model 2)
    'EloModel',
    
    # Baseline Models (Model 1)
    'BaselineModel',
    'GlobalMeanBaseline',
    'TeamMeanBaseline',
    'HomeAwayBaseline',
    'MovingAverageBaseline',
    'WeightedHistoryBaseline',
    'PoissonBaseline',
    'compare_baselines',
    
    # Linear Regression Model (Model 3)
    'LinearRegressionModel',
    'LinearGoalPredictor',
    'grid_search_linear',
    'random_search_linear',
    'compare_regularization',
    'create_polynomial_features',
    
    # XGBoost Model (Model 4)
    'XGBoostModel',
    'XGBoostGoalPredictor',
    'grid_search_xgboost',
    'random_search_xgboost',
]
