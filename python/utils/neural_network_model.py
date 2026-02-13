"""
Neural Network Model - Production-Ready Hockey Goal Prediction

This module provides neural network models for regression using sklearn's
MLPRegressor. Includes proper feature scaling, architecture configuration,
and hyperparameter search utilities.

CRITICAL: Neural networks REQUIRE feature scaling for proper training.
This module handles scaling automatically.

Classes:
    - NeuralNetworkModel: MLP-based regression model with auto-scaling
    - NeuralNetworkGoalPredictor: Dual NN for home/away goal prediction

Functions:
    - grid_search_nn: Grid search for neural network hyperparameters
    - random_search_nn: Random search for neural network hyperparameters
    - create_architecture: Helper to create hidden layer configurations

Usage:
    from utils.neural_network_model import NeuralNetworkModel, NeuralNetworkGoalPredictor
    
    # Basic usage
    model = NeuralNetworkModel(hidden_layer_sizes=(64, 32))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Dual prediction
    predictor = NeuralNetworkGoalPredictor()
    predictor.fit(games_df)
    home_pred, away_pred = predictor.predict_goals(new_games)
"""

import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

# Configure module logger
logger = logging.getLogger(__name__)


# Column name mappings - consistent with other models
COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_goals': ['home_goals', 'home_score', 'h_goals', 'goals_home', 'home_pts'],
    'away_goals': ['away_goals', 'away_score', 'a_goals', 'goals_away', 'away_pts', 'visitor_goals'],
}


def get_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """Find the correct column name in a DataFrame using aliases."""
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


# Default hyperparameters for hockey goal prediction
DEFAULT_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20,
    'random_state': 42,
    'batch_size': 'auto',
}


class NeuralNetworkModel:
    """
    Neural network model with automatic feature scaling.
    
    Wraps sklearn's MLPRegressor with built-in scaling to ensure proper
    training. Neural networks are sensitive to feature scales - this class
    handles that automatically.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(64, 32)
        Architecture: each element is the number of neurons in that layer.
        (64, 32) = 2 hidden layers with 64 and 32 neurons.
    activation : str, default='relu'
        Activation function: 'relu', 'tanh', 'logistic', 'identity'.
    solver : str, default='adam'
        Optimizer: 'adam', 'sgd', 'lbfgs'.
    alpha : float, default=0.001
        L2 regularization strength.
    learning_rate : str, default='adaptive'
        Learning rate schedule: 'constant', 'invscaling', 'adaptive'.
    learning_rate_init : float, default=0.001
        Initial learning rate.
    max_iter : int, default=500
        Maximum training epochs.
    early_stopping : bool, default=True
        Stop training when validation score stops improving.
    scaler_type : str, default='standard'
        Feature scaler: 'standard', 'robust', 'minmax'.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        Print training progress.
    **kwargs
        Additional parameters passed to MLPRegressor.
    
    Attributes
    ----------
    model : MLPRegressor
        The underlying neural network.
    scaler : StandardScaler or similar
        Feature scaler.
    is_fitted : bool
        Whether the model has been trained.
    training_history : dict
        Training information (loss curve, etc.).
    
    Examples
    --------
    >>> model = NeuralNetworkModel(hidden_layer_sizes=(128, 64, 32))
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.001,
        learning_rate: str = 'adaptive',
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = True,
        scaler_type: str = 'standard',
        random_state: int = 42,
        verbose: bool = False,
        **kwargs
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.verbose = verbose
        self.extra_params = kwargs
        
        # Create scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        # Create model
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=kwargs.get('validation_fraction', 0.1),
            n_iter_no_change=kwargs.get('n_iter_no_change', 20),
            random_state=random_state,
            verbose=verbose,
            batch_size=kwargs.get('batch_size', 'auto'),
        )
        
        self.is_fitted = False
        self.training_history = {}
        self.feature_names = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> 'NeuralNetworkModel':
        """
        Fit the neural network with automatic scaling.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features.
        y : pd.Series or np.ndarray
            Training target.
        sample_weight : np.ndarray, optional
            Sample weights (currently not supported by MLPRegressor).
        
        Returns
        -------
        self
            Fitted model.
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model (suppress convergence warnings, we use early stopping)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        self.training_history = {
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_iter': self.model.n_iter_,
            'best_loss': self.model.best_loss_ if hasattr(self.model, 'best_loss_') else None,
            'loss_curve': self.model.loss_curve_ if hasattr(self.model, 'loss_curve_') else None,
            'timestamp': datetime.now().isoformat(),
        }
        
        n_iter = self.model.n_iter_
        logger.info(f"Fitted neural network in {n_iter} iterations")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction.
        
        Returns
        -------
        np.ndarray
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns
        -------
        dict
            Dictionary with rmse, mae, r2, mse metrics.
        """
        predictions = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
        }
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation with proper scaling per fold (no data leakage).
        
        Uses a Pipeline(scaler, model) so the scaler is fit only on each
        training fold. Creates a fresh clone of the model for CV.
        
        Returns
        -------
        dict
            Cross-validation results with mean and std scores.
        """
        from sklearn.pipeline import Pipeline
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Build pipeline: scaler fits per-fold (no leakage), fresh model clone
        scaler_class = type(self.scaler)
        cv_pipeline = Pipeline([
            ('scaler', scaler_class()),
            ('model', clone(self.model)),
        ])
        
        cv_scores = cross_val_score(
            cv_pipeline, X, y,
            cv=cv,
            scoring='neg_mean_squared_error'
        )
        
        rmse_scores = np.sqrt(-cv_scores)
        
        return {
            'cv_rmse_scores': rmse_scores,
            'cv_rmse_mean': float(rmse_scores.mean()),
            'cv_rmse_std': float(rmse_scores.std()),
        }
    
    def get_loss_curve(self) -> Optional[List[float]]:
        """Get training loss curve for plotting."""
        if hasattr(self.model, 'loss_curve_'):
            return self.model.loss_curve_
        return None
    
    def get_architecture(self) -> Dict[str, Any]:
        """Get neural network architecture details."""
        return {
            'hidden_layers': len(self.hidden_layer_sizes),
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'total_neurons': sum(self.hidden_layer_sizes),
            'activation': self.activation,
            'n_features_in': getattr(self.model, 'n_features_in_', None),
            'n_outputs': getattr(self.model, 'n_outputs_', None),
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get all model parameters."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'scaler_type': self.scaler_type,
            'random_state': self.random_state,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model AND scaler to disk.
        
        IMPORTANT: Both the model and scaler must be saved together
        for predictions to work correctly.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved neural network to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NeuralNetworkModel':
        """Load model and scaler from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            hidden_layer_sizes=model_data['hidden_layer_sizes'],
            activation=model_data['activation'],
            solver=model_data['solver'],
            alpha=model_data['alpha'],
            learning_rate=model_data['learning_rate'],
            learning_rate_init=model_data['learning_rate_init'],
            max_iter=model_data['max_iter'],
            early_stopping=model_data['early_stopping'],
            scaler_type=model_data['scaler_type'],
            random_state=model_data['random_state'],
        )
        
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data.get('feature_names')
        instance.training_history = model_data.get('training_history', {})
        instance.is_fitted = True
        
        return instance


class NeuralNetworkGoalPredictor:
    """
    Dual neural network model for predicting both home and away goals.
    
    Uses separate neural networks for home and away goal prediction,
    each with its own scaling.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(64, 32)
        Architecture for both networks.
    feature_columns : list, optional
        Column names to use as features.
    scaler_type : str, default='standard'
        Feature scaler type.
    **kwargs
        Additional parameters passed to NeuralNetworkModel.
    
    Examples
    --------
    >>> predictor = NeuralNetworkGoalPredictor(hidden_layer_sizes=(128, 64))
    >>> predictor.fit(games_df)
    >>> home_pred, away_pred = predictor.predict_goals(new_games)
    """
    
    DEFAULT_FEATURES = [
        'home_elo', 'away_elo', 'elo_diff',
        'home_win_pct', 'away_win_pct',
        'home_goals_avg', 'away_goals_avg',
        'home_goals_against_avg', 'away_goals_against_avg',
    ]
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        feature_columns: Optional[List[str]] = None,
        scaler_type: str = 'standard',
        **kwargs
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feature_columns = feature_columns
        self.scaler_type = scaler_type
        self.model_params = kwargs
        
        self.home_model = None
        self.away_model = None
        self.is_fitted = False
        self.training_info = {}
    
    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns from dataframe."""
        if self.feature_columns:
            available = [c for c in self.feature_columns if c in df.columns]
            return df[available]
        
        # Auto-detect features
        available = [c for c in self.DEFAULT_FEATURES if c in df.columns]
        if available:
            return df[available]
        
        # Fallback: use all numeric columns except targets
        exclude = ['home_goals', 'away_goals', 'total_goals']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[[c for c in numeric_cols if c not in exclude]]
    
    def fit(
        self,
        df: pd.DataFrame,
        home_goals_col: str = 'home_goals',
        away_goals_col: str = 'away_goals'
    ) -> 'NeuralNetworkGoalPredictor':
        """
        Fit home and away goal prediction models.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with features and goal columns.
        home_goals_col : str, default='home_goals'
            Column name for home goals.
        away_goals_col : str, default='away_goals'
            Column name for away goals.
        
        Returns
        -------
        self
            Fitted predictor.
        """
        # Get features
        X = self._get_features(df)
        self.feature_columns = list(X.columns)
        
        # Get targets
        home_col = get_column(df, 'home_goals') or home_goals_col
        away_col = get_column(df, 'away_goals') or away_goals_col
        
        y_home = df[home_col]
        y_away = df[away_col]
        
        # Create models
        self.home_model = NeuralNetworkModel(
            hidden_layer_sizes=self.hidden_layer_sizes,
            scaler_type=self.scaler_type,
            **self.model_params
        )
        
        self.away_model = NeuralNetworkModel(
            hidden_layer_sizes=self.hidden_layer_sizes,
            scaler_type=self.scaler_type,
            **self.model_params
        )
        
        # Fit models
        self.home_model.fit(X, y_home)
        self.away_model.fit(X, y_away)
        
        self.is_fitted = True
        self.training_info = {
            'n_samples': len(df),
            'n_features': len(self.feature_columns),
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'home_iterations': self.home_model.model.n_iter_,
            'away_iterations': self.away_model.model.n_iter_,
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"Fitted dual neural network predictor")
        
        return self
    
    def predict_goals(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict home and away goals.
        
        Returns
        -------
        tuple
            (home_goals_predictions, away_goals_predictions)
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        X = self._get_features(df)
        
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        
        return home_pred, away_pred
    
    def predict_winner(self, df: pd.DataFrame) -> pd.Series:
        """Predict game winners."""
        home_pred, away_pred = self.predict_goals(df)
        
        results = []
        for h, a in zip(home_pred, away_pred):
            if h > a + 0.5:
                results.append('home')
            elif a > h + 0.5:
                results.append('away')
            else:
                results.append('tie')
        
        return pd.Series(results, index=df.index)
    
    def evaluate(
        self,
        df: pd.DataFrame,
        home_goals_col: str = 'home_goals',
        away_goals_col: str = 'away_goals'
    ) -> Dict[str, float]:
        """Evaluate prediction performance."""
        home_col = get_column(df, 'home_goals') or home_goals_col
        away_col = get_column(df, 'away_goals') or away_goals_col
        
        home_pred, away_pred = self.predict_goals(df)
        
        y_home = df[home_col].values
        y_away = df[away_col].values
        
        return {
            'home_rmse': np.sqrt(mean_squared_error(y_home, home_pred)),
            'home_mae': mean_absolute_error(y_home, home_pred),
            'away_rmse': np.sqrt(mean_squared_error(y_away, away_pred)),
            'away_mae': mean_absolute_error(y_away, away_pred),
            'combined_rmse': np.sqrt(mean_squared_error(
                np.concatenate([y_home, y_away]),
                np.concatenate([home_pred, away_pred])
            )),
        }
    
    def get_loss_curves(self) -> Dict[str, List[float]]:
        """Get training loss curves for both models."""
        return {
            'home': self.home_model.get_loss_curve(),
            'away': self.away_model.get_loss_curve(),
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save predictor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'home_model': self.home_model,
            'away_model': self.away_model,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'feature_columns': self.feature_columns,
            'scaler_type': self.scaler_type,
            'model_params': self.model_params,
            'training_info': self.training_info,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NeuralNetworkGoalPredictor':
        """Load predictor from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            hidden_layer_sizes=model_data['hidden_layer_sizes'],
            feature_columns=model_data['feature_columns'],
            scaler_type=model_data['scaler_type'],
            **model_data.get('model_params', {})
        )
        instance.home_model = model_data['home_model']
        instance.away_model = model_data['away_model']
        instance.training_info = model_data.get('training_info', {})
        instance.is_fitted = True
        
        return instance


def grid_search_nn(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Perform grid search for neural network hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Training features.
    y : pd.Series or np.ndarray
        Training target.
    param_grid : dict, optional
        Parameter grid. Defaults to reasonable architectures.
    cv : int, default=5
        Cross-validation folds.
    n_jobs : int, default=-1
        Parallel jobs.
    verbose : int, default=1
        Verbosity level.
    
    Returns
    -------
    dict
        Best parameters, score, and fitted model.
    """
    # Scale features first
    scaler = StandardScaler()
    if isinstance(X, pd.DataFrame):
        X_scaled = scaler.fit_transform(X.values)
    else:
        X_scaled = scaler.fit_transform(X)
    
    if isinstance(y, pd.Series):
        y = y.values
    
    if param_grid is None:
        param_grid = {
            'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
        }
    
    # Base model
    model = MLPRegressor(
        solver='adam',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    grid_search.fit(X_scaled, y)
    
    best_rmse = np.sqrt(-grid_search.best_score_)
    
    logger.info(f"Best NN params: {grid_search.best_params_}")
    logger.info(f"Best CV RMSE: {best_rmse:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': best_rmse,
        'best_model': grid_search.best_estimator_,
        'scaler': scaler,
        'cv_results': pd.DataFrame(grid_search.cv_results_),
    }


def random_search_nn(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    param_distributions: Optional[Dict] = None,
    n_iter: int = 50,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Perform random search for neural network hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Training features.
    y : pd.Series or np.ndarray
        Training target.
    param_distributions : dict, optional
        Parameter distributions. Defaults to reasonable ranges.
    n_iter : int, default=50
        Number of random configurations to try.
    cv : int, default=5
        Cross-validation folds.
    n_jobs : int, default=-1
        Parallel jobs.
    random_state : int, default=42
        Random seed.
    verbose : int, default=1
        Verbosity level.
    
    Returns
    -------
    dict
        Best parameters, score, and fitted model.
    """
    try:
        from scipy.stats import loguniform, uniform
    except ImportError:
        logger.warning("scipy not available, using basic distributions")
        loguniform = None
    
    # Scale features first
    scaler = StandardScaler()
    if isinstance(X, pd.DataFrame):
        X_scaled = scaler.fit_transform(X.values)
    else:
        X_scaled = scaler.fit_transform(X)
    
    if isinstance(y, pd.Series):
        y = y.values
    
    if param_distributions is None:
        architectures = [
            (32,), (64,), (128,),
            (32, 16), (64, 32), (128, 64),
            (64, 32, 16), (128, 64, 32),
            (256, 128, 64),
        ]
        
        if loguniform:
            param_distributions = {
                'hidden_layer_sizes': architectures,
                'activation': ['relu', 'tanh'],
                'alpha': loguniform(1e-5, 1e-1),
                'learning_rate_init': loguniform(1e-4, 1e-1),
            }
        else:
            param_distributions = {
                'hidden_layer_sizes': architectures,
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate_init': [0.0001, 0.001, 0.01],
            }
    
    # Base model
    model = MLPRegressor(
        solver='adam',
        max_iter=500,
        early_stopping=True,
        random_state=random_state
    )
    
    random_search = RandomizedSearchCV(
        model, param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )
    
    random_search.fit(X_scaled, y)
    
    best_rmse = np.sqrt(-random_search.best_score_)
    
    logger.info(f"Best NN params: {random_search.best_params_}")
    logger.info(f"Best CV RMSE: {best_rmse:.4f}")
    
    return {
        'best_params': random_search.best_params_,
        'best_score': best_rmse,
        'best_model': random_search.best_estimator_,
        'scaler': scaler,
        'cv_results': pd.DataFrame(random_search.cv_results_),
    }


def create_architecture(
    n_features: int,
    depth: int = 2,
    width_factor: float = 2.0,
    min_neurons: int = 16
) -> Tuple[int, ...]:
    """
    Create a neural network architecture based on input size.
    
    Common heuristic: first layer ~2x features, then halve each layer.
    
    Parameters
    ----------
    n_features : int
        Number of input features.
    depth : int, default=2
        Number of hidden layers.
    width_factor : float, default=2.0
        Multiplier for first layer width relative to features.
    min_neurons : int, default=16
        Minimum neurons in any layer.
    
    Returns
    -------
    tuple
        Hidden layer sizes.
    
    Examples
    --------
    >>> create_architecture(n_features=10, depth=3)
    (20, 10, 5)
    >>> create_architecture(n_features=50, depth=2)
    (100, 50)
    """
    first_layer = max(int(n_features * width_factor), min_neurons)
    layers = [first_layer]
    
    for _ in range(depth - 1):
        next_layer = max(layers[-1] // 2, min_neurons)
        layers.append(next_layer)
    
    return tuple(layers)
