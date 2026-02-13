"""
Ensemble Model - Production-Ready Hockey Goal Prediction

This module provides ensemble methods for combining multiple models to improve
prediction accuracy. Supports averaging, weighted averaging, and stacking.

Classes:
    - EnsembleModel: Combines multiple base models
    - StackedEnsemble: Meta-learning ensemble with stacking
    - EnsembleGoalPredictor: Dual ensemble for home/away goal prediction

Functions:
    - optimize_weights: Find optimal ensemble weights
    - create_default_ensemble: Create ensemble with standard models

Usage:
    from utils.ensemble_model import EnsembleModel, StackedEnsemble
    
    # Simple averaging
    ensemble = EnsembleModel(models=[model1, model2, model3])
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    
    # Stacking
    stacked = StackedEnsemble(base_models=[rf, xgb, ridge], meta_model=Ridge())
    stacked.fit(X_train, y_train)
    predictions = stacked.predict(X_test)
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    Ensemble model that combines multiple base models.
    
    Supports simple averaging, weighted averaging, and custom combination functions.
    
    Parameters
    ----------
    models : list
        List of fitted or unfitted model instances.
    weights : list or str, default='equal'
        Model weights for averaging:
        - 'equal': Equal weights for all models
        - 'inverse_error': Weights inversely proportional to validation error
        - list of floats: Custom weights (must sum to 1)
    combination : str, default='mean'
        How to combine predictions:
        - 'mean': Weighted average
        - 'median': Median of predictions
        - 'min': Minimum prediction
        - 'max': Maximum prediction
    name : str, optional
        Ensemble name for identification.
    
    Attributes
    ----------
    models : list
        Base model instances.
    weights_ : np.ndarray
        Learned or specified model weights.
    is_fitted : bool
        Whether the ensemble has been fitted.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import Ridge
    >>> 
    >>> rf = RandomForestRegressor(n_estimators=100)
    >>> ridge = Ridge(alpha=1.0)
    >>> 
    >>> ensemble = EnsembleModel(models=[rf, ridge], weights='inverse_error')
    >>> ensemble.fit(X_train, y_train, X_val, y_val)
    >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(
        self,
        models: List[Any] = None,
        weights: Union[str, List[float]] = 'equal',
        combination: str = 'mean',
        name: Optional[str] = None
    ):
        self.models = models or []
        self.weights = weights
        self.combination = combination
        self.name = name or f"Ensemble_{len(self.models)}models"
        
        self.weights_ = None
        self.is_fitted = False
        self.model_names = []
        self.training_info = {}
    
    def add_model(self, model: Any, name: Optional[str] = None) -> 'EnsembleModel':
        """
        Add a model to the ensemble.
        
        Parameters
        ----------
        model : estimator
            Model to add (fitted or unfitted).
        name : str, optional
            Model name for identification.
        
        Returns
        -------
        self
            The ensemble instance.
        """
        self.models.append(model)
        self.model_names.append(name or f"model_{len(self.models)}")
        return self
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'EnsembleModel':
        """
        Fit all base models and compute weights.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features.
        y : pd.Series or np.ndarray
            Training target.
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features (required for 'inverse_error' weights).
        y_val : pd.Series or np.ndarray, optional
            Validation target (required for 'inverse_error' weights).
        
        Returns
        -------
        self
            Fitted ensemble.
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Use add_model() first.")
        
        # Preserve DataFrames for sub-models that need feature names
        # Only convert y to numpy (targets don't need column names)
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Fit each model — pass X as-is (DataFrame or numpy)
        # Sub-models handle their own conversion internally
        for i, model in enumerate(self.models):
            if not hasattr(model, 'fit'):
                raise ValueError(f"Model {i} does not have a fit() method")
            model.fit(X, y)
            logger.debug(f"Fitted model {i+1}/{len(self.models)}")
        
        # Compute weights
        if isinstance(self.weights, list):
            self.weights_ = np.array(self.weights)
            if len(self.weights_) != len(self.models):
                raise ValueError(f"Weights length ({len(self.weights_)}) != models ({len(self.models)})")
        
        elif self.weights == 'equal':
            self.weights_ = np.ones(len(self.models)) / len(self.models)
        
        elif self.weights == 'inverse_error':
            if X_val is None or y_val is None:
                raise ValueError("Validation data required for 'inverse_error' weights")
            
            errors = []
            for model in self.models:
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                errors.append(rmse)
            
            # Inverse error (lower error = higher weight)
            inverse = 1.0 / np.array(errors)
            self.weights_ = inverse / inverse.sum()
        
        else:
            raise ValueError(f"Unknown weight strategy: {self.weights}")
        
        # Ensure weights sum to 1
        self.weights_ = self.weights_ / self.weights_.sum()
        
        self.is_fitted = True
        self.training_info = {
            'n_models': len(self.models),
            'n_samples': len(y),
            'weights': self.weights_.tolist(),
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"Fitted ensemble with {len(self.models)} models. "
                    f"Weights: {[f'{w:.3f}' for w in self.weights_]}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions by combining base model predictions.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction.
        
        Returns
        -------
        np.ndarray
            Combined predictions.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Pass X as-is — sub-models handle their own conversion
        # Get predictions from each model
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        # Combine predictions
        if self.combination == 'mean':
            return np.average(predictions, axis=1, weights=self.weights_)
        elif self.combination == 'median':
            return np.median(predictions, axis=1)
        elif self.combination == 'min':
            return np.min(predictions, axis=1)
        elif self.combination == 'max':
            return np.max(predictions, axis=1)
        else:
            raise ValueError(f"Unknown combination: {self.combination}")
    
    def predict_all(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get predictions from all base models.
        
        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_models).
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return np.column_stack([model.predict(X) for model in self.models])
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate ensemble and individual model performance.
        
        Returns
        -------
        dict
            Metrics for ensemble and each base model.
        """
        predictions = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        results = {
            'ensemble_rmse': np.sqrt(mean_squared_error(y, predictions)),
            'ensemble_mae': mean_absolute_error(y, predictions),
            'ensemble_r2': r2_score(y, predictions),
        }
        
        # Individual model metrics
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            name = self.model_names[i] if i < len(self.model_names) else f"model_{i}"
            results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y, pred))
        
        return results
    
    def get_weights(self) -> pd.Series:
        """Get model weights as a Series."""
        if self.weights_ is None:
            return None
        
        names = self.model_names if self.model_names else [f"model_{i}" for i in range(len(self.models))]
        return pd.Series(self.weights_, index=names)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'weights_': self.weights_,
            'model_names': self.model_names,
            'combination': self.combination,
            'weights': self.weights,
            'name': self.name,
            'training_info': self.training_info,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved ensemble to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EnsembleModel':
        """Load ensemble from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            models=model_data['models'],
            weights=model_data['weights'],
            combination=model_data['combination'],
            name=model_data.get('name')
        )
        instance.weights_ = model_data['weights_']
        instance.model_names = model_data.get('model_names', [])
        instance.training_info = model_data.get('training_info', {})
        instance.is_fitted = True
        
        return instance


class StackedEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble that uses a meta-model to combine base model predictions.
    
    Uses cross-validation to generate out-of-fold predictions for training
    the meta-model, preventing overfitting.
    
    Parameters
    ----------
    base_models : list
        List of (name, model) tuples for base estimators.
    meta_model : estimator, default=Ridge(alpha=0.1)
        Meta-model that learns to combine base predictions.
    cv : int, default=5
        Number of cross-validation folds for generating OOF predictions.
    passthrough : bool, default=False
        Whether to include original features in meta-model input.
    n_jobs : int, default=-1
        Parallel jobs for cross-validation.
    name : str, optional
        Ensemble name for identification.
    
    Attributes
    ----------
    stacking_model : StackingRegressor
        The sklearn stacking regressor.
    is_fitted : bool
        Whether the ensemble has been fitted.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import Ridge
    >>> 
    >>> base_models = [
    ...     ('rf', RandomForestRegressor(n_estimators=100)),
    ...     ('ridge', Ridge(alpha=1.0)),
    ... ]
    >>> stacked = StackedEnsemble(base_models=base_models, meta_model=Ridge())
    >>> stacked.fit(X_train, y_train)
    >>> predictions = stacked.predict(X_test)
    """
    
    def __init__(
        self,
        base_models: List[Tuple[str, Any]] = None,
        meta_model: Any = None,
        cv: int = 5,
        passthrough: bool = False,
        n_jobs: int = -1,
        name: Optional[str] = None
    ):
        self.base_models = base_models or []
        self.meta_model = meta_model or Ridge(alpha=0.1)
        self.cv = cv
        self.passthrough = passthrough
        self.n_jobs = n_jobs
        self.name = name or "StackedEnsemble"
        
        self.stacking_model = None
        self.is_fitted = False
        self.training_info = {}
    
    def add_model(self, name: str, model: Any) -> 'StackedEnsemble':
        """Add a base model to the stack."""
        self.base_models.append((name, model))
        return self
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'StackedEnsemble':
        """
        Fit the stacking ensemble.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features.
        y : pd.Series or np.ndarray
            Training target.
        
        Returns
        -------
        self
            Fitted ensemble.
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models. Use add_model() first.")
        
        # Create stacking regressor
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=clone(self.meta_model),
            cv=self.cv,
            passthrough=self.passthrough,
            n_jobs=self.n_jobs
        )
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = None
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Fit
        self.stacking_model.fit(X, y)
        
        self.is_fitted = True
        self.training_info = {
            'n_base_models': len(self.base_models),
            'n_samples': len(y),
            'cv_folds': self.cv,
            'passthrough': self.passthrough,
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"Fitted stacking ensemble with {len(self.base_models)} base models")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.stacking_model.predict(X)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate stacking ensemble performance."""
        predictions = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
        }
    
    def get_meta_model_weights(self) -> Optional[pd.Series]:
        """Get learned weights from the meta-model (if linear)."""
        if not self.is_fitted:
            return None
        
        meta = self.stacking_model.final_estimator_
        
        if hasattr(meta, 'coef_'):
            names = [name for name, _ in self.base_models]
            if self.passthrough and self.feature_names:
                names.extend(self.feature_names)
            
            n_coef = len(meta.coef_)
            if len(names) >= n_coef:
                return pd.Series(meta.coef_, index=names[:n_coef])
            return pd.Series(meta.coef_)
        
        return None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save stacking ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'stacking_model': self.stacking_model,
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'cv': self.cv,
            'passthrough': self.passthrough,
            'name': self.name,
            'training_info': self.training_info,
            'feature_names': getattr(self, 'feature_names', None),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved stacking ensemble to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'StackedEnsemble':
        """Load stacking ensemble from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            base_models=model_data['base_models'],
            meta_model=model_data['meta_model'],
            cv=model_data['cv'],
            passthrough=model_data['passthrough'],
            name=model_data.get('name')
        )
        instance.stacking_model = model_data['stacking_model']
        instance.training_info = model_data.get('training_info', {})
        instance.feature_names = model_data.get('feature_names')
        instance.is_fitted = True
        
        return instance


class EnsembleGoalPredictor:
    """
    Dual ensemble model for predicting both home and away goals.
    
    Uses separate ensembles for home and away goal prediction.
    
    Parameters
    ----------
    base_models : list
        List of (name, model) tuples to use in each ensemble.
    weights : str or list, default='inverse_error'
        Weight strategy for the ensembles.
    feature_columns : list, optional
        Column names to use as features.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import Ridge
    >>> 
    >>> base_models = [
    ...     ('rf', RandomForestRegressor(n_estimators=100)),
    ...     ('ridge', Ridge(alpha=1.0)),
    ... ]
    >>> predictor = EnsembleGoalPredictor(base_models=base_models)
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
        base_models: List[Tuple[str, Any]] = None,
        weights: Union[str, List[float]] = 'inverse_error',
        feature_columns: Optional[List[str]] = None
    ):
        self.base_models = base_models or []
        self.weights = weights
        self.feature_columns = feature_columns
        
        self.home_ensemble = None
        self.away_ensemble = None
        self.is_fitted = False
        self.training_info = {}
    
    def _clone_models(self) -> List[Tuple[str, Any]]:
        """Clone base models for fresh instances."""
        return [(name, clone(model)) for name, model in self.base_models]
    
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
        away_goals_col: str = 'away_goals',
        val_fraction: float = 0.2
    ) -> 'EnsembleGoalPredictor':
        """
        Fit home and away goal prediction ensembles.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with features and goal columns.
        home_goals_col : str, default='home_goals'
            Column name for home goals.
        away_goals_col : str, default='away_goals'
            Column name for away goals.
        val_fraction : float, default=0.2
            Fraction of data for validation (for weight optimization).
        
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
        
        # Split for weight optimization
        n_val = int(len(df) * val_fraction)
        X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_home_train, y_home_val = y_home.iloc[:-n_val], y_home.iloc[-n_val:]
        y_away_train, y_away_val = y_away.iloc[:-n_val], y_away.iloc[-n_val:]
        
        # Create ensembles with cloned models
        self.home_ensemble = EnsembleModel(
            models=[clone(m) for _, m in self.base_models],
            weights=self.weights,
            name='home_goals_ensemble'
        )
        self.home_ensemble.model_names = [n for n, _ in self.base_models]
        
        self.away_ensemble = EnsembleModel(
            models=[clone(m) for _, m in self.base_models],
            weights=self.weights,
            name='away_goals_ensemble'
        )
        self.away_ensemble.model_names = [n for n, _ in self.base_models]
        
        # Fit ensembles
        self.home_ensemble.fit(X_train, y_home_train, X_val, y_home_val)
        self.away_ensemble.fit(X_train, y_away_train, X_val, y_away_val)
        
        self.is_fitted = True
        self.training_info = {
            'n_samples': len(df),
            'n_features': len(self.feature_columns),
            'n_models': len(self.base_models),
            'home_weights': self.home_ensemble.weights_.tolist(),
            'away_weights': self.away_ensemble.weights_.tolist(),
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"Fitted dual ensemble predictor with {len(self.base_models)} models")
        
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
        
        home_pred = self.home_ensemble.predict(X)
        away_pred = self.away_ensemble.predict(X)
        
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
    
    def save(self, path: Union[str, Path]) -> None:
        """Save predictor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'home_ensemble': self.home_ensemble,
            'away_ensemble': self.away_ensemble,
            'base_models': self.base_models,
            'weights': self.weights,
            'feature_columns': self.feature_columns,
            'training_info': self.training_info,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EnsembleGoalPredictor':
        """Load predictor from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            base_models=model_data['base_models'],
            weights=model_data['weights'],
            feature_columns=model_data['feature_columns']
        )
        instance.home_ensemble = model_data['home_ensemble']
        instance.away_ensemble = model_data['away_ensemble']
        instance.training_info = model_data.get('training_info', {})
        instance.is_fitted = True
        
        return instance


def optimize_weights(
    predictions: np.ndarray,
    y_true: np.ndarray,
    method: str = 'minimize'
) -> np.ndarray:
    """
    Find optimal weights for combining predictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Array of shape (n_samples, n_models) with predictions from each model.
    y_true : np.ndarray
        True target values.
    method : str, default='minimize'
        Optimization method: 'minimize' uses scipy.optimize.
    
    Returns
    -------
    np.ndarray
        Optimal weights (sum to 1).
    """
    try:
        from scipy.optimize import minimize
        
        n_models = predictions.shape[1]
        
        def objective(weights):
            combined = predictions @ weights
            return mean_squared_error(y_true, combined)
        
        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial guess: equal weights
        initial = np.ones(n_models) / n_models
        
        result = minimize(
            objective, initial,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    except ImportError:
        logger.warning("scipy not available, using equal weights")
        n_models = predictions.shape[1]
        return np.ones(n_models) / n_models


def create_default_ensemble() -> EnsembleModel:
    """
    Create an ensemble with recommended default models.
    
    Returns
    -------
    EnsembleModel
        Pre-configured ensemble (unfitted).
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    
    models = [
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        Ridge(alpha=1.0),
        ElasticNet(alpha=0.1, l1_ratio=0.5),
    ]
    
    ensemble = EnsembleModel(
        models=models,
        weights='inverse_error',
        name='DefaultEnsemble'
    )
    ensemble.model_names = ['rf', 'gbm', 'ridge', 'elastic']
    
    return ensemble
