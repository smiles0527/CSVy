"""
XGBoost Model v2 - Production-Ready Hockey Goal Prediction

This module provides XGBoost-based models for predicting hockey game outcomes.
Fully production-ready with serialization, logging, and comprehensive utilities.

Classes:
    - XGBoostModel: Single-target XGBoost regressor
    - XGBoostGoalPredictor: Dual model for home/away goal prediction

Functions:
    - grid_search_xgboost: Exhaustive hyperparameter search
    - random_search_xgboost: Randomized hyperparameter search
    - bayesian_search_xgboost: Optuna-based Bayesian optimization

Usage:
    from utils.xgboost_model_v2 import XGBoostModel, XGBoostGoalPredictor
    
    # Single target prediction
    model = XGBoostModel(params={'max_depth': 6, 'learning_rate': 0.05})
    model.fit(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_test)
    
    # Goal prediction (both home and away)
    predictor = XGBoostGoalPredictor()
    predictor.fit(games_df)
    home_pred, away_pred = predictor.predict_goals(game)
"""

import json
import logging
import pickle
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Configure module logger
logger = logging.getLogger(__name__)

# Optional imports with availability flags
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    XGB_VERSION = xgb.__version__
except ImportError:
    XGB_AVAILABLE = False
    XGB_VERSION = None
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# Column name mappings - consistent with other models
COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_goals': ['home_goals', 'home_score', 'h_goals', 'goals_home', 'home_pts'],
    'away_goals': ['away_goals', 'away_score', 'a_goals', 'goals_away', 'away_pts', 'visitor_goals'],
    'game_date': ['game_date', 'date', 'Date', 'game_datetime', 'datetime', 'game_time'],
}


def get_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """
    Find the correct column name in a DataFrame using aliases.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search
    field : str
        Logical field name (e.g., 'home_goals')
    
    Returns
    -------
    str or None
        Actual column name if found, None otherwise
    """
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


class XGBoostModel:
    """
    XGBoost regression model for hockey goal prediction.
    
    A production-ready XGBoost wrapper with full serialization support,
    early stopping, feature importance analysis, and prediction intervals.
    
    Parameters
    ----------
    params : dict, optional
        XGBoost hyperparameters. Missing keys use defaults from DEFAULT_PARAMS.
    scale_features : bool, default=False
        Whether to standardize features before training.
    name : str, optional
        Model name for logging and identification.
    
    Attributes
    ----------
    model : xgb.XGBRegressor
        The underlying XGBoost model
    scaler : StandardScaler or None
        Feature scaler if scale_features=True
    feature_names : list
        Names of features used in training
    feature_importances_ : pd.Series
        Feature importance scores (after fitting)
    is_fitted : bool
        Whether the model has been trained
    training_history : dict
        Training and validation loss history
    metadata : dict
        Model metadata (creation time, version, etc.)
    
    Examples
    --------
    >>> model = XGBoostModel({'max_depth': 6, 'learning_rate': 0.05})
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)
    >>> metrics = model.evaluate(X_test, y_test)
    >>> model.save('models/xgboost_home.pkl')
    """
    
    # Default hyperparameters optimized for hockey prediction
    DEFAULT_PARAMS = {
        'learning_rate': 0.05,
        'n_estimators': 500,
        'max_depth': 6,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'verbosity': 0,
    }
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_features: bool = False,
        name: Optional[str] = None
    ):
        if not XGB_AVAILABLE:
            raise ImportError(
                "XGBoost is required but not installed. "
                "Install with: pip install xgboost"
            )
        
        self.name = name or f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.scale_features = scale_features
        
        # Model components
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = StandardScaler() if scale_features else None
        
        # Feature tracking
        self.feature_names: Optional[List[str]] = None
        self.feature_importances_: Optional[pd.Series] = None
        self.n_features_: int = 0
        
        # State
        self.is_fitted: bool = False
        self.training_history: Optional[Dict] = None
        self.best_iteration_: Optional[int] = None
        
        # Metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'xgboost_version': XGB_VERSION,
            'model_name': self.name,
        }
        
        logger.debug(f"Initialized XGBoostModel: {self.name}")
    
    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Prepare features for training/prediction.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features
        fit_scaler : bool, default=False
            Whether to fit the scaler (only True during training)
        
        Returns
        -------
        np.ndarray
            Prepared feature matrix
        """
        # Extract feature names from DataFrame
        if isinstance(X, pd.DataFrame):
            if fit_scaler or self.feature_names is None:
                self.feature_names = list(X.columns)
            X = X.values
        
        # Handle NaN values
        if np.isnan(X).any():
            logger.warning("NaN values detected in features, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        # Apply scaling if enabled
        if self.scale_features and self.scaler is not None:
            if fit_scaler:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        return X.astype(np.float32)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : pd.Series or np.ndarray
            Training targets
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features for early stopping
        y_val : pd.Series or np.ndarray, optional
            Validation targets for early stopping
        early_stopping_rounds : int, default=50
            Stop training if validation metric doesn't improve
        verbose : bool, default=False
            Print training progress
        sample_weight : np.ndarray, optional
            Sample weights for training
        
        Returns
        -------
        XGBoostModel
            Fitted model instance (self)
        """
        logger.info(f"Training {self.name}...")
        
        # Prepare training data
        X_train = self._prepare_features(X, fit_scaler=True)
        y_train = np.asarray(y).ravel().astype(np.float32)
        self.n_features_ = X_train.shape[1]
        
        # Create model with early stopping params
        model_params = self.params.copy()
        if X_val is not None and y_val is not None:
            model_params['early_stopping_rounds'] = early_stopping_rounds
        
        self.model = xgb.XGBRegressor(**model_params)
        
        # Prepare fit kwargs
        fit_kwargs = {'verbose': verbose}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
        # Fit with or without validation set
        if X_val is not None and y_val is not None:
            X_val_prep = self._prepare_features(X_val, fit_scaler=False)
            y_val_prep = np.asarray(y_val).ravel().astype(np.float32)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val_prep, y_val_prep)],
                **fit_kwargs
            )
            
            self.training_history = self.model.evals_result()
            self.best_iteration_ = self.model.best_iteration
            
            logger.info(
                f"Training complete. Best iteration: {self.best_iteration_}, "
                f"Best validation RMSE: {self.training_history['validation_1']['rmse'][self.best_iteration_]:.4f}"
            )
        else:
            self.model.fit(X_train, y_train, **fit_kwargs)
            logger.info("Training complete (no validation set)")
        
        # Store feature importances
        self._compute_feature_importance()
        
        # Update metadata
        self.metadata['fitted_at'] = datetime.now().isoformat()
        self.metadata['n_samples'] = len(y_train)
        self.metadata['n_features'] = self.n_features_
        
        self.is_fitted = True
        return self
    
    def _compute_feature_importance(self) -> None:
        """Compute and store feature importance scores."""
        if self.model is None:
            return
        
        importances = self.model.feature_importances_
        
        if self.feature_names and len(self.feature_names) == len(importances):
            self.feature_importances_ = pd.Series(
                importances,
                index=self.feature_names,
                name='importance'
            ).sort_values(ascending=False)
        else:
            self.feature_importances_ = pd.Series(
                importances,
                name='importance'
            ).sort_values(ascending=False)
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        clip_negative: bool = True
    ) -> np.ndarray:
        """
        Predict target values.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
        clip_negative : bool, default=True
            Clip predictions to be non-negative (goals can't be negative)
        
        Returns
        -------
        np.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_prep = self._prepare_features(X, fit_scaler=False)
        predictions = self.model.predict(X_prep)
        
        if clip_negative:
            predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_uncertainty(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        confidence: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
        confidence : float, default=0.95
            Confidence level for intervals
        
        Returns
        -------
        dict
            Dictionary with 'mean', 'std', 'lower', 'upper' arrays
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_prep = self._prepare_features(X, fit_scaler=False)
        
        # Get base prediction
        base_pred = self.model.predict(X_prep)
        
        # Estimate std from learning rate and number of trees
        lr = self.params.get('learning_rate', 0.05)
        n_trees = self.params.get('n_estimators', 500)
        estimated_std = np.abs(base_pred) * lr * np.sqrt(n_trees) / n_trees
        estimated_std = np.maximum(estimated_std, 0.1)  # Minimum uncertainty
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        
        return {
            'mean': base_pred,
            'std': estimated_std,
            'lower': np.maximum(base_pred - z_score * estimated_std, 0),
            'upper': base_pred + z_score * estimated_std
        }
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Test features
        y : pd.Series or np.ndarray
            True values
        
        Returns
        -------
        dict
            Dictionary with RMSE, MAE, R², MAPE metrics
        """
        predictions = self.predict(X)
        y_true = np.asarray(y).ravel()
        
        # Avoid division by zero for MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100 if mask.any() else 0.0
        
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, predictions))),
            'mae': float(mean_absolute_error(y_true, predictions)),
            'r2': float(r2_score(y_true, predictions)),
            'mape': float(mape),
            'n_samples': int(len(y_true)),
            'mean_prediction': float(predictions.mean()),
            'std_prediction': float(predictions.std()),
        }
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: str = 'neg_root_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            Targets
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='neg_root_mean_squared_error'
            Scoring metric
        
        Returns
        -------
        dict
            Cross-validation results with mean, std, and fold scores
        """
        X_prep = self._prepare_features(X, fit_scaler=True)
        y_prep = np.asarray(y).ravel()
        
        cv_model = xgb.XGBRegressor(**self.params)
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(cv_model, X_prep, y_prep, cv=kfold, scoring=scoring)
        
        # Convert negative scores if needed
        if scoring.startswith('neg_'):
            scores = -scores
        
        return {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'scores': scores.tolist(),
            'cv_folds': cv,
            'scoring': scoring,
        }
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_n: Optional[int] = None
    ) -> pd.Series:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        importance_type : str, default='gain'
            Type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
        top_n : int, optional
            Return only top N features
        
        Returns
        -------
        pd.Series
            Feature importance scores, sorted descending
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Get importance from booster with specified type
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Map to feature names
        if self.feature_names:
            importance_series = pd.Series(
                index=self.feature_names,
                dtype=float,
                name=f'importance_{importance_type}'
            ).fillna(0)
            
            for feat, score in importance.items():
                if feat.startswith('f') and feat[1:].isdigit():
                    idx = int(feat[1:])
                    if idx < len(self.feature_names):
                        importance_series.iloc[idx] = score
                elif feat in importance_series.index:
                    importance_series[feat] = score
            
            importance_series = importance_series.sort_values(ascending=False)
        else:
            importance_series = pd.Series(importance).sort_values(ascending=False)
        
        if top_n:
            return importance_series.head(top_n)
        return importance_series
    
    def get_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_samples: int = 1000
    ) -> Optional[Any]:
        """
        Calculate SHAP values for model interpretation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features to explain
        max_samples : int, default=1000
            Maximum samples (SHAP can be slow on large datasets)
        
        Returns
        -------
        shap.Explanation or None
            SHAP values if shap is installed
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return None
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        X_prep = self._prepare_features(X, fit_scaler=False)
        
        # Subsample if needed
        if len(X_prep) > max_samples:
            indices = np.random.choice(len(X_prep), max_samples, replace=False)
            X_prep = X_prep[indices]
        
        explainer = shap.TreeExplainer(self.model)
        return explainer(X_prep)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save complete model state to disk.
        
        Saves the XGBoost model, scaler, feature names, parameters,
        and all metadata for full reproducibility.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the model (recommended: .pkl extension)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare state dict
        state = {
            'model_state': self.model.get_booster().save_raw('json'),
            'params': self.params,
            'feature_names': self.feature_names,
            'n_features': self.n_features_,
            'scale_features': self.scale_features,
            'scaler': self.scaler,
            'feature_importances': self.feature_importances_.to_dict() if self.feature_importances_ is not None else None,
            'training_history': self.training_history,
            'best_iteration': self.best_iteration_,
            'metadata': self.metadata,
            'name': self.name,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'XGBoostModel':
        """
        Load a saved model from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved model
        
        Returns
        -------
        XGBoostModel
            Loaded model instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance
        instance = cls(
            params=state['params'],
            scale_features=state['scale_features'],
            name=state.get('name')
        )
        
        # Restore model
        instance.model = xgb.XGBRegressor(**state['params'])
        booster = xgb.Booster()
        booster.load_model(bytearray(state['model_state']))
        instance.model._Booster = booster
        instance.model.n_features_in_ = state['n_features']
        
        # Restore other state
        instance.feature_names = state['feature_names']
        instance.n_features_ = state['n_features']
        instance.scaler = state['scaler']
        instance.training_history = state['training_history']
        instance.best_iteration_ = state['best_iteration']
        instance.metadata = state['metadata']
        
        if state['feature_importances']:
            instance.feature_importances_ = pd.Series(state['feature_importances'])
        
        instance.is_fitted = True
        instance.metadata['loaded_at'] = datetime.now().isoformat()
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def save_xgb_format(self, filepath: Union[str, Path]) -> None:
        """
        Save model in native XGBoost format (for interoperability).
        
        Parameters
        ----------
        filepath : str or Path
            Path to save (.json or .ubj extension)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(filepath))
        
        # Also save metadata as companion file
        meta_path = filepath.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'params': {k: v for k, v in self.params.items() if not callable(v)},
                'metadata': self.metadata,
            }, f, indent=2, default=str)
        
        logger.info(f"Model saved to {filepath} (XGBoost format)")
    
    def get_params(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> 'XGBoostModel':
        """Set hyperparameters (must refit after changing)."""
        self.params.update(params)
        self.is_fitted = False
        return self
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"XGBoostModel(name='{self.name}', status={status}, n_features={self.n_features_})"


class XGBoostGoalPredictor:
    """
    Dual XGBoost model for predicting both home and away goals.
    
    Uses two separate XGBoost models internally - one for home goals,
    one for away goals. Provides unified interface consistent with
    EloModel and BaselineModel.
    
    Parameters
    ----------
    params : dict, optional
        XGBoost hyperparameters (applied to both models)
    scale_features : bool, default=False
        Whether to scale features
    home_params : dict, optional
        Override params for home model only
    away_params : dict, optional
        Override params for away model only
    
    Attributes
    ----------
    home_model : XGBoostModel
        Model for predicting home team goals
    away_model : XGBoostModel
        Model for predicting away team goals
    feature_columns : list
        Feature columns used for prediction
    
    Examples
    --------
    >>> predictor = XGBoostGoalPredictor({'max_depth': 6})
    >>> predictor.fit(train_df)
    >>> home_pred, away_pred = predictor.predict_goals(game)
    >>> metrics = predictor.evaluate(test_df)
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_features: bool = False,
        home_params: Optional[Dict[str, Any]] = None,
        away_params: Optional[Dict[str, Any]] = None
    ):
        self.base_params = params or {}
        self.scale_features = scale_features
        
        # Merge params for each model
        home_p = {**self.base_params, **(home_params or {})}
        away_p = {**self.base_params, **(away_params or {})}
        
        self.home_model = XGBoostModel(home_p, scale_features, name='xgb_home_goals')
        self.away_model = XGBoostModel(away_p, scale_features, name='xgb_away_goals')
        
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted: bool = False
        
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': 'XGBoostGoalPredictor',
        }
    
    def _extract_features(self, df: pd.DataFrame) -> List[str]:
        """Extract numeric feature columns, excluding targets and identifiers."""
        exclude_cols = {
            'home_goals', 'away_goals', 'home_score', 'away_score',
            'home_team', 'away_team', 'game_date', 'date', 'game_id',
            'season', 'game_type', 'venue'
        }
        
        feature_cols = [
            col for col in df.columns
            if col.lower() not in {c.lower() for c in exclude_cols}
            and df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int', 'float']
        ]
        
        return feature_cols
    
    def fit(
        self,
        games_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_home_val: Optional[pd.Series] = None,
        y_away_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False
    ) -> 'XGBoostGoalPredictor':
        """
        Train both models on game data.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            Training data with features and target columns
        feature_columns : list, optional
            Specific columns to use as features. Auto-detected if None.
        X_val : pd.DataFrame, optional
            Validation features
        y_home_val : pd.Series, optional
            Validation home goals
        y_away_val : pd.Series, optional
            Validation away goals
        early_stopping_rounds : int, default=50
            Early stopping patience
        verbose : bool, default=False
            Print training progress
        
        Returns
        -------
        XGBoostGoalPredictor
            Fitted predictor (self)
        """
        logger.info("Training XGBoostGoalPredictor...")
        
        # Identify feature columns
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = self._extract_features(games_df)
        
        if len(self.feature_columns) == 0:
            raise ValueError("No numeric feature columns found in games_df")
        
        logger.info(f"Using {len(self.feature_columns)} features")
        
        # Get target columns
        home_col = get_column(games_df, 'home_goals')
        away_col = get_column(games_df, 'away_goals')
        
        if home_col is None or away_col is None:
            raise ValueError(
                "games_df must have home_goals and away_goals columns. "
                f"Found columns: {list(games_df.columns)}"
            )
        
        X = games_df[self.feature_columns]
        y_home = games_df[home_col]
        y_away = games_df[away_col]
        
        # Prepare validation data
        X_val_prep = X_val[self.feature_columns] if X_val is not None else None
        
        # Fit both models
        self.home_model.fit(
            X, y_home,
            X_val_prep, y_home_val,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
        
        self.away_model.fit(
            X, y_away,
            X_val_prep, y_away_val,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
        
        self.is_fitted = True
        self.metadata['fitted_at'] = datetime.now().isoformat()
        self.metadata['n_games'] = len(games_df)
        self.metadata['n_features'] = len(self.feature_columns)
        
        logger.info("Training complete")
        return self
    
    def predict_goals(
        self,
        game: Union[Dict, pd.Series],
        with_uncertainty: bool = False
    ) -> Union[Tuple[float, float], Dict[str, Any]]:
        """
        Predict goals for a single game.
        
        Parameters
        ----------
        game : dict or pd.Series
            Game with feature values
        with_uncertainty : bool, default=False
            Return prediction intervals
        
        Returns
        -------
        tuple or dict
            (home_goals, away_goals) or dict with predictions and intervals
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if isinstance(game, dict):
            game = pd.Series(game)
        
        # Check for missing features
        missing = set(self.feature_columns) - set(game.index)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X = pd.DataFrame([game[self.feature_columns]])
        
        if with_uncertainty:
            home_result = self.home_model.predict_with_uncertainty(X)
            away_result = self.away_model.predict_with_uncertainty(X)
            
            return {
                'home_goals': float(home_result['mean'][0]),
                'away_goals': float(away_result['mean'][0]),
                'home_lower': float(home_result['lower'][0]),
                'home_upper': float(home_result['upper'][0]),
                'away_lower': float(away_result['lower'][0]),
                'away_upper': float(away_result['upper'][0]),
            }
        
        home_pred = self.home_model.predict(X)[0]
        away_pred = self.away_model.predict(X)[0]
        
        return float(home_pred), float(away_pred)
    
    def predict_batch(
        self,
        games_df: pd.DataFrame,
        with_uncertainty: bool = False
    ) -> pd.DataFrame:
        """
        Predict goals for multiple games.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            Games with feature values
        with_uncertainty : bool, default=False
            Include prediction intervals
        
        Returns
        -------
        pd.DataFrame
            DataFrame with predictions (and intervals if requested)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = games_df[self.feature_columns]
        
        result = pd.DataFrame({
            'home_pred': self.home_model.predict(X),
            'away_pred': self.away_model.predict(X),
        }, index=games_df.index)
        
        if with_uncertainty:
            home_unc = self.home_model.predict_with_uncertainty(X)
            away_unc = self.away_model.predict_with_uncertainty(X)
            
            result['home_lower'] = home_unc['lower']
            result['home_upper'] = home_unc['upper']
            result['away_lower'] = away_unc['lower']
            result['away_upper'] = away_unc['upper']
        
        return result
    
    def predict_winner(
        self,
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict game winners and probabilities.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            Games with feature values
        
        Returns
        -------
        pd.DataFrame
            DataFrame with winner predictions and confidence
        """
        predictions = self.predict_batch(games_df, with_uncertainty=True)
        
        goal_diff = predictions['home_pred'] - predictions['away_pred']
        
        # Estimate win probability from prediction spread
        predictions['predicted_winner'] = np.where(goal_diff > 0, 'home', 'away')
        predictions['goal_difference'] = goal_diff
        predictions['confidence'] = np.abs(goal_diff) / (
            np.abs(goal_diff) + 1
        )  # Logistic-like confidence
        
        return predictions
    
    def evaluate(self, games_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate predictor on test set.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            Test games with features and targets
        
        Returns
        -------
        dict
            Metrics for home, away, combined, and win prediction
        """
        home_col = get_column(games_df, 'home_goals')
        away_col = get_column(games_df, 'away_goals')
        
        X = games_df[self.feature_columns]
        
        home_metrics = self.home_model.evaluate(X, games_df[home_col])
        away_metrics = self.away_model.evaluate(X, games_df[away_col])
        
        # Combined predictions
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        all_preds = np.concatenate([home_pred, away_pred])
        all_actual = np.concatenate([
            games_df[home_col].values,
            games_df[away_col].values
        ])
        
        combined_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(all_actual, all_preds))),
            'mae': float(mean_absolute_error(all_actual, all_preds)),
            'r2': float(r2_score(all_actual, all_preds)),
        }
        
        # Win prediction accuracy
        actual_winner = np.where(
            games_df[home_col] > games_df[away_col], 'home',
            np.where(games_df[home_col] < games_df[away_col], 'away', 'tie')
        )
        pred_winner = np.where(home_pred > away_pred, 'home', 'away')
        
        # Exclude ties from win accuracy
        non_tie_mask = actual_winner != 'tie'
        win_accuracy = (
            (actual_winner[non_tie_mask] == pred_winner[non_tie_mask]).mean()
            if non_tie_mask.any() else 0.0
        )
        
        return {
            'home': home_metrics,
            'away': away_metrics,
            'combined': combined_metrics,
            'win_accuracy': float(win_accuracy),
            'n_games': len(games_df),
        }
    
    def get_feature_importance(
        self,
        target: str = 'combined',
        top_n: int = 20
    ) -> pd.Series:
        """
        Get feature importance from models.
        
        Parameters
        ----------
        target : str, default='combined'
            Which model: 'home', 'away', or 'combined' (average)
        top_n : int, default=20
            Number of top features
        
        Returns
        -------
        pd.Series
            Feature importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        if target == 'home':
            return self.home_model.feature_importances_.head(top_n)
        elif target == 'away':
            return self.away_model.feature_importances_.head(top_n)
        else:
            combined = (
                self.home_model.feature_importances_.reindex(self.feature_columns).fillna(0) +
                self.away_model.feature_importances_.reindex(self.feature_columns).fillna(0)
            ) / 2
            return combined.sort_values(ascending=False).head(top_n)
    
    def save(self, dirpath: Union[str, Path]) -> None:
        """
        Save both models and metadata.
        
        Parameters
        ----------
        dirpath : str or Path
            Directory to save models
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        self.home_model.save(dirpath / 'home_model.pkl')
        self.away_model.save(dirpath / 'away_model.pkl')
        
        # Save predictor metadata
        meta = {
            'feature_columns': self.feature_columns,
            'base_params': self.base_params,
            'scale_features': self.scale_features,
            'metadata': self.metadata,
        }
        
        with open(dirpath / 'predictor_meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        
        logger.info(f"Predictor saved to {dirpath}")
    
    @classmethod
    def load(cls, dirpath: Union[str, Path]) -> 'XGBoostGoalPredictor':
        """
        Load a saved predictor.
        
        Parameters
        ----------
        dirpath : str or Path
            Directory containing saved models
        
        Returns
        -------
        XGBoostGoalPredictor
            Loaded predictor
        """
        dirpath = Path(dirpath)
        
        # Load metadata
        with open(dirpath / 'predictor_meta.json', 'r') as f:
            meta = json.load(f)
        
        # Create instance
        instance = cls(
            params=meta['base_params'],
            scale_features=meta['scale_features']
        )
        
        # Load models
        instance.home_model = XGBoostModel.load(dirpath / 'home_model.pkl')
        instance.away_model = XGBoostModel.load(dirpath / 'away_model.pkl')
        instance.feature_columns = meta['feature_columns']
        instance.metadata = meta['metadata']
        instance.is_fitted = True
        
        logger.info(f"Predictor loaded from {dirpath}")
        return instance
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_feat = len(self.feature_columns) if self.feature_columns else 0
        return f"XGBoostGoalPredictor(status={status}, n_features={n_feat})"


# =============================================================================
# Hyperparameter Search Functions
# =============================================================================

def grid_search_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict[str, List],
    cv: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Exhaustive grid search for XGBoost hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training targets
    param_grid : dict
        Parameter grid {param_name: [values]}
    cv : int, default=5
        Cross-validation folds
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    dict
        {'best_params', 'best_score', 'all_results'}
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    results = []
    best_score = float('inf')
    best_params = None
    
    if verbose:
        print(f"Grid Search: Testing {len(combinations)} combinations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        try:
            model = XGBoostModel(params)
            cv_result = model.cross_validate(X, y, cv=cv)
            rmse = cv_result['mean']
            
            results.append({
                **params,
                'rmse_mean': rmse,
                'rmse_std': cv_result['std']
            })
            
            if rmse < best_score:
                best_score = rmse
                best_params = params.copy()
            
            if verbose and (i + 1) % max(1, len(combinations) // 10) == 0:
                print(f"  Progress: {i + 1}/{len(combinations)}, Best RMSE: {best_score:.4f}")
                
        except Exception as e:
            logger.warning(f"Error with params {params}: {e}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': pd.DataFrame(results).sort_values('rmse_mean'),
        'n_combinations': len(combinations),
    }


def random_search_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    param_distributions: Dict[str, List],
    n_iter: int = 50,
    cv: int = 5,
    verbose: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Randomized search for XGBoost hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training targets
    param_distributions : dict
        Parameter distributions {param_name: [values]}
    n_iter : int, default=50
        Number of random combinations to try
    cv : int, default=5
        Cross-validation folds
    verbose : bool, default=True
        Print progress
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    dict
        {'best_params', 'best_score', 'all_results'}
    """
    np.random.seed(random_state)
    
    results = []
    best_score = float('inf')
    best_params = None
    
    if verbose:
        print(f"Random Search: Testing {n_iter} combinations...")
    
    for i in range(n_iter):
        params = {
            key: np.random.choice(values) if isinstance(values, (list, tuple)) else values
            for key, values in param_distributions.items()
        }
        
        try:
            model = XGBoostModel(params)
            cv_result = model.cross_validate(X, y, cv=cv)
            rmse = cv_result['mean']
            
            results.append({
                **{k: (float(v) if isinstance(v, np.floating) else v) for k, v in params.items()},
                'rmse_mean': rmse,
                'rmse_std': cv_result['std']
            })
            
            if rmse < best_score:
                best_score = rmse
                best_params = params.copy()
            
            if verbose and (i + 1) % max(1, n_iter // 10) == 0:
                print(f"  Progress: {i + 1}/{n_iter}, Best RMSE: {best_score:.4f}")
                
        except Exception as e:
            logger.warning(f"Error with params {params}: {e}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': pd.DataFrame(results).sort_values('rmse_mean'),
        'n_iterations': n_iter,
    }


def bayesian_search_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    cv: int = 5,
    verbose: bool = True,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Bayesian optimization using Optuna.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training targets
    n_trials : int, default=100
        Number of optimization trials
    cv : int, default=5
        Cross-validation folds
    verbose : bool, default=True
        Print progress
    timeout : int, optional
        Maximum time in seconds
    
    Returns
    -------
    dict
        {'best_params', 'best_score', 'study'}
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required. Install with: pip install optuna")
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        }
        
        model = XGBoostModel(params)
        cv_result = model.cross_validate(X, y, cv=cv)
        return cv_result['mean']
    
    # Suppress Optuna logs if not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study,
        'n_trials': len(study.trials),
    }
