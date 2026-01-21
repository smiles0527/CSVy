"""
Linear Regression Model - Production-Ready Hockey Goal Prediction

This module provides linear regression-based models for predicting hockey game outcomes.
Supports ElasticNet regularization (L1/L2), polynomial features, and feature scaling.

Classes:
    - LinearRegressionModel: Single-target linear regression
    - LinearGoalPredictor: Dual model for home/away goal prediction

Functions:
    - grid_search_linear: Exhaustive hyperparameter search
    - random_search_linear: Randomized hyperparameter search
    - create_polynomial_features: Generate polynomial feature combinations

Usage:
    from utils.linear_model import LinearRegressionModel, LinearGoalPredictor
    
    # Single target prediction
    model = LinearRegressionModel(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Goal prediction (both home and away)
    predictor = LinearGoalPredictor()
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
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Configure module logger
logger = logging.getLogger(__name__)


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


def create_polynomial_features(
    X: Union[pd.DataFrame, np.ndarray],
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False
) -> Tuple[np.ndarray, PolynomialFeatures]:
    """
    Create polynomial features from input data.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input features
    degree : int, default=2
        Polynomial degree
    interaction_only : bool, default=False
        Only include interaction terms
    include_bias : bool, default=False
        Include bias column
    
    Returns
    -------
    tuple
        (transformed_features, poly_transformer)
    """
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias
    )
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    X_poly = poly.fit_transform(X_array)
    
    return X_poly, poly


class LinearRegressionModel:
    """
    Linear regression model with regularization for hockey goal prediction.
    
    Supports Ridge (L2), Lasso (L1), ElasticNet (L1+L2), and plain linear regression.
    Includes polynomial feature generation, feature scaling, and full serialization.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. 0 = no regularization.
    l1_ratio : float, default=0.5
        ElasticNet mixing: 0.0 = Ridge (L2), 1.0 = Lasso (L1), between = ElasticNet.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    scaling : str, default='standard'
        Scaling method: 'standard', 'robust', or None.
    poly_degree : int, default=1
        Polynomial feature degree. 1 = linear only.
    max_iter : int, default=1000
        Maximum iterations for convergence.
    name : str, optional
        Model name for identification.
    
    Attributes
    ----------
    model : sklearn estimator
        The underlying linear model
    scaler : StandardScaler or RobustScaler or None
        Feature scaler
    poly : PolynomialFeatures or None
        Polynomial feature transformer
    feature_names : list
        Names of input features
    coef_ : np.ndarray
        Learned coefficients (after fitting)
    intercept_ : float
        Learned intercept (after fitting)
    is_fitted : bool
        Whether the model has been trained
    
    Examples
    --------
    >>> model = LinearRegressionModel(alpha=0.1, l1_ratio=0.5)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> metrics = model.evaluate(X_test, y_test)
    >>> model.save('models/linear_home.pkl')
    """
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'fit_intercept': True,
        'scaling': 'standard',
        'poly_degree': 1,
        'max_iter': 1000,
        'solver': 'auto',
    }
    
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        scaling: Optional[str] = 'standard',
        poly_degree: int = 1,
        max_iter: int = 1000,
        solver: str = 'auto',
        name: Optional[str] = None
    ):
        self.name = name or f"linear_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store parameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.scaling = scaling
        self.poly_degree = poly_degree
        self.max_iter = max_iter
        self.solver = solver
        
        # Build model based on regularization type
        self.model = self._create_model()
        
        # Scaler
        if scaling == 'standard':
            self.scaler: Optional[StandardScaler] = StandardScaler()
        elif scaling == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        # Polynomial features
        self.poly: Optional[PolynomialFeatures] = None
        if poly_degree > 1:
            self.poly = PolynomialFeatures(
                degree=poly_degree,
                include_bias=False
            )
        
        # Feature tracking
        self.feature_names: Optional[List[str]] = None
        self.poly_feature_names: Optional[List[str]] = None
        self.n_features_: int = 0
        self.n_features_poly_: int = 0
        
        # Coefficients
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        
        # State
        self.is_fitted: bool = False
        
        # Metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_name': self.name,
            'model_type': self._get_model_type(),
        }
        
        logger.debug(f"Initialized LinearRegressionModel: {self.name}")
    
    def _create_model(self):
        """Create the appropriate sklearn model based on parameters."""
        if self.alpha == 0:
            # No regularization
            return LinearRegression(fit_intercept=self.fit_intercept)
        elif self.l1_ratio == 0:
            # Pure Ridge (L2)
            return Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                solver='auto' if self.solver == 'auto' else self.solver
            )
        elif self.l1_ratio == 1.0:
            # Pure Lasso (L1)
            return Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter
            )
        else:
            # ElasticNet (L1 + L2)
            return ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter
            )
    
    def _get_model_type(self) -> str:
        """Get human-readable model type."""
        if self.alpha == 0:
            return 'OLS'
        elif self.l1_ratio == 0:
            return 'Ridge'
        elif self.l1_ratio == 1.0:
            return 'Lasso'
        else:
            return 'ElasticNet'
    
    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        fit: bool = False
    ) -> np.ndarray:
        """
        Prepare features for training/prediction.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features
        fit : bool, default=False
            Whether to fit transformers (only True during training)
        
        Returns
        -------
        np.ndarray
            Prepared feature matrix
        """
        # Extract feature names from DataFrame
        if isinstance(X, pd.DataFrame):
            if fit or self.feature_names is None:
                self.feature_names = list(X.columns)
            X = X.values
        
        # Handle NaN values
        if np.isnan(X).any():
            logger.warning("NaN values detected in features, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        X = X.astype(np.float64)
        self.n_features_ = X.shape[1]
        
        # Apply polynomial features
        if self.poly is not None:
            if fit:
                X = self.poly.fit_transform(X)
                if self.feature_names:
                    self.poly_feature_names = self.poly.get_feature_names_out(
                        self.feature_names
                    ).tolist()
            else:
                X = self.poly.transform(X)
            self.n_features_poly_ = X.shape[1]
        else:
            self.n_features_poly_ = self.n_features_
        
        # Apply scaling
        if self.scaler is not None:
            if fit:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        return X
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> 'LinearRegressionModel':
        """
        Train the linear regression model.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : pd.Series or np.ndarray
            Training targets
        sample_weight : np.ndarray, optional
            Sample weights for training
        
        Returns
        -------
        LinearRegressionModel
            Fitted model instance (self)
        """
        logger.info(f"Training {self.name} ({self._get_model_type()})...")
        
        # Prepare training data
        X_train = self._prepare_features(X, fit=True)
        y_train = np.asarray(y).ravel().astype(np.float64)
        
        # Fit model
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        
        # Store coefficients
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # Update metadata
        self.metadata['fitted_at'] = datetime.now().isoformat()
        self.metadata['n_samples'] = len(y_train)
        self.metadata['n_features'] = self.n_features_
        self.metadata['n_features_poly'] = self.n_features_poly_
        
        self.is_fitted = True
        
        logger.info(f"Training complete. R² on training data: {self.model.score(X_train, y_train):.4f}")
        return self
    
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
        
        X_prep = self._prepare_features(X, fit=False)
        predictions = self.model.predict(X_prep)
        
        if clip_negative:
            predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_interval(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_train: Optional[np.ndarray] = None,
        X_train: Optional[np.ndarray] = None,
        confidence: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Predict with confidence intervals.
        
        For linear regression, we estimate the prediction interval using
        the residual standard error from training data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
        y_train : np.ndarray, optional
            Training targets (for variance estimation)
        X_train : np.ndarray, optional
            Training features (for variance estimation)
        confidence : float, default=0.95
            Confidence level for intervals
        
        Returns
        -------
        dict
            Dictionary with 'mean', 'lower', 'upper' arrays
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        predictions = self.predict(X, clip_negative=False)
        
        # Estimate standard error
        if y_train is not None and X_train is not None:
            train_pred = self.predict(X_train, clip_negative=False)
            residuals = y_train - train_pred
            se = np.std(residuals)
        else:
            # Use a reasonable default based on typical hockey game scores
            se = 1.0
        
        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        return {
            'mean': predictions,
            'lower': np.maximum(predictions - z * se, 0),
            'upper': predictions + z * se,
            'std': np.full_like(predictions, se)
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
        X_prep = self._prepare_features(X, fit=True)
        y_prep = np.asarray(y).ravel()
        
        # Create fresh model for CV
        cv_model = self._create_model()
        
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
    
    def get_coefficients(
        self,
        sort_by_abs: bool = True,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get model coefficients with feature names.
        
        Parameters
        ----------
        sort_by_abs : bool, default=True
            Sort by absolute value
        top_n : int, optional
            Return only top N coefficients
        
        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and coefficients
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Get feature names
        if self.poly_feature_names:
            names = self.poly_feature_names
        elif self.feature_names:
            names = self.feature_names
        else:
            names = [f'feature_{i}' for i in range(len(self.coef_))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': names,
            'coefficient': self.coef_,
            'abs_coefficient': np.abs(self.coef_)
        })
        
        if sort_by_abs:
            df = df.sort_values('abs_coefficient', ascending=False)
        
        if top_n:
            df = df.head(top_n)
        
        return df.reset_index(drop=True)
    
    def get_feature_importance(
        self,
        top_n: Optional[int] = None
    ) -> pd.Series:
        """
        Get feature importance (absolute coefficients).
        
        For linear models, importance is the absolute value of coefficients.
        
        Parameters
        ----------
        top_n : int, optional
            Return only top N features
        
        Returns
        -------
        pd.Series
            Feature importance scores
        """
        df = self.get_coefficients(sort_by_abs=True, top_n=top_n)
        return pd.Series(
            df['abs_coefficient'].values,
            index=df['feature'].values,
            name='importance'
        )
    
    def get_nonzero_features(self) -> List[str]:
        """
        Get list of features with non-zero coefficients.
        
        Useful for Lasso/ElasticNet which do feature selection.
        
        Returns
        -------
        list
            Feature names with non-zero coefficients
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        df = self.get_coefficients(sort_by_abs=True)
        return df[df['coefficient'] != 0]['feature'].tolist()
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save complete model state to disk.
        
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
            'model': self.model,
            'scaler': self.scaler,
            'poly': self.poly,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.fit_intercept,
            'scaling': self.scaling,
            'poly_degree': self.poly_degree,
            'max_iter': self.max_iter,
            'solver': self.solver,
            'feature_names': self.feature_names,
            'poly_feature_names': self.poly_feature_names,
            'n_features': self.n_features_,
            'n_features_poly': self.n_features_poly_,
            'coef': self.coef_,
            'intercept': self.intercept_,
            'metadata': self.metadata,
            'name': self.name,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'LinearRegressionModel':
        """
        Load a saved model from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved model
        
        Returns
        -------
        LinearRegressionModel
            Loaded model instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance
        instance = cls(
            alpha=state['alpha'],
            l1_ratio=state['l1_ratio'],
            fit_intercept=state['fit_intercept'],
            scaling=state['scaling'],
            poly_degree=state['poly_degree'],
            max_iter=state['max_iter'],
            solver=state.get('solver', 'auto'),
            name=state.get('name')
        )
        
        # Restore state
        instance.model = state['model']
        instance.scaler = state['scaler']
        instance.poly = state['poly']
        instance.feature_names = state['feature_names']
        instance.poly_feature_names = state['poly_feature_names']
        instance.n_features_ = state['n_features']
        instance.n_features_poly_ = state['n_features_poly']
        instance.coef_ = state['coef']
        instance.intercept_ = state['intercept']
        instance.metadata = state['metadata']
        
        instance.is_fitted = True
        instance.metadata['loaded_at'] = datetime.now().isoformat()
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.fit_intercept,
            'scaling': self.scaling,
            'poly_degree': self.poly_degree,
            'max_iter': self.max_iter,
            'solver': self.solver,
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        model_type = self._get_model_type()
        return f"LinearRegressionModel(type='{model_type}', alpha={self.alpha}, status={status})"


class LinearGoalPredictor:
    """
    Dual linear regression model for predicting both home and away goals.
    
    Uses two separate linear models internally - one for home goals,
    one for away goals. Provides unified interface consistent with
    EloModel, XGBoostGoalPredictor, and other predictors.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter
    scaling : str, default='standard'
        Feature scaling method
    poly_degree : int, default=1
        Polynomial feature degree
    home_params : dict, optional
        Override params for home model
    away_params : dict, optional
        Override params for away model
    
    Attributes
    ----------
    home_model : LinearRegressionModel
        Model for predicting home team goals
    away_model : LinearRegressionModel
        Model for predicting away team goals
    feature_columns : list
        Feature columns used for prediction
    
    Examples
    --------
    >>> predictor = LinearGoalPredictor(alpha=0.1, l1_ratio=0.5)
    >>> predictor.fit(train_df)
    >>> home_pred, away_pred = predictor.predict_goals(game)
    >>> metrics = predictor.evaluate(test_df)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        scaling: str = 'standard',
        poly_degree: int = 1,
        max_iter: int = 1000,
        home_params: Optional[Dict[str, Any]] = None,
        away_params: Optional[Dict[str, Any]] = None
    ):
        self.base_params = {
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'scaling': scaling,
            'poly_degree': poly_degree,
            'max_iter': max_iter,
        }
        
        # Merge params for each model
        home_p = {**self.base_params, **(home_params or {})}
        away_p = {**self.base_params, **(away_params or {})}
        
        self.home_model = LinearRegressionModel(**home_p, name='linear_home_goals')
        self.away_model = LinearRegressionModel(**away_p, name='linear_away_goals')
        
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted: bool = False
        
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': 'LinearGoalPredictor',
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
        verbose: bool = False
    ) -> 'LinearGoalPredictor':
        """
        Train both models on game data.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            Training data with features and target columns
        feature_columns : list, optional
            Specific columns to use as features. Auto-detected if None.
        verbose : bool, default=False
            Print training progress
        
        Returns
        -------
        LinearGoalPredictor
            Fitted predictor (self)
        """
        logger.info("Training LinearGoalPredictor...")
        
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
        
        # Fit both models
        self.home_model.fit(X, y_home)
        self.away_model.fit(X, y_away)
        
        self.is_fitted = True
        self.metadata['fitted_at'] = datetime.now().isoformat()
        self.metadata['n_games'] = len(games_df)
        self.metadata['n_features'] = len(self.feature_columns)
        
        logger.info("Training complete")
        return self
    
    def predict_goals(
        self,
        game: Union[Dict, pd.Series]
    ) -> Tuple[float, float]:
        """
        Predict goals for a single game.
        
        Parameters
        ----------
        game : dict or pd.Series
            Game with feature values
        
        Returns
        -------
        tuple
            (home_goals_pred, away_goals_pred)
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
        
        home_pred = self.home_model.predict(X)[0]
        away_pred = self.away_model.predict(X)[0]
        
        return float(home_pred), float(away_pred)
    
    def predict_batch(
        self,
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict goals for multiple games.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            Games with feature values
        
        Returns
        -------
        pd.DataFrame
            DataFrame with home_pred and away_pred columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = games_df[self.feature_columns]
        
        return pd.DataFrame({
            'home_pred': self.home_model.predict(X),
            'away_pred': self.away_model.predict(X),
        }, index=games_df.index)
    
    def predict_winner(
        self,
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict game winners.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            Games with feature values
        
        Returns
        -------
        pd.DataFrame
            DataFrame with winner predictions
        """
        predictions = self.predict_batch(games_df)
        
        goal_diff = predictions['home_pred'] - predictions['away_pred']
        
        predictions['predicted_winner'] = np.where(goal_diff > 0, 'home', 'away')
        predictions['goal_difference'] = goal_diff
        predictions['confidence'] = np.abs(goal_diff) / (np.abs(goal_diff) + 1)
        
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
    
    def get_coefficients(self, target: str = 'home') -> pd.DataFrame:
        """
        Get coefficients from specified model.
        
        Parameters
        ----------
        target : str, default='home'
            Which model: 'home' or 'away'
        
        Returns
        -------
        pd.DataFrame
            Coefficients DataFrame
        """
        if target == 'home':
            return self.home_model.get_coefficients()
        else:
            return self.away_model.get_coefficients()
    
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
            return self.home_model.get_feature_importance(top_n)
        elif target == 'away':
            return self.away_model.get_feature_importance(top_n)
        else:
            home_imp = self.home_model.get_feature_importance()
            away_imp = self.away_model.get_feature_importance()
            combined = (home_imp + away_imp.reindex(home_imp.index).fillna(0)) / 2
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
            'metadata': self.metadata,
        }
        
        with open(dirpath / 'predictor_meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        
        logger.info(f"Predictor saved to {dirpath}")
    
    @classmethod
    def load(cls, dirpath: Union[str, Path]) -> 'LinearGoalPredictor':
        """
        Load a saved predictor.
        
        Parameters
        ----------
        dirpath : str or Path
            Directory containing saved models
        
        Returns
        -------
        LinearGoalPredictor
            Loaded predictor
        """
        dirpath = Path(dirpath)
        
        # Load metadata
        with open(dirpath / 'predictor_meta.json', 'r') as f:
            meta = json.load(f)
        
        # Create instance
        instance = cls(**meta['base_params'])
        
        # Load models
        instance.home_model = LinearRegressionModel.load(dirpath / 'home_model.pkl')
        instance.away_model = LinearRegressionModel.load(dirpath / 'away_model.pkl')
        instance.feature_columns = meta['feature_columns']
        instance.metadata = meta['metadata']
        instance.is_fitted = True
        
        logger.info(f"Predictor loaded from {dirpath}")
        return instance
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_feat = len(self.feature_columns) if self.feature_columns else 0
        return f"LinearGoalPredictor(status={status}, n_features={n_feat})"


# =============================================================================
# Hyperparameter Search Functions
# =============================================================================

def grid_search_linear(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Optional[Dict[str, List]] = None,
    cv: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Exhaustive grid search for linear regression hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training targets
    param_grid : dict, optional
        Parameter grid {param_name: [values]}. Uses defaults if None.
    cv : int, default=5
        Cross-validation folds
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    dict
        {'best_params', 'best_score', 'all_results'}
    """
    if param_grid is None:
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.0, 0.5, 1.0],
            'poly_degree': [1, 2],
            'scaling': ['standard', 'robust'],
        }
    
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
            model = LinearRegressionModel(**params)
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


def random_search_linear(
    X: pd.DataFrame,
    y: pd.Series,
    param_distributions: Optional[Dict[str, List]] = None,
    n_iter: int = 50,
    cv: int = 5,
    verbose: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Randomized search for linear regression hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training targets
    param_distributions : dict, optional
        Parameter distributions {param_name: [values]}. Uses defaults if None.
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
    if param_distributions is None:
        param_distributions = {
            'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'poly_degree': [1, 2],
            'scaling': ['standard', 'robust'],
            'max_iter': [1000, 5000],
        }
    
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
            model = LinearRegressionModel(**params)
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


def compare_regularization(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: Optional[List[float]] = None,
    cv: int = 5
) -> pd.DataFrame:
    """
    Compare Ridge, Lasso, and ElasticNet across different alpha values.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Targets
    alphas : list, optional
        Alpha values to test
    cv : int, default=5
        Cross-validation folds
    
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    results = []
    
    # Test each regularization type
    for alpha in alphas:
        for l1_ratio, name in [(0.0, 'Ridge'), (0.5, 'ElasticNet'), (1.0, 'Lasso')]:
            model = LinearRegressionModel(alpha=alpha, l1_ratio=l1_ratio)
            cv_result = model.cross_validate(X, y, cv=cv)
            
            results.append({
                'model': name,
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'rmse_mean': cv_result['mean'],
                'rmse_std': cv_result['std'],
            })
    
    return pd.DataFrame(results).sort_values('rmse_mean')
