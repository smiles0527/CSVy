"""
XGBoost Model

This module provides an XGBoost-based model for predicting hockey game outcomes.
Includes feature engineering, hyperparameter tuning, and early stopping support.

Features:
    - Automatic feature preparation
    - Built-in cross-validation
    - Early stopping support
    - Feature importance analysis
    - SHAP value integration (optional)
    - Same interface as other models

Usage:
    from utils.xgboost_model import XGBoostModel
    
    model = XGBoostModel(params={'max_depth': 6, 'learning_rate': 0.05})
    model.fit(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Column name mappings - consistent with other models
COLUMN_ALIASES = {
    'home_team': ['home_team', 'home', 'team_home', 'h_team'],
    'away_team': ['away_team', 'away', 'team_away', 'a_team', 'visitor', 'visiting_team'],
    'home_goals': ['home_goals', 'home_score', 'h_goals', 'goals_home', 'home_pts'],
    'away_goals': ['away_goals', 'away_score', 'a_goals', 'goals_away', 'away_pts', 'visitor_goals'],
    'game_date': ['game_date', 'date', 'Date', 'game_datetime', 'datetime', 'game_time'],
}


def get_column(df, field):
    """Find the correct column name in a DataFrame."""
    aliases = COLUMN_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


class XGBoostModel:
    """
    XGBoost regression model for hockey goal prediction.
    
    This model uses gradient boosting to predict game outcomes based on
    engineered features like team statistics, recent form, and contextual factors.
    
    Attributes
    ----------
    params : dict
        XGBoost hyperparameters
    model : xgb.XGBRegressor
        The trained XGBoost model
    scaler : StandardScaler
        Feature scaler (optional)
    feature_names : list
        Names of features used in training
    feature_importances_ : pd.Series
        Feature importance scores
    
    Examples
    --------
    >>> model = XGBoostModel({'max_depth': 6, 'learning_rate': 0.05})
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """
    
    # Default hyperparameters (from config/hyperparams/model4_xgboost.yaml)
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
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
    }
    
    def __init__(self, params=None, scale_features=False):
        """
        Initialize XGBoost model.
        
        Parameters
        ----------
        params : dict, optional
            XGBoost hyperparameters. Missing keys use defaults.
        scale_features : bool, default=False
            Whether to standardize features before training.
            XGBoost doesn't require scaling, but it can help
            with interpretation.
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.model = None
        self.feature_names = None
        self.feature_importances_ = None
        self.is_fitted = False
        self.training_history = None
    
    def _prepare_features(self, X, fit_scaler=False):
        """
        Prepare features for training/prediction.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features
        fit_scaler : bool
            Whether to fit the scaler (True during training)
        
        Returns
        -------
        np.ndarray
            Prepared features
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        if self.scale_features and self.scaler is not None:
            if fit_scaler:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        return X
    
    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=50, verbose=False):
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
            Stop if no improvement for this many rounds
        verbose : bool, default=False
            Print training progress
        
        Returns
        -------
        self
            Fitted model instance
        """
        X_train = self._prepare_features(X, fit_scaler=True)
        y_train = np.array(y).ravel()
        
        # Create model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Fit with or without validation set
        if X_val is not None and y_val is not None:
            X_val_prep = self._prepare_features(X_val, fit_scaler=False)
            y_val_prep = np.array(y_val).ravel()
            
            # Pass early_stopping_rounds so training actually stops early
            fit_kwargs = {
                'eval_set': [(X_train, y_train), (X_val_prep, y_val_prep)],
                'verbose': verbose,
            }
            if early_stopping_rounds and early_stopping_rounds > 0:
                fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
            
            self.model.fit(X_train, y_train, **fit_kwargs)
            
            # Store training history
            self.training_history = self.model.evals_result()
        else:
            self.model.fit(X_train, y_train, verbose=verbose)
        
        # Store feature importances
        if self.feature_names:
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        else:
            self.feature_importances_ = pd.Series(self.model.feature_importances_)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict target values.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
        
        Returns
        -------
        np.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_prep = self._prepare_features(X, fit_scaler=False)
        return self.model.predict(X_prep)
    
    def evaluate(self, X, y):
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
            Dictionary with RMSE, MAE, R² metrics
        """
        predictions = self.predict(X)
        y_true = np.array(y).ravel()
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
            'mae': mean_absolute_error(y_true, predictions),
            'r2': r2_score(y_true, predictions),
            'n_samples': len(y_true)
        }
    
    def cross_validate(self, X, y, cv=5, scoring='neg_root_mean_squared_error'):
        """
        Perform cross-validation with proper scaling per fold (no data leakage).
        
        Uses a Pipeline(scaler, model) so the scaler is fit only on each
        training fold, not on the full dataset.
        
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
            Cross-validation results with mean and std
        """
        from sklearn.pipeline import Pipeline
        
        # Extract numpy but do NOT fit the scaler on all data
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.array(X)
        y_prep = np.array(y).ravel()
        
        # Build a pipeline so scaler fits per-fold (no leakage)
        steps = []
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', xgb.XGBRegressor(**self.params)))
        cv_pipeline = Pipeline(steps)
        
        scores = cross_val_score(cv_pipeline, X_arr, y_prep, cv=cv, scoring=scoring)
        
        # Negate scores if using neg_ metrics
        if scoring.startswith('neg_'):
            scores = -scores
        
        return {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'scores': scores.tolist(),
            'cv_folds': cv
        }
    
    def get_feature_importance(self, importance_type='gain', top_n=None):
        """
        Get feature importance scores.
        
        Parameters
        ----------
        importance_type : str, default='gain'
            Type of importance: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
        top_n : int, optional
            Return only top N features
        
        Returns
        -------
        pd.Series
            Feature importance scores, sorted descending
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Get importance with specified type
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to Series
        if self.feature_names:
            # Map feature indices to names
            importance_series = pd.Series(index=self.feature_names, dtype=float)
            for feat, score in importance.items():
                # Handle both 'f0', 'f1' format and feature names
                if feat.startswith('f') and feat[1:].isdigit():
                    idx = int(feat[1:])
                    if idx < len(self.feature_names):
                        importance_series[self.feature_names[idx]] = score
                elif feat in self.feature_names:
                    importance_series[feat] = score
            importance_series = importance_series.fillna(0).sort_values(ascending=False)
        else:
            importance_series = pd.Series(importance).sort_values(ascending=False)
        
        if top_n:
            return importance_series.head(top_n)
        return importance_series
    
    def get_shap_values(self, X, max_samples=1000):
        """
        Calculate SHAP values for feature interpretation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features to explain
        max_samples : int, default=1000
            Maximum samples to use (SHAP can be slow)
        
        Returns
        -------
        shap.Explanation or None
            SHAP values if shap is installed
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP not installed. Install with: pip install shap")
            return None
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        X_prep = self._prepare_features(X, fit_scaler=False)
        
        # Subsample if too large
        if len(X_prep) > max_samples:
            indices = np.random.choice(len(X_prep), max_samples, replace=False)
            X_prep = X_prep[indices]
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X_prep)
        
        return shap_values
    
    def save_model(self, filepath):
        """
        Save the full model state to disk (model + scaler + feature names + params).
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the model (uses pickle)
        """
        import pickle
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        if not str(filepath).endswith('.pkl'):
            filepath = Path(str(filepath) + '.pkl')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'params': self.params,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importances_': getattr(self, 'feature_importances_', None),
            'training_history': getattr(self, 'training_history', None),
            'use_scaler': self.use_scaler,
            'is_fitted': True,
        }
        
        # Save XGBoost model separately then bundle
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            self.model.save_model(tmp.name)
            with open(tmp.name, 'r') as f:
                state['xgb_model_json'] = f.read()
            os.unlink(tmp.name)
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"XGBoost model saved to {filepath} ({len(self.feature_names or [])} features)")
        return str(filepath)
    
    def load_model(self, filepath):
        """
        Load a full model state from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved model
        
        Returns
        -------
        self
        """
        import pickle
        
        filepath = Path(filepath)
        if not filepath.exists() and not Path(str(filepath) + '.pkl').exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        if not filepath.exists():
            filepath = Path(str(filepath) + '.pkl')
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.params = state.get('params', self.params)
        self.scaler = state.get('scaler', self.scaler)
        self.feature_names = state.get('feature_names', None)
        self.feature_importances_ = state.get('feature_importances_', None)
        self.training_history = state.get('training_history', None)
        self.use_scaler = state.get('use_scaler', False)
        self.is_fitted = True
        
        # Restore XGBoost model from JSON
        import tempfile, os
        self.model = xgb.XGBRegressor(**self.params)
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as tmp:
            tmp.write(state['xgb_model_json'])
            tmp.flush()
            self.model.load_model(tmp.name)
        os.unlink(tmp.name)
        
        print(f"XGBoost model loaded ({len(self.feature_names or [])} features)")
        return self
    
    def get_params(self):
        """Get current hyperparameters."""
        return self.params.copy()
    
    def set_params(self, **params):
        """Set hyperparameters."""
        self.params.update(params)
        return self


class XGBoostGoalPredictor:
    """
    High-level wrapper for predicting both home and away goals.
    
    Uses two separate XGBoost models - one for home goals, one for away goals.
    Follows the same interface as EloModel and BaselineModel.
    
    Attributes
    ----------
    home_model : XGBoostModel
        Model for predicting home team goals
    away_model : XGBoostModel
        Model for predicting away team goals
    
    Examples
    --------
    >>> predictor = XGBoostGoalPredictor()
    >>> predictor.fit(train_df)
    >>> home_pred, away_pred = predictor.predict_goals(game)
    """
    
    def __init__(self, params=None, scale_features=False):
        """
        Initialize goal predictor with two XGBoost models.
        
        Parameters
        ----------
        params : dict, optional
            XGBoost hyperparameters (applied to both models)
        scale_features : bool, default=False
            Whether to scale features
        """
        self.params = params or {}
        self.scale_features = scale_features
        self.home_model = XGBoostModel(params, scale_features)
        self.away_model = XGBoostModel(params, scale_features)
        self.feature_columns = None
        self.is_fitted = False
    
    def _extract_features(self, df):
        """
        Extract feature columns from dataframe.
        
        Excludes target columns and identifiers.
        """
        exclude_cols = {
            'home_goals', 'away_goals', 'home_score', 'away_score',
            'home_team', 'away_team', 'game_date', 'date', 'game_id'
        }
        
        # Get numeric columns only
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols 
            and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]
        
        return feature_cols
    
    def fit(self, games_df, X_val=None, y_home_val=None, y_away_val=None):
        """
        Train models on game data.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            DataFrame with features and target columns (home_goals, away_goals)
        X_val : pd.DataFrame, optional
            Validation features
        y_home_val : pd.Series, optional
            Validation home goals
        y_away_val : pd.Series, optional
            Validation away goals
        
        Returns
        -------
        self
        """
        # Identify feature columns
        self.feature_columns = self._extract_features(games_df)
        
        if len(self.feature_columns) == 0:
            raise ValueError("No numeric feature columns found in games_df")
        
        # Get targets
        home_col = get_column(games_df, 'home_goals')
        away_col = get_column(games_df, 'away_goals')
        
        if home_col is None or away_col is None:
            raise ValueError("games_df must have home_goals and away_goals columns")
        
        X = games_df[self.feature_columns]
        y_home = games_df[home_col]
        y_away = games_df[away_col]
        
        # Fit both models
        self.home_model.fit(X, y_home, X_val, y_home_val)
        self.away_model.fit(X, y_away, X_val, y_away_val)
        
        self.is_fitted = True
        return self
    
    def predict_goals(self, game):
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
        
        # Convert to DataFrame for prediction
        if isinstance(game, dict):
            game = pd.Series(game)
        
        X = game[self.feature_columns].values.reshape(1, -1)
        
        home_pred = self.home_model.predict(X)[0]
        away_pred = self.away_model.predict(X)[0]
        
        return float(home_pred), float(away_pred)
    
    def predict_batch(self, games_df):
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
            'away_pred': self.away_model.predict(X)
        })
    
    def evaluate(self, games_df):
        """
        Evaluate model on test set.
        
        Parameters
        ----------
        games_df : pd.DataFrame
            Test games with features and targets
        
        Returns
        -------
        dict
            Dictionary with metrics for home, away, and combined
        """
        home_col = get_column(games_df, 'home_goals')
        away_col = get_column(games_df, 'away_goals')
        
        X = games_df[self.feature_columns]
        
        home_metrics = self.home_model.evaluate(X, games_df[home_col])
        away_metrics = self.away_model.evaluate(X, games_df[away_col])
        
        # Combined metrics
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        all_preds = np.concatenate([home_pred, away_pred])
        all_actual = np.concatenate([games_df[home_col].values, games_df[away_col].values])
        
        combined_metrics = {
            'rmse': np.sqrt(mean_squared_error(all_actual, all_preds)),
            'mae': mean_absolute_error(all_actual, all_preds),
            'r2': r2_score(all_actual, all_preds),
        }
        
        return {
            'home': home_metrics,
            'away': away_metrics,
            'combined': combined_metrics,
            'n_games': len(games_df)
        }
    
    def get_feature_importance(self, target='combined', top_n=20):
        """
        Get feature importance from both models.
        
        Parameters
        ----------
        target : str, default='combined'
            Which model: 'home', 'away', or 'combined' (average)
        top_n : int, default=20
            Number of top features to return
        
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
        else:  # combined
            combined = (
                self.home_model.feature_importances_ + 
                self.away_model.feature_importances_
            ) / 2
            return combined.sort_values(ascending=False).head(top_n)


def grid_search_xgboost(X, y, param_grid, cv=5, verbose=True):
    """
    Perform grid search for XGBoost hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training targets
    param_grid : dict
        Dictionary with parameter names as keys and lists of values
    cv : int, default=5
        Number of cross-validation folds
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    dict
        Best parameters and all results
    """
    from itertools import product
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    results = []
    best_score = float('inf')
    best_params = None
    
    if verbose:
        print(f"Testing {len(combinations)} parameter combinations...")
    
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
                best_params = params
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(combinations)}, best RMSE: {best_score:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  Error with {params}: {e}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': pd.DataFrame(results).sort_values('rmse_mean')
    }


def random_search_xgboost(X, y, param_distributions, n_iter=50, cv=5, verbose=True):
    """
    Perform random search for XGBoost hyperparameters.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training targets
    param_distributions : dict
        Dictionary with parameter names as keys and value ranges/lists
    n_iter : int, default=50
        Number of parameter combinations to try
    cv : int, default=5
        Number of cross-validation folds
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    dict
        Best parameters and all results
    """
    results = []
    best_score = float('inf')
    best_params = None
    
    if verbose:
        print(f"Testing {n_iter} random parameter combinations...")
    
    for i in range(n_iter):
        # Sample random parameters
        params = {}
        for key, values in param_distributions.items():
            if isinstance(values, (list, tuple)):
                params[key] = np.random.choice(values)
            else:
                params[key] = values
        
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
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_iter}, best RMSE: {best_score:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  Error with {params}: {e}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': pd.DataFrame(results).sort_values('rmse_mean')
    }


if __name__ == "__main__":
    # Demo usage
    print("XGBoost Model - Demo")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic features
    X = pd.DataFrame({
        'home_win_pct': np.random.uniform(0.3, 0.7, n_samples),
        'away_win_pct': np.random.uniform(0.3, 0.7, n_samples),
        'home_goals_avg': np.random.uniform(2, 4, n_samples),
        'away_goals_avg': np.random.uniform(2, 4, n_samples),
        'home_goals_against_avg': np.random.uniform(2, 4, n_samples),
        'away_goals_against_avg': np.random.uniform(2, 4, n_samples),
        'home_rest_days': np.random.randint(1, 5, n_samples),
        'away_rest_days': np.random.randint(1, 5, n_samples),
    })
    
    # Generate target (home goals)
    y = (
        X['home_goals_avg'] * 0.5 +
        (1 - X['away_goals_against_avg'] / 4) * 2 +
        X['home_win_pct'] * 2 +
        np.random.normal(0, 0.5, n_samples)
    ).clip(0, 10)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    model = XGBoostModel({'max_depth': 4, 'n_estimators': 100})
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    # Feature importance
    print(f"\nTop 5 Features:")
    for feat, imp in model.feature_importances_.head(5).items():
        print(f"  {feat}: {imp:.4f}")
    
    print("\n XGBoost Model demo complete!")
