"""
Experiment Tracker - Comprehensive ML Experiment Tracking with MLflow

This module provides a unified interface for tracking experiments, logging
metrics, parameters, and artifacts during model training.

Features:
    - MLflow integration for experiment tracking
    - Automatic parameter logging
    - Real-time metric logging with history
    - Model artifact storage
    - Experiment comparison
    - Training run management

Usage:
    from utils.experiment_tracker import ExperimentTracker
    
    tracker = ExperimentTracker(experiment_name="hockey_prediction")
    
    with tracker.start_run(run_name="xgboost_v1"):
        tracker.log_params({"learning_rate": 0.05, "n_estimators": 500})
        
        for epoch in range(100):
            # Training...
            tracker.log_metrics({"loss": loss, "rmse": rmse}, step=epoch)
        
        tracker.log_model(model, "xgboost_model")
        tracker.log_artifact("feature_importance.png")
"""

import os
import json
import logging
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Run: pip install mlflow")


class MetricHistory:
    """Stores metric history for a training run."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    def add(self, name: str, value: float, step: Optional[int] = None):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'step': step if step is not None else len(self.metrics[name]),
            'timestamp': datetime.now().isoformat()
        })
    
    def get(self, name: str) -> List[float]:
        """Get all values for a metric."""
        if name not in self.metrics:
            return []
        return [m['value'] for m in self.metrics[name]]
    
    def get_steps(self, name: str) -> List[int]:
        """Get all steps for a metric."""
        if name not in self.metrics:
            return []
        return [m['step'] for m in self.metrics[name]]
    
    def get_last(self, name: str) -> Optional[float]:
        """Get the last value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1]['value']
    
    def get_best(self, name: str, mode: str = 'min') -> Optional[float]:
        """Get the best value for a metric."""
        values = self.get(name)
        if not values:
            return None
        return min(values) if mode == 'min' else max(values)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all metrics to a DataFrame."""
        records = []
        for name, history in self.metrics.items():
            for entry in history:
                records.append({
                    'metric': name,
                    'value': entry['value'],
                    'step': entry['step'],
                    'timestamp': entry['timestamp']
                })
        return pd.DataFrame(records)


class ExperimentTracker:
    """
    Comprehensive experiment tracking with MLflow backend.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment (groups related runs).
    tracking_uri : str, optional
        MLflow tracking URI. Defaults to local './mlruns'.
    artifact_location : str, optional
        Location for storing artifacts.
    tags : dict, optional
        Default tags for all runs.
    
    Attributes
    ----------
    experiment_name : str
        Current experiment name.
    current_run : mlflow.ActiveRun
        The currently active run.
    metric_history : MetricHistory
        History of logged metrics.
    
    Examples
    --------
    >>> tracker = ExperimentTracker("hockey_models")
    >>> with tracker.start_run("xgboost_baseline"):
    ...     tracker.log_params({"lr": 0.05})
    ...     for epoch in range(10):
    ...         tracker.log_metric("loss", loss_value, step=epoch)
    ...     tracker.log_model(model, "model")
    """
    
    def __init__(
        self,
        experiment_name: str = "default",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "./mlruns"
        self.artifact_location = artifact_location
        self.default_tags = tags or {}
        
        self.current_run = None
        self.metric_history = MetricHistory()
        self._run_params = {}
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                self.experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
        else:
            self.experiment_id = None
            logger.warning("MLflow not available. Using local tracking only.")
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        Start a new tracking run.
        
        Parameters
        ----------
        run_name : str, optional
            Name for this run.
        tags : dict, optional
            Additional tags for this run.
        description : str, optional
            Description of the run.
        
        Yields
        ------
        ExperimentTracker
            The tracker instance for chaining.
        """
        self.metric_history = MetricHistory()
        self._run_params = {}
        
        run_tags = {**self.default_tags, **(tags or {})}
        if description:
            run_tags['mlflow.note.content'] = description
        
        if MLFLOW_AVAILABLE:
            self.current_run = mlflow.start_run(run_name=run_name, tags=run_tags)
            try:
                yield self
            finally:
                mlflow.end_run()
                self.current_run = None
        else:
            # Fallback: just track locally
            self._local_run = {
                'name': run_name,
                'tags': run_tags,
                'start_time': datetime.now().isoformat()
            }
            try:
                yield self
            finally:
                self._local_run['end_time'] = datetime.now().isoformat()
                self._save_local_run()
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._run_params[key] = value
        
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self._run_params.update(params)
        
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_params(params)
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log a single metric value.
        
        Parameters
        ----------
        key : str
            Metric name.
        value : float
            Metric value.
        step : int, optional
            Step number (epoch, iteration, etc.).
        """
        self.metric_history.add(key, value, step)
        
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_metric(key, value, step=step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.metric_history.add(key, value, step)
        
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        flavor: str = "sklearn"
    ) -> None:
        """
        Log a model as an artifact.
        
        Parameters
        ----------
        model : Any
            The model to log.
        artifact_path : str
            Path within the artifact store.
        flavor : str
            Model flavor: 'sklearn', 'xgboost', 'pytorch', 'keras'.
        """
        if MLFLOW_AVAILABLE and self.current_run:
            if flavor == 'sklearn':
                mlflow.sklearn.log_model(model, artifact_path)
            elif flavor == 'xgboost':
                try:
                    mlflow.xgboost.log_model(model, artifact_path)
                except:
                    mlflow.sklearn.log_model(model, artifact_path)
            elif flavor == 'pytorch':
                mlflow.pytorch.log_model(model, artifact_path)
            elif flavor == 'keras':
                mlflow.keras.log_model(model, artifact_path)
            else:
                # Fallback: pickle the model
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    pickle.dump(model, f)
                    mlflow.log_artifact(f.name, artifact_path)
                    os.unlink(f.name)
        else:
            # Local save
            output_dir = Path(self.tracking_uri) / "models"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / f"{artifact_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file or directory as an artifact."""
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_artifact(local_path, artifact_path)
        else:
            logger.info(f"Would log artifact: {local_path}")
    
    def log_figure(self, figure, artifact_name: str) -> None:
        """Log a matplotlib figure as an artifact."""
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_figure(figure, artifact_name)
        else:
            # Save locally
            output_dir = Path(self.tracking_uri) / "figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_dir / artifact_name)
    
    def log_dict(self, dictionary: Dict, artifact_name: str) -> None:
        """Log a dictionary as a JSON artifact."""
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_dict(dictionary, artifact_name)
        else:
            output_dir = Path(self.tracking_uri) / "artifacts"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / artifact_name, 'w') as f:
                json.dump(dictionary, f, indent=2, default=str)
    
    def log_dataframe(self, df: pd.DataFrame, artifact_name: str) -> None:
        """Log a DataFrame as a CSV artifact."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            df.to_csv(f, index=False)
            temp_path = f.name
        
        self.log_artifact(temp_path, artifact_name.replace('.csv', ''))
        os.unlink(temp_path)
    
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.set_tag(key, value)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags on the current run."""
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.set_tags(tags)
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get the history of a metric."""
        return self.metric_history.get(metric_name)
    
    def get_best_metric(self, metric_name: str, mode: str = 'min') -> Optional[float]:
        """Get the best value of a metric."""
        return self.metric_history.get_best(metric_name, mode)
    
    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self.current_run:
            return self.current_run.info.run_id
        return None
    
    def _save_local_run(self):
        """Save run data locally when MLflow is not available."""
        output_dir = Path(self.tracking_uri) / "local_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_data = {
            **self._local_run,
            'params': self._run_params,
            'metrics': self.metric_history.to_dataframe().to_dict('records')
        }
        
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_path = output_dir / f"run_{run_id}.json"
        
        with open(run_path, 'w') as f:
            json.dump(run_data, f, indent=2, default=str)
        
        logger.info(f"Local run saved to {run_path}")
    
    @staticmethod
    def get_best_run(
        experiment_name: str,
        metric: str = "rmse",
        mode: str = "min"
    ) -> Optional[Dict]:
        """
        Get the best run from an experiment.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        metric : str
            Metric to optimize.
        mode : str
            'min' or 'max'.
        
        Returns
        -------
        dict or None
            Best run info with params and metrics.
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available")
            return None
        
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            return None
        
        order = "ASC" if mode == "min" else "DESC"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )
        
        if not runs:
            return None
        
        best_run = runs[0]
        return {
            'run_id': best_run.info.run_id,
            'params': best_run.data.params,
            'metrics': best_run.data.metrics,
            'tags': best_run.data.tags,
        }
    
    @staticmethod
    def compare_runs(
        experiment_name: str,
        metric: str = "rmse",
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Compare top runs from an experiment.
        
        Returns DataFrame with params and metrics for comparison.
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available")
            return pd.DataFrame()
        
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            return pd.DataFrame()
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=top_n
        )
        
        records = []
        for run in runs:
            record = {
                'run_id': run.info.run_id[:8],
                'run_name': run.data.tags.get('mlflow.runName', 'unnamed'),
                **run.data.params,
                **{f"metric_{k}": v for k, v in run.data.metrics.items()}
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    @staticmethod
    def load_model(run_id: str, artifact_path: str = "model"):
        """Load a model from a run."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available")
            return None
        
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.sklearn.load_model(model_uri)
    
    @staticmethod
    def launch_ui(port: int = 5000):
        """Launch the MLflow tracking UI."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available")
            return
        
        import subprocess
        print(f"Starting MLflow UI at http://localhost:{port}")
        print("Press Ctrl+C to stop")
        subprocess.run(["mlflow", "ui", "--port", str(port)])


def create_tracker(experiment_name: str = "hockey_prediction") -> ExperimentTracker:
    """Create a default experiment tracker."""
    return ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri="./mlruns",
        tags={
            "project": "CSVy",
            "domain": "hockey_prediction"
        }
    )
