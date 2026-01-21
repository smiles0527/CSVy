"""
Training Callbacks - Live Progress & Metric Reporting During Training

This module provides callback systems for monitoring training progress
in real-time across different ML frameworks.

Features:
    - Progress bars with tqdm/rich
    - Live metric logging
    - Early stopping with tracking
    - Learning rate scheduling
    - Checkpoint saving
    - Integration with ExperimentTracker

Usage:
    from utils.training_callbacks import TrainingCallback, ProgressBar
    
    callback = TrainingCallback(tracker=tracker, verbose=True)
    
    # During training loop:
    for epoch in range(n_epochs):
        callback.on_epoch_start(epoch)
        # ... training ...
        callback.on_epoch_end(epoch, {'loss': loss, 'rmse': rmse})
"""

import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Try to import progress bar libraries
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class CallbackBase(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def on_train_start(self, total_epochs: int, **kwargs) -> None:
        """Called at the start of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, **kwargs) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, **kwargs) -> None:
        """Called at the start of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> bool:
        """Called at the end of each epoch. Returns True to continue, False to stop."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of each batch."""
        pass


class TrainingCallback(CallbackBase):
    """
    Comprehensive training callback with logging and tracking.
    
    Parameters
    ----------
    tracker : ExperimentTracker, optional
        Experiment tracker for MLflow logging.
    verbose : bool
        Whether to print progress updates.
    log_frequency : int
        How often to log metrics (in epochs).
    early_stopping_patience : int, optional
        Epochs to wait before early stopping.
    early_stopping_metric : str
        Metric to monitor for early stopping.
    early_stopping_mode : str
        'min' or 'max' for early stopping.
    
    Examples
    --------
    >>> callback = TrainingCallback(tracker, verbose=True, early_stopping_patience=10)
    >>> callback.on_train_start(total_epochs=100)
    >>> for epoch in range(100):
    ...     callback.on_epoch_start(epoch)
    ...     # ... training ...
    ...     should_continue = callback.on_epoch_end(epoch, {'loss': loss})
    ...     if not should_continue:
    ...         break
    >>> callback.on_train_end()
    """
    
    def __init__(
        self,
        tracker=None,
        verbose: bool = True,
        log_frequency: int = 1,
        early_stopping_patience: Optional[int] = None,
        early_stopping_metric: str = "val_loss",
        early_stopping_mode: str = "min",
        checkpoint_path: Optional[str] = None
    ):
        self.tracker = tracker
        self.verbose = verbose
        self.log_frequency = log_frequency
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.checkpoint_path = checkpoint_path
        
        # State
        self.total_epochs = 0
        self.current_epoch = 0
        self.start_time = None
        self.epoch_start_time = None
        
        # Metrics tracking
        self.metric_history: Dict[str, List[float]] = {}
        self.best_metric = None
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        
        # Progress bar
        self.pbar = None
    
    def on_train_start(self, total_epochs: int, **kwargs) -> None:
        """Initialize training."""
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_time = time.time()
        self.metric_history = {}
        self.best_metric = None
        self.epochs_without_improvement = 0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total epochs: {total_epochs}")
            print(f"{'='*60}\n")
        
        # Initialize progress bar
        if TQDM_AVAILABLE and self.verbose:
            self.pbar = tqdm(total=total_epochs, desc="Training", unit="epoch")
    
    def on_train_end(self, **kwargs) -> None:
        """Finalize training."""
        if self.pbar:
            self.pbar.close()
        
        total_time = time.time() - self.start_time
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
            print(f"Epochs completed: {self.current_epoch + 1}")
            
            if self.best_metric is not None:
                print(f"Best {self.early_stopping_metric}: {self.best_metric:.6f} (epoch {self.best_epoch})")
            
            print(f"{'='*60}\n")
        
        # Log final summary
        if self.tracker:
            self.tracker.log_params({
                "total_training_time": total_time,
                "epochs_trained": self.current_epoch + 1,
                "best_epoch": self.best_epoch
            })
            
            if self.best_metric is not None:
                self.tracker.log_metric(f"best_{self.early_stopping_metric}", self.best_metric)
    
    def on_epoch_start(self, epoch: int, **kwargs) -> None:
        """Start epoch tracking."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> bool:
        """
        Log metrics and check early stopping.
        
        Returns True to continue training, False to stop.
        """
        epoch_time = time.time() - self.epoch_start_time
        
        # Store metrics
        for name, value in metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = []
            self.metric_history[name].append(value)
        
        # Log to tracker
        if self.tracker and epoch % self.log_frequency == 0:
            self.tracker.log_metrics(metrics, step=epoch)
        
        # Update progress bar
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix(metrics)
        elif self.verbose and epoch % self.log_frequency == 0:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            print(f"Epoch {epoch+1}/{self.total_epochs} - {epoch_time:.2f}s - {metrics_str}")
        
        # Early stopping check
        if self.early_stopping_patience is not None:
            if self.early_stopping_metric in metrics:
                current_value = metrics[self.early_stopping_metric]
                
                if self.best_metric is None:
                    self.best_metric = current_value
                    self.best_epoch = epoch
                else:
                    improved = (
                        (self.early_stopping_mode == "min" and current_value < self.best_metric) or
                        (self.early_stopping_mode == "max" and current_value > self.best_metric)
                    )
                    
                    if improved:
                        self.best_metric = current_value
                        self.best_epoch = epoch
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                        
                        if self.epochs_without_improvement >= self.early_stopping_patience:
                            if self.verbose:
                                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                                print(f"Best {self.early_stopping_metric}: {self.best_metric:.6f}")
                            return False
        
        return True
    
    def on_batch_end(self, batch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Log batch-level metrics (optional)."""
        # For batch-level logging, only log every N batches to avoid overhead
        pass
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get the history of a specific metric."""
        return self.metric_history.get(metric_name, [])
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best value for each metric."""
        best = {}
        for name, values in self.metric_history.items():
            if 'loss' in name.lower() or 'error' in name.lower():
                best[name] = min(values)
            else:
                best[name] = max(values)
        return best


class ProgressBar:
    """
    Flexible progress bar wrapper supporting tqdm and rich.
    
    Parameters
    ----------
    total : int
        Total number of iterations.
    desc : str
        Description text.
    backend : str
        'auto', 'tqdm', 'rich', or 'simple'.
    
    Examples
    --------
    >>> with ProgressBar(100, "Training") as pbar:
    ...     for i in range(100):
    ...         # ... work ...
    ...         pbar.update(1, {'loss': loss})
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        backend: str = "auto"
    ):
        self.total = total
        self.desc = desc
        self.backend = self._select_backend(backend)
        
        self._pbar = None
        self._rich_progress = None
        self._task_id = None
        self._current = 0
    
    def _select_backend(self, backend: str) -> str:
        """Select the best available backend."""
        if backend == "auto":
            if RICH_AVAILABLE:
                return "rich"
            elif TQDM_AVAILABLE:
                return "tqdm"
            else:
                return "simple"
        return backend
    
    def __enter__(self):
        """Start the progress bar."""
        if self.backend == "tqdm" and TQDM_AVAILABLE:
            self._pbar = tqdm(total=self.total, desc=self.desc, unit="it")
        elif self.backend == "rich" and RICH_AVAILABLE:
            self._rich_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
            self._rich_progress.start()
            self._task_id = self._rich_progress.add_task(self.desc, total=self.total)
        else:
            print(f"{self.desc}: starting...")
        
        return self
    
    def __exit__(self, *args):
        """Close the progress bar."""
        if self._pbar:
            self._pbar.close()
        if self._rich_progress:
            self._rich_progress.stop()
        if self.backend == "simple":
            print(f"{self.desc}: complete!")
    
    def update(self, n: int = 1, metrics: Optional[Dict[str, float]] = None):
        """Update the progress bar."""
        self._current += n
        
        if self._pbar:
            self._pbar.update(n)
            if metrics:
                self._pbar.set_postfix(metrics)
        
        elif self._rich_progress and self._task_id is not None:
            self._rich_progress.update(self._task_id, advance=n)
            if metrics:
                desc = f"{self.desc} - " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                self._rich_progress.update(self._task_id, description=desc)
        
        elif self.backend == "simple" and self._current % max(1, self.total // 10) == 0:
            pct = 100 * self._current / self.total
            metrics_str = ""
            if metrics:
                metrics_str = " - " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"{self.desc}: {pct:.0f}%{metrics_str}")


class EarlyStopping:
    """
    Standalone early stopping monitor.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping.
    metric : str
        Name of the metric to monitor.
    mode : str
        'min' or 'max'.
    min_delta : float
        Minimum change to qualify as an improvement.
    
    Examples
    --------
    >>> early_stopping = EarlyStopping(patience=10, metric='val_loss', mode='min')
    >>> for epoch in range(1000):
    ...     # ... training ...
    ...     if early_stopping(metrics['val_loss']):
    ...         print("Early stopping triggered!")
    ...         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = None
    
    def __call__(self, current_value: float, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Returns True if training should stop.
        """
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            return False
        
        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        
        return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_value = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = None


class LearningRateScheduler:
    """
    Learning rate scheduler with various decay strategies.
    
    Parameters
    ----------
    initial_lr : float
        Starting learning rate.
    schedule : str
        'constant', 'step', 'exponential', 'cosine', 'warmup'.
    **kwargs
        Schedule-specific parameters.
    
    Examples
    --------
    >>> scheduler = LearningRateScheduler(0.01, 'cosine', total_epochs=100)
    >>> for epoch in range(100):
    ...     lr = scheduler(epoch)
    ...     model.learning_rate = lr
    """
    
    def __init__(
        self,
        initial_lr: float = 0.01,
        schedule: str = "constant",
        **kwargs
    ):
        self.initial_lr = initial_lr
        self.schedule = schedule
        self.kwargs = kwargs
        
        # History
        self.lr_history: List[float] = []
    
    def __call__(self, epoch: int) -> float:
        """Get the learning rate for the given epoch."""
        if self.schedule == "constant":
            lr = self.initial_lr
        
        elif self.schedule == "step":
            step_size = self.kwargs.get("step_size", 10)
            gamma = self.kwargs.get("gamma", 0.1)
            lr = self.initial_lr * (gamma ** (epoch // step_size))
        
        elif self.schedule == "exponential":
            gamma = self.kwargs.get("gamma", 0.95)
            lr = self.initial_lr * (gamma ** epoch)
        
        elif self.schedule == "cosine":
            total_epochs = self.kwargs.get("total_epochs", 100)
            min_lr = self.kwargs.get("min_lr", 1e-6)
            lr = min_lr + 0.5 * (self.initial_lr - min_lr) * (
                1 + np.cos(np.pi * epoch / total_epochs)
            )
        
        elif self.schedule == "warmup":
            warmup_epochs = self.kwargs.get("warmup_epochs", 5)
            if epoch < warmup_epochs:
                lr = self.initial_lr * (epoch + 1) / warmup_epochs
            else:
                lr = self.initial_lr
        
        else:
            lr = self.initial_lr
        
        self.lr_history.append(lr)
        return lr


class XGBoostCallback:
    """
    Callback for XGBoost training.
    
    Wraps the training callback interface for XGBoost's callback system.
    
    Examples
    --------
    >>> callback = XGBoostCallback(tracker, verbose=True)
    >>> model = xgb.train(params, dtrain, callbacks=[callback])
    """
    
    def __init__(
        self,
        tracker=None,
        verbose: bool = True,
        early_stopping_rounds: Optional[int] = None,
        log_frequency: int = 10
    ):
        self.tracker = tracker
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.log_frequency = log_frequency
        
        self.best_score = None
        self.best_iteration = 0
        self.metric_history = []
    
    def __call__(self, env):
        """XGBoost callback interface."""
        iteration = env.iteration
        evaluation_result = env.evaluation_result_list
        
        metrics = {}
        for data_name, metric_name, value, is_higher_better in evaluation_result:
            key = f"{data_name}_{metric_name}"
            metrics[key] = value
            self.metric_history.append({'iteration': iteration, 'metric': key, 'value': value})
        
        # Log to tracker
        if self.tracker and iteration % self.log_frequency == 0:
            self.tracker.log_metrics(metrics, step=iteration)
        
        # Verbose output
        if self.verbose and iteration % self.log_frequency == 0:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            print(f"[{iteration}] {metrics_str}")
    
    @property
    def callback(self):
        """Return self as the callback."""
        return self


def create_callback(
    tracker=None,
    verbose: bool = True,
    early_stopping_patience: Optional[int] = None,
    **kwargs
) -> TrainingCallback:
    """Create a training callback with sensible defaults."""
    return TrainingCallback(
        tracker=tracker,
        verbose=verbose,
        early_stopping_patience=early_stopping_patience,
        **kwargs
    )
