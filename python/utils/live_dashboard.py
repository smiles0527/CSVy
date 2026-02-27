"""
Live Dashboard - Real-Time Training Visualization

This module provides live visualization of training progress with
matplotlib and optional web-based dashboards.

Features:
    - Real-time loss curve plotting
    - Multi-metric comparison charts
    - Experiment comparison visualizations
    - Interactive web dashboard
    - Auto-updating plots

Usage:
    from utils.live_dashboard import LivePlotter, TrainingDashboard
    
    plotter = LivePlotter()
    plotter.start()
    
    for epoch in range(100):
        # Training...
        plotter.update({'loss': loss, 'val_loss': val_loss})
    
    plotter.finalize()
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Check for IPython display
try:
    from IPython.display import display, clear_output
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


class LivePlotter:
    """
    Real-time training visualization with matplotlib.
    
    Parameters
    ----------
    metrics : list, optional
        Names of metrics to plot.
    figsize : tuple
        Figure size (width, height).
    update_interval : int
        Minimum epochs between visual updates.
    style : str
        matplotlib style ('default', 'dark_background', 'seaborn', etc.).
    
    Examples
    --------
    >>> plotter = LivePlotter(metrics=['loss', 'val_loss', 'rmse'])
    >>> plotter.start()
    >>> for epoch in range(100):
    ...     plotter.update({'loss': 0.5, 'val_loss': 0.6}, epoch=epoch)
    >>> plotter.finalize()
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        update_interval: int = 1,
        style: str = "default"
    ):
        self.metrics = metrics or []
        self.figsize = figsize
        self.update_interval = update_interval
        self.style = style
        
        # Data storage
        self.data: Dict[str, List[float]] = defaultdict(list)
        self.epochs: List[int] = []
        
        # Plot state
        self.fig = None
        self.axes = None
        self.lines: Dict[str, Any] = {}
        self._update_count = 0
        self._is_running = False
        
        # Colors for different metrics
        self.colors = plt.cm.tab10.colors if MATPLOTLIB_AVAILABLE else []
    
    def start(self, title: str = "Training Progress"):
        """Initialize the live plot."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping visualization")
            return
        
        plt.style.use(self.style)
        plt.ion()  # Enable interactive mode
        
        self.fig, self.axes = plt.subplots(1, 2, figsize=self.figsize)
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Left plot: Loss curves
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].set_title('Loss Curves')
        self.axes[0].grid(True, alpha=0.3)
        
        # Right plot: Other metrics
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Value')
        self.axes[1].set_title('Metrics')
        self.axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._is_running = True
        
        # Show the plot
        if IPYTHON_AVAILABLE:
            display(self.fig)
        else:
            plt.show(block=False)
    
    def update(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """
        Update the plot with new metric values.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of metric name -> value.
        epoch : int, optional
            Current epoch number.
        """
        if not MATPLOTLIB_AVAILABLE or self.fig is None:
            return
        
        # Store data
        current_epoch = epoch if epoch is not None else len(self.epochs)
        if len(self.epochs) == 0 or self.epochs[-1] != current_epoch:
            self.epochs.append(current_epoch)
        
        for name, value in metrics.items():
            self.data[name].append(value)
            
            if name not in self.metrics:
                self.metrics.append(name)
        
        self._update_count += 1
        
        # Only update visual every N epochs to reduce overhead
        if self._update_count % self.update_interval == 0:
            self._redraw()
    
    def _redraw(self):
        """Redraw the plot with current data."""
        if not self._is_running:
            return
        
        # Clear axes
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Separate loss metrics from others
        loss_metrics = [m for m in self.data.keys() if 'loss' in m.lower()]
        other_metrics = [m for m in self.data.keys() if 'loss' not in m.lower()]
        
        # Plot loss curves
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].set_title('Loss Curves')
        self.axes[0].grid(True, alpha=0.3)
        
        for i, name in enumerate(loss_metrics):
            color = self.colors[i % len(self.colors)]
            values = self.data[name]
            epochs = list(range(len(values)))
            self.axes[0].plot(epochs, values, label=name, color=color, linewidth=2)
        
        if loss_metrics:
            self.axes[0].legend(loc='upper right')
        
        # Plot other metrics
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Value')
        self.axes[1].set_title('Metrics')
        self.axes[1].grid(True, alpha=0.3)
        
        for i, name in enumerate(other_metrics):
            color = self.colors[(i + len(loss_metrics)) % len(self.colors)]
            values = self.data[name]
            epochs = list(range(len(values)))
            self.axes[1].plot(epochs, values, label=name, color=color, linewidth=2)
        
        if other_metrics:
            self.axes[1].legend(loc='best')
        
        plt.tight_layout()
        
        # Refresh display
        if IPYTHON_AVAILABLE:
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def finalize(self, save_path: Optional[str] = None):
        """Finalize the plot and optionally save it."""
        if not MATPLOTLIB_AVAILABLE or self.fig is None:
            return
        
        self._is_running = False
        self._redraw()
        
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.ioff()
    
    def close(self):
        """Close the plot."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
    
    def get_figure(self):
        """Get the matplotlib figure object."""
        return self.fig


class MetricComparison:
    """
    Compare metrics across multiple experiments.
    
    Parameters
    ----------
    experiments : dict
        Dictionary of experiment_name -> {metric_name -> values}.
    
    Examples
    --------
    >>> comparison = MetricComparison()
    >>> comparison.add_experiment("baseline", {"rmse": [0.5, 0.4, 0.3]})
    >>> comparison.add_experiment("improved", {"rmse": [0.4, 0.3, 0.2]})
    >>> comparison.plot()
    
    For k-sweep (x-axis = k values):
    >>> comparison.add_experiment("goals", {"accuracy": [0.6, 0.62, 0.61]}, x_values=[5, 10, 15])
    >>> comparison.add_experiment("xG", {"accuracy": [0.58, 0.63, 0.62]}, x_values=[5, 10, 15])
    >>> comparison.plot()
    """
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, List[float]]] = {}
        self.x_values: Dict[str, Optional[List[float]]] = {}
        self.colors = plt.cm.tab10.colors if MATPLOTLIB_AVAILABLE else []
    
    def add_experiment(
        self,
        name: str,
        metrics: Dict[str, List[float]],
        x_values: Optional[List[float]] = None
    ):
        """
        Add an experiment's metrics.
        
        Parameters
        ----------
        name : str
            Experiment name.
        metrics : dict
            Metric name -> list of values.
        x_values : list, optional
            X-axis values (e.g. k values for k-sweep). If None, uses 0, 1, 2, ...
        """
        self.experiments[name] = metrics
        self.x_values[name] = x_values
    
    def add_from_tracker(self, tracker, experiment_name: str):
        """Add metrics from an ExperimentTracker (uses epoch index for x-axis)."""
        if hasattr(tracker, 'metric_history'):
            metrics = {}
            for name in tracker.metric_history.metrics:
                metrics[name] = tracker.metric_history.get(name)
            self.experiments[experiment_name] = metrics
            self.x_values[experiment_name] = None
    
    def plot(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Create comparison plots for specified metrics.
        
        Parameters
        ----------
        metrics_to_plot : list, optional
            Specific metrics to plot. If None, plots all common metrics.
        figsize : tuple
            Figure size.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available")
            return
        
        if not self.experiments:
            logger.warning("No experiments to compare")
            return
        
        # Find common metrics
        all_metrics = set()
        for exp_metrics in self.experiments.values():
            all_metrics.update(exp_metrics.keys())
        
        if metrics_to_plot:
            metrics_to_show = [m for m in metrics_to_plot if m in all_metrics]
        else:
            metrics_to_show = list(all_metrics)
        
        if not metrics_to_show:
            logger.warning("No metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(metrics_to_show)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle("Experiment Comparison", fontsize=14, fontweight='bold')
        
        for idx, metric_name in enumerate(metrics_to_show):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            use_k_axis = False
            for exp_idx, (exp_name, exp_metrics) in enumerate(self.experiments.items()):
                if metric_name in exp_metrics:
                    values = exp_metrics[metric_name]
                    x_vals = self.x_values.get(exp_name)
                    if x_vals is not None and len(x_vals) == len(values):
                        x_axis = x_vals
                        use_k_axis = True
                    else:
                        x_axis = list(range(len(values)))
                    color = self.colors[exp_idx % len(self.colors)]
                    ax.plot(x_axis, values, label=exp_name, color=color, linewidth=2)
            ax.set_xlabel('k' if use_k_axis else 'Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig


class TrainingDashboard:
    """
    Comprehensive training dashboard with multiple views.
    
    Parameters
    ----------
    metrics : list
        Metrics to track.
    refresh_rate : float
        Seconds between dashboard updates.
    
    Examples
    --------
    >>> dashboard = TrainingDashboard(['loss', 'rmse', 'mae'])
    >>> dashboard.start()
    >>> for epoch in range(100):
    ...     dashboard.log_epoch(epoch, {'loss': 0.5, 'rmse': 0.3})
    >>> dashboard.stop()
    """
    
    def __init__(
        self,
        metrics: List[str] = None,
        refresh_rate: float = 0.5
    ):
        self.metrics = metrics or ['loss', 'val_loss']
        self.refresh_rate = refresh_rate
        
        # Data storage
        self.epoch_data: List[Dict[str, Any]] = []
        self.current_metrics: Dict[str, float] = {}
        self.best_metrics: Dict[str, float] = {}
        self.start_time: Optional[float] = None
        
        # Dashboard state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Plotting
        self.live_plotter = LivePlotter(metrics=self.metrics)
    
    def start(self):
        """Start the dashboard."""
        self.start_time = time.time()
        self._running = True
        self.live_plotter.start("Training Dashboard")
        logger.info("Training dashboard started")
    
    def stop(self):
        """Stop the dashboard."""
        self._running = False
        self.live_plotter.finalize()
        logger.info("Training dashboard stopped")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log metrics for an epoch."""
        self.current_metrics = metrics.copy()
        
        # Update best metrics
        for name, value in metrics.items():
            if name not in self.best_metrics:
                self.best_metrics[name] = value
            elif 'loss' in name.lower() or 'error' in name.lower():
                self.best_metrics[name] = min(self.best_metrics[name], value)
            else:
                self.best_metrics[name] = max(self.best_metrics[name], value)
        
        # Store epoch data
        self.epoch_data.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
        
        # Update plot
        self.live_plotter.update(metrics, epoch=epoch)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training progress."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'epochs_completed': len(self.epoch_data),
            'elapsed_time': elapsed_time,
            'current_metrics': self.current_metrics,
            'best_metrics': self.best_metrics,
            'epochs_per_second': len(self.epoch_data) / elapsed_time if elapsed_time > 0 else 0
        }
    
    def print_summary(self):
        """Print a summary of training progress."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"Epochs completed: {summary['epochs_completed']}")
        print(f"Elapsed time: {summary['elapsed_time']:.2f}s")
        print(f"Speed: {summary['epochs_per_second']:.2f} epochs/sec")
        print("\nCurrent metrics:")
        for name, value in summary['current_metrics'].items():
            print(f"  {name}: {value:.6f}")
        print("\nBest metrics:")
        for name, value in summary['best_metrics'].items():
            print(f"  {name}: {value:.6f}")
        print("=" * 50 + "\n")
    
    def save_figure(self, path: str):
        """Save the current dashboard figure."""
        if self.live_plotter.fig:
            self.live_plotter.fig.savefig(path, dpi=150, bbox_inches='tight')


class PlotlyDashboard:
    """
    Interactive web-based dashboard using Plotly.
    
    More feature-rich than matplotlib but requires plotly.
    
    Examples
    --------
    >>> dashboard = PlotlyDashboard()
    >>> for epoch in range(100):
    ...     dashboard.update({'loss': loss})
    >>> dashboard.show()
    """
    
    def __init__(self):
        self.data: Dict[str, List[float]] = defaultdict(list)
        self.epochs: List[int] = []
    
    def update(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """Add new data point."""
        current_epoch = epoch if epoch is not None else len(self.epochs)
        self.epochs.append(current_epoch)
        
        for name, value in metrics.items():
            self.data[name].append(value)
    
    def create_figure(self) -> Optional[Any]:
        """Create a Plotly figure."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available")
            return None
        
        # Separate loss and other metrics
        loss_metrics = [m for m in self.data.keys() if 'loss' in m.lower()]
        other_metrics = [m for m in self.data.keys() if 'loss' not in m.lower()]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss Curves", "Other Metrics")
        )
        
        # Add loss traces
        for name in loss_metrics:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.data[name]))),
                    y=self.data[name],
                    name=name,
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Add other metric traces
        for name in other_metrics:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.data[name]))),
                    y=self.data[name],
                    name=name,
                    mode='lines'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Training Dashboard",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def show(self):
        """Display the dashboard."""
        fig = self.create_figure()
        if fig:
            fig.show()
    
    def save_html(self, path: str):
        """Save dashboard as HTML."""
        fig = self.create_figure()
        if fig:
            fig.write_html(path)


def create_live_plotter(metrics: List[str] = None) -> LivePlotter:
    """Create a default live plotter."""
    return LivePlotter(
        metrics=metrics or ['loss', 'val_loss', 'rmse'],
        figsize=(12, 5),
        update_interval=1
    )


def create_dashboard(metrics: List[str] = None) -> TrainingDashboard:
    """Create a default training dashboard."""
    return TrainingDashboard(
        metrics=metrics or ['loss', 'val_loss', 'rmse', 'mae']
    )
