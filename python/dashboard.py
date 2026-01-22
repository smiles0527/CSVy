"""
Real-Time Hockey ML Dashboard
Auto-updating plots, live experiment tracking, interactive controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from pathlib import Path
import json
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Hockey ML Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: 600;
        color: #ffffff;
    }
    .status-success {
        color: #10b981;
        font-weight: 600;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    .status-error {
        color: #ef4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


class HockeyDashboard:
    """Interactive dashboard for hockey ML pipeline."""
    
    def __init__(self):
        self.mlflow_path = Path("mlruns")
        self.data_path = Path("../data")
        
    def load_mlflow_experiments(self):
        """Load all MLflow experiments and runs."""
        try:
            mlflow.set_tracking_uri("file:///" + str(self.mlflow_path.absolute()))
            
            # All available experiments
            experiments = {
                "XGBoost": "xgboost_hyperparam_search",
                "Linear": "linear_hyperparam_search",
                "RandomForest": "random_forest_hyperparam_search",
                "Elo": "elo_hyperparam_search",
                "NeuralNetwork": "neural_network_hyperparam_search",
                "TestTracking": "test_tracking"
            }
            
            all_runs = []
            for model_name, exp_name in experiments.items():
                try:
                    exp = mlflow.get_experiment_by_name(exp_name)
                    if exp:
                        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                        if not runs.empty:
                            runs['model'] = model_name
                            all_runs.append(runs)
                except:
                    pass
            
            if all_runs:
                return pd.concat(all_runs, ignore_index=True)
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def render_header(self):
        """Render dashboard header."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown('<p class="big-font">Hockey ML Pipeline Dashboard</p>', unsafe_allow_html=True)
            st.caption("Real-time experiment tracking and model comparison")
        
        with col2:
            if st.button("↻ Refresh Data", use_container_width=True):
                st.rerun()
        
        with col3:
            auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
            if auto_refresh:
                time.sleep(5)
                st.rerun()
    
    def render_metrics_overview(self, runs_df):
        """Render key metrics cards."""
        if runs_df.empty:
            st.error("**No experiments found in MLflow**")
            st.info("""
            **Getting Started:**
            
            1. Run hyperparameter searches:
               ```
               python training/linear_hyperparam_search.py
               python training/xgboost_hyperparam_search.py
               python training/random_forest_hyperparam_search.py
               ```
            
            2. Or run the complete pipeline:
               ```
               python training/automated_model_selection.py
               ```
            
            3. Refresh this dashboard to see results
            """)
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_runs = len(runs_df)
            st.metric("Total Runs", total_runs, delta=None)
        
        with col2:
            best_r2 = runs_df['metrics.test_r2'].max() if 'metrics.test_r2' in runs_df.columns else 0
            st.metric("Best R² Score", f"{best_r2:.4f}", delta=None)
        
        with col3:
            models_tested = runs_df['model'].nunique()
            st.metric("Models Tested", models_tested, delta=None)
        
        with col4:
            latest_run = pd.to_datetime(runs_df['start_time']).max() if 'start_time' in runs_df.columns else None
            if latest_run:
                # Convert to timezone-naive for comparison
                if latest_run.tzinfo is not None:
                    latest_run = latest_run.replace(tzinfo=None)
                time_ago = (datetime.now() - latest_run).total_seconds() / 60
                if time_ago < 0:
                    # Future timestamp - just show the date
                    st.metric("Last Run", latest_run.strftime("%b %d"), delta=None)
                elif time_ago < 60:
                    st.metric("Last Run", f"{int(time_ago)}m ago", delta=None)
                elif time_ago < 1440:
                    st.metric("Last Run", f"{int(time_ago/60)}h ago", delta=None)
                else:
                    st.metric("Last Run", f"{int(time_ago/1440)}d ago", delta=None)
            else:
                st.metric("Last Run", "N/A", delta=None)
        
        # Easy-to-understand accuracy section
        st.divider()
        st.subheader("How Good Are Your Models?")
        
        if 'metrics.test_r2' in runs_df.columns:
            best_r2_val = runs_df['metrics.test_r2'].max()
            best_model = runs_df.loc[runs_df['metrics.test_r2'].idxmax(), 'model'] if not runs_df.empty else "N/A"
            
            # Convert R² to percentage accuracy (simplified explanation)
            accuracy_pct = max(0, best_r2_val * 100)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Color based on quality
                if accuracy_pct >= 70:
                    st.success(f"**Prediction Accuracy: {accuracy_pct:.1f}%**")
                    st.caption("Excellent - Model explains most of the variance")
                elif accuracy_pct >= 40:
                    st.warning(f"**Prediction Accuracy: {accuracy_pct:.1f}%**")
                    st.caption("Moderate - Room for improvement")
                else:
                    st.error(f"**Prediction Accuracy: {accuracy_pct:.1f}%**")
                    st.caption("Low - Needs more features or data")
            
            with col2:
                st.info(f"**Best Model: {best_model}**")
                st.caption("This model performed best on test data")
            
            with col3:
                # Simple interpretation guide
                st.markdown("""
                **What does this mean?**
                - 80%+ = Great predictions
                - 50-80% = Decent predictions
                - Below 50% = Needs work
                """)
    
    def render_performance_chart(self, runs_df):
        """Interactive performance comparison chart (AUTO-UPDATES)."""
        st.subheader("Model Performance Comparison")
        
        if 'metrics.test_r2' not in runs_df.columns or runs_df.empty:
            st.info("No performance data available. Run experiments to see model comparisons.")
            return
        
        # Aggregate by model - handle missing columns gracefully
        agg_dict = {'metrics.test_r2': ['mean', 'max', 'min', 'std']}
        if 'metrics.test_rmse' in runs_df.columns:
            agg_dict['metrics.test_rmse'] = ['mean', 'min']
        
        model_perf = runs_df.groupby('model').agg(agg_dict).reset_index()
        
        # Flatten column names
        model_perf.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in model_perf.columns]
        
        # Rename for consistency
        col_rename = {
            'metrics.test_r2_mean': 'r2_mean',
            'metrics.test_r2_max': 'r2_max', 
            'metrics.test_r2_min': 'r2_min',
            'metrics.test_r2_std': 'r2_std',
            'metrics.test_rmse_mean': 'rmse_mean',
            'metrics.test_rmse_min': 'rmse_min'
        }
        model_perf = model_perf.rename(columns=col_rename)
        
        # Create dual-axis chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("R² Score (Higher = Better)", "RMSE (Lower = Better)"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # R² bars
        fig.add_trace(
            go.Bar(
                x=model_perf['model'],
                y=model_perf['r2_max'],
                name='Best R²',
                marker_color='#667eea',
                text=model_perf['r2_max'].round(4),
                textposition='auto',
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=model_perf['model'],
                y=model_perf['r2_mean'],
                name='Avg R²',
                marker_color='#764ba2',
                text=model_perf['r2_mean'].round(4),
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # RMSE bars (only if data available)
        if 'rmse_min' in model_perf.columns:
            fig.add_trace(
                go.Bar(
                    x=model_perf['model'],
                    y=model_perf['rmse_min'],
                    name='Best RMSE',
                    marker_color='#f093fb',
                    text=model_perf['rmse_min'].round(4),
                    textposition='auto',
                ),
                row=1, col=2
            )
        else:
            fig.add_annotation(
                text="RMSE data not available",
                xref="x2", yref="y2",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template="plotly_dark",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_hyperparameter_space(self, runs_df):
        """3D hyperparameter exploration (INTERACTIVE)."""
        st.subheader("Hyperparameter Space Explorer")
        
        if runs_df.empty or 'model' not in runs_df.columns:
            st.info("No experiment data available for visualization.")
            return
        
        models = runs_df['model'].unique()
        if len(models) == 0:
            st.info("No models found.")
            return
            
        model_choice = st.selectbox("Select Model", models)
        model_runs = runs_df[runs_df['model'] == model_choice]
        
        if len(model_runs) < 5:
            st.warning(f"Not enough runs for {model_choice} visualization (need at least 5, have {len(model_runs)})")
            return
        
        # Get numeric params
        param_cols = [col for col in model_runs.columns if col.startswith('params.')]
        numeric_params = []
        
        for col in param_cols:
            try:
                pd.to_numeric(model_runs[col])
                numeric_params.append(col)
            except:
                pass
        
        if len(numeric_params) >= 2 and 'metrics.test_r2' in model_runs.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                x_param = st.selectbox("X-axis", numeric_params, key='x')
            
            other_params = [p for p in numeric_params if p != x_param]
            if not other_params:
                st.warning("Need at least 2 different numeric parameters for 3D plot.")
                return
                
            with col2:
                y_param = st.selectbox("Y-axis", other_params, key='y')
            
            if x_param is None or y_param is None:
                st.warning("Please select both X and Y parameters.")
                return
            
            # Convert to numeric
            x_vals = pd.to_numeric(model_runs[x_param], errors='coerce')
            y_vals = pd.to_numeric(model_runs[y_param], errors='coerce')
            r2_vals = model_runs['metrics.test_r2']
            
            # Filter out NaN values
            valid_mask = ~(x_vals.isna() | y_vals.isna() | r2_vals.isna())
            if valid_mask.sum() < 3:
                st.warning("Not enough valid data points for 3D visualization.")
                return
            
            x_vals = x_vals[valid_mask]
            y_vals = y_vals[valid_mask]
            r2_vals = r2_vals[valid_mask]
            
            # Create 3D scatter
            x_label = str(x_param).replace('params.', '')
            y_label = str(y_param).replace('params.', '')
            
            fig = px.scatter_3d(
                x=x_vals,
                y=y_vals,
                z=r2_vals,
                color=r2_vals,
                color_continuous_scale='Viridis',
                labels={
                    'x': x_label,
                    'y': y_label,
                    'z': 'R² Score'
                }
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=500,
                scene=dict(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    zaxis_title='R² Score'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric parameters to create 3D visualization.")
    
    def render_training_progress(self, runs_df):
        """Live training progress over time."""
        st.subheader("Training Progress Over Time")
        
        if 'start_time' not in runs_df.columns or runs_df.empty:
            st.info("No training history yet")
            return
        
        # Convert timestamps
        runs_df['timestamp'] = pd.to_datetime(runs_df['start_time'])
        runs_df = runs_df.sort_values('timestamp')
        
        # Create rolling best R² chart
        if 'metrics.test_r2' in runs_df.columns:
            fig = go.Figure()
            
            for model in runs_df['model'].unique():
                model_data = runs_df[runs_df['model'] == model].copy()
                model_data['best_r2_so_far'] = model_data['metrics.test_r2'].cummax()
                
                fig.add_trace(go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['best_r2_so_far'],
                    mode='lines+markers',
                    name=model,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Best R² Score Over Time (Cumulative)",
                xaxis_title="Time",
                yaxis_title="Best R² Score",
                template="plotly_dark",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_data_uploader(self):
        """Interactive data upload and preview."""
        st.subheader("Data Upload & Preview")
        
        uploaded_file = st.file_uploader(
            "Upload Hockey CSV",
            type=['csv'],
            help="Upload your competition data for analysis"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.metric("Missing %", f"{missing_pct:.1f}%")
            
            # Preview
            with st.expander("Preview Data (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Missing data heatmap
            if df.isnull().any().any():
                st.write("**Missing Data Heatmap:**")
                missing_matrix = df.isnull().astype(int)
                fig = px.imshow(
                    missing_matrix.T,
                    labels=dict(x="Row", y="Column", color="Missing"),
                    color_continuous_scale=['#0e1117', '#ff4b4b'],
                    aspect="auto"
                )
                fig.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature engineering toggle
            if st.button("Apply Hockey Feature Engineering", use_container_width=True):
                with st.spinner("Engineering features..."):
                    try:
                        import sys
                        sys.path.insert(0, str(Path(__file__).parent / 'training'))
                        from hockey_feature_engineering import HockeyFeatureEngineer
                        engineer = HockeyFeatureEngineer(df)
                        df_enhanced, features = engineer.create_all_features()
                        
                        st.success(f"Successfully created {len(features)} new features")
                        st.download_button(
                            "Download Enhanced CSV",
                            df_enhanced.to_csv(index=False).encode('utf-8'),
                            "hockey_data_enhanced.csv",
                            "text/csv"
                        )
                    except ImportError as e:
                        st.error(f"Could not load feature engineering module: {e}")
                        st.info("Make sure hockey_feature_engineering.py exists in the training folder.")
                    except Exception as e:
                        st.error(f"Error during feature engineering: {e}")
    
    def render_advanced_metrics(self, runs_df):
        """Render comprehensive advanced metrics panel."""
        st.subheader("Advanced Competition Metrics")
        
        if runs_df.empty:
            st.info("No experiment data available.")
            return
        
        # Collect all available metrics
        metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
        
        if not metric_cols:
            st.warning("No metrics logged yet.")
            return
        
        # Create tabs for different metric categories
        m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs([
            "Regression", "Classification", "Model Health", "Competition Score"
        ])
        
        with m_tab1:
            st.markdown("**Regression Metrics** - For predicting scores/margins")
            
            reg_metrics = {
                'metrics.test_r2': ('R² Score', 'How much variance explained (1.0 = perfect)', True),
                'metrics.train_r2': ('Train R²', 'Training set R²', True),
                'metrics.test_rmse': ('RMSE', 'Root Mean Square Error (lower = better)', False),
                'metrics.test_mae': ('MAE', 'Mean Absolute Error (lower = better)', False),
                'metrics.test_mape': ('MAPE %', 'Mean Absolute Percentage Error', False),
                'metrics.explained_variance': ('Explained Var', 'Explained variance score', True),
            }
            
            cols = st.columns(3)
            col_idx = 0
            for metric, (name, desc, higher_better) in reg_metrics.items():
                if metric in runs_df.columns:
                    with cols[col_idx % 3]:
                        best_val = runs_df[metric].max() if higher_better else runs_df[metric].min()
                        st.metric(name, f"{best_val:.4f}")
                        st.caption(desc)
                    col_idx += 1
        
        with m_tab2:
            st.markdown("**Classification Metrics** - For predicting win/loss")
            
            class_metrics = {
                'metrics.accuracy': ('Accuracy', 'Correct predictions / Total', True),
                'metrics.precision': ('Precision', 'True positives / Predicted positives', True),
                'metrics.recall': ('Recall', 'True positives / Actual positives', True),
                'metrics.f1': ('F1 Score', 'Harmonic mean of precision & recall', True),
                'metrics.auc_roc': ('AUC-ROC', 'Area under ROC curve (0.5 = random)', True),
                'metrics.log_loss': ('Log Loss', 'Cross-entropy loss (lower = better)', False),
                'metrics.brier_score': ('Brier Score', 'Probability calibration (lower = better)', False),
            }
            
            cols = st.columns(3)
            col_idx = 0
            found_any = False
            for metric, (name, desc, higher_better) in class_metrics.items():
                if metric in runs_df.columns:
                    found_any = True
                    with cols[col_idx % 3]:
                        best_val = runs_df[metric].max() if higher_better else runs_df[metric].min()
                        st.metric(name, f"{best_val:.4f}")
                        st.caption(desc)
                    col_idx += 1
            
            if not found_any:
                st.info("No classification metrics found. Add accuracy, precision, recall, f1, auc_roc, log_loss to your training scripts.")
        
        with m_tab3:
            st.markdown("**Model Health** - Overfitting & stability checks")
            
            # Overfitting analysis
            if 'metrics.train_r2' in runs_df.columns and 'metrics.test_r2' in runs_df.columns:
                runs_df['overfit_gap'] = runs_df['metrics.train_r2'] - runs_df['metrics.test_r2']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_gap = runs_df['overfit_gap'].mean()
                    if avg_gap > 0.1:
                        st.error(f"Overfit Gap: {avg_gap:.4f}")
                        st.caption("High gap = overfitting. Try regularization.")
                    elif avg_gap > 0.05:
                        st.warning(f"Overfit Gap: {avg_gap:.4f}")
                        st.caption("Moderate gap. Monitor closely.")
                    else:
                        st.success(f"Overfit Gap: {avg_gap:.4f}")
                        st.caption("Good! Train/test performance aligned.")
                
                with col2:
                    # Variance in performance
                    if 'metrics.test_r2' in runs_df.columns:
                        r2_std = runs_df['metrics.test_r2'].std()
                        st.metric("R² Std Dev", f"{r2_std:.4f}")
                        st.caption("Lower = more stable across runs")
                
                with col3:
                    # Best vs average gap
                    if 'metrics.test_r2' in runs_df.columns:
                        best_r2 = runs_df['metrics.test_r2'].max()
                        avg_r2 = runs_df['metrics.test_r2'].mean()
                        consistency = avg_r2 / best_r2 if best_r2 > 0 else 0
                        st.metric("Consistency", f"{consistency:.1%}")
                        st.caption("Avg/Best ratio. Higher = reproducible.")
            else:
                st.info("Log both train_r2 and test_r2 to see overfitting analysis.")
            
            # Cross-validation stability (if available)
            cv_metrics = [col for col in runs_df.columns if 'cv_' in col.lower()]
            if cv_metrics:
                st.markdown("**Cross-Validation Scores:**")
                for cv_metric in cv_metrics[:5]:
                    mean_val = runs_df[cv_metric].mean()
                    std_val = runs_df[cv_metric].std()
                    st.write(f"- {cv_metric}: {mean_val:.4f} (+/- {std_val:.4f})")
        
        with m_tab4:
            st.markdown("**Competition Score Estimator**")
            st.caption("Composite score based on multiple factors")
            
            # Calculate composite competition score
            scores = {}
            
            # R² contribution (0-40 points)
            if 'metrics.test_r2' in runs_df.columns:
                best_r2 = max(0, runs_df['metrics.test_r2'].max())
                scores['Prediction Power'] = min(40, best_r2 * 40)
            
            # Consistency contribution (0-20 points)
            if 'metrics.test_r2' in runs_df.columns:
                r2_std = runs_df['metrics.test_r2'].std()
                consistency_score = max(0, 20 - (r2_std * 100))
                scores['Consistency'] = min(20, consistency_score)
            
            # Low overfitting contribution (0-20 points)
            if 'metrics.train_r2' in runs_df.columns and 'metrics.test_r2' in runs_df.columns:
                overfit = abs(runs_df['metrics.train_r2'].max() - runs_df['metrics.test_r2'].max())
                overfit_score = max(0, 20 - (overfit * 100))
                scores['Generalization'] = min(20, overfit_score)
            
            # Model diversity (0-10 points)
            if 'model' in runs_df.columns:
                diversity = runs_df['model'].nunique()
                scores['Model Diversity'] = min(10, diversity * 2.5)
            
            # Experiment volume (0-10 points)
            total_runs = len(runs_df)
            scores['Experiment Volume'] = min(10, total_runs / 100)
            
            total_score = sum(scores.values())
            
            # Display as progress bars
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Big score display
                if total_score >= 70:
                    st.success(f"### {total_score:.0f}/100")
                elif total_score >= 40:
                    st.warning(f"### {total_score:.0f}/100")
                else:
                    st.error(f"### {total_score:.0f}/100")
                st.caption("Competition Readiness Score")
            
            with col2:
                for component, points in scores.items():
                    max_points = 40 if component == 'Prediction Power' else 20 if component in ['Consistency', 'Generalization'] else 10
                    pct = points / max_points
                    st.progress(pct, text=f"{component}: {points:.1f}/{max_points}")
            
            # Recommendations
            st.markdown("**Recommendations to Improve:**")
            recs = []
            if scores.get('Prediction Power', 0) < 30:
                recs.append("- Add more features (hockey-specific: fatigue, momentum, goalie stats)")
            if scores.get('Consistency', 0) < 15:
                recs.append("- Reduce hyperparameter search variance, use more stable ranges")
            if scores.get('Generalization', 0) < 15:
                recs.append("- Add regularization (L1/L2), reduce model complexity")
            if scores.get('Model Diversity', 0) < 7:
                recs.append("- Try more model types (XGBoost, Neural Net, Ensemble)")
            if scores.get('Experiment Volume', 0) < 7:
                recs.append("- Run more experiments to find optimal configurations")
            
            if recs:
                for rec in recs:
                    st.write(rec)
            else:
                st.success("Your pipeline looks competition-ready!")
    
    def render_sidebar(self):
        """Sidebar with controls and info."""
        with st.sidebar:
            st.header("Controls")
            
            # MLflow connection status
            st.subheader("MLflow Status")
            if self.mlflow_path.exists():
                st.markdown('<p class="status-success">● Connected</p>', unsafe_allow_html=True)
                st.caption(f"Path: {self.mlflow_path.absolute()}")
            else:
                st.markdown('<p class="status-error">● Disconnected</p>', unsafe_allow_html=True)
                st.caption("MLflow directory not found")
            
            st.divider()
            
            # Quick actions
            st.subheader("Quick Actions")
            
            if st.button("▶ Run Hyperparameter Search", use_container_width=True):
                st.code("python training/linear_hyperparam_search.py", language="bash")
            
            if st.button("▶ Train Final Models", use_container_width=True):
                st.code("python training/automated_model_selection.py", language="bash")
            
            if st.button("▶ Open MLflow UI", use_container_width=True):
                st.code("mlflow ui --backend-store-uri file:///./mlruns", language="bash")
            
            st.divider()
            
            # Info
            st.subheader("About")
            st.caption("""
            This dashboard provides real-time tracking of your ML experiments.
            
            **Features:**
            - Live experiment monitoring
            - Interactive hyperparameter exploration
            - Data upload & preview
            - Feature engineering integration
            
            **Auto-refresh:** Enable in header to see live updates every 5 seconds.
            """)
    
    def run(self):
        """Main dashboard loop."""
        self.render_header()
        
        # Load data
        runs_df = self.load_mlflow_experiments()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Experiments",
            "Advanced Metrics",
            "Data",
            "Leaderboard"
        ])
        
        with tab1:
            self.render_metrics_overview(runs_df)
            st.divider()
            self.render_performance_chart(runs_df)
            st.divider()
            self.render_training_progress(runs_df)
        
        with tab2:
            self.render_hyperparameter_space(runs_df)
        
        with tab3:
            self.render_advanced_metrics(runs_df)
        
        with tab4:
            self.render_data_uploader()
        
        with tab5:
            st.subheader("Model Leaderboard")
            if not runs_df.empty and 'metrics.test_r2' in runs_df.columns:
                # Build column list dynamically
                cols_to_show = ['model', 'metrics.test_r2']
                col_config = {
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "model": st.column_config.TextColumn("Model"),
                    "metrics.test_r2": st.column_config.NumberColumn("R²", format="%.4f")
                }
                
                if 'metrics.test_rmse' in runs_df.columns:
                    cols_to_show.append('metrics.test_rmse')
                    col_config["metrics.test_rmse"] = st.column_config.NumberColumn("RMSE", format="%.4f")
                
                if 'metrics.test_mae' in runs_df.columns:
                    cols_to_show.append('metrics.test_mae')
                    col_config["metrics.test_mae"] = st.column_config.NumberColumn("MAE", format="%.4f")
                
                leaderboard = runs_df[cols_to_show].sort_values('metrics.test_r2', ascending=False).head(10).copy()
                leaderboard.insert(0, 'rank', range(1, len(leaderboard) + 1))
                
                st.dataframe(
                    leaderboard,
                    use_container_width=True,
                    hide_index=True,
                    column_config=col_config
                )
            else:
                st.info("No model results available yet. Run experiments to see the leaderboard.")


if __name__ == "__main__":
    dashboard = HockeyDashboard()
    dashboard.run()
