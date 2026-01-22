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
            
            experiments = {
                "XGBoost": "xgboost_hyperparam_search",
                "Linear": "linear_hyperparam_search",
                "RandomForest": "random_forest_hyperparam_search",
                "Elo": "elo_hyperparam_search"
            }
            
            all_runs = []
            for model_name, exp_name in experiments.items():
                try:
                    exp = mlflow.get_experiment_by_name(exp_name)
                    if exp:
                        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
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
                # Remove timezone to avoid comparison issues
                if hasattr(latest_run, 'tz') and latest_run.tz is not None:
                    latest_run = latest_run.tz_localize(None)
                time_ago = (datetime.now() - latest_run).total_seconds() / 60
                st.metric("Last Run", f"{int(time_ago)}m ago", delta=None)
            else:
                st.metric("Last Run", "N/A", delta=None)
    
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview",
            "Experiments",
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
            self.render_data_uploader()
        
        with tab4:
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
