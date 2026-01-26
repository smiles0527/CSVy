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
import os
from datetime import datetime
from scipy import stats as scipy_stats

# Configure MLflow tracking URI (supports Docker or local)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)

# Page config
st.set_page_config(
    page_title="Hockey ML Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    
    .stMetric {
        background-color: #1a1a2e;
        padding: 12px;
        border-radius: 6px;
        border-left: 3px solid #4f46e5;
    }
    
    h1, h2, h3 {
        color: #e0e0e0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important;
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
            # Use environment variable if set (Docker), otherwise use local path
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            else:
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
        """Comprehensive professional data analysis suite."""
        st.markdown("## 📊 Comprehensive Data Analysis Suite")
        st.caption("Drop your CSV for enterprise-grade statistical analysis, benchmarking, and insights")
        
        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=['csv'],
            help="Upload any CSV for comprehensive analysis"
        )
        
        if not uploaded_file:
            st.info("👆 Upload a CSV file to begin comprehensive analysis")
            
            # Show example of what analysis includes
            with st.expander("📋 What's Included in the Analysis"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    **📈 Statistical Analysis**
                    - Descriptive statistics
                    - Distribution analysis
                    - Normality tests
                    - Skewness & kurtosis
                    - Confidence intervals
                    """)
                with col2:
                    st.markdown("""
                    **🔍 Data Quality**
                    - Missing value patterns
                    - Duplicate detection
                    - Outlier analysis (IQR, Z-score)
                    - Data type inference
                    - Cardinality analysis
                    """)
                with col3:
                    st.markdown("""
                    **🎯 Advanced Insights**
                    - Correlation matrix
                    - Feature importance
                    - Multicollinearity (VIF)
                    - Cluster analysis
                    - Automated recommendations
                    """)
            return
        
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        # Store in session state for cross-tab access
        st.session_state['uploaded_df'] = df
        
        # Create analysis tabs
        analysis_tabs = st.tabs([
            "📋 Overview",
            "📊 Statistics", 
            "🔍 Data Quality",
            "📈 Distributions",
            "🔗 Correlations",
            "🎯 Feature Analysis",
            "⚡ Quick Model",
            "📑 Report"
        ])
        
        # TAB 1: Overview
        with analysis_tabs[0]:
            self._render_data_overview(df)
        
        # TAB 2: Statistics
        with analysis_tabs[1]:
            self._render_statistics(df)
        
        # TAB 3: Data Quality
        with analysis_tabs[2]:
            self._render_data_quality(df)
        
        # TAB 4: Distributions
        with analysis_tabs[3]:
            self._render_distributions(df)
        
        # TAB 5: Correlations
        with analysis_tabs[4]:
            self._render_correlations(df)
        
        # TAB 6: Feature Analysis
        with analysis_tabs[5]:
            self._render_feature_analysis(df)
        
        # TAB 7: Quick Model Benchmark
        with analysis_tabs[6]:
            self._render_quick_model(df)
        
        # TAB 8: Report
        with analysis_tabs[7]:
            self._render_analysis_report(df)
    
    def _render_data_overview(self, df):
        """Executive summary of the dataset."""
        st.markdown("### 📋 Dataset Executive Summary")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Rows", f"{len(df):,}")
        with col2:
            st.metric("📋 Columns", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("🔢 Numeric", len(numeric_cols))
        with col4:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            st.metric("📝 Categorical", len(cat_cols))
        with col5:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("💾 Memory", f"{memory_mb:.2f} MB")
        
        st.divider()
        
        # Data health score
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Calculate health score
            missing_score = 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            dup_score = 100 - (df.duplicated().sum() / len(df) * 100)
            type_score = 100  # Assume good type inference
            health_score = (missing_score * 0.4 + dup_score * 0.3 + type_score * 0.3)
            
            if health_score >= 90:
                st.success(f"### Data Health: {health_score:.0f}/100")
                st.caption("✅ Excellent - Ready for modeling")
            elif health_score >= 70:
                st.warning(f"### Data Health: {health_score:.0f}/100")
                st.caption("⚠️ Good - Minor cleaning needed")
            else:
                st.error(f"### Data Health: {health_score:.0f}/100")
                st.caption("🔴 Needs attention")
        
        with col2:
            # Health breakdown
            health_data = pd.DataFrame({
                'Metric': ['Completeness', 'Uniqueness', 'Type Consistency'],
                'Score': [missing_score, dup_score, type_score],
                'Status': [
                    '✅' if missing_score >= 90 else '⚠️' if missing_score >= 70 else '❌',
                    '✅' if dup_score >= 95 else '⚠️' if dup_score >= 80 else '❌',
                    '✅'
                ]
            })
            st.dataframe(health_data, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Column summary
        st.markdown("### Column Overview")
        
        col_summary = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            missing_pct = missing / len(df) * 100
            unique = df[col].nunique()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                sample = f"μ={df[col].mean():.2f}, σ={df[col].std():.2f}"
            else:
                top_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                sample = f"Top: {str(top_val)[:20]}"
            
            col_summary.append({
                'Column': col,
                'Type': dtype,
                'Missing': f"{missing} ({missing_pct:.1f}%)",
                'Unique': unique,
                'Sample/Stats': sample
            })
        
        st.dataframe(pd.DataFrame(col_summary), use_container_width=True, hide_index=True)
        
        # Preview
        with st.expander("🔎 Preview Data (first 100 rows)"):
            st.dataframe(df.head(100), use_container_width=True)
    
    def _render_statistics(self, df):
        """Comprehensive statistical analysis."""
        st.markdown("### 📊 Comprehensive Statistical Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.warning("No numeric columns found for statistical analysis")
            return
        
        # Descriptive statistics tabs
        stat_tabs = st.tabs(["Descriptive", "Moments", "Percentiles", "Confidence Intervals"])
        
        with stat_tabs[0]:
            st.markdown("#### Descriptive Statistics")
            desc_stats = numeric_df.describe().T
            desc_stats['range'] = desc_stats['max'] - desc_stats['min']
            desc_stats['iqr'] = desc_stats['75%'] - desc_stats['25%']
            desc_stats['cv'] = (desc_stats['std'] / desc_stats['mean'] * 100).round(2)
            st.dataframe(desc_stats.round(4), use_container_width=True)
        
        with stat_tabs[1]:
            st.markdown("#### Higher-Order Moments")
            moments_data = []
            for col in numeric_df.columns:
                data = numeric_df[col].dropna()
                if len(data) > 3:
                    from scipy import stats as scipy_stats
                    moments_data.append({
                        'Column': col,
                        'Mean': data.mean(),
                        'Variance': data.var(),
                        'Skewness': scipy_stats.skew(data),
                        'Kurtosis': scipy_stats.kurtosis(data),
                        'Skew Interpretation': 'Right-skewed' if scipy_stats.skew(data) > 0.5 else 'Left-skewed' if scipy_stats.skew(data) < -0.5 else 'Symmetric',
                        'Kurt Interpretation': 'Heavy-tailed' if scipy_stats.kurtosis(data) > 1 else 'Light-tailed' if scipy_stats.kurtosis(data) < -1 else 'Normal-tailed'
                    })
            if moments_data:
                st.dataframe(pd.DataFrame(moments_data).round(4), use_container_width=True, hide_index=True)
        
        with stat_tabs[2]:
            st.markdown("#### Percentile Analysis")
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            pct_data = numeric_df.quantile([p/100 for p in percentiles]).T
            pct_data.columns = [f"P{p}" for p in percentiles]
            st.dataframe(pct_data.round(4), use_container_width=True)
        
        with stat_tabs[3]:
            st.markdown("#### 95% Confidence Intervals for Means")
            from scipy import stats as scipy_stats
            ci_data = []
            for col in numeric_df.columns:
                data = numeric_df[col].dropna()
                if len(data) > 2:
                    mean = data.mean()
                    sem = scipy_stats.sem(data)
                    ci = scipy_stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
                    ci_data.append({
                        'Column': col,
                        'Mean': mean,
                        'Std Error': sem,
                        'CI Lower (95%)': ci[0],
                        'CI Upper (95%)': ci[1],
                        'CI Width': ci[1] - ci[0]
                    })
            if ci_data:
                st.dataframe(pd.DataFrame(ci_data).round(4), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Normality tests
        st.markdown("#### Normality Tests (Shapiro-Wilk)")
        from scipy import stats as scipy_stats
        
        normality_results = []
        for col in numeric_df.columns[:15]:  # Limit to first 15 columns
            data = numeric_df[col].dropna()
            if len(data) >= 20 and len(data) <= 5000:
                stat, p_value = scipy_stats.shapiro(data.sample(min(len(data), 5000)))
                normality_results.append({
                    'Column': col,
                    'Statistic': stat,
                    'P-Value': p_value,
                    'Normal?': '✅ Yes' if p_value > 0.05 else '❌ No',
                    'Interpretation': 'Normally distributed' if p_value > 0.05 else 'Non-normal distribution'
                })
        
        if normality_results:
            st.dataframe(pd.DataFrame(normality_results).round(4), use_container_width=True, hide_index=True)
            st.caption("Note: Shapiro-Wilk test with α=0.05. H₀: Data is normally distributed.")
    
    def _render_data_quality(self, df):
        """Data quality assessment."""
        st.markdown("### 🔍 Data Quality Assessment")
        
        quality_tabs = st.tabs(["Missing Values", "Duplicates", "Outliers", "Data Types", "Cardinality"])
        
        with quality_tabs[0]:
            st.markdown("#### Missing Value Analysis")
            
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
                'Data Type': df.dtypes.astype(str).values
            }).sort_values('Missing %', ascending=False)
            
            missing_df['Status'] = missing_df['Missing %'].apply(
                lambda x: '✅ Complete' if x == 0 else '⚠️ Minor' if x < 5 else '🔴 Significant' if x < 30 else '❌ Critical'
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Missing value bar chart
                missing_nonzero = missing_df[missing_df['Missing %'] > 0]
                if not missing_nonzero.empty:
                    fig = px.bar(
                        missing_nonzero.head(20),
                        x='Column',
                        y='Missing %',
                        color='Missing %',
                        color_continuous_scale='Reds',
                        title='Missing Values by Column (Top 20)'
                    )
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing values in this dataset!")
            
            with col2:
                st.dataframe(missing_df, use_container_width=True, hide_index=True, height=400)
            
            # Missing pattern heatmap
            if df.isnull().any().any():
                st.markdown("#### Missing Value Patterns (Sample)")
                sample_size = min(100, len(df))
                missing_sample = df.isnull().iloc[:sample_size].astype(int)
                
                fig = px.imshow(
                    missing_sample.T,
                    labels=dict(x="Row", y="Column", color="Missing"),
                    color_continuous_scale=['#1e1e1e', '#ff4b4b'],
                    aspect="auto",
                    title="Missing Value Pattern Matrix"
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with quality_tabs[1]:
            st.markdown("#### Duplicate Analysis")
            
            total_dups = df.duplicated().sum()
            dup_pct = total_dups / len(df) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Duplicates", f"{total_dups:,}")
            with col2:
                st.metric("Duplicate %", f"{dup_pct:.2f}%")
            with col3:
                st.metric("Unique Rows", f"{len(df) - total_dups:,}")
            
            if total_dups > 0:
                st.warning(f"⚠️ Found {total_dups} duplicate rows ({dup_pct:.2f}%)")
                
                with st.expander("View Duplicate Rows"):
                    dup_rows = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))
                    st.dataframe(dup_rows.head(100), use_container_width=True)
            else:
                st.success("✅ No duplicate rows found")
            
            # Column-wise uniqueness
            st.markdown("#### Column Uniqueness Analysis")
            uniqueness_df = pd.DataFrame({
                'Column': df.columns,
                'Unique Values': [df[col].nunique() for col in df.columns],
                'Uniqueness %': [(df[col].nunique() / len(df) * 100) for col in df.columns],
                'Potential ID?': [df[col].nunique() == len(df) for col in df.columns]
            }).round(2)
            st.dataframe(uniqueness_df, use_container_width=True, hide_index=True)
        
        with quality_tabs[2]:
            st.markdown("#### Outlier Detection")
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                st.warning("No numeric columns for outlier analysis")
                return
            
            method = st.radio("Detection Method", ["IQR (1.5x)", "Z-Score (3σ)", "Modified Z-Score"], horizontal=True)
            
            outlier_summary = []
            outlier_details = {}
            
            for col in numeric_df.columns:
                data = numeric_df[col].dropna()
                
                if method == "IQR (1.5x)":
                    Q1, Q3 = data.quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    outliers = (data < lower) | (data > upper)
                elif method == "Z-Score (3σ)":
                    from scipy import stats as scipy_stats
                    z_scores = np.abs(scipy_stats.zscore(data))
                    outliers = z_scores > 3
                    lower, upper = data.mean() - 3*data.std(), data.mean() + 3*data.std()
                else:  # Modified Z-Score
                    median = data.median()
                    mad = np.median(np.abs(data - median))
                    modified_z = 0.6745 * (data - median) / (mad + 1e-10)
                    outliers = np.abs(modified_z) > 3.5
                    lower, upper = None, None
                
                n_outliers = outliers.sum()
                outlier_summary.append({
                    'Column': col,
                    'Outliers': n_outliers,
                    'Outlier %': n_outliers / len(data) * 100,
                    'Lower Bound': lower if lower else 'N/A',
                    'Upper Bound': upper if upper else 'N/A'
                })
                outlier_details[col] = data[outliers].values[:10]
            
            outlier_df = pd.DataFrame(outlier_summary)
            outlier_df = outlier_df.sort_values('Outlier %', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(outlier_df.round(3), use_container_width=True, hide_index=True)
            
            with col2:
                # Outlier visualization
                fig = px.bar(
                    outlier_df.head(15),
                    x='Column',
                    y='Outlier %',
                    color='Outlier %',
                    color_continuous_scale='YlOrRd',
                    title=f'Outlier % by Column ({method})'
                )
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Box plot for selected column
            selected_col = st.selectbox("Inspect Column", numeric_df.columns, key="outlier_inspect")
            fig = px.box(numeric_df, y=selected_col, title=f"Box Plot: {selected_col}")
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with quality_tabs[3]:
            st.markdown("#### Data Type Analysis")
            
            type_summary = []
            for col in df.columns:
                inferred = pd.api.types.infer_dtype(df[col])
                actual = str(df[col].dtype)
                
                # Type recommendation
                if inferred == 'string' and df[col].nunique() < 20:
                    recommendation = 'Consider: category'
                elif inferred == 'integer' and df[col].min() >= 0 and df[col].max() < 256:
                    recommendation = 'Consider: uint8'
                elif 'date' in col.lower() or 'time' in col.lower():
                    recommendation = 'Consider: datetime'
                else:
                    recommendation = 'OK'
                
                type_summary.append({
                    'Column': col,
                    'Current Type': actual,
                    'Inferred Type': inferred,
                    'Recommendation': recommendation
                })
            
            st.dataframe(pd.DataFrame(type_summary), use_container_width=True, hide_index=True)
        
        with quality_tabs[4]:
            st.markdown("#### Cardinality Analysis")
            st.caption("High cardinality in categorical columns may need special handling")
            
            cardinality_data = []
            for col in df.columns:
                n_unique = df[col].nunique()
                cardinality_ratio = n_unique / len(df)
                
                if cardinality_ratio == 1:
                    card_type = "🔑 Unique (ID)"
                elif cardinality_ratio > 0.5:
                    card_type = "📊 High"
                elif cardinality_ratio > 0.1:
                    card_type = "📈 Medium"
                elif n_unique <= 2:
                    card_type = "🔘 Binary"
                else:
                    card_type = "📉 Low"
                
                cardinality_data.append({
                    'Column': col,
                    'Unique Values': n_unique,
                    'Cardinality Ratio': f"{cardinality_ratio:.4f}",
                    'Type': card_type,
                    'Sample Values': str(df[col].unique()[:5].tolist())[:50]
                })
            
            card_df = pd.DataFrame(cardinality_data).sort_values('Unique Values', ascending=False)
            st.dataframe(card_df, use_container_width=True, hide_index=True)
    
    def _render_distributions(self, df):
        """Distribution analysis and visualization."""
        st.markdown("### 📈 Distribution Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        cat_df = df.select_dtypes(include=['object', 'category'])
        
        dist_tabs = st.tabs(["Numeric Distributions", "Categorical Distributions", "Bivariate"])
        
        with dist_tabs[0]:
            if numeric_df.empty:
                st.warning("No numeric columns found")
                return
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_num = st.selectbox("Select Column", numeric_df.columns, key="dist_num")
                bin_count = st.slider("Bins", 10, 100, 30)
                show_kde = st.checkbox("Show KDE", value=True)
            
            with col2:
                data = numeric_df[selected_num].dropna()
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Histogram", "Box Plot", "Violin Plot", "Q-Q Plot"),
                    specs=[[{}, {}], [{}, {}]]
                )
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=data, nbinsx=bin_count, name="Histogram", marker_color='#667eea'),
                    row=1, col=1
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=data, name="Box", marker_color='#764ba2'),
                    row=1, col=2
                )
                
                # Violin
                fig.add_trace(
                    go.Violin(y=data, name="Violin", marker_color='#f093fb'),
                    row=2, col=1
                )
                
                # Q-Q plot
                from scipy import stats as scipy_stats
                qq = scipy_stats.probplot(data, dist="norm")
                fig.add_trace(
                    go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Q-Q', marker_color='#4fd1c5'),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(x=qq[0][0], y=qq[1][0] + qq[1][1]*qq[0][0], mode='lines', name='Fit', line_color='red'),
                    row=2, col=2
                )
                
                fig.update_layout(template="plotly_dark", height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats summary
                st.markdown(f"**{selected_num} Statistics:**")
                stat_cols = st.columns(6)
                with stat_cols[0]:
                    st.metric("Mean", f"{data.mean():.4f}")
                with stat_cols[1]:
                    st.metric("Median", f"{data.median():.4f}")
                with stat_cols[2]:
                    st.metric("Std Dev", f"{data.std():.4f}")
                with stat_cols[3]:
                    st.metric("Skewness", f"{data.skew():.4f}")
                with stat_cols[4]:
                    st.metric("Kurtosis", f"{data.kurtosis():.4f}")
                with stat_cols[5]:
                    st.metric("Range", f"{data.max() - data.min():.4f}")
        
        with dist_tabs[1]:
            if cat_df.empty:
                st.warning("No categorical columns found")
                return
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_cat = st.selectbox("Select Column", cat_df.columns, key="dist_cat")
                top_n = st.slider("Top N Categories", 5, 50, 15)
            
            with col2:
                value_counts = df[selected_cat].value_counts().head(top_n)
                
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])
                
                fig.add_trace(
                    go.Bar(x=value_counts.index.astype(str), y=value_counts.values, marker_color='#667eea'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Pie(labels=value_counts.index.astype(str), values=value_counts.values),
                    row=1, col=2
                )
                
                fig.update_layout(template="plotly_dark", height=400, title=f"{selected_cat} Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with dist_tabs[2]:
            st.markdown("#### Bivariate Analysis")
            
            if len(numeric_df.columns) < 2:
                st.warning("Need at least 2 numeric columns")
                return
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numeric_df.columns, key="bi_x")
            with col2:
                y_options = [c for c in numeric_df.columns if c != x_col]
                y_col = st.selectbox("Y-axis", y_options, key="bi_y")
            with col3:
                color_col = st.selectbox("Color by", ["None"] + list(df.columns), key="bi_color")
            
            if color_col == "None":
                fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"{x_col} vs {y_col}")
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, trendline="ols", title=f"{x_col} vs {y_col}")
            
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation for selected pair
            corr = df[x_col].corr(df[y_col])
            st.metric(f"Pearson Correlation: {x_col} ↔ {y_col}", f"{corr:.4f}")
    
    def _render_correlations(self, df):
        """Correlation analysis."""
        st.markdown("### 🔗 Correlation Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis")
            return
        
        corr_tabs = st.tabs(["Correlation Matrix", "Top Correlations", "Multicollinearity (VIF)"])
        
        with corr_tabs[0]:
            method = st.radio("Correlation Method", ["pearson", "spearman", "kendall"], horizontal=True)
            
            corr_matrix = numeric_df.corr(method=method)
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto",
                title=f"{method.title()} Correlation Matrix"
            )
            fig.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with corr_tabs[1]:
            st.markdown("#### Strongest Correlations")
            
            # Get upper triangle correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_pairs_df = pd.DataFrame(corr_pairs)
            corr_pairs_df['Abs Correlation'] = corr_pairs_df['Correlation'].abs()
            corr_pairs_df = corr_pairs_df.sort_values('Abs Correlation', ascending=False)
            corr_pairs_df['Strength'] = corr_pairs_df['Abs Correlation'].apply(
                lambda x: '🔴 Strong' if x > 0.7 else '🟡 Moderate' if x > 0.4 else '🟢 Weak'
            )
            
            st.dataframe(corr_pairs_df.head(20).round(4), use_container_width=True, hide_index=True)
            
            # Highlight strong correlations
            strong_corr = corr_pairs_df[corr_pairs_df['Abs Correlation'] > 0.7]
            if not strong_corr.empty:
                st.warning(f"⚠️ Found {len(strong_corr)} pairs with strong correlation (|r| > 0.7)")
        
        with corr_tabs[2]:
            st.markdown("#### Variance Inflation Factor (VIF)")
            st.caption("VIF > 5 indicates high multicollinearity. VIF > 10 is critical.")
            
            from sklearn.linear_model import LinearRegression
            
            # Calculate VIF for each feature
            vif_data = []
            clean_df = numeric_df.dropna()
            
            if len(clean_df) < len(numeric_df.columns) + 1:
                st.warning("Not enough complete cases for VIF calculation")
            else:
                for i, col in enumerate(clean_df.columns):
                    X = clean_df.drop(columns=[col])
                    y = clean_df[col]
                    
                    if X.shape[1] > 0 and len(y) > X.shape[1]:
                        try:
                            r2 = LinearRegression().fit(X, y).score(X, y)
                            vif = 1 / (1 - r2) if r2 < 1 else float('inf')
                            
                            vif_data.append({
                                'Feature': col,
                                'VIF': vif,
                                'R² (with others)': r2,
                                'Status': '✅ OK' if vif < 5 else '⚠️ High' if vif < 10 else '❌ Critical'
                            })
                        except:
                            pass
                
                if vif_data:
                    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.dataframe(vif_df.round(4), use_container_width=True, hide_index=True)
                    with col2:
                        fig = px.bar(
                            vif_df.head(15),
                            x='Feature',
                            y='VIF',
                            color='VIF',
                            color_continuous_scale='YlOrRd',
                            title='VIF by Feature'
                        )
                        fig.add_hline(y=5, line_dash="dash", line_color="yellow", annotation_text="Warning (5)")
                        fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Critical (10)")
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_analysis(self, df):
        """Feature importance and target analysis."""
        st.markdown("### 🎯 Feature Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            st.warning("Need at least 2 numeric columns")
            return
        
        feat_tabs = st.tabs(["Feature Importance", "Target Analysis", "Feature Clustering"])
        
        with feat_tabs[0]:
            st.markdown("#### Quick Feature Importance (Random Forest)")
            
            target_col = st.selectbox("Select Target Variable", numeric_df.columns, key="fi_target")
            
            if st.button("Calculate Feature Importance", use_container_width=True):
                with st.spinner("Training Random Forest..."):
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.preprocessing import StandardScaler
                    
                    X = numeric_df.drop(columns=[target_col]).dropna()
                    y = df.loc[X.index, target_col]
                    
                    # Handle any remaining NaN
                    valid_idx = ~y.isna()
                    X = X[valid_idx]
                    y = y[valid_idx]
                    
                    if len(X) < 10:
                        st.error("Not enough valid samples")
                        return
                    
                    # Scale and train
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                    rf.fit(X_scaled, y)
                    
                    # Get importance
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.dataframe(importance_df.round(4), use_container_width=True, hide_index=True)
                    
                    with col2:
                        fig = px.bar(
                            importance_df.head(15),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Viridis',
                            title='Feature Importance (Top 15)'
                        )
                        fig.update_layout(template="plotly_dark", height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Quick model performance
                    from sklearn.model_selection import cross_val_score
                    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
                    st.metric("Cross-Validation R² (5-fold)", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        with feat_tabs[1]:
            st.markdown("#### Target Variable Analysis")
            
            target_col = st.selectbox("Select Target", numeric_df.columns, key="ta_target")
            target_data = df[target_col].dropna()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = px.histogram(target_data, nbins=50, title=f"{target_col} Distribution")
                fig.add_vline(x=target_data.mean(), line_dash="dash", line_color="red", annotation_text="Mean")
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Target correlations
                target_corr = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
                
                fig = px.bar(
                    x=target_corr.values[:10],
                    y=target_corr.index[:10],
                    orientation='h',
                    title=f'Features Most Correlated with {target_col}',
                    labels={'x': 'Absolute Correlation', 'y': 'Feature'}
                )
                fig.update_layout(template="plotly_dark", height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with feat_tabs[2]:
            st.markdown("#### Feature Clustering")
            
            if len(numeric_df.columns) < 3:
                st.warning("Need at least 3 numeric columns for clustering")
                return
            
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            
            clean_df = numeric_df.dropna()
            
            if len(clean_df) < 10:
                st.warning("Not enough complete samples")
                return
            
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(clean_df)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            cluster_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': clusters.astype(str)
            })
            
            fig = px.scatter(
                cluster_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title=f'Feature Clustering (K={n_clusters}, PCA Visualization)',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance")
    
    def _render_quick_model(self, df):
        """Quick model benchmarking."""
        st.markdown("### ⚡ Quick Model Benchmark")
        st.caption("Compare multiple models on your data in seconds")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            st.warning("Need at least 2 numeric columns")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox("Target Variable", numeric_df.columns, key="qm_target")
        with col2:
            test_size = st.slider("Test Size %", 10, 40, 20) / 100
        
        feature_cols = [c for c in numeric_df.columns if c != target_col]
        selected_features = st.multiselect(
            "Select Features (or leave empty for all)",
            feature_cols,
            default=[]
        )
        
        if not selected_features:
            selected_features = feature_cols
        
        if st.button("🚀 Run Benchmark", use_container_width=True, type="primary"):
            with st.spinner("Training models..."):
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.svm import SVR
                from sklearn.neighbors import KNeighborsRegressor
                
                # Prepare data
                X = df[selected_features].dropna()
                y = df.loc[X.index, target_col]
                valid_idx = ~y.isna()
                X = X[valid_idx]
                y = y[valid_idx]
                
                if len(X) < 20:
                    st.error("Not enough valid samples (need at least 20)")
                    return
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Define models
                models = {
                    'Linear Regression': LinearRegression(),
                    'Ridge': Ridge(alpha=1.0),
                    'Lasso': Lasso(alpha=0.1),
                    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
                    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
                    'KNN': KNeighborsRegressor(n_neighbors=5),
                }
                
                try:
                    from xgboost import XGBRegressor
                    models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
                except ImportError:
                    pass
                
                results = []
                progress = st.progress(0)
                
                for i, (name, model) in enumerate(models.items()):
                    try:
                        # Train
                        if name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                        else:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring='r2')
                        
                        # Metrics
                        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                        
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        results.append({
                            'Model': name,
                            'R²': r2,
                            'RMSE': rmse,
                            'MAE': mae,
                            'CV R² (mean)': cv_scores.mean(),
                            'CV R² (std)': cv_scores.std()
                        })
                    except Exception as e:
                        st.warning(f"Error with {name}: {e}")
                    
                    progress.progress((i + 1) / len(models))
                
                progress.empty()
                
                if results:
                    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
                    results_df['Rank'] = range(1, len(results_df) + 1)
                    results_df = results_df[['Rank', 'Model', 'R²', 'RMSE', 'MAE', 'CV R² (mean)', 'CV R² (std)']]
                    
                    # Winner
                    winner = results_df.iloc[0]
                    st.success(f"🏆 **Best Model: {winner['Model']}** with R² = {winner['R²']:.4f}")
                    
                    # Results table
                    st.markdown("#### Benchmark Results")
                    st.dataframe(results_df.round(4), use_container_width=True, hide_index=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            results_df,
                            x='Model',
                            y='R²',
                            color='R²',
                            color_continuous_scale='Viridis',
                            title='R² Score by Model'
                        )
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            results_df,
                            x='Model',
                            y='RMSE',
                            color='RMSE',
                            color_continuous_scale='Reds_r',
                            title='RMSE by Model (Lower is Better)'
                        )
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # CV stability
                    fig = px.bar(
                        results_df,
                        x='Model',
                        y='CV R² (mean)',
                        error_y='CV R² (std)',
                        title='Cross-Validation R² with Std Dev',
                        color='CV R² (mean)',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_analysis_report(self, df):
        """Generate comprehensive analysis report."""
        st.markdown("### 📑 Analysis Report")
        
        if st.button("📄 Generate Full Report", use_container_width=True, type="primary"):
            with st.spinner("Generating comprehensive report..."):
                report = []
                report.append("# Comprehensive Data Analysis Report")
                report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report.append(f"\n**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
                
                # Executive Summary
                report.append("\n\n## Executive Summary")
                
                numeric_df = df.select_dtypes(include=[np.number])
                cat_df = df.select_dtypes(include=['object', 'category'])
                
                missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                dup_pct = df.duplicated().sum() / len(df) * 100
                
                report.append(f"\n- **Numeric Columns:** {len(numeric_df.columns)}")
                report.append(f"- **Categorical Columns:** {len(cat_df.columns)}")
                report.append(f"- **Missing Data:** {missing_pct:.2f}%")
                report.append(f"- **Duplicate Rows:** {dup_pct:.2f}%")
                report.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                
                # Data Quality
                report.append("\n\n## Data Quality Assessment")
                
                if missing_pct < 5:
                    report.append("\n✅ **Missing Data:** Excellent - Less than 5% missing")
                elif missing_pct < 20:
                    report.append("\n⚠️ **Missing Data:** Moderate - Consider imputation strategies")
                else:
                    report.append("\n❌ **Missing Data:** Critical - Significant missing data requires attention")
                
                if dup_pct < 1:
                    report.append("\n✅ **Duplicates:** Minimal duplicate rows")
                else:
                    report.append(f"\n⚠️ **Duplicates:** {df.duplicated().sum()} duplicate rows detected")
                
                # Column Summary
                report.append("\n\n## Column Summary")
                report.append("\n| Column | Type | Missing % | Unique |")
                report.append("|--------|------|-----------|--------|")
                for col in df.columns:
                    missing = df[col].isnull().sum() / len(df) * 100
                    unique = df[col].nunique()
                    dtype = str(df[col].dtype)
                    report.append(f"| {col} | {dtype} | {missing:.1f}% | {unique} |")
                
                # Statistical Summary
                if not numeric_df.empty:
                    report.append("\n\n## Numeric Summary Statistics")
                    desc = numeric_df.describe().round(4)
                    report.append("\n" + desc.to_markdown())
                
                # Correlations
                if len(numeric_df.columns) >= 2:
                    report.append("\n\n## Top Correlations")
                    corr_matrix = numeric_df.corr()
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                'Pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                    
                    top_corr = sorted(corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True)[:10]
                    report.append("\n| Feature Pair | Correlation |")
                    report.append("|--------------|-------------|")
                    for pair in top_corr:
                        report.append(f"| {pair['Pair']} | {pair['Correlation']:.4f} |")
                
                # Recommendations
                report.append("\n\n## Recommendations")
                
                if missing_pct > 5:
                    report.append("\n1. **Handle Missing Values:** Consider imputation (mean/median for numeric, mode for categorical) or row removal")
                
                if dup_pct > 0:
                    report.append("\n2. **Remove Duplicates:** Drop duplicate rows to prevent data leakage")
                
                high_card = [col for col in cat_df.columns if df[col].nunique() > 50]
                if high_card:
                    report.append(f"\n3. **High Cardinality:** Columns {high_card} have many unique values - consider encoding strategies")
                
                if len(numeric_df.columns) >= 2:
                    strong_corr = [(c1, c2) for c1 in numeric_df.columns for c2 in numeric_df.columns 
                                   if c1 < c2 and abs(numeric_df[c1].corr(numeric_df[c2])) > 0.8]
                    if strong_corr:
                        report.append(f"\n4. **Multicollinearity:** High correlations detected between {len(strong_corr)} feature pairs - consider feature selection")
                
                report.append("\n\n---\n*Report generated by Hockey ML Pipeline Dashboard*")
                
                # Display report
                full_report = "\n".join(report)
                st.markdown(full_report)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Report (Markdown)",
                        full_report,
                        file_name="data_analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col2:
                    # CSV summary
                    summary_df = df.describe(include='all').round(4)
                    st.download_button(
                        "📥 Download Statistics (CSV)",
                        summary_df.to_csv(),
                        file_name="data_statistics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

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
    
    def load_hyperparam_csvs(self):
        """Load hyperparameter search results from CSV files."""
        hyperparam_path = Path("../output/hyperparams")
        if not hyperparam_path.exists():
            hyperparam_path = Path("output/hyperparams")
        
        all_data = {}
        if hyperparam_path.exists():
            for csv_file in hyperparam_path.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    # Only include if it has some results (non-empty metrics)
                    if not df.empty:
                        model_name = csv_file.stem.replace("_grid_search", "").replace("_random_search", "")
                        search_type = "grid" if "grid" in csv_file.stem else "random"
                        key = f"{model_name} ({search_type})"
                        all_data[key] = df
                except Exception as e:
                    pass
        return all_data
    
    def render_hyperparam_graphs(self):
        """Render hyperparameter visualization with error metrics."""
        st.subheader("📊 Hyperparameter Search Results")
        
        hyperparam_data = self.load_hyperparam_csvs()
        
        if not hyperparam_data:
            st.warning("No hyperparameter CSV files found in output/hyperparams/")
            st.info("""
            **Expected file location:** `output/hyperparams/*.csv`
            
            Run hyperparameter searches to generate these files:
            ```
            python training/linear_hyperparam_search.py
            python training/xgboost_hyperparam_search.py
            ```
            """)
            return
        
        # Model selector
        selected_model = st.selectbox(
            "Select Model/Search", 
            list(hyperparam_data.keys()),
            key="hyperparam_model_select"
        )
        
        df = hyperparam_data[selected_model]
        
        # Show data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Configurations", len(df))
        with col2:
            has_results = df['r2'].notna().sum() if 'r2' in df.columns else 0
            st.metric("Completed Runs", has_results)
        with col3:
            if 'r2' in df.columns and df['r2'].notna().any():
                best_r2 = df['r2'].max()
                st.metric("Best R²", f"{best_r2:.4f}")
            else:
                st.metric("Best R²", "N/A")
        
        # Identify numeric hyperparameters and metrics
        metric_cols = ['rmse', 'mae', 'r2', 'mse', 'mape']
        available_metrics = [col for col in metric_cols if col in df.columns and df[col].notna().any()]
        
        # Exclude metrics and metadata to get hyperparameter columns
        exclude_cols = metric_cols + ['experiment_id', 'notes', 'timestamp', 'run_id']
        hyperparam_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not available_metrics:
            st.warning("No completed experiments with metrics found. The CSV has configurations but no results yet.")
            
            # Still show the configuration distribution
            st.markdown("### Configuration Distribution")
            numeric_params = []
            for col in hyperparam_cols:
                try:
                    if pd.to_numeric(df[col], errors='coerce').notna().any():
                        numeric_params.append(col)
                except:
                    pass
            
            if numeric_params:
                param_to_show = st.selectbox("Select parameter to visualize", numeric_params)
                fig = px.histogram(df, x=param_to_show, title=f"Distribution of {param_to_show}")
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data preview
            with st.expander("View Raw Configuration Data"):
                st.dataframe(df.head(50), use_container_width=True)
            return
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📈 Error by Hyperparameter",
            "🔥 Heatmaps", 
            "📉 Error Distribution",
            "🏆 Top Configurations"
        ])
        
        with viz_tab1:
            st.markdown("### Error Metrics vs Hyperparameters")
            
            # Get numeric hyperparameters
            numeric_params = []
            for col in hyperparam_cols:
                try:
                    numeric_vals = pd.to_numeric(df[col], errors='coerce')
                    if numeric_vals.notna().sum() > 1 and numeric_vals.nunique() > 1:
                        numeric_params.append(col)
                except:
                    pass
            
            # Get categorical hyperparameters
            categorical_params = [col for col in hyperparam_cols 
                                  if col not in numeric_params 
                                  and df[col].nunique() > 1 
                                  and df[col].nunique() <= 20]
            
            col1, col2 = st.columns(2)
            with col1:
                selected_metric = st.selectbox("Select Error Metric", available_metrics, key="error_metric")
            with col2:
                all_params = numeric_params + categorical_params
                if all_params:
                    selected_param = st.selectbox("Select Hyperparameter", all_params, key="hyperparam")
                else:
                    st.warning("No suitable hyperparameters found")
                    selected_param = None
            
            if selected_param and selected_metric:
                # Filter rows with valid metric values
                plot_df = df[df[selected_metric].notna()].copy()
                
                if len(plot_df) > 0:
                    if selected_param in numeric_params:
                        # Scatter plot for numeric params
                        fig = px.scatter(
                            plot_df,
                            x=selected_param,
                            y=selected_metric,
                            color=selected_metric,
                            color_continuous_scale='RdYlGn_r' if selected_metric in ['rmse', 'mae', 'mse', 'mape'] else 'RdYlGn',
                            title=f"{selected_metric.upper()} vs {selected_param}",
                            trendline="lowess" if len(plot_df) > 10 else None
                        )
                        fig.update_layout(template="plotly_dark", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Also show aggregated box plot
                        if plot_df[selected_param].nunique() <= 20:
                            fig2 = px.box(
                                plot_df,
                                x=selected_param,
                                y=selected_metric,
                                title=f"{selected_metric.upper()} Distribution by {selected_param}"
                            )
                            fig2.update_layout(template="plotly_dark", height=400)
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        # Box plot for categorical params
                        fig = px.box(
                            plot_df,
                            x=selected_param,
                            y=selected_metric,
                            color=selected_param,
                            title=f"{selected_metric.upper()} by {selected_param}"
                        )
                        fig.update_layout(template="plotly_dark", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Bar chart of means
                        agg_df = plot_df.groupby(selected_param)[selected_metric].agg(['mean', 'std', 'count']).reset_index()
                        fig2 = px.bar(
                            agg_df,
                            x=selected_param,
                            y='mean',
                            error_y='std',
                            title=f"Mean {selected_metric.upper()} by {selected_param}",
                            text='count'
                        )
                        fig2.update_traces(texttemplate='n=%{text}', textposition='outside')
                        fig2.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No data points with valid metric values")
        
        with viz_tab2:
            st.markdown("### Hyperparameter Interaction Heatmaps")
            
            if len(numeric_params) >= 2 and available_metrics:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_param = st.selectbox("X-axis Parameter", numeric_params, key="heatmap_x")
                with col2:
                    y_options = [p for p in numeric_params if p != x_param]
                    y_param = st.selectbox("Y-axis Parameter", y_options, key="heatmap_y") if y_options else None
                with col3:
                    heat_metric = st.selectbox("Metric", available_metrics, key="heatmap_metric")
                
                if x_param and y_param and heat_metric:
                    plot_df = df[df[heat_metric].notna()].copy()
                    
                    if len(plot_df) > 0:
                        # Create pivot table for heatmap
                        try:
                            pivot = plot_df.pivot_table(
                                values=heat_metric,
                                index=y_param,
                                columns=x_param,
                                aggfunc='mean'
                            )
                            
                            fig = px.imshow(
                                pivot,
                                labels=dict(x=x_param, y=y_param, color=heat_metric),
                                color_continuous_scale='RdYlGn_r' if heat_metric in ['rmse', 'mae', 'mse', 'mape'] else 'RdYlGn',
                                aspect="auto",
                                title=f"{heat_metric.upper()} Heatmap: {x_param} vs {y_param}"
                            )
                            fig.update_layout(template="plotly_dark", height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create heatmap: {e}")
            elif len(categorical_params) >= 1 and len(numeric_params) >= 1:
                st.info("Select a categorical and numeric parameter for the heatmap")
            else:
                st.info("Need at least 2 numeric hyperparameters for heatmap visualization")
        
        with viz_tab3:
            st.markdown("### Error Distribution Analysis")
            
            for metric in available_metrics:
                metric_data = df[metric].dropna()
                if len(metric_data) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            metric_data,
                            nbins=30,
                            title=f"{metric.upper()} Distribution",
                            labels={'value': metric.upper(), 'count': 'Count'}
                        )
                        fig.add_vline(x=metric_data.mean(), line_dash="dash", line_color="yellow",
                                      annotation_text=f"Mean: {metric_data.mean():.4f}")
                        fig.add_vline(x=metric_data.min(), line_dash="dot", line_color="green",
                                      annotation_text=f"Best: {metric_data.min():.4f}" if metric != 'r2' else f"Best: {metric_data.max():.4f}")
                        fig.update_layout(template="plotly_dark", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Stats summary
                        st.markdown(f"**{metric.upper()} Statistics:**")
                        stats_df = pd.DataFrame({
                            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                            'Value': [
                                f"{metric_data.mean():.4f}",
                                f"{metric_data.std():.4f}",
                                f"{metric_data.min():.4f}",
                                f"{metric_data.max():.4f}",
                                f"{metric_data.median():.4f}",
                                f"{metric_data.quantile(0.25):.4f}",
                                f"{metric_data.quantile(0.75):.4f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with viz_tab4:
            st.markdown("### Top Performing Configurations")
            
            # Sort by best metric
            sort_metric = st.selectbox("Rank by", available_metrics, key="rank_metric")
            ascending = sort_metric in ['rmse', 'mae', 'mse', 'mape']  # Lower is better for errors
            
            # Filter and sort
            valid_df = df[df[sort_metric].notna()].copy()
            valid_df = valid_df.sort_values(sort_metric, ascending=ascending)
            
            # Show top 10
            top_n = min(10, len(valid_df))
            if top_n > 0:
                top_configs = valid_df.head(top_n).copy()
                top_configs.insert(0, 'Rank', range(1, top_n + 1))
                
                # Highlight the display columns
                display_cols = ['Rank'] + hyperparam_cols + available_metrics
                display_cols = [c for c in display_cols if c in top_configs.columns]
                
                st.dataframe(
                    top_configs[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Parallel coordinates plot for top configs
                if len(numeric_params) >= 2:
                    st.markdown("### Parallel Coordinates - Top 10 Configurations")
                    
                    plot_cols = numeric_params[:5] + [sort_metric]  # Limit to 5 params + metric
                    plot_cols = [c for c in plot_cols if c in top_configs.columns]
                    
                    if len(plot_cols) >= 3:
                        fig = px.parallel_coordinates(
                            top_configs,
                            dimensions=plot_cols,
                            color=sort_metric,
                            color_continuous_scale='RdYlGn_r' if ascending else 'RdYlGn',
                            title="Hyperparameter Patterns in Top Configurations"
                        )
                        fig.update_layout(template="plotly_dark", height=500)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No configurations with valid metrics found")
        
        # Raw data expander
        with st.expander("📋 View All Data"):
            st.dataframe(df, use_container_width=True)

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
            
            if st.button("Run Hyperparameter Search", use_container_width=True):
                st.code("python training/linear_hyperparam_search.py", language="bash")
            
            if st.button("Train Final Models", use_container_width=True):
                st.code("python training/automated_model_selection.py", language="bash")
            
            if st.button("Open MLflow UI", use_container_width=True):
                import subprocess
                import webbrowser
                mlruns_path = Path(__file__).parent / "mlruns"
                if not mlruns_path.exists():
                    mlruns_path = Path(__file__).parent.parent / "mlruns"
                try:
                    subprocess.Popen(
                        ["mlflow", "ui", "--backend-store-uri", f"file:///{mlruns_path.absolute()}"],
                        cwd=str(mlruns_path.parent),
                        creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
                    )
                    time.sleep(2)
                    webbrowser.open("http://localhost:5000")
                    st.success("MLflow UI launched at http://localhost:5000")
                except Exception as e:
                    st.error(f"Failed to launch MLflow: {e}")
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
    
    def render_terminal_view(self, runs_df):
        """Professional trading terminal style multi-panel view."""
        
        # Top status bar
        status_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
        
        with status_cols[0]:
            st.markdown('<div class="panel-header">STATUS</div>', unsafe_allow_html=True)
            st.markdown('<span class="status-live">LIVE</span>', unsafe_allow_html=True)
        
        with status_cols[1]:
            st.markdown('<div class="panel-header">RUNS</div>', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#fff;font-family:monospace">{len(runs_df) if not runs_df.empty else 0}</span>', unsafe_allow_html=True)
        
        with status_cols[2]:
            st.markdown('<div class="panel-header">MODELS</div>', unsafe_allow_html=True)
            n_models = runs_df['model'].nunique() if not runs_df.empty and 'model' in runs_df.columns else 0
            st.markdown(f'<span style="color:#fff;font-family:monospace">{n_models}</span>', unsafe_allow_html=True)
        
        with status_cols[3]:
            st.markdown('<div class="panel-header">BEST R2</div>', unsafe_allow_html=True)
            best_r2 = runs_df['metrics.test_r2'].max() if not runs_df.empty and 'metrics.test_r2' in runs_df.columns else 0
            color = "#00d4aa" if best_r2 > 0.5 else "#ffaa00" if best_r2 > 0.2 else "#ff4444"
            st.markdown(f'<span style="color:{color};font-family:monospace">{best_r2:.4f}</span>', unsafe_allow_html=True)
        
        with status_cols[4]:
            st.markdown('<div class="panel-header">BEST RMSE</div>', unsafe_allow_html=True)
            best_rmse = runs_df['metrics.test_rmse'].min() if not runs_df.empty and 'metrics.test_rmse' in runs_df.columns else 0
            st.markdown(f'<span style="color:#00d4aa;font-family:monospace">{best_rmse:.4f}</span>', unsafe_allow_html=True)
        
        with status_cols[5]:
            st.markdown('<div class="panel-header">AVG R2</div>', unsafe_allow_html=True)
            avg_r2 = runs_df['metrics.test_r2'].mean() if not runs_df.empty and 'metrics.test_r2' in runs_df.columns else 0
            st.markdown(f'<span style="color:#888;font-family:monospace">{avg_r2:.4f}</span>', unsafe_allow_html=True)
        
        with status_cols[6]:
            st.markdown('<div class="panel-header">UPDATED</div>', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#888;font-family:monospace">{datetime.now().strftime("%H:%M:%S")}</span>', unsafe_allow_html=True)
        
        with status_cols[7]:
            if st.button("REFRESH", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        
        # Main chart grid - 2x2 layout like trading terminal
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Main performance chart
            self._render_main_chart(runs_df)
            
            # Secondary charts row
            chart_row = st.columns(2)
            with chart_row[0]:
                self._render_model_comparison_chart(runs_df)
            with chart_row[1]:
                self._render_rmse_chart(runs_df)
        
        with col_right:
            # Leaderboard table
            self._render_compact_leaderboard(runs_df)
            
            # Recent runs
            self._render_recent_runs(runs_df)
            
            # Model stats
            self._render_model_stats(runs_df)
    
    def _render_main_chart(self, runs_df):
        """Main time series chart like a price chart."""
        st.markdown('<div class="panel-header">PERFORMANCE OVER TIME</div>', unsafe_allow_html=True)
        
        if runs_df.empty or 'start_time' not in runs_df.columns or 'metrics.test_r2' not in runs_df.columns:
            # Generate sample data if no real data
            fig = go.Figure()
            fig.add_annotation(text="No experiment data - run training to see results", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                             font=dict(color="#666", size=14))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#0a0a0a',
                height=300,
                margin=dict(l=40, r=20, t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            return
        
        runs_df = runs_df.copy()
        runs_df['timestamp'] = pd.to_datetime(runs_df['start_time'])
        runs_df = runs_df.sort_values('timestamp')
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           row_heights=[0.7, 0.3], vertical_spacing=0.02)
        
        # Main R2 line (like price)
        for model in runs_df['model'].unique():
            model_data = runs_df[runs_df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['metrics.test_r2'],
                    mode='lines+markers',
                    name=model,
                    line=dict(width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
        
        # Volume-style bar chart (number of runs per time period)
        runs_df['date'] = runs_df['timestamp'].dt.date
        run_counts = runs_df.groupby('date').size().reset_index(name='count')
        run_counts['date'] = pd.to_datetime(run_counts['date'])
        
        fig.add_trace(
            go.Bar(
                x=run_counts['date'],
                y=run_counts['count'],
                name='Run Count',
                marker_color='#1a4a6e',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0a0a0a',
            height=350,
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                       font=dict(size=10)),
            hovermode='x unified'
        )
        
        fig.update_xaxes(gridcolor='#1a1a2e', zeroline=False)
        fig.update_yaxes(gridcolor='#1a1a2e', zeroline=False)
        fig.update_yaxes(title_text="R2", row=1, col=1, title_font=dict(size=10))
        fig.update_yaxes(title_text="Runs", row=2, col=1, title_font=dict(size=10))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_comparison_chart(self, runs_df):
        """Compact model comparison bar chart."""
        st.markdown('<div class="panel-header">MODEL COMPARISON</div>', unsafe_allow_html=True)
        
        if runs_df.empty or 'metrics.test_r2' not in runs_df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                             font=dict(color="#666", size=12))
        else:
            model_stats = runs_df.groupby('model').agg({
                'metrics.test_r2': ['max', 'mean']
            }).reset_index()
            model_stats.columns = ['model', 'max_r2', 'mean_r2']
            model_stats = model_stats.sort_values('max_r2', ascending=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=model_stats['model'],
                x=model_stats['max_r2'],
                orientation='h',
                name='Best',
                marker_color='#00d4aa'
            ))
            
            fig.add_trace(go.Bar(
                y=model_stats['model'],
                x=model_stats['mean_r2'],
                orientation='h',
                name='Avg',
                marker_color='#1a4a6e'
            ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0a0a0a',
            height=200,
            margin=dict(l=80, r=20, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                       font=dict(size=9)),
            barmode='group',
            bargap=0.3
        )
        fig.update_xaxes(gridcolor='#1a1a2e', zeroline=False, title_text="R2", title_font=dict(size=9))
        fig.update_yaxes(gridcolor='#1a1a2e', zeroline=False, tickfont=dict(size=9))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_rmse_chart(self, runs_df):
        """RMSE distribution chart."""
        st.markdown('<div class="panel-header">ERROR DISTRIBUTION</div>', unsafe_allow_html=True)
        
        if runs_df.empty or 'metrics.test_rmse' not in runs_df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No RMSE data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                             font=dict(color="#666", size=12))
        else:
            rmse_data = runs_df['metrics.test_rmse'].dropna()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=rmse_data,
                nbinsx=30,
                marker_color='#ff4444',
                opacity=0.7
            ))
            
            # Add mean line
            fig.add_vline(x=rmse_data.mean(), line_dash="dash", line_color="#ffaa00",
                         annotation_text=f"Mean: {rmse_data.mean():.3f}",
                         annotation_font_size=9)
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0a0a0a',
            height=200,
            margin=dict(l=40, r=20, t=10, b=30),
            showlegend=False
        )
        fig.update_xaxes(gridcolor='#1a1a2e', zeroline=False, title_text="RMSE", title_font=dict(size=9))
        fig.update_yaxes(gridcolor='#1a1a2e', zeroline=False, title_text="Count", title_font=dict(size=9))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_compact_leaderboard(self, runs_df):
        """Compact leaderboard table."""
        st.markdown('<div class="panel-header">LEADERBOARD</div>', unsafe_allow_html=True)
        
        if runs_df.empty or 'metrics.test_r2' not in runs_df.columns:
            st.markdown('<span style="color:#666">No results yet</span>', unsafe_allow_html=True)
            return
        
        # Top 8 runs
        top_runs = runs_df.nlargest(8, 'metrics.test_r2')[['model', 'metrics.test_r2']].copy()
        top_runs.columns = ['Model', 'R2']
        top_runs['R2'] = top_runs['R2'].apply(lambda x: f"{x:.4f}")
        top_runs.insert(0, '#', range(1, len(top_runs) + 1))
        
        # Display as styled table
        st.dataframe(top_runs, use_container_width=True, hide_index=True, height=220)
    
    def _render_recent_runs(self, runs_df):
        """Recent experiment runs."""
        st.markdown('<div class="panel-header">RECENT RUNS</div>', unsafe_allow_html=True)
        
        if runs_df.empty or 'start_time' not in runs_df.columns:
            st.markdown('<span style="color:#666">No runs yet</span>', unsafe_allow_html=True)
            return
        
        runs_df = runs_df.copy()
        runs_df['time'] = pd.to_datetime(runs_df['start_time']).dt.strftime('%m/%d %H:%M')
        
        recent = runs_df.nlargest(5, 'start_time')[['model', 'time']].copy()
        recent.columns = ['Model', 'Time']
        
        st.dataframe(recent, use_container_width=True, hide_index=True, height=150)
    
    def _render_model_stats(self, runs_df):
        """Per-model statistics."""
        st.markdown('<div class="panel-header">MODEL STATS</div>', unsafe_allow_html=True)
        
        if runs_df.empty or 'model' not in runs_df.columns:
            st.markdown('<span style="color:#666">No stats</span>', unsafe_allow_html=True)
            return
        
        stats = runs_df.groupby('model').agg({
            'run_id': 'count'
        }).reset_index()
        stats.columns = ['Model', 'Runs']
        
        if 'metrics.test_r2' in runs_df.columns:
            r2_stats = runs_df.groupby('model')['metrics.test_r2'].max().reset_index()
            r2_stats.columns = ['Model', 'Best']
            stats = stats.merge(r2_stats, on='Model')
            stats['Best'] = stats['Best'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(stats, use_container_width=True, hide_index=True, height=150)

    def run(self):
        """Main dashboard loop."""
        self.render_header()
        
        # Load data
        runs_df = self.load_mlflow_experiments()
        
        # Render sidebar
        self.render_sidebar()
        
        st.markdown("---")
        
        # Main content
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview",
            "Experiments",
            "Hyperparams",
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
            self.render_hyperparam_graphs()
        
        with tab4:
            self.render_advanced_metrics(runs_df)
        
        with tab5:
            self.render_data_uploader()
        
        with tab6:
            st.subheader("Model Leaderboard")
            if not runs_df.empty and 'metrics.test_r2' in runs_df.columns:
                cols_to_show = ['model', 'metrics.test_r2']
                col_config = {
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "model": st.column_config.TextColumn("Model"),
                    "metrics.test_r2": st.column_config.NumberColumn("R2", format="%.4f")
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
