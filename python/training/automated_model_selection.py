"""
Automated Model Selection Pipeline

This script:
1. Extracts best hyperparameters from MLflow experiments
2. Trains final models with optimal settings
3. Compares all models
4. Saves the best model for production
5. Generates a comprehensive report

Run this after hyperparameter searches complete.
"""

import mlflow
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class AutomatedModelSelection:
    def __init__(self):
        self.mlruns_path = Path(__file__).parent.parent / "mlruns"
        mlflow.set_tracking_uri(f"file:///{str(self.mlruns_path).replace(chr(92), '/')}")
        
        self.output_dir = Path(__file__).parent.parent / "output"
        self.models_dir = Path(__file__).parent.parent / "models"
        self.hyperparams_dir = self.output_dir / "hyperparams"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.hyperparams_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        
    def extract_best_hyperparameters(self):
        """Extract best hyperparameters from each MLflow experiment."""
        print("=" * 80)
        print("STEP 1: Extracting Best Hyperparameters from MLflow")
        print("=" * 80)
        
        experiments = {
            "XGBoost": "xgboost_hyperparam_search",
            "Linear": "linear_hyperparam_search",
            "Elo": "elo_hyperparam_search",
            "RandomForest": "random_forest_hyperparam_search"
        }
        
        best_params = {}
        
        for model_name, exp_name in experiments.items():
            try:
                # Get experiment
                experiment = mlflow.get_experiment_by_name(exp_name)
                if not experiment:
                    print(f"  WARNING: {model_name}: No experiment found (skipping)")
                    continue
                
                # Get all runs
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.test_r2 DESC", "metrics.test_accuracy DESC"]
                )
                
                if runs.empty:
                    print(f"  WARNING: {model_name}: No runs found (skipping)")
                    continue
                
                # Get best run
                best_run = runs.iloc[0]
                
                # Extract parameters (filter columns starting with 'params.')
                params = {}
                for col in best_run.index:
                    if col.startswith('params.'):
                        param_name = col.replace('params.', '')
                        param_value = best_run[col]
                        if pd.notna(param_value):
                            params[param_name] = param_value
                
                # Extract metrics
                metrics = {}
                for col in best_run.index:
                    if col.startswith('metrics.'):
                        metric_name = col.replace('metrics.', '')
                        metrics[metric_name] = best_run[col]
                
                best_params[model_name] = {
                    "parameters": params,
                    "metrics": metrics,
                    "run_id": best_run['run_id']
                }
                
                # Print summary
                score_metric = metrics.get('test_r2', metrics.get('test_accuracy', 0))
                print(f"  [OK] {model_name}: Score = {score_metric:.4f}")
                
            except Exception as e:
                print(f"  [ERROR] {model_name}: Error - {str(e)}")
        
        # Save to JSON
        output_file = self.hyperparams_dir / "best_hyperparameters.json"
        with open(output_file, 'w') as f:
            json.dump(best_params, f, indent=2, default=str)
        
        print(f"\n   Saved to: {output_file}")
        return best_params
    
    def load_data(self):
        """Load and prepare training data."""
        print("\n" + "=" * 80)
        print("STEP 2: Loading Training Data")
        print("=" * 80)
        
        # Look for data files
        data_dir = Path(__file__).parent.parent / "data"
        
        # Try to find CSV files
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            print("  WARNING: No CSV files found in data directory")
            print("  Using synthetic data for demonstration...")
            
            # Create synthetic data
            np.random.seed(42)
            n_samples = 1000
            X = np.random.randn(n_samples, 10)
            y = X[:, 0] * 2 + X[:, 1] * -1 + X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"  [OK] Created synthetic dataset: {n_samples} samples, 10 features")
        else:
            # Use first file found (handles CSV, Excel, JSON)
            data_file = csv_files[0]
            print(f"   Loading: {data_file.name}")
            
            # Load based on file extension
            file_ext = data_file.suffix.lower()
            if file_ext == '.csv':
                df = pd.read_csv(data_file)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(data_file)
            elif file_ext == '.json':
                df = pd.read_json(data_file)
            else:
                # Try CSV as default
                df = pd.read_csv(data_file)
            
            print(f"  [OK] Loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Try to identify target column (flexible keywords)
            target_cols = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['target', 'label', 'y', 'outcome', 'result', 
                                                        'prediction', 'score', 'value', 'price', 
                                                        'amount', 'total', 'win', 'loss']
            )]
            
            if target_cols:
                target = target_cols[0]
            else:
                # Use last column as target
                target = df.columns[-1]
            
            print(f"   Target: {target}")
            
            # Prepare features and target
            X = df.drop(columns=[target])
            y = df[target]
            
            # Handle missing values in target
            if y.isnull().any():
                print(f"   WARNING: Dropping {y.isnull().sum()} rows with missing target values")
                valid_mask = ~y.isnull()
                X = X[valid_mask]
                y = y[valid_mask]
            
            # Handle non-numeric columns in features
            # Step 1: Drop columns with >50% missing values
            missing_pct = X.isnull().mean()
            high_missing = missing_pct[missing_pct > 0.5].index.tolist()
            if high_missing:
                print(f"   WARNING: Dropping {len(high_missing)} columns with >50% missing: {high_missing[:3]}...")
                X = X.drop(columns=high_missing)
            
            # Step 2: Identify categorical vs numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                print(f"   Found {len(categorical_cols)} categorical columns")
                # One-hot encode categoricals with <=10 unique values
                for col in categorical_cols:
                    if X[col].nunique() <= 10:
                        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                        X = pd.concat([X, dummies], axis=1)
                    X = X.drop(columns=[col])
            
            # Step 3: Fill missing numeric values with median
            if X.isnull().any().any():
                print(f"   Filling {X.isnull().sum().sum()} missing numeric values with median")
                X = X.fillna(X.median())
            
            # Final check: only numeric data
            X = X.select_dtypes(include=[np.number])
            
            print(f"  [OK] Final features: {X.shape[1]} columns")
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"  [OK] Train: {len(self.X_train)} | Test: {len(self.X_test)}")
    
    def train_final_models(self, best_params):
        """Train final models with best hyperparameters."""
        print("\n" + "=" * 80)
        print("STEP 3: Training Final Models")
        print("=" * 80)
        
        trained_models = {}
        
        # XGBoost
        if "XGBoost" in best_params:
            print("\n   Training XGBoost...")
            params = best_params["XGBoost"]["parameters"]
            
            # Convert params
            xgb_params = {
                'max_depth': int(float(params.get('max_depth', 5))),
                'learning_rate': float(params.get('learning_rate', 0.1)),
                'n_estimators': int(float(params.get('n_estimators', 100))),
                'subsample': float(params.get('subsample', 0.8)),
                'colsample_bytree': float(params.get('colsample_bytree', 0.8)),
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(self.X_train, self.y_train)
            trained_models['XGBoost'] = model
            print("  [OK] XGBoost trained")
        
        # Linear Models
        if "Linear" in best_params:
            print("\n   Training Linear Model...")
            params = best_params["Linear"]["parameters"]
            
            model_type = params.get('model_type', 'ridge')
            alpha = float(params.get('alpha', 1.0))
            
            if model_type == 'ridge':
                model = Ridge(alpha=alpha, random_state=42)
            else:
                model = Lasso(alpha=alpha, random_state=42)
            
            model.fit(self.X_train, self.y_train)
            trained_models['Linear'] = model
            print(f"  [OK] {model_type.capitalize()} trained")
        
        # Random Forest (if Ensemble exists)
        if "Ensemble" in best_params:
            print("\n   Training Random Forest...")
            # Use sensible defaults
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(self.X_train, self.y_train)
            trained_models['RandomForest'] = model
            print("  [OK] Random Forest trained")
        
        return trained_models
    
    def evaluate_models(self, models):
        """Evaluate all trained models."""
        print("\n" + "=" * 80)
        print("STEP 4: Evaluating Models")
        print("=" * 80)
        
        results = []
        
        for name, model in models.items():
            print(f"\n   Evaluating {name}...")
            
            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            result = {
                'Model': name,
                'Train R2': train_r2,
                'Test R2': test_r2,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Overfit Gap': train_r2 - test_r2
            }
            
            results.append(result)
            
            print(f"    R2 (train/test): {train_r2:.4f} / {test_r2:.4f}")
            print(f"    RMSE (train/test): {train_rmse:.4f} / {test_rmse:.4f}")
            
            # Track best model
            if test_r2 > self.best_score:
                self.best_score = test_r2
                self.best_model = model
                self.best_model_name = name
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def save_best_model(self):
        """Save the best model for production."""
        print("\n" + "=" * 80)
        print("STEP 5: Saving Best Model")
        print("=" * 80)
        
        if self.best_model is None:
            print("  WARNING: No models were trained")
            return
        
        # Save model
        model_file = self.models_dir / "production_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'test_r2': float(self.best_score),
            'trained_at': datetime.now().isoformat(),
            'model_file': str(model_file)
        }
        
        metadata_file = self.models_dir / "production_model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n   WINNER: {self.best_model_name}")
        print(f"   Test R2: {self.best_score:.4f}")
        print(f"   Saved to: {model_file}")
        print(f"   Metadata: {metadata_file}")
    
    def generate_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "=" * 80)
        print("STEP 6: Generating Report")
        print("=" * 80)
        
        # Save results CSV
        csv_file = self.output_dir / "model_comparison.csv"
        self.results.to_csv(csv_file, index=False)
        print(f"\n   Comparison CSV: {csv_file}")
        
        # Generate markdown report
        report = []
        report.append("# Model Selection Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Summary")
        report.append(f"\n**Best Model:** {self.best_model_name}")
        report.append(f"**Test R2 Score:** {self.best_score:.4f}")
        
        report.append("\n## All Models Comparison")
        report.append("\n" + self.results.to_markdown(index=False))
        
        report.append("\n## Rankings")
        sorted_results = self.results.sort_values('Test R2', ascending=False)
        for idx, row in sorted_results.iterrows():
            rank = sorted_results.index.get_loc(idx) + 1
            report.append(f"\n{rank}. **{row['Model']}** - R2 = {row['Test R2']:.4f}")
        
        report.append("\n## Production Deployment")
        report.append(f"\n The **{self.best_model_name}** model is ready for production.")
        report.append(f"\n Model file: `models/production_model.pkl`")
        report.append("\n### Usage Example")
        report.append("```python")
        report.append("import pickle")
        report.append("")
        report.append("# Load model")
        report.append("with open('models/production_model.pkl', 'rb') as f:")
        report.append("    model = pickle.load(f)")
        report.append("")
        report.append("# Make predictions")
        report.append("predictions = model.predict(X_new)")
        report.append("```")
        
        # Save report
        report_file = self.output_dir / "MODEL_SELECTION_REPORT.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"   Report: {report_file}")
        
        # Print to console
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(self.results.to_string(index=False))
        print("\n" + "=" * 80)
        print(f" WINNER: {self.best_model_name} (R2 = {self.best_score:.4f})")
        print("=" * 80)
    
    def run(self):
        """Run the complete automated pipeline."""
        print("\n" + "" * 40)
        print("AUTOMATED MODEL SELECTION PIPELINE")
        print("" * 40 + "\n")
        
        try:
            # Step 1: Extract best hyperparameters
            best_params = self.extract_best_hyperparameters()
            
            if not best_params:
                print("\nERROR: No hyperparameter results found in MLflow.")
                print("   Run the hyperparameter search scripts first!")
                return
            
            # Step 2: Load data
            self.load_data()
            
            # Step 3: Train models
            models = self.train_final_models(best_params)
            
            if not models:
                print("\nWARNING: No models could be trained")
                return
            
            # Step 4: Evaluate
            self.evaluate_models(models)
            
            # Step 5: Save best
            self.save_best_model()
            
            # Step 6: Report
            self.generate_report()
            
            print("\n" + "" * 40)
            print("PIPELINE COMPLETE!")
            print("" * 40 + "\n")
            
        except Exception as e:
            print(f"\nERROR: Error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    pipeline = AutomatedModelSelection()
    pipeline.run()
