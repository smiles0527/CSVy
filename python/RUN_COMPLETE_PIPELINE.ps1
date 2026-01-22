# Complete Automated ML Pipeline
# This runs everything from start to finish

Write-Host "`n" -NoNewline
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "COMPLETE AUTOMATED ML PIPELINE" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Change to python directory
Set-Location $PSScriptRoot

# Activate virtual environment if it exists
if (Test-Path "..\..\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ..\..\.venv\Scripts\Activate.ps1
}

Write-Host "`nPHASE 1: Hyperparameter Search" -ForegroundColor Green
Write-Host "-" * 80
Write-Host "This will run all 5 model hyperparameter searches..."
Write-Host "Estimated time: 5-10 minutes"
Write-Host ""

$response = Read-Host "Run hyperparameter searches? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y' -or $response -eq '') {
    Write-Host "`n[1/5] XGBoost..." -ForegroundColor Cyan
    python training/xgboost_hyperparam_search.py
    
    Write-Host "`n[2/5] Linear Regression..." -ForegroundColor Cyan
    python training/linear_hyperparam_search.py
    
    Write-Host "`n[3/5] Elo..." -ForegroundColor Cyan
    python training/elo_hyperparam_search.py
    
    Write-Host "`n[4/5] Ensemble..." -ForegroundColor Cyan
    python training/ensemble_hyperparam_search.py
    
    Write-Host "`n[5/5] Neural Network..." -ForegroundColor Cyan
    python training/neural_network_hyperparam_search.py
    
    Write-Host "`nAll hyperparameter searches complete!" -ForegroundColor Green
} else {
    Write-Host "Skipping hyperparameter searches..." -ForegroundColor Yellow
}

Write-Host "`n"
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "PHASE 2: Automated Model Selection" -ForegroundColor Green
Write-Host "-" * 80
Write-Host "This will:"
Write-Host "  1. Extract best hyperparameters from MLflow"
Write-Host "  2. Train final models"
Write-Host "  3. Compare all models"
Write-Host "  4. Save the best model for production"
Write-Host "  5. Generate a detailed report"
Write-Host ""

python training/automated_model_selection.py

Write-Host "`n"
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Check these files:" -ForegroundColor Yellow
Write-Host "   - output/MODEL_SELECTION_REPORT.md  - Full report"
Write-Host "   - output/model_comparison.csv       - Results comparison"
Write-Host "   - models/production_model.pkl       - Best model (ready to use)"
Write-Host "   - output/hyperparams/               - Best hyperparameters"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "   1. Review MODEL_SELECTION_REPORT.md"
Write-Host "   2. Use production_model.pkl in your application"
Write-Host "   3. Run: .\start_mlflow.ps1 to view experiment details"
Write-Host ""
