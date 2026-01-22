# Start MLflow UI pointing to the correct mlruns directory
# Run this from anywhere with: .\python\start_mlflow.ps1

$MLRUNS_PATH = Join-Path $PSScriptRoot "mlruns"
Write-Host "Starting MLflow UI..."
Write-Host "Tracking URI: $MLRUNS_PATH"
Write-Host "Open: http://localhost:5000"
Write-Host ""
mlflow ui --backend-store-uri "file:///$($MLRUNS_PATH -replace '\\', '/')" --port 5000
