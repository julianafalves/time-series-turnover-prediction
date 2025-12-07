# PowerShell script to run the full pipeline using venv
# This script will create a .venv, install dependencies and run the pipeline with the venv's Python.

$venvDir = ".\.venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
	Write-Host "Creating virtual environment at $venvDir..."
	python -m venv $venvDir
	Write-Host "Upgrading pip and ensuring setuptools/wheel installed..."
	& $venvPython -m pip install --upgrade pip
	& $venvPython -m pip install --upgrade setuptools wheel
	Write-Host "Installing requirements (preferring binary wheels to avoid building from source)..."
	& $venvPython -m pip install --prefer-binary -r requirements.txt
} else {
	Write-Host "Using existing virtual environment at $venvDir"
}

# Install project in editable mode
& $venvPython -m pip install -e .

# Ensure output directories exist
New-Item -ItemType Directory -Force -Path "models","reports","data" | Out-Null

Write-Host "Running pipeline via venv Python: $venvPython"
# Pipeline mode: 'tabular' or 'ts' (time series). We use the user-provided CSV file by default.
$pipelineMode = 'ts'

# Use the CSV the user added. Prefer `data/time_series_turnover.csv` otherwise fallback to existing file name if present.
$userCsv = "data/time_series_turnover.csv"
$fallbackCsv = "data/Time_Series___Turnover (1).csv"
if (Test-Path $userCsv) {
	$csvFile = $userCsv
} elseif (Test-Path $fallbackCsv) {
	$csvFile = $fallbackCsv
	Write-Host "Using fallback CSV file: $fallbackCsv. To prefer a different CSV, put your file at data/time_series_turnover.csv"
} else {
	Write-Error "No user CSV file found. Please place your dataset at data/time_series_turnover.csv"
	exit 1
}
if ($pipelineMode -eq 'ts') {
    & $venvPython -m turnover_prediction.preprocessing --input $csvFile --out-dir "data" --mode ts --n-lags 12 --date-col date
	& $venvPython -m turnover_prediction.train --prepared "data/prepared.joblib" --model-output "models/xgb_turnover.joblib" --task ts --n-estimators 100
	& $venvPython -m turnover_prediction.evaluate --prepared "data/prepared.joblib" --model "models/xgb_turnover.joblib" --report-dir "reports" --task ts
	& $venvPython -m turnover_prediction.explain --model "models/xgb_turnover.joblib" --prepared "data/prepared.joblib" --out-dir "reports" --task ts
	& $venvPython -m turnover_prediction.benchmark --input $csvFile --out-dir "reports/benchmark" --n-lags 12
} else {
	& $venvPython -m turnover_prediction.preprocessing --input $csvFile --out-dir "data" --mode tabular
	& $venvPython -m turnover_prediction.train --prepared "data/prepared.joblib" --model-output "models/rf_turnover.joblib" --task classification --n-estimators 100
	& $venvPython -m turnover_prediction.evaluate --prepared "data/prepared.joblib" --model "models/rf_turnover.joblib" --report-dir "reports" --task classification
	& $venvPython -m turnover_prediction.explain --model "models/rf_turnover.joblib" --prepared "data/prepared.joblib" --out-dir "reports" --task classification
}
