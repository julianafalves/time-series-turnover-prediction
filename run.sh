#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
VENV_DIR="./.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
# Exact name of your input file
CSV_FILE="data/Time_Series___Turnover (1).csv"

# --- 1. Environment Setup ---
echo "--- 1. Checking Virtual Environment ---"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
fi

# Always ensure dependencies are installed (including tensorflow, statsmodels)
echo "Installing/Updating dependencies..."
"$VENV_PYTHON" -m pip install --prefer-binary -r requirements.txt > /dev/null
echo "Installing project in editable mode..."
"$VENV_PYTHON" -m pip install -e . > /dev/null

# --- 2. Create Output Directories ---
echo "--- 2. Creating Output Directories ---"
mkdir -p models reports/benchmark reports/eda data

# --- 3. Pipeline Execution ---

echo "--- 3. Running Exploratory Data Analysis (EDA) ---"
# Generates seasonal decomposition and global trend plots
"$VENV_PYTHON" -m turnover_prediction.eda \
    --input "$CSV_FILE" \
    --out-dir "reports/eda"

echo "--- 4. Running Preprocessing (Global Forecasting Model) ---"
# Generates group-aware lag features and saves .joblib artifacts
"$VENV_PYTHON" -m turnover_prediction.preprocessing \
    --input "$CSV_FILE" \
    --out-dir "data" \
    --mode ts \
    --n-lags 12

echo "--- 5. Running Benchmark (RF vs XGB vs LSTM) ---"
# Trains and compares models, generating comparative metrics
"$VENV_PYTHON" -m turnover_prediction.benchmark \
    --input "$CSV_FILE" \
    --out-dir "reports/benchmark" \
    --n-lags 12

echo "--- 5.5. Running Hyperparameter Tuning (Time Series CV) ---"
# Optimizes the model and saves the best parameters
"$VENV_PYTHON" -m turnover_prediction.tuning \
    --input "$CSV_FILE" \
    --out-dir "reports"

echo "--- 6. Training Final Model with Tuned Hyperparameters ---"
# Train the final model using the best hyperparameters from tuning
"$VENV_PYTHON" -m turnover_prediction.train \
    --prepared "data/prepared.joblib" \
    --model-output "models/xgb_turnover.joblib" \
    --task ts \
    --params "reports/best_params.json"

echo "--- 7. Running Advanced Evaluation (Diebold-Mariano Test) ---"
# Generates ACF plots of residuals and performs statistical significance tests
"$VENV_PYTHON" -m turnover_prediction.evaluate \
    --prepared "data/prepared.joblib" \
    --model "models/xgb_turnover.joblib" \
    --report-dir "reports"

echo "========================================================"
echo "Pipeline completed successfully!"
echo "Reports generated in: reports/"
echo "Model saved in: models/xgb_turnover.joblib"
echo "========================================================"