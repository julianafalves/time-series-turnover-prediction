#!/bin/bash

# Setup script for Python virtual environment and dependencies
# This ensures the project is isolated from the system Python

VENV_DIR="./.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "--- Configuring Python Environment ---"

# Check if venv exists, create if not
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "Upgrading pip and build tools..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo "Installing libraries (tensorflow, statsmodels, xgboost, etc)..."
# The --prefer-binary flag speeds up installation on WSL by avoiding unnecessary compilation
"$VENV_PYTHON" -m pip install --prefer-binary -r requirements.txt

echo "Installing project in editable mode..."
"$VENV_PYTHON" -m pip install -e .

echo "Environment ready! Activate it with: source $VENV_DIR/bin/activate"