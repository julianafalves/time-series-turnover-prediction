#!/bin/bash

# Script to clean up generated artifacts (models, processed data, reports)
# Useful to ensure the next run is a "fresh start" and not using cached data.

echo "Starting cleanup..."

# 1. Clean processed data (keeps the original CSV)
if [ -d "data" ]; then
    # Remove joblib (serialized data) and pickle files
    rm -f data/*.joblib
    rm -f data/*.pkl
    echo "- Processed data removed (.joblib, .pkl)"
fi

# 2. Clean trained models
if [ -d "models" ]; then
    rm -f models/*.joblib
    rm -f models/*.json
    echo "- Trained models removed"
fi

# 3. Clean reports and plots
if [ -d "reports" ]; then
    rm -rf reports/eda/*
    rm -rf reports/benchmark/*
    rm -f reports/*.png
    rm -f reports/*.json
    echo "- Reports and plots removed"
fi

# 4. Clean Python cache (optional, but good for avoiding import errors)
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
echo "- Python caches removed"

echo "Cleanup complete! The environment is ready for a clean run."