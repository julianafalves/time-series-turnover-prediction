"""
Hyperparameter Tuning using Time Series Cross-Validation.
Implements RandomizedSearchCV with TimeSeriesSplit.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from turnover_prediction.preprocessing import preprocess_data


def run_tuning(input_file, out_dir, n_splits=5, n_iter=20):
    """
    Perform hyperparameter tuning for XGBoost using time series cross-validation.

    Parameters
    ----------
    input_file : str
        Path to the CSV file containing the time series data.
    out_dir : str
        Directory where the best parameters JSON will be saved.
    n_splits : int, default=5
        Number of splits for TimeSeriesSplit cross-validation.
    n_iter : int, default=20
        Number of random search iterations.

    Returns
    -------
    None
        Saves the best model to 'models/xgb_tuned.joblib' and best parameters to
        '{out_dir}/best_params.json'.
    """
    print(f"--- Starting Hyperparameter Tuning (Rolling Window with {n_splits} splits) ---")

    # 1. Load and prepare data
    df = pd.read_csv(input_file)

    # Column mapping (keeping consistency with the project)
    date_col = 'MES_REF'
    target_col = 'TO_TURNOVER_TO-GERAL'
    group_col = 'area_anonn'

    # Preprocessing (mode='ts' to generate lag features)
    # Note: preprocess_data returns train/test split; we use the training portion for CV.
    X, _, y, _, _, col_names, _ = preprocess_data(
        df, mode='ts', n_lags=12,
        date_col=date_col, target_col=target_col, group_col=group_col
    )

    # preprocess_data already splits data into train/test (80/20). We use only the training
    # portion for hyperparameter tuning to avoid leakage into the final test set.
    print(f"Training data available: {X.shape[0]} samples.")

    # 2. Define parameter search space
    # This covers the requirement "Hyperparameter Optimization"
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    # 3. Configure Time Series Cross‑Validation (Rolling Window)
    # This covers the requirement "Robust experimental methodology"
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Use negative MAE because Scikit‑learn always tries to maximize the score
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    print(f"Running RandomizedSearchCV with {n_iter} iterations...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,  # Time Series CV magic happens here
        verbose=1,
        n_jobs=-1,  # Use all available CPU cores
        random_state=42
    )

    search.fit(X, y)

    # 4. Results
    print("\n--- Tuning Results ---")
    print(f"Best Score (average MAE across folds): {-search.best_score_:.4f}")
    print("Best Hyperparameters found:")
    print(search.best_params_)

    # Save the best model
    best_model = search.best_estimator_
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgb_tuned.joblib'
    joblib.dump(best_model, model_path)

    # Save parameters as JSON for later use (e.g., in the paper)
    with open(f"{out_dir}/best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=4)

    print(f"Best model saved to: {model_path}")
    print(f"Parameters saved to: {out_dir}/best_params.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/Time_Series___Turnover (1).csv')
    parser.add_argument('--out-dir', type=str, default='reports')
    args = parser.parse_args()

    run_tuning(args.input, args.out_dir)