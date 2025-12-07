"""
Evaluate the saved model and produce plots and advanced statistical metrics.
"""
import argparse
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

def diebold_mariano_test(actual, pred1, pred2, h=1):
    """
    Simplified Diebold-Mariano test.
    H0: Two models have the same forecast accuracy.
    Returns: DM-statistic (Compare with 1.96 for 95% confidence)
    """
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2 # using MSE loss
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=0)
    
    if var_d == 0:
        return 0.0
        
    # Standard DM stat (approximate for large N)
    dm_stat = mean_d / np.sqrt(var_d / len(actual))
    return float(dm_stat)

def evaluate(prepared_path, model_path, report_dir='reports', task='ts'):
    data = joblib.load(prepared_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    
    metrics = {}
    
    # 1. Standard Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    metrics['mae'] = float(mae)
    metrics['rmse'] = float(rmse)
    
    # 2. Residual Analysis (ACF)
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(residuals, ax=ax, lags=min(20, len(residuals)-1))
    plt.title("Residual Autocorrelation Function (ACF)")
    plt.savefig(f"{report_dir}/residuals_acf.png")
    plt.close()
    
    # 3. Diebold-Mariano vs Naive (Lag 1)
    # We try to reconstruct a naive baseline from the X_test if 'lag_1' exists,
    # otherwise we use 0 as a dummy baseline or mean
    # Note: X_test is scaled, so we can't easily extract raw lag_1. 
    # We will use mean(y_test) as a 'Dummy Baseline' for statistical comparison.
    dummy_pred = np.full(shape=y_test.shape, fill_value=np.mean(y_test))
    
    dm_stat = diebold_mariano_test(y_test, preds, dummy_pred)
    metrics['dm_test_stat_vs_mean'] = dm_stat
    metrics['dm_test_significant'] = abs(dm_stat) > 1.96
    
    print(f"Metrics: {metrics}")
    
    with open(f"{report_dir}/eval_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=str, default='data/prepared.joblib')
    parser.add_argument('--model', type=str, default='models/xgb_turnover.joblib')
    parser.add_argument('--report-dir', type=str, default='reports')
    args = parser.parse_args()

    evaluate(args.prepared, args.model, args.report_dir)