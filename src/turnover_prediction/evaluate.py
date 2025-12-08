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

def diebold_mariano_test(actual: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, h: int = 1) -> float:
    """
    Diebold-Mariano test for forecast accuracy comparison.
    
    Tests H0: Two models have equal forecast accuracy.
    
    Parameters
    ----------
    actual : ndarray
        Actual values
    pred1 : ndarray
        Predictions from model 1
    pred2 : ndarray
        Predictions from model 2 (baseline)
    h : int, default=1
        Forecast horizon (not used in simplified version, kept for interface compatibility)
        
    Returns
    -------
    float
        DM test statistic (compare with ±1.96 for 95% significance)
    """
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=0)
    
    if var_d == 0:
        return 0.0
        
    dm_stat = mean_d / np.sqrt(var_d / len(actual))
    return float(dm_stat)

def evaluate(prepared_path: str, model_path: str, report_dir: str = 'reports') -> dict:
    """
    Evaluate model performance and produce diagnostic plots and metrics.
    
    Computes MAE, RMSE, and performs residual autocorrelation analysis. Includes 
    Diebold-Mariano test comparing model against mean baseline.

    Parameters
    ----------
    prepared_path : str
        Path to prepared.joblib containing X_test and y_test
    model_path : str
        Path to trained model (.joblib)
    report_dir : str, default='reports'
        Output directory for plots and metrics JSON
        
    Returns
    -------
    dict
        Dictionary with keys: mae, rmse, dm_test_stat_vs_mean, dm_test_significant
    """
    data = joblib.load(prepared_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    
    metrics = {}
    
    # Standard Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    metrics['mae'] = float(mae)
    metrics['rmse'] = float(rmse)
    
    # Residual Autocorrelation Analysis
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(residuals, ax=ax, lags=min(20, len(residuals)-1))
    plt.title("Residual Autocorrelation Function (ACF)")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(f"{report_dir}/residuals_acf.png", dpi=300)
    plt.close()
    
    # Diebold-Mariano Test vs Mean Baseline
    mean_baseline = np.full(shape=y_test.shape, fill_value=np.mean(y_test))
    dm_stat = diebold_mariano_test(y_test, preds, mean_baseline)
    metrics['dm_test_stat_vs_mean'] = dm_stat
    metrics['dm_test_significant'] = abs(dm_stat) > 1.96
    
    with open(f"{report_dir}/eval_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate model performance and produce diagnostic metrics and plots.'
    )
    parser.add_argument('--prepared', type=str, default='data/prepared.joblib',
                        help='Path to prepared.joblib file')
    parser.add_argument('--model', type=str, default='models/xgb_turnover.joblib',
                        help='Path to trained model')
    parser.add_argument('--report-dir', type=str, default='reports',
                        help='Output directory for results')
    args = parser.parse_args()

    evaluate(args.prepared, args.model, args.report_dir)