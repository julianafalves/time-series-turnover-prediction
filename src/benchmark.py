"""
Benchmark time-series forecasting methods on the turnover dataset.

Methods included:
- Naive/last step
- Seasonal naive (if seasonal period known)
- Moving average
- SARIMAX (statsmodels)
- RandomForestRegressor (windowed features)
- XGBRegressor (windowed features)

Evaluation: rolling-origin evaluation (expanding window) with horizon 1 by default.

Outputs:
- CSV with metrics for each method
- Plots comparing actual vs predicted time series

Usage:
python src/benchmark.py --input data/time_series_turnover.csv --out-dir reports/benchmark --n-lags 12

"""
import argparse
import os
import json
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# SARIMAX
try:
    import statsmodels.api as sm
    HAS_STATS = True
except Exception:
    sm = None
    HAS_STATS = False

import matplotlib.pyplot as plt


def create_lag_features(df: pd.DataFrame, date_col: str = 'date', target_col: str = 'turnover_rate', n_lags: int = 12) -> pd.DataFrame:
    ts = df.copy()
    # Infer common date/target columns if not present and normalize to 'date' / 'turnover_rate'
    if date_col not in ts.columns:
        date_candidates = [c for c in ts.columns if 'date' in c.lower() or 'mes' in c.lower() or c.lower().startswith('mes_ref')]
        if date_candidates:
            date_col = date_candidates[0]
    if target_col not in ts.columns:
        candidates = [c for c in ts.columns if 'turnover' in c.lower()] or [c for c in ts.columns if 'to_' in c.lower() and 'turn' in c.lower()]
        if candidates:
            target_col = candidates[0]
    if date_col != 'date' and date_col in ts.columns:
        ts = ts.rename(columns={date_col: 'date'})
        date_col = 'date'
    if target_col != 'turnover_rate' and target_col in ts.columns:
        ts = ts.rename(columns={target_col: 'turnover_rate'})
        target_col = 'turnover_rate'
    ts = ts.sort_values(by=date_col).reset_index(drop=True)
    for lag in range(1, n_lags + 1):
        ts[f'lag_{lag}'] = ts[target_col].shift(lag)
    ts['rolling_mean_3'] = ts[target_col].rolling(3).mean()
    ts['rolling_mean_6'] = ts[target_col].rolling(6).mean()
    ts = ts.dropna().reset_index(drop=True)
    return ts


def naive_last(train_y: np.ndarray, train_X: pd.DataFrame = None, horizon: int = 1) -> float:
    return float(train_y[-1])


def moving_average(train_y: np.ndarray, k: int = 3) -> float:
    k = min(k, len(train_y))
    return float(np.mean(train_y[-k:]))


def seasonal_naive(train_y: np.ndarray, season_period: int = 12) -> float:
    if len(train_y) >= season_period:
        return float(train_y[-season_period])
    return float(train_y[-1])


def fit_predict_sarimax(train_y: np.ndarray, exog_train: pd.DataFrame = None, steps: int = 1,
                        order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)) -> np.ndarray:
    # Fit SARIMAX; pass exog if available
    if exog_train is not None:
        model = sm.tsa.SARIMAX(train_y, exog=exog_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = sm.tsa.SARIMAX(train_y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.forecast(steps=steps, exog=None if exog_train is None else exog_train.iloc[-steps:])
    return np.array(pred)


def fit_predict_rf(train_X: pd.DataFrame, train_y: np.ndarray, test_X: pd.DataFrame):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_X, train_y)
    return model.predict(test_X)


def fit_predict_xgb(train_X: pd.DataFrame, train_y: np.ndarray, test_X: pd.DataFrame):
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    model.fit(train_X, train_y)
    return model.predict(test_X)


def rolling_origin_evaluation(df: pd.DataFrame, date_col: str = 'date', target_col: str = 'turnover_rate', n_lags: int = 12, min_train_periods: int = 12, methods: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    # Returns predictions DataFrame and metrics per method
    ts = create_lag_features(df, date_col=date_col, target_col=target_col, n_lags=n_lags)
    X_cols = [c for c in ts.columns if c not in [date_col, 'leavers', target_col]]
    y = ts[target_col].values
    X_df = ts[X_cols]
    dates = ts[date_col].values

    n = len(ts)
    if methods is None:
        methods = ['naive_last', 'moving_avg', 'rf', 'xgb']
        # include seasonal naive if there is at least a seasonal period
        if len(ts) > 12:
            methods.insert(2, 'seasonal_naive')
        if HAS_STATS:
            methods.insert(2, 'sarimax')

    preds = {m: [] for m in methods}
    actuals = []
    pred_dates = []

    # expanding window: train on indices [0:i], test is i (1-step forecast)
    start = min_train_periods
    for i in range(start, n):
        train_X = X_df.iloc[:i]
        train_y = y[:i]
        test_X = X_df.iloc[i:i + 1]
        actual = y[i]
        for method in methods:
            if method == 'naive_last':
                pred = naive_last(train_y)
            elif method == 'moving_avg':
                pred = moving_average(train_y, k=min(3, len(train_y)))
            elif method == 'seasonal_naive':
                pred = seasonal_naive(train_y, season_period=12)
            elif method == 'sarimax':
                if HAS_STATS:
                    try:
                        exog = train_X[['economy', 'avg_salary', 'avg_satisfaction']] if 'economy' in train_X.columns else None
                        pred = fit_predict_sarimax(train_y, exog_train=exog, steps=1)[0]
                    except Exception:
                        # fallback to naive if SARIMAX fails
                        pred = naive_last(train_y)
                else:
                    pred = naive_last(train_y)
            elif method == 'rf':
                try:
                    pred = fit_predict_rf(train_X, train_y, test_X)[0]
                except Exception:
                    pred = naive_last(train_y)
            elif method == 'xgb':
                try:
                    pred = fit_predict_xgb(train_X, train_y, test_X)[0]
                except Exception:
                    pred = naive_last(train_y)
            else:
                pred = np.nan
            preds[method].append(pred)
        actuals.append(actual)
        pred_dates.append(dates[i])

    # Compute metrics
    metrics = {}
    actuals = np.array(actuals)
    for method in methods:
        method_preds = np.array(preds[method])
        mae = float(mean_absolute_error(actuals, method_preds))
        rmse = float(mean_squared_error(actuals, method_preds, squared=False))
        mape = float(np.mean(np.abs((actuals - method_preds) / np.where(actuals == 0, 1e-7, actuals))))
        metrics[method] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
        }

    preds_df = pd.DataFrame({'date': pred_dates})
    for method in methods:
        preds_df[method] = preds[method]
    preds_df['actual'] = actuals

    return preds_df, metrics


def run_benchmark(input_file: str, out_dir: str, n_lags: int = 12):
    df = pd.read_csv(input_file)
    preds_df, metrics = rolling_origin_evaluation(df, n_lags=n_lags)

    os.makedirs(out_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    # Save metrics
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot actual vs predictions for each method
    for method in [c for c in preds_df.columns if c not in ['date', 'actual']]:
        plt.figure(figsize=(10, 3))
        plt.plot(pd.to_datetime(preds_df['date']), preds_df['actual'], label='actual')
        plt.plot(pd.to_datetime(preds_df['date']), preds_df[method], label=method)
        plt.title(f'Actual vs Predicted - {method}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'actual_vs_{method}.png'))
        plt.close()

    # Bar chart of metrics
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.plot.bar(rot=0)
    plt.title('Benchmark metrics')
    plt.ylabel('error')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'benchmark_metrics.png'))
    plt.close()

    print('Benchmark complete. Results saved to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/time_series_turnover.csv')
    parser.add_argument('--out-dir', type=str, default='reports/benchmark')
    parser.add_argument('--n-lags', type=int, default=12)
    args = parser.parse_args()

    run_benchmark(args.input, args.out_dir, args.n_lags)
