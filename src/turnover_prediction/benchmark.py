"""
Benchmark forecasting methods including Tree-based and Deep Learning (LSTM) models.
Implements a Global Forecasting approach.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Deep Learning Imports (LSTM not currently used; requires tensorflow)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_DL = True
except ImportError:
    HAS_DL = False

# Import the robust feature engineering from preprocessing to ensure consistency
from turnover_prediction.preprocessing import feature_engineering_ts, preprocess_data

def evaluate_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return {'mae': mae, 'rmse': rmse}

def fit_predict_lstm(X_train, y_train, X_test, time_steps=1):
    """
    Train a simple LSTM model for forecasting.
    
    Reshapes tabular input (samples, features) → (samples, 1, features) for LSTM compatibility.
    
    Parameters
    ----------
    X_train : ndarray
        Training features (n_samples, n_features)
    y_train : ndarray
        Training target values
    X_test : ndarray
        Test features
    time_steps : int, default=1
        Number of timesteps (here fixed at 1 for GFM compatibility)
        
    Returns
    -------
    ndarray
        Predicted values for X_test
    """
    if not HAS_DL:
        return np.zeros(len(X_test))

    X_train_rs = X_train.reshape((X_train.shape[0], time_steps, X_train.shape[1]))
    X_test_rs = X_test.reshape((X_test.shape[0], time_steps, X_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(time_steps, X_train.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train_rs, y_train, epochs=30, batch_size=32, verbose=0, callbacks=[es])
    
    pred = model.predict(X_test_rs, verbose=0)
    return pred.flatten()

def run_benchmark(input_file: str, out_dir: str, n_lags: int = 12):
    """
    Benchmark multiple forecasting methods on turnover data.
    
    Compares Random Forest, XGBoost, and LSTM (if TensorFlow available) using a Global Forecasting Model
    approach with lagged features.
    
    Parameters
    ----------
    input_file : str
        Path to input CSV (must contain MES_REF, TO_TURNOVER_TO-GERAL, area_anonn columns)
    out_dir : str
        Output directory for metrics, predictions, and plots
    n_lags : int, default=12
        Number of lag features to engineer
    """
    df = pd.read_csv(input_file)
    
    date_col = 'MES_REF'
    target_col = 'TO_TURNOVER_TO-GERAL'
    group_col = 'area_anonn'
    
    X_train, X_test, y_train, y_test, _, _, test_dates = preprocess_data(
        df, mode='ts', n_lags=n_lags, 
        date_col=date_col, target_col=target_col, group_col=group_col
    )
    
    results = {}
    predictions = pd.DataFrame({'date': test_dates, 'actual': y_test})

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['RandomForest'] = evaluate_metrics(y_test, rf_pred)
    predictions['RandomForest'] = rf_pred

    # XGBoost Regressor
    xg = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    xg.fit(X_train, y_train)
    xg_pred = xg.predict(X_test)
    results['XGBoost'] = evaluate_metrics(y_test, xg_pred)
    predictions['XGBoost'] = xg_pred
    
    # LSTM (if TensorFlow available)
    if HAS_DL:
        lstm_pred = fit_predict_lstm(X_train, y_train, X_test)
        results['LSTM'] = evaluate_metrics(y_test, lstm_pred)
        predictions['LSTM'] = lstm_pred

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    
    with open(f"{out_dir}/benchmark_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    predictions.to_csv(f"{out_dir}/benchmark_predictions.csv", index=False)
    
    # Plot aggregate predictions across all areas
    plt.figure(figsize=(12, 6))
    agg_pred = predictions.groupby('date').mean()
    
    plt.plot(agg_pred.index, agg_pred['actual'], label='Actual', color='black', linewidth=2)
    if 'RandomForest' in agg_pred:
        plt.plot(agg_pred.index, agg_pred['RandomForest'], label='Random Forest', linestyle='--')
    if 'XGBoost' in agg_pred:
        plt.plot(agg_pred.index, agg_pred['XGBoost'], label='XGBoost', linestyle='--')
    if 'LSTM' in agg_pred:
        plt.plot(agg_pred.index, agg_pred['LSTM'], label='LSTM', linestyle='-.')
        
    plt.title("Benchmark: Global Average Turnover Rate Forecast")
    plt.xlabel("Date")
    plt.ylabel("Average Turnover Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/benchmark_plot.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark turnover forecasting methods using Global Forecasting Model.'
    )
    parser.add_argument('--input', type=str, default='data/Time_Series___Turnover (1).csv',
                        help='Path to input CSV file')
    parser.add_argument('--out-dir', type=str, default='reports/benchmark',
                        help='Output directory for results')
    parser.add_argument('--n-lags', type=int, default=12,
                        help='Number of lag features to create')
    args = parser.parse_args()

    run_benchmark(args.input, args.out_dir, args.n_lags)