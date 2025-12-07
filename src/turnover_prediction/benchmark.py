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

# Deep Learning Imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_DL = True
except ImportError:
    HAS_DL = False
    print("Tensorflow not found. LSTM model will be skipped.")

# Import the robust feature engineering from preprocessing to ensure consistency
from turnover_prediction.preprocessing import feature_engineering_ts, preprocess_data

def evaluate_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return {'mae': mae, 'rmse': rmse}

def fit_predict_lstm(X_train, y_train, X_test, time_steps=1):
    """
    Fits a simple LSTM. 
    Reshapes tabular input (samples, features) -> (samples, 1, features) for LSTM.
    """
    if not HAS_DL:
        return np.zeros(len(X_test))

    # Reshape for LSTM: [samples, time steps, features]
    # Here we treat the flattened feature row as 1 timestep with N features
    # Ideally, we would construct sequences, but using the lag-features as input state is a valid simplification for benchmarking
    X_train_rs = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_rs = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train_rs, y_train, epochs=30, batch_size=32, verbose=0, callbacks=[es])
    
    pred = model.predict(X_test_rs, verbose=0)
    return pred.flatten()

def run_benchmark(input_file: str, out_dir: str, n_lags: int = 12):
    df = pd.read_csv(input_file)
    
    # Configuration based on file
    date_col = 'MES_REF'
    target_col = 'TO_TURNOVER_TO-GERAL'
    group_col = 'area_anonn'
    
    # Use the centralized preprocessing logic to get clean Train/Test sets
    # We reuse the logic from src/preprocessing.py to ensure identical feature engineering
    X_train, X_test, y_train, y_test, _, _, test_dates = preprocess_data(
        df, mode='ts', n_lags=n_lags, 
        date_col=date_col, target_col=target_col, group_col=group_col
    )
    
    results = {}
    predictions = pd.DataFrame({'date': test_dates, 'actual': y_test})
    
    print(f"Benchmarking on {len(y_train)} training samples and {len(y_test)} test samples.")

    # 1. Naive Baseline (Last observed value)
    # In a GFM with feature engineering, the 'lag_1' feature (if standard scaled) 
    # corresponds to the last value. However, X_train is scaled. 
    # We cannot simply look at X_test column 0. We assume Naive isn't easily extractable 
    # from scaled matrix without inverse transform. 
    # Instead, we will rely on ML models. 
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['RandomForest'] = evaluate_metrics(y_test, rf_pred)
    predictions['RandomForest'] = rf_pred

    # 3. XGBoost
    print("Training XGBoost...")
    xg = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    xg.fit(X_train, y_train)
    xg_pred = xg.predict(X_test)
    results['XGBoost'] = evaluate_metrics(y_test, xg_pred)
    predictions['XGBoost'] = xg_pred
    
    # 4. LSTM (Deep Learning)
    if HAS_DL:
        print("Training LSTM...")
        lstm_pred = fit_predict_lstm(X_train, y_train, X_test)
        results['LSTM'] = evaluate_metrics(y_test, lstm_pred)
        predictions['LSTM'] = lstm_pred

    # Save Output
    os.makedirs(out_dir, exist_ok=True)
    
    # Save Metrics
    with open(f"{out_dir}/benchmark_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Save Predictions CSV
    predictions.to_csv(f"{out_dir}/benchmark_predictions.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    # Aggregate predictions by date for plotting (average turnover rate across areas per date)
    # This gives a "Global" view of performance
    agg_pred = predictions.groupby('date').mean()
    
    plt.plot(agg_pred.index, agg_pred['actual'], label='Actual (Avg)', color='black', linewidth=2)
    if 'RandomForest' in agg_pred:
        plt.plot(agg_pred.index, agg_pred['RandomForest'], label='RF', linestyle='--')
    if 'XGBoost' in agg_pred:
        plt.plot(agg_pred.index, agg_pred['XGBoost'], label='XGB', linestyle='--')
    if 'LSTM' in agg_pred:
        plt.plot(agg_pred.index, agg_pred['LSTM'], label='LSTM', linestyle='-.')
        
    plt.title("Global Average Turnover Rate Forecast")
    plt.xlabel("Date")
    plt.ylabel("Avg Turnover Rate")
    plt.legend()
    plt.savefig(f"{out_dir}/benchmark_plot.png")
    plt.close()
    
    print("Benchmark complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/Time_Series___Turnover (1).csv')
    parser.add_argument('--out-dir', type=str, default='reports/benchmark')
    parser.add_argument('--n-lags', type=int, default=12)
    args = parser.parse_args()

    run_benchmark(args.input, args.out_dir, args.n_lags)