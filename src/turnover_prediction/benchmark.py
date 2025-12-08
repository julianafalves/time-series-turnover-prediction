"""
Benchmark forecasting methods including Tree-based and Deep Learning (LSTM) models.

This script implements a Global Forecasting Model (GFM) approach.
It generates a comprehensive suite of 6 visualizations for the final report:
1. Weighted Average Plot (Executive View).
2. Confidence Interval Plot (Scientific View).
3. Feature Importance (Explainability).
4. Residual Analysis (Error bias).
5. Scatter Fit (Accuracy).
6. Seasonality Boxplot (Data behavior).
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap

# Deep Learning Imports (Optional - handles missing TensorFlow gracefully)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_DL = True
except ImportError:
    HAS_DL = False
    print("Tensorflow not found. LSTM model will be skipped.")

from turnover_prediction.preprocessing import preprocess_data

def evaluate_metrics(actual, pred):
    """Calculates MAE and RMSE for model evaluation."""
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return {'mae': mae, 'rmse': rmse}

def fit_predict_lstm(X_train, y_train, X_test):
    """
    Trains a simple LSTM model.
    Reshapes tabular data (samples, features) into (samples, 1, features) for the RNN.
    """
    if not HAS_DL:
        return np.zeros(len(X_test))

    # Reshape for LSTM: [samples, time steps, features]
    # We treat the flattened lag features as a single time step with N features
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
    # 1. Load Original Data
    df = pd.read_csv(input_file)
    
    # Column Configuration (Hardcoded to ensure consistency with the specific dataset structure)
    date_col = 'MES_REF'
    target_col = 'TO_TURNOVER_TO-GERAL'
    group_col = 'area_anonn'
    headcount_col = 'TO_HEADCOUNT_HEADCOUNT-MES-ATUAL' 
    
    # 2. Preprocessing (Train/Test Split)
    # Note: capturing 'col_names' is essential for Plot 3 (Feature Importance)
    X_train, X_test, y_train, y_test, _, col_names, test_dates, _, _ = preprocess_data(
        df, mode='ts', n_lags=n_lags,
        date_col=date_col, target_col=target_col, group_col=group_col
    )
    
    # 3. Model Training & Forecasting
    print(f"Benchmarking on {len(y_train)} training samples and {len(y_test)} test samples.")
    
    # --- Prepare Results DataFrame ---
    # Ensure date column matches the format for merging/filtering
    df[date_col] = pd.to_datetime(df[date_col])
    
    results_df = pd.DataFrame({
        'date': test_dates,
        'actual': y_test
    })
    
    # --- Retrieve Headcount for Test Set ---
    # Logic: Sort original DF and take the last 20% of dates (same split as preprocessing)
    unique_dates = df[date_col].sort_values().unique()
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]
    
    # Filter original DF to match the test period
    df_test_raw = df[df[date_col] >= split_date].copy()
    df_test_raw = df_test_raw.dropna(subset=[target_col]).sort_values(by=[group_col, date_col])
    
    # Alignment Check
    if len(df_test_raw) == len(results_df):
        results_df['headcount'] = df_test_raw[headcount_col].values
    else:
        print("Warning: Could not perfectly align Headcount. Using simple average for aggregation.")
        results_df['headcount'] = 1

    # --- Train Models ---
    metrics_dict = {}

    # A. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results_df['RandomForest'] = rf.predict(X_test)
    metrics_dict['RandomForest'] = evaluate_metrics(y_test, results_df['RandomForest'])

    # B. XGBoost (Primary Model)
    print("Training XGBoost...")
    xg = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    xg.fit(X_train, y_train)
    results_df['XGBoost'] = xg.predict(X_test)
    metrics_dict['XGBoost'] = evaluate_metrics(y_test, results_df['XGBoost'])
    
    # C. LSTM (Optional)
    if HAS_DL:
        print("Training LSTM...")
        results_df['LSTM'] = fit_predict_lstm(X_train, y_train, X_test)
        metrics_dict['LSTM'] = evaluate_metrics(y_test, results_df['LSTM'])

    # Save Outputs
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/benchmark_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    results_df.to_csv(f"{out_dir}/benchmark_predictions_detailed.csv", index=False)
    
    # Set global style for all plots
    sns.set_style("whitegrid")

    # =========================================================
    # PLOT 1: WEIGHTED AVERAGE (Executive View)
    # =========================================================
    def calculate_weighted_mean(x):
        return np.average(x, weights=results_df.loc[x.index, 'headcount'])

    weighted_df = results_df.groupby('date').agg({
        'actual': calculate_weighted_mean,
        'XGBoost': calculate_weighted_mean
    }).sort_index()

    plt.figure(figsize=(14, 6))
    plt.plot(weighted_df.index, weighted_df['actual'], label='Actual (Weighted Avg)', color='black', linewidth=2.5)
    plt.plot(weighted_df.index, weighted_df['XGBoost'], label='XGBoost Forecast', color='#E63946', linestyle='--', linewidth=2)
    plt.title("1. Global Turnover Weighted by Headcount (Executive View)", fontsize=14)
    plt.ylabel("Weighted Turnover Rate (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_benchmark_weighted_avg.png", dpi=300)
    plt.close()

    # =========================================================
    # PLOT 2: CONFIDENCE INTERVAL (Scientific View)
    # =========================================================
    plot_long = pd.melt(results_df, id_vars=['date'], 
                        value_vars=['actual', 'XGBoost'], 
                        var_name='Type', value_name='Turnover')
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=plot_long, x='date', y='Turnover', hue='Type', 
                 style='Type', markers=True, dashes=False, ci=95,
                 palette=['black', 'red'])
    plt.title("2. Turnover Trend with Confidence Interval (Sector Variability)", fontsize=14)
    plt.ylabel("Turnover Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_benchmark_confidence_interval.png", dpi=300)
    plt.close()

    # =========================================================
    # PLOT 3: FEATURE IMPORTANCE (XGBoost)
    # Explains "WHY" the model is making these predictions.
    # =========================================================
    if 'XGBoost' in metrics_dict:
        plt.figure(figsize=(10, 8))
        # Use SHAP for more robust feature importance
        explainer = shap.TreeExplainer(xg)
        shap_values = explainer.shap_values(X_test)
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance = pd.DataFrame({
            'Feature': col_names,
            'Importance': mean_abs_shap
        }).sort_values(by='Importance', ascending=False)
        
        # Filter out non-predictive features before plotting
        importance = importance[importance['Importance'] > 1e-6].head(20)

        sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
        plt.title("3. Top 20 Feature Importance (SHAP)", fontsize=14)
        plt.xlabel("mean(|SHAP value|) average impact on model output")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/3_feature_importance.png", dpi=300)
        plt.close()

    # =========================================================
    # PLOT 4: RESIDUAL ANALYSIS
    # Shows the bias/error distribution over time (Actual - Predicted).
    # =========================================================
    # Calculate residuals on the weighted average to reduce noise
    weighted_df['Residual'] = weighted_df['actual'] - weighted_df['XGBoost']

    plt.figure(figsize=(14, 6))
    colors = ['#D62828' if x < 0 else '#2A9D8F' for x in weighted_df['Residual']]
    plt.bar(weighted_df.index, weighted_df['Residual'], color=colors, alpha=0.8)
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    
    plt.title("4. Residual Analysis (Actual - Predicted)", fontsize=14)
    plt.ylabel("Error Magnitude (p.p.)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_benchmark_residuals.png", dpi=300)
    plt.close()

    # =========================================================
    # PLOT 5: SCATTER FIT (R2 Visualization)
    # Shows how close the predictions are to the perfect line.
    # =========================================================
    plt.figure(figsize=(8, 8))
    # Using full non-aggregated data to show dispersion
    sns.scatterplot(x=results_df['actual'], y=results_df['XGBoost'], alpha=0.3, color='#457B9D', s=40)
    
    # Perfect identity line
    max_val = max(results_df['actual'].max(), results_df['XGBoost'].max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Fit', linewidth=1.5)
    
    plt.title("5. Model Fit: Actual vs Predicted", fontsize=14)
    plt.xlabel("Actual Turnover (%)")
    plt.ylabel("Predicted Turnover (%)")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/5_benchmark_scatter_fit.png", dpi=300)
    plt.close()

    # =========================================================
    # PLOT 6: MONTHLY SEASONALITY BOXPLOT
    # Shows the distribution of turnover across different months.
    # =========================================================
    # Create a month column for grouping
    results_df['Month'] = results_df['date'].dt.strftime('%b') # Jan, Feb...
    # Ensure correct month order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month', y='actual', data=results_df, order=month_order, palette="Blues")
    plt.title("6. Monthly Seasonality Distribution (Actual Data)", fontsize=14)
    plt.ylabel("Turnover Rate (%)")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/6_eda_seasonality_boxplot.png", dpi=300)
    plt.close()

    print("All 6 benchmark plots generated successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/Time_Series___Turnover (1).csv')
    parser.add_argument('--out-dir', type=str, default='reports/benchmark')
    parser.add_argument('--n-lags', type=int, default=12)
    args = parser.parse_args()

    run_benchmark(args.input, args.out_dir, args.n_lags)