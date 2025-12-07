"""
Exploratory Data Analysis script.
Performs Seasonal Decomposition and Trend Analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import argparse

def plot_custom_decomposition(decomp, output_path):
    """
    Manually plots seasonal decomposition components for better readability.
    """
    # Create a figure with 4 subplots sharing the x-axis
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 1. Observed (Actual Data)
    axes[0].plot(decomp.observed, label='Observed', color='navy')
    axes[0].set_title('1. Observed: Actual Global Turnover Rate', fontsize=14, loc='left')
    axes[0].set_ylabel('Turnover Rate')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Trend (Moving Average / Smoothed)
    axes[1].plot(decomp.trend, label='Trend', color='darkorange', linewidth=2)
    axes[1].set_title('2. Trend: Underlying Direction (Noise & Seasonality Removed)', fontsize=14, loc='left')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Seasonal (Repeating Pattern)
    axes[2].plot(decomp.seasonal, label='Seasonality', color='green')
    axes[2].set_title('3. Seasonality: Repeating 12-Month Pattern', fontsize=14, loc='left')
    axes[2].set_ylabel('Seasonal Effect')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # 4. Residuals (Noise)
    # Plot as scatter to see outliers clearly
    axes[3].scatter(decomp.resid.index, decomp.resid, label='Residuals', color='red', alpha=0.7, s=30)
    # Add a zero line to check for randomness
    axes[3].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[3].set_title('4. Residuals: Noise/Unexplained Variance', fontsize=14, loc='left')
    axes[3].set_ylabel('Residue')
    axes[3].grid(True, linestyle='--', alpha=0.6)
    
    # Final Layout adjustments
    plt.xlabel('Date', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_eda(input_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Data
    df = pd.read_csv(input_file)
    
    # Identify Columns
    date_col = 'MES_REF'
    target_col = 'TO_TURNOVER_TO-GERAL'
    if date_col not in df.columns: # Fallback
         date_col = [c for c in df.columns if 'date' in c.lower() or 'mes' in c.lower()][0]
            
    # Process Dates
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 1. Global Aggregation (Mean across all areas per month)
    global_ts = df.groupby(date_col)[target_col].mean().sort_index()
    
    # 2. Plot Global Trend (Simple)
    plt.figure(figsize=(12, 6))
    plt.plot(global_ts, label='Avg Turnover Rate', color='#333333')
    plt.title('Global Turnover Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Turnover Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/global_trend.png")
    plt.close()
    
    # 3. Seasonal Decomposition
    # Set frequency to Month (12)
    # Handle missing dates if any by forward filling
    ts_resampled = global_ts.asfreq('MS').fillna(method='ffill')
    
    if len(ts_resampled) >= 24: # Need at least 2 cycles (24 months)
        # We use additive model because Turnover can be 0 (Multiplicative fails with 0)
        decomp = sm.tsa.seasonal_decompose(ts_resampled, model='additive', period=12)
        
        # Use custom plotter instead of decomp.plot()
        output_path = f"{out_dir}/seasonal_decomposition.png"
        plot_custom_decomposition(decomp, output_path)
        
        print("Seasonal decomposition plot saved.")
    else:
        print(f"Not enough data points for seasonal decomposition. Need 24 months, got {len(ts_resampled)}.")

    print(f"EDA complete. Reports saved to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/Time_Series___Turnover (1).csv')
    parser.add_argument('--out-dir', type=str, default='reports/eda')
    args = parser.parse_args()
    
    run_eda(args.input, args.out_dir)