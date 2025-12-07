"""
Exploratory Data Analysis script.
Performs Seasonal Decomposition and Trend Analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import argparse

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
    
    # 1. Global Aggregation (Sum/Mean across all areas)
    # Assuming the target is a RATE, we take the mean. If it's a COUNT, we sum.
    # CSV name implies Rate ("TO_TURNOVER..."), so MEAN is safer for global trend.
    global_ts = df.groupby(date_col)[target_col].mean().sort_index()
    
    # 2. Plot Global Trend
    plt.figure(figsize=(12, 6))
    plt.plot(global_ts, label='Avg Turnover Rate')
    plt.title('Global Turnover Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/global_trend.png")
    plt.close()
    
    # 3. Seasonal Decomposition
    # Set frequency to Month (12)
    # Handle missing dates if any
    ts_resampled = global_ts.asfreq('MS').fillna(method='ffill')
    
    if len(ts_resampled) >= 24: # Need at least 2 cycles
        decomp = sm.tsa.seasonal_decompose(ts_resampled, model='additive', period=12)
        
        fig = decomp.plot()
        fig.set_size_inches(12, 10)
        plt.savefig(f"{out_dir}/seasonal_decomposition.png")
        plt.close()
        print("Seasonal decomposition plot saved.")
    else:
        print("Not enough data points for seasonal decomposition (need 24+ months).")

    print(f"EDA complete. Reports saved to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/Time_Series___Turnover (1).csv')
    parser.add_argument('--out-dir', type=str, default='reports/eda')
    args = parser.parse_args()
    
    run_eda(args.input, args.out_dir)