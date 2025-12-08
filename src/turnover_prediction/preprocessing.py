"""
Preprocessing pipeline for Turnover Prediction.

This module is responsible for transforming raw HR data into a format suitable for
Supervised Time-Series Machine Learning. Key functionalities include:

1. Time-Series Feature Engineering: Creation of lags, sliding windows, and temporal features.
2. Volatility Handling: Calculation of rolling standard deviations to capture process stability.
3. Sample Weighting: Generation of instance weights based on team size (headcount) to prioritize stable samples.
"""
import argparse
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def feature_engineering_ts(df: pd.DataFrame, date_col: str, target_col: str, group_col: str, n_lags: int) -> pd.DataFrame:
    """
    Generates temporal features and cleans data for time-series forecasting.
    
    This function handles the chronological structure of the data, ensuring that
    past values (lags) and trends (rolling stats) are correctly calculated within
    each specific group/area.
    """
    df = df.copy()
    
    # 1. Temporal Ordering
    # Ensures data is sorted by entity and date so that .shift() and .rolling() work correctly.
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[group_col, date_col])
    
    # Extraction of cyclical calendar features (Seasonality indicators)
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['is_december'] = (df[date_col].dt.month == 12).astype(int)
    
    # 2. Standardization of Column Names
    # Maps specific raw dataset names to generic internal identifiers.
    col_map = {
        'TO_ADMISSOES_ADMISSOES-MES-ATUAL': 'admissions',
        'TO_HEADCOUNT_HEADCOUNT-MES-ATUAL': 'headcount'
    }
    for orig, new in col_map.items():
        if orig in df.columns:
            df = df.rename(columns={orig: new})

    # Component columns for lag features
    component_columns = [
        'TO_TURNOVER_TO-VOL',
        'TO_TURNOVER_TO-INVOL',
        'TO_DESLIGAMENTO_DESLIGAMENTO-MES-ATUAL',
        'TO_DESLIGAMENTO_DESLIGAMENTO-VOLUNTARIO-MES-ATUAL',
        'TO_DESLIGAMENTO_DESLIGAMENTO-INVOLUNTARIO-MES-ATUAL',
        'TO_HEADCOUNT-TO_HEADCOUNT-MES-ATUAL',
        'TO_TURNOVER_TO-VOL-HEAD-GT-25'
    ]
    
    # 3. Feature Engineering per Group
    grouped = df.groupby(group_col)
    
    # A. Autoregressive Features (Target Lags)
    # Captures the direct dependency of the target on its own past values.
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = grouped[target_col].shift(lag)
        
    # B. Exogenous Lags (Leading Indicators)
    # Uses past admissions and headcount to predict future turnover risk.
    if 'admissions' in df.columns:
        for lag in [1, 3, 6]: 
            df[f'admissions_lag_{lag}'] = grouped['admissions'].shift(lag)
            
    if 'headcount' in df.columns:
        for lag in [1, 3, 6]:
            df[f'headcount_lag_{lag}'] = grouped['headcount'].shift(lag)

    # C. Historical Components of Turnover
    # Uses past breakdown (voluntary vs involuntary) to inform future trends.
    for col in component_columns:
        if col in df.columns:
            for lag in [1, 3]:
                df[f'{col}_lag_{lag}'] = grouped[col].shift(lag)

    # D. Rolling Statistics (Trend and Volatility)
    # shift(1) ensures we strictly use past data for the window calculation.
    
    # Moving Average: Captures the short-term and medium-term trend.
    df['rolling_mean_3'] = grouped[target_col].transform(lambda x: x.shift(1).rolling(3).mean())
    df['rolling_mean_6'] = grouped[target_col].transform(lambda x: x.shift(1).rolling(6).mean())
    
    # Moving Std Dev: Quantifies volatility. 
    # Allows the model to distinguish between stable areas and erratic areas.
    df['rolling_std_6'] = grouped[target_col].transform(lambda x: x.shift(1).rolling(6).std())

    # 4. Data Cleaning
    # Removes rows with NaN values resulting from lag/rolling operations (the "warm-up" period).
    df = df.dropna().reset_index(drop=True)
    
    return df

def preprocess_data(df, mode='ts', n_lags=12, date_col='MES_REF', target_col='TO_TURNOVER_TO-GERAL', group_col='area_anonn'):
    """
    Orchestrates the data preparation process: cleaning, engineering, splitting, and scaling.
    Returns processed arrays and sample weights for training.
    """
    # Column Discovery Heuristics
    if date_col not in df.columns:
        possible_dates = [c for c in df.columns if 'date' in c.lower() or 'mes' in c.lower()]
        if possible_dates: date_col = possible_dates[0]
    if target_col not in df.columns:
        possible_targets = [c for c in df.columns if 'turnover' in c.lower()]
        if possible_targets: target_col = possible_targets[0]

    # Type safety for target
    if df[target_col].dtype == object:
         df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)

    if mode == 'ts':
        # 1. Feature Engineering
        ts_df = feature_engineering_ts(df, date_col, target_col, group_col, n_lags)
        
        # 2. Categorical Encoding
        # Converts Area IDs to numerical format required by algorithms.
        le = LabelEncoder()
        ts_df[f'{group_col}_encoded'] = le.fit_transform(ts_df[group_col].astype(str))
        
        # 3. Temporal Train/Test Split
        # Splits data by time (not random) to prevent future data leaking into training.
        dates = ts_df[date_col].sort_values().unique()
        split_idx = int(len(dates) * 0.8) # 80% history for training
        split_date = dates[split_idx]
        
        train_df = ts_df[ts_df[date_col] < split_date]
        test_df = ts_df[ts_df[date_col] >= split_date]
        
        print(f"Time Split Date: {split_date}")
        
        # 4. Sample Weight Calculation
        # Calculates weights based on team size (headcount).
        # Used during training to penalize errors in large/stable teams more heavily 
        # than errors in small/volatile teams, reducing noise impact.
        if 'headcount' in train_df.columns:
            w_train = train_df['headcount'].values
            w_test = test_df['headcount'].values
        else:
            w_train = np.ones(len(train_df))
            w_test = np.ones(len(test_df))
            
        # Feature Selection
        drop_cols_X = [date_col, target_col, group_col]
        
        X_train = train_df.drop(columns=drop_cols_X)
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=drop_cols_X)
        y_test = test_df[target_col]
        
        test_dates = test_df[date_col].values
        
        # 5. Scaling Pipeline & Feature Name Reconstruction
        cat_feats = [f'{group_col}_encoded', 'is_december']
        num_feats = [c for c in X_train.columns if c not in cat_feats]

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', 'passthrough', cat_feats)
        ], remainder='passthrough')
        
        # Fit scaler ONLY on training data to maintain statistical integrity.
        pipeline = Pipeline([('preprocessor', preprocessor)])
        pipeline.fit(X_train)
        
        X_train_trans = pipeline.transform(X_train)
        X_test_trans = pipeline.transform(X_test)

        # Feature Name Reconstruction (for explainability)
        # This is the correct way to get feature names post-ColumnTransformer
        # It respects the order in which the transformer outputs the columns.
        try:
            col_names = num_feats + cat_feats
        except:
            col_names = [f'feat_{i}' for i in range(X_train_trans.shape[1])]
            
        # Returns transformed data + sample weights (w_train, w_test)
        return X_train_trans, X_test_trans, y_train.values, y_test.values, pipeline, col_names, test_dates, w_train, w_test
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/Time_Series___Turnover (1).csv')
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--mode', type=str, default='ts', choices=['tabular','ts'])
    parser.add_argument('--n-lags', type=int, default=12)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    date_col = 'MES_REF'
    target_col = 'TO_TURNOVER_TO-GERAL'
    group_col = 'area_anonn'
    
    results = preprocess_data(df, mode=args.mode, n_lags=args.n_lags, 
                              date_col=date_col, target_col=target_col, group_col=group_col)
    
    # Unpack including weights
    X_train, X_test, y_train, y_test, pipeline, col_names, test_dates, w_train, w_test = results
    
    import joblib
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save artifacts for training and evaluation phases
    prepared = {
        'X_train': X_train, 'y_train': y_train, 
        'X_test': X_test, 'y_test': y_test, 
        'col_names': col_names,
        'test_dates': test_dates,
        'w_train': w_train, 'w_test': w_test 
    }
    
    joblib.dump(prepared, f"{args.out_dir}/prepared.joblib")
    joblib.dump(pipeline, f"{args.out_dir}/preprocessor.joblib")
    
    with open(f"{args.out_dir}/col_names.pkl", 'wb') as f:
        pickle.dump(col_names, f)
        
    print(f"Preprocessing ({args.mode}) complete with weights. Train shape: {X_train.shape}")