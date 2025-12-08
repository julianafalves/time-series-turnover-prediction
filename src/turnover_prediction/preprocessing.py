"""
Preprocessing utilities for the turnover dataset.
Implements Global Forecasting Model (GFM) logic with group-aware feature engineering.
"""
import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def feature_engineering_ts(df: pd.DataFrame, date_col: str, target_col: str, 
                           group_col: str, n_lags: int) -> pd.DataFrame:
    """
    Time-series feature engineering respecting groups (areas).
    
    Creates lag features, rolling statistics, and temporal features while
    maintaining group integrity to prevent data leakage across areas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with time-series turnover data
    date_col : str
        Column name for date/time variable
    target_col : str
        Column name for target variable (turnover rate)
    group_col : str
        Column name for grouping variable (area identifier)
    n_lags : int
        Number of lag features to create
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features, cleaned and sorted
    """
    df = df.copy()
    
    # 1. Temporal Features
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[group_col, date_col])
    
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['is_december'] = (df[date_col].dt.month == 12).astype(int)
    
    # 2. Exogenous Features (Admissions, Headcount)
    col_map = {
        'TO_ADMISSOES_ADMISSOES-MES-ATUAL': 'admissions',
        'TO_HEADCOUNT_HEADCOUNT-MES-ATUAL': 'headcount'
    }
    
    exog_cols = []
    for orig, new in col_map.items():
        if orig in df.columns:
            df = df.rename(columns={orig: new})
            exog_cols.append(new)
            
    # 3. Create Lags and Rolling features (Grouped by Area)
    # We group by area so lag_1 of Area B doesn't come from Area A
    grouped = df.groupby(group_col)
    
    # Target Lags
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = grouped[target_col].shift(lag)
        
    # Exogenous Lags (Admissions, Headcount)
    # We lag these because current month's admissions might not be known at prediction time, 
    # or we want to see historical impact.
    for exog in exog_cols:
        for lag in [1, 3, 6]: # Selected lags for exogenous
            df[f'{exog}_lag_{lag}'] = grouped[exog].shift(lag)

    # Rolling Means (Lagged to prevent leakage)
    # We shift by 1 first, then roll, to ensure we don't use current value
    df['rolling_mean_3'] = grouped[target_col].transform(lambda x: x.shift(1).rolling(3).mean())
    df['rolling_mean_6'] = grouped[target_col].transform(lambda x: x.shift(1).rolling(6).mean())

    # 4. Cleanup
    df = df.dropna().reset_index(drop=True)
    
    return df

def preprocess_data(df, mode='tabular', n_lags=12, date_col='MES_REF', target_col='TO_TURNOVER_TO-GERAL', group_col='area_anonn'):
    # Normalize column names if defaults provided differ from actual CSV
    # Simple heuristic to find the date column if the default doesn't exist
    if date_col not in df.columns:
        possible_dates = [c for c in df.columns if 'date' in c.lower() or 'mes' in c.lower()]
        if possible_dates: date_col = possible_dates[0]
        
    if target_col not in df.columns:
        possible_targets = [c for c in df.columns if 'turnover' in c.lower() and 'geral' in c.lower()]
        if not possible_targets: # fallback
             possible_targets = [c for c in df.columns if 'turnover' in c.lower()]
        if possible_targets: target_col = possible_targets[0]

    # Ensure target is float
    # Clean percentage strings if necessary or ensure float
    if df[target_col].dtype == object:
         df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)

    if mode == 'tabular':
        # (Existing tabular logic preserved roughly as is, simplifying for brevity)
        X = df.select_dtypes(include=[np.number]).fillna(0)
        if target_col in X.columns:
             X = X.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        preprocessor = Pipeline([('scaler', StandardScaler())])
        preprocessor.fit(X_train)
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        col_names = list(X.columns)
        return X_train_trans, X_test_trans, y_train.values, y_test.values, preprocessor, col_names, None

    else: # mode == 'ts' (Global Forecasting Model)
        
        # 1. Feature Engineering (Group Aware)
        ts_df = feature_engineering_ts(df, date_col, target_col, group_col, n_lags)
        
        # 2. Encode Group ID (Area)
        le = LabelEncoder()
        ts_df[f'{group_col}_encoded'] = le.fit_transform(ts_df[group_col].astype(str))
        
        # 3. Train/Test Split (Time-based split, Global)
        # We split by date, not random. Last 20% of DATES.
        dates = ts_df[date_col].sort_values().unique()
        split_idx = int(len(dates) * 0.8)
        split_date = dates[split_idx]
        
        train_df = ts_df[ts_df[date_col] < split_date]
        test_df = ts_df[ts_df[date_col] >= split_date]
        
        # Columns to drop for X
        drop_cols = [date_col, target_col, group_col] # Keep encoded group, drop original string
        
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df[target_col]
        
        test_dates = test_df[date_col].values
        
        # 4. Scaling
        # Identify categorical (our encoded group) and numerical
        # Ensure categorical features exist in the frame
        cat_feats = [f'{group_col}_encoded', 'is_december']
        cat_feats = [c for c in cat_feats if c in X_train.columns]

        # Select numeric features only to avoid attempting to scale strings (emails/ids)
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude categorical features from numeric list if they are numeric-encoded
        num_feats = [c for c in numeric_cols if c not in cat_feats]

        # If there are no numeric features (unlikely), raise a clear error
        if len(num_feats) == 0:
            raise ValueError('No numeric features found for scaling. Check input dataframe and feature engineering output.')

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('passthrough', 'passthrough', cat_feats)
        ])
        
        pipeline = Pipeline([('preprocessor', preprocessor)])
        pipeline.fit(X_train)
        
        X_train_trans = pipeline.transform(X_train)
        X_test_trans = pipeline.transform(X_test)
        
        # Get feature names
        # Transformer names might be generic, let's reconstruct
        col_names = num_feats + cat_feats
        
        return X_train_trans, X_test_trans, y_train.values, y_test.values, pipeline, col_names, test_dates

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/Time_Series___Turnover (1).csv')
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--mode', type=str, default='ts', choices=['tabular','ts'])
    parser.add_argument('--n-lags', type=int, default=12)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    # Mapping CSV raw names if not passed explicitly (Hardcoded based on user context)
    date_col = 'MES_REF'
    target_col = 'TO_TURNOVER_TO-GERAL'
    group_col = 'area_anonn'
    
    results = preprocess_data(df, mode=args.mode, n_lags=args.n_lags, 
                              date_col=date_col, target_col=target_col, group_col=group_col)
    
    X_train, X_test, y_train, y_test, pipeline, col_names, test_dates = results
    
    import joblib
    os.makedirs(args.out_dir, exist_ok=True)
    
    prepared = {
        'X_train': X_train, 'y_train': y_train, 
        'X_test': X_test, 'y_test': y_test, 
        'col_names': col_names,
        'test_dates': test_dates
    }
    
    joblib.dump(prepared, f"{args.out_dir}/prepared.joblib")
    joblib.dump(pipeline, f"{args.out_dir}/preprocessor.joblib")
    
    with open(f"{args.out_dir}/col_names.pkl", 'wb') as f:
        pickle.dump(col_names, f)
        
    print(f"Preprocessing ({args.mode}) complete. Train shape: {X_train.shape}")