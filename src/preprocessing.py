"""
Preprocessing utilities for the turnover dataset.
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
from datetime import datetime


def preprocess_data(df, mode='tabular', n_lags=12, date_col='date', target_col='turnover_rate'):
    if mode == 'tabular':
        cat_cols = ['role', 'overtime']
        num_cols = ['age', 'tenure', 'salary', 'satisfaction', 'projects', 'promotion_last_5yrs', 'performance']
    else:
        # time series: numeric exogenous columns (do not include target)
        cat_cols = []
        # Try to infer date column if not provided
        if date_col not in df.columns:
            date_candidates = [c for c in df.columns if 'date' in c.lower() or 'mes' in c.lower() or c.lower().startswith('mes_ref')]
            if date_candidates:
                date_col = date_candidates[0]
        # Try to infer target column if not provided
        if target_col not in df.columns:
            candidates = [c for c in df.columns if 'turnover' in c.lower()] or [c for c in df.columns if 'to_' in c.lower() and 'turn' in c.lower()]
            if candidates:
                target_col = candidates[0]
        # Normalize date/target column names so downstream code uses 'date' and 'turnover_rate'
        if date_col != 'date' and date_col in df.columns:
            df = df.rename(columns={date_col: 'date'})
            date_col = 'date'
        if target_col != 'turnover_rate' and target_col in df.columns:
            df = df.rename(columns={target_col: 'turnover_rate'})
        # if target_col not found, try to infer from column names
        if target_col not in df.columns:
            candidates = [c for c in df.columns if 'turnover' in c.lower() or 'turnover' in c.lower() or c.lower().startswith('to_turnover')]
            if candidates:
                # prefer exact 'turnover_rate' if present in the candidates, else pick the first
                target_col = 'turnover_rate' if 'turnover_rate' in candidates else candidates[0]
                # normalize column name for downstream code by renaming if necessary
                if target_col != 'turnover_rate':
                    df = df.rename(columns={target_col: 'turnover_rate'})
                    target_col = 'turnover_rate'
        # Normalize target column to 'turnover_rate' if needed
        if target_col != 'turnover_rate' and target_col in df.columns:
            df = df.rename(columns={target_col: 'turnover_rate'})
        # Select numeric columns only for scaling (drop string identifiers like area_anonn)
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        num_cols = [c for c in numeric_cols if c not in [date_col, 'turnover_rate', 'leavers']]
        # If target column not available, try to compute from 'turnover' and 'num_employees'
        if 'turnover_rate' not in df.columns:
            if 'turnover' in df.columns and 'num_employees' in df.columns:
                df['turnover_rate'] = df['turnover'] / df['num_employees']
            elif 'turnover' in df.columns:
                # If turnover is binary/0-1, approximate as rate per period
                if df['turnover'].max() <= 1:
                    df['turnover_rate'] = df['turnover']
                else:
                    # Fallback: compute normalized values
                    df['turnover_rate'] = df['turnover'] / df['turnover'].max()

    # For ts mode, we'll build the preprocessor after creating lag features so it contains all numeric columns.
    if mode == 'tabular':
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_cols),
        ])

    if mode == 'tabular':
        X = df.drop(columns=['turnover'])
        y = df['turnover']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # time series processing: create lag features for target
        ts = df.copy()
        # normalize the date column to 'date' for consistency
        if date_col != 'date' and date_col in ts.columns:
            ts = ts.rename(columns={date_col: 'date'})
            date_col = 'date'
        target = 'turnover_rate'
        for lag in range(1, n_lags + 1):
            ts[f'lag_{lag}'] = ts[target].shift(lag)
        # rolling means
        ts['rolling_mean_3'] = ts[target].rolling(3).mean()
        ts['rolling_mean_6'] = ts[target].rolling(6).mean()

        ts = ts.dropna().reset_index(drop=True)
        drop_cols = [c for c in [date_col, 'leavers', target] if c in ts.columns]
        X = ts.drop(columns=drop_cols)
        y = ts[target]
        # chronological train/test split: last 12 periods as test
        test_periods = int(max(1, round(0.2 * len(ts))))
        X_train = X[:-test_periods]
        X_test = X[-test_periods:]
        y_train = y[:-test_periods]
        y_test = y[-test_periods:]

    if mode == 'ts':
        # now num_cols should be only numeric columns in X_train (lag_1..lag_n + numeric exogenous)
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
        ])
    pipeline = Pipeline([('preprocessor', preprocessor)])
    pipeline.fit(X_train)
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # We need column names for later explainability
    if mode == 'tabular':
        cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_cols_transformed = list(cat_encoder.get_feature_names_out(cat_cols))
        col_names = num_cols + cat_cols_transformed
    else:
        # For TS, when we have only numeric columns, the feature names should be the numeric columns
        col_names = num_cols

    # Optionally return test dates for time series tasks
    test_dates = None
    if mode == 'ts' and 'date' in ts.columns:
        # ts is the post-lag dataset; the last test_periods are the test dates
        test_dates = list(ts['date'][-test_periods:])
    return X_train_transformed, X_test_transformed, y_train.values, y_test.values, pipeline, col_names, test_dates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/time_series_turnover.csv')
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--mode', type=str, default='tabular', choices=['tabular','ts'])
    parser.add_argument('--date-col', type=str, default='date')
    parser.add_argument('--target-col', type=str, default='turnover_rate')
    parser.add_argument('--n-lags', type=int, default=12)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X_train, X_test, y_train, y_test, pipeline, col_names, test_dates = preprocess_data(df, mode=args.mode, n_lags=args.n_lags, date_col=args.date_col, target_col=args.target_col)
    # No additional CLI output by default

    # Save datasets and preprocessor
    import joblib
    prepared = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'col_names': col_names}
    if args.mode == 'ts' and test_dates is not None:
        prepared['test_dates'] = test_dates

    # Save files with mode-specific suffix so tests can locate them
    prepared_name = 'prepared_ts.joblib' if args.mode == 'ts' else 'prepared.joblib'
    preprocessor_name = 'preprocessor_ts.joblib' if args.mode == 'ts' else 'preprocessor.joblib'
    joblib.dump(prepared, f"{args.out_dir}/{prepared_name}")
    joblib.dump(pipeline, f"{args.out_dir}/{preprocessor_name}")
    # Also save a default 'prepared.joblib' / 'preprocessor.joblib' for backward compatibility
    joblib.dump(prepared, f"{args.out_dir}/prepared.joblib")
    joblib.dump(pipeline, f"{args.out_dir}/preprocessor.joblib")
    with open(f"{args.out_dir}/col_names.pkl", 'wb') as f:
        pickle.dump(col_names, f)
    print('Preprocessing complete, saved to', args.out_dir)
