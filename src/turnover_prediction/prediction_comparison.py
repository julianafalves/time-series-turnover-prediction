"""
Model predictions comparison at area level.

Generates predictions from both Fala AI and Turnover pipelines and compares accuracy.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer


def load_models_and_data(turnover_csv: str, fala_csv: str, prepared_joblib: str, 
                         fala_model_path: str, turnover_model_path: str) -> tuple:
    """
    Load datasets, preprocessor, and trained models.
    
    Parameters
    ----------
    turnover_csv : str
        Path to turnover dataset
    fala_csv : str
        Path to Fala AI dataset
    prepared_joblib : str
        Path to prepared.joblib with preprocessor
    fala_model_path : str
        Path to trained Fala AI model
    turnover_model_path : str
        Path to trained turnover model
        
    Returns
    -------
    tuple
        (turnover_df, fala_df, prepared, fala_model, turnover_model)
    """
    turnover_df = pd.read_csv(turnover_csv, low_memory=False)
    fala_df = pd.read_csv(fala_csv, low_memory=False)
    prepared = joblib.load(prepared_joblib)
    fala_model = joblib.load(fala_model_path)
    turnover_model = joblib.load(turnover_model_path)
    
    return turnover_df, fala_df, prepared, fala_model, turnover_model


def get_turnover_predictions_by_area_res(turnover_df: pd.DataFrame, prepared: dict, 
                                         turnover_model) -> dict:
    """
    Generate turnover model predictions and compute error metrics.
    
    Parameters
    ----------
    turnover_df : pd.DataFrame
        Turnover dataset
    prepared : dict
        Prepared data with X_test and y_test
    turnover_model : sklearn estimator
        Trained turnover model
        
    Returns
    -------
    dict
        Dictionary with predictions and metrics
    """
    X_test = prepared['X_test']
    y_test = prepared['y_test']
    
    predictions = turnover_model.predict(X_test)
    
    metrics = {
        'avg_prediction': float(np.mean(predictions)),
        'avg_actual': float(np.mean(y_test)),
        'mae': float(np.mean(np.abs(predictions - y_test))),
        'rmse': float(np.sqrt(np.mean((predictions - y_test) ** 2)))
    }
    
    return metrics


def get_fala_predictions_by_area(fala_df: pd.DataFrame, fala_model, 
                                  area_level: str = 'n2_owner_area') -> pd.DataFrame:
    """
    Generate Fala AI predictions aggregated by area level.
    
    Counts absolute number of unique employees (by pseudo_person_id + month) 
    predicted to leave by area.
    
    Parameters
    ----------
    fala_df : pd.DataFrame
        Fala AI dataset with pseudo_person_id and year_month columns
    fala_model : sklearn estimator
        Trained Fala AI classification model
    area_level : str, default='n2_owner_area'
        Area hierarchy level to aggregate by
        
    Returns
    -------
    pd.DataFrame
        Employee counts (predicted to leave and actual) aggregated by area
    """
    # Deduplicate by unique pseudo_person_id + year_month (keep first occurrence)
    fala_df_dedup = fala_df.drop_duplicates(subset=['pseudo_person_id', 'year_month'], keep='first').reset_index(drop=True)
    
    X = fala_df_dedup.drop(['voluntary_turnover_one_month_flag'], axis=1)
    y = fala_df_dedup['voluntary_turnover_one_month_flag']
    
    # Keep numeric features only
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_numeric)
    
    # Generate predictions
    predictions_binary = fala_model.predict(X_imputed)
    
    # Group by area and aggregate
    area_col = fala_df_dedup[area_level].reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    results = []
    for area in area_col.unique():
        if pd.isna(area):
            continue
        
        mask = area_col == area
        n_samples = mask.sum()
        
        # Skip very small areas
        if n_samples < 10:
            continue
        
        # Count absolute number of employees predicted/actual to leave
        n_pred_leave = int(predictions_binary[mask].sum())
        n_actual_leave = int(y_reset[mask].sum())
        
        results.append({
            'area': str(area),
            'n_employees': int(n_samples),
            'pred_leave': int(n_pred_leave),
            'actual_leave': int(n_actual_leave)
        })
    
    return pd.DataFrame(results)


def compare_all_predictions(turnover_csv: str, fala_csv: str, prepared_joblib: str, 
                            fala_model_path: str, turnover_model_path: str, 
                            out_dir: str = 'reports/prediction_comparison') -> dict:
    """
    Compare predictions from Fala AI and turnover-only pipelines.
    
    Generates predictions at area level from both models and produces summary statistics
    and visualizations comparing prediction accuracy.
    
    Parameters
    ----------
    turnover_csv : str
        Path to turnover dataset
    fala_csv : str
        Path to Fala AI dataset
    prepared_joblib : str
        Path to prepared.joblib with preprocessor
    fala_model_path : str
        Path to trained Fala AI model
    turnover_model_path : str
        Path to trained turnover model
    out_dir : str, default='reports/prediction_comparison'
        Output directory for results
        
    Returns
    -------
    dict
        Summary statistics with metrics for both pipelines
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load models and data
    turnover_df, fala_df, prepared, fala_model, turnover_model = load_models_and_data(
        turnover_csv, fala_csv, prepared_joblib, fala_model_path, turnover_model_path
    )
    
    # Generate predictions for all three area levels
    # Add year_month to fala_df for deduplication
    fala_df['datereport'] = pd.to_datetime(fala_df['datereport'])
    fala_df['year_month'] = fala_df['datereport'].dt.to_period('M')
        
    # Generate predictions for all three area levels
    fala_n2 = get_fala_predictions_by_area(fala_df, fala_model, 'n2_owner_area')
    fala_n3 = get_fala_predictions_by_area(fala_df, fala_model, 'n3_owner_area')
    fala_n4 = get_fala_predictions_by_area(fala_df, fala_model, 'n4_owner_area')
    
    # Get turnover model metrics
    turnover_metrics = get_turnover_predictions_by_area_res(turnover_df, prepared, turnover_model)
    
    # Save prediction dataframes
    fala_n2.to_csv(os.path.join(out_dir, 'fala_predictions_n2.csv'), index=False)
    fala_n3.to_csv(os.path.join(out_dir, 'fala_predictions_n3.csv'), index=False)
    fala_n4.to_csv(os.path.join(out_dir, 'fala_predictions_n4.csv'), index=False)
    
    # Compute summary statistics
    summary = {
        'turnover_model': turnover_metrics,
        'fala_ai_n2': {
            'n_areas': len(fala_n2),
            'total_employees': int(fala_n2['n_employees'].sum()),
            'total_predicted_leave': int(fala_n2['pred_leave'].sum()),
            'total_actual_leave': int(fala_n2['actual_leave'].sum()),
            'mae_count': float(np.mean(np.abs(fala_n2['pred_leave'] - fala_n2['actual_leave'])))
        },
        'fala_ai_n3': {
            'n_areas': len(fala_n3),
            'total_employees': int(fala_n3['n_employees'].sum()),
            'total_predicted_leave': int(fala_n3['pred_leave'].sum()),
            'total_actual_leave': int(fala_n3['actual_leave'].sum()),
            'mae_count': float(np.mean(np.abs(fala_n3['pred_leave'] - fala_n3['actual_leave'])))
        },
        'fala_ai_n4': {
            'n_areas': len(fala_n4),
            'total_employees': int(fala_n4['n_employees'].sum()),
            'total_predicted_leave': int(fala_n4['pred_leave'].sum()),
            'total_actual_leave': int(fala_n4['actual_leave'].sum()),
            'mae_count': float(np.mean(np.abs(fala_n4['pred_leave'] - fala_n4['actual_leave'])))
        }
    }
    
    with open(os.path.join(out_dir, 'prediction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Visualization 1: Actual vs Predicted for Fala AI (N4) with error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.scatter(fala_n4['actual_leave'], fala_n4['pred_leave'], alpha=0.6, s=50)
    max_val = max(fala_n4['actual_leave'].max(), fala_n4['pred_leave'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Employees Left')
    ax.set_ylabel('Predicted Employees to Leave')
    ax.set_title('Fala AI N4 Areas: Predictions vs Actuals (Employee Count)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    errors = fala_n4['pred_leave'] - fala_n4['actual_leave']
    ax.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error (Employee Count)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution (Fala AI N4)', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fala_ai_predictions_n4.png'), dpi=300)
    plt.close()
    
    # Visualization 2: Error metrics across area levels
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['MAE (Employees)']
    n2_vals = [summary['fala_ai_n2']['mae_count']]
    n3_vals = [summary['fala_ai_n3']['mae_count']]
    n4_vals = [summary['fala_ai_n4']['mae_count']]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    ax.bar(x - width, n2_vals, width, label='N2 Areas', color='steelblue')
    ax.bar(x, n3_vals, width, label='N3 Areas', color='coral')
    ax.bar(x + width, n4_vals, width, label='N4 Areas', color='mediumseagreen')
    
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Error (Employee Count) by Area Level', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_comparison_by_level.png'), dpi=300)
    plt.close()
    
    # Visualization 3: Top areas with largest errors
    fig, ax = plt.subplots(figsize=(12, 7))
    
    errors_n4 = fala_n4.copy()
    errors_n4['abs_error'] = np.abs(errors_n4['pred_leave'] - errors_n4['actual_leave'])
    top_errors = errors_n4.nlargest(15, 'abs_error')
    
    ax.barh(range(len(top_errors)), top_errors['abs_error'].values, color='salmon')
    ax.set_yticks(range(len(top_errors)))
    ax.set_yticklabels([str(a)[:35] for a in top_errors['area']], fontsize=10)
    ax.set_xlabel('Absolute Prediction Error (Employees)')
    ax.set_title('Top 15 Areas by Prediction Error (Fala AI N4)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'largest_errors_n4.png'), dpi=300)
    plt.close()
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare predictions from Fala AI and turnover-only pipelines.'
    )
    parser.add_argument('--turnover-csv', type=str, 
                        default='data/juliana_alves_turnover_with_label.csv',
                        help='Path to turnover dataset')
    parser.add_argument('--fala-csv', type=str, 
                        default='data/juliana_alves_turnover_and_fala_ai_annon_with_label.csv',
                        help='Path to Fala AI dataset')
    parser.add_argument('--prepared', type=str, default='data/prepared.joblib',
                        help='Path to prepared.joblib')
    parser.add_argument('--fala-model', type=str, default='models/fala_rf.joblib',
                        help='Path to trained Fala AI model')
    parser.add_argument('--turnover-model', type=str, default='models/xgb_turnover.joblib',
                        help='Path to trained turnover model')
    parser.add_argument('--out-dir', type=str, default='reports/prediction_comparison',
                        help='Output directory')
    args = parser.parse_args()
    
    compare_all_predictions(
        args.turnover_csv, args.fala_csv, args.prepared,
        args.fala_model, args.turnover_model, args.out_dir
    )
