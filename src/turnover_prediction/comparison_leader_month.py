"""
Granular prediction comparison by leader and month.

Generates predictions for each leader × month combination from both pipelines
and compares accuracy at the highest available granularity.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer


def load_data(turnover_csv: str, fala_csv: str, prepared_joblib: str, 
              fala_model_path: str, turnover_model_path: str) -> tuple:
    """
    Load datasets, preprocess, and load trained models.
    
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
    
    # Parse dates and extract year-month
    turnover_df['MES_REF'] = pd.to_datetime(turnover_df['MES_REF'])
    turnover_df['year_month'] = turnover_df['MES_REF'].dt.to_period('M')
    
    fala_df['datereport'] = pd.to_datetime(fala_df['datereport'])
    fala_df['year_month'] = fala_df['datereport'].dt.to_period('M')
    
    # Load models
    prepared = joblib.load(prepared_joblib)
    fala_model = joblib.load(fala_model_path)
    turnover_model = joblib.load(turnover_model_path)
    
    return turnover_df, fala_df, prepared, fala_model, turnover_model


def get_fala_predictions_by_leader_month(fala_df: pd.DataFrame, fala_model, 
                                         area_level: str = 'n4_owner_area') -> pd.DataFrame:
    """
    Generate Fala AI predictions aggregated by leader and year-month.
    
    Counts absolute number of unique employees (by pseudo_person_id + month) 
    predicted to leave (and actual count).
    
    Parameters
    ----------
    fala_df : pd.DataFrame
        Fala AI dataset with year_month and pseudo_person_id columns
    fala_model : sklearn estimator
        Trained Fala AI classification model
    area_level : str, default='n4_owner_area'
        Column name for leader/area identifier
        
    Returns
    -------
    pd.DataFrame
        Employee counts (predicted to leave and actual) aggregated by leader and year-month
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
    
    # Group by leader and month, aggregate predictions
    results = []
    
    for (leader, year_month), group_idx in fala_df_dedup.groupby([area_level, 'year_month']).groups.items():
        if pd.isna(leader):
            continue
        
        n_samples = len(group_idx)
        if n_samples < 5:
            continue
        
        group_idx_array = np.array(group_idx)
        
        # Count absolute number of employees predicted/actual to leave
        n_pred_leave = int(predictions_binary[group_idx_array].sum())
        n_actual_leave = int(y.iloc[group_idx_array].sum())
        
        results.append({
            'leader': str(leader),
            'year_month': str(year_month),
            'n_employees': int(n_samples),
            'fala_pred_leave': int(n_pred_leave),
            'fala_actual_leave': int(n_actual_leave)
        })
    
    return pd.DataFrame(results)


def get_turnover_predictions_by_leader_month(turnover_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate actual employee terminations by leader and year-month.
    
    Counts absolute number of employees who actually left.
    
    Parameters
    ----------
    turnover_df : pd.DataFrame
        Turnover dataset with year_month column
        
    Returns
    -------
    pd.DataFrame
        Employee termination counts by leader and year-month
    """
    results = []
    
    for (leader, year_month), group in turnover_df.groupby(['EMAILADDRESS_AREA_RES', 'year_month']):
        if pd.isna(leader):
            continue
        
        n_employees = len(group)
        
        # Count actual terminations
        n_actual_left = 0
        if 'TO_DESLIGAMENTO_DESLIGAMENTO-MES-ATUAL' in group.columns:
            n_actual_left = int(group['TO_DESLIGAMENTO_DESLIGAMENTO-MES-ATUAL'].sum())
        
        results.append({
            'leader': str(leader),
            'year_month': str(year_month),
            'n_employees': int(n_employees),
            'turnover_actual_leave': int(n_actual_left)
        })
    
    return pd.DataFrame(results)


def compare_by_leader_month(turnover_csv: str, fala_csv: str, prepared_joblib: str, 
                            fala_model_path: str, turnover_model_path: str, 
                            out_dir: str = 'reports/comparison_leader_month') -> dict:
    """
    Compare employee predictions from both pipelines at leader × month granularity.
    
    Compares predicted number of employees leaving vs actual count for each leader-month.
    
    Parameters
    ----------
    turnover_csv : str
        Path to turnover dataset
    fala_csv : str
        Path to Fala AI dataset
    prepared_joblib : str
        Path to prepared.joblib
    fala_model_path : str
        Path to trained Fala AI model
    turnover_model_path : str
        Path to trained turnover model
    out_dir : str, default='reports/comparison_leader_month'
        Output directory for results
        
    Returns
    -------
    dict
        Summary statistics
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data and models
    turnover_df, fala_df, prepared, fala_model, turnover_model = load_data(
        turnover_csv, fala_csv, prepared_joblib, fala_model_path, turnover_model_path
    )
    
    # Generate predictions by leader and month
    fala_by_leader_month = get_fala_predictions_by_leader_month(fala_df, fala_model, 'n4_owner_area')
    turnover_by_leader_month = get_turnover_predictions_by_leader_month(turnover_df)
    
    # Merge predictions on leader and year-month
    comparison_table = fala_by_leader_month.merge(
        turnover_by_leader_month,
        on=['leader', 'year_month'],
        how='left',
        suffixes=('_fala', '_turnover')
    )
    
    # Calculate prediction errors (in absolute employee counts)
    comparison_table['fala_error'] = comparison_table['fala_pred_leave'] - comparison_table['fala_actual_leave']
    comparison_table['fala_abs_error'] = np.abs(comparison_table['fala_error'])
    
    # Save results
    comparison_table.to_csv(os.path.join(out_dir, 'prediction_comparison_leader_month.csv'), index=False)
    fala_by_leader_month.to_csv(os.path.join(out_dir, 'fala_predictions_leader_month.csv'), index=False)
    turnover_by_leader_month.to_csv(os.path.join(out_dir, 'turnover_actual_leader_month.csv'), index=False)
    
    # Compute summary statistics
    summary = {
        'fala_ai': {
            'n_leader_month_records': len(fala_by_leader_month),
            'n_unique_leaders': fala_by_leader_month['leader'].nunique(),
            'n_months': fala_by_leader_month['year_month'].nunique(),
            'total_employees_surveyed': int(fala_by_leader_month['n_employees'].sum()),
            'total_predicted_leave': int(fala_by_leader_month['fala_pred_leave'].sum()),
            'total_actual_leave': int(fala_by_leader_month['fala_actual_leave'].sum()),
            'mae_count': float(fala_by_leader_month['fala_pred_leave'].sub(fala_by_leader_month['fala_actual_leave']).abs().mean()),
        },
        'turnover': {
            'n_leader_month_records': len(turnover_by_leader_month),
            'n_unique_leaders': turnover_by_leader_month['leader'].nunique(),
            'n_months': turnover_by_leader_month['year_month'].nunique(),
            'total_employees': int(turnover_by_leader_month['n_employees'].sum()),
            'total_actual_leave': int(turnover_by_leader_month['turnover_actual_leave'].sum()),
        },
        'comparison': {
            'n_merged_records': len(comparison_table.dropna(subset=['turnover_actual_leave'])),
            'fala_mae_count': float(comparison_table['fala_abs_error'].mean())
        }
    }
    
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Visualization 1: Time series for top leaders
    top_leaders = fala_by_leader_month['leader'].value_counts().head(5).index
    
    fig, axes = plt.subplots(len(top_leaders), 1, figsize=(14, 3 * len(top_leaders)))
    if len(top_leaders) == 1:
        axes = [axes]
    
    for ax, leader in zip(axes, top_leaders):
        leader_data = fala_by_leader_month[fala_by_leader_month['leader'] == leader].sort_values('year_month')
        
        ax.plot(range(len(leader_data)), leader_data['fala_pred_leave'].values, 
                marker='o', linewidth=2, label='Fala AI Predicted to Leave')
        ax.plot(range(len(leader_data)), leader_data['fala_actual_leave'].values,
                marker='s', linewidth=2, label='Fala AI Actually Left')
        ax.set_title(f'Leader: {leader[:45]}', fontsize=11)
        ax.set_ylabel('Number of Employees')
        ax.set_xlabel('Month')
        ax.legend(loc='upper right')
        ax.set_xticks(range(len(leader_data)))
        ax.set_xticklabels([str(m)[:7] for m in leader_data['year_month'].values], rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'time_series_top_leaders.png'), dpi=300)
    plt.close()
    
    # Visualization 2: Error analysis
    merged_data = comparison_table.dropna(subset=['turnover_actual_leave'])
    
    if len(merged_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error over time
        ax = axes[0]
        merged_sorted = merged_data.sort_values('year_month')
        ax.scatter(range(len(merged_sorted)), merged_sorted['fala_abs_error'].values, 
                  alpha=0.5, s=40, color='steelblue')
        ax.axhline(y=merged_sorted['fala_abs_error'].mean(), color='r', 
                  linestyle='--', linewidth=2, label=f'Mean Error={merged_sorted["fala_abs_error"].mean():.1f} employees')
        ax.set_xlabel('Leader-Month Index')
        ax.set_ylabel('Absolute Prediction Error (Employees)')
        ax.set_title('Fala AI Prediction Error Over Time', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Actual vs Predicted scatter
        ax = axes[1]
        ax.scatter(merged_data['turnover_actual_leave'], merged_data['fala_pred_leave'], 
                  alpha=0.6, s=40, color='steelblue')
        max_val = max(merged_data['turnover_actual_leave'].max(), merged_data['fala_pred_leave'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Employees Left (Turnover Data)')
        ax.set_ylabel('Predicted Employees to Leave (Fala AI)')
        ax.set_title('Fala AI vs Turnover: Predicted vs Actual (Employee Count)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'error_analysis.png'), dpi=300)
        plt.close()
    
    # Visualization 3: Top errors
    fig, ax = plt.subplots(figsize=(12, 7))
    
    top_errors = comparison_table.nlargest(15, 'fala_abs_error')[['leader', 'year_month', 'fala_abs_error']]
    top_errors['label'] = top_errors['leader'].str[:25] + ' (' + top_errors['year_month'].astype(str) + ')'
    
    ax.barh(range(len(top_errors)), top_errors['fala_abs_error'].values, color='salmon')
    ax.set_yticks(range(len(top_errors)))
    ax.set_yticklabels(top_errors['label'].values, fontsize=10)
    ax.set_xlabel('Absolute Prediction Error (Employees)')
    ax.set_title('Top 15 Leader-Month Combinations with Largest Errors (Fala AI)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'largest_errors_leader_month.png'), dpi=300)
    plt.close()
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare predictions at leader × month granularity.'
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
    parser.add_argument('--out-dir', type=str, default='reports/comparison_leader_month',
                        help='Output directory')
    args = parser.parse_args()
    
    compare_by_leader_month(
        args.turnover_csv, args.fala_csv, args.prepared,
        args.fala_model, args.turnover_model, args.out_dir
    )
