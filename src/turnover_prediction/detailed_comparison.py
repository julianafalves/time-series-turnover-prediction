"""
Granular comparison: Turnover vs Fala AI predictions by month and area.

Aggregates both datasets by temporal and spatial dimensions for detailed analysis.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_prepare_data(turnover_csv: str, fala_csv: str) -> tuple:
    """
    Load and prepare both datasets for comparison.
    
    Parameters
    ----------
    turnover_csv : str
        Path to turnover dataset
    fala_csv : str
        Path to Fala AI pulse dataset
        
    Returns
    -------
    tuple
        (turnover_df, fala_df) with datetime columns converted and year_month periods extracted
    """
    turnover_df = pd.read_csv(turnover_csv, low_memory=False)
    turnover_df['MES_REF'] = pd.to_datetime(turnover_df['MES_REF'])
    turnover_df['year_month'] = turnover_df['MES_REF'].dt.to_period('M')
    
    fala_df = pd.read_csv(fala_csv, low_memory=False)
    fala_df['datereport'] = pd.to_datetime(fala_df['datereport'])
    fala_df['year_month'] = fala_df['datereport'].dt.to_period('M')
    
    return turnover_df, fala_df


def aggregate_turnover_by_month_area(turnover_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate turnover employee terminations by month and area responsible.
    
    Counts absolute number of employees who terminated in each month-area combination.
    
    Parameters
    ----------
    turnover_df : pd.DataFrame
        Turnover dataset with MES_REF, year_month, and area columns
        
    Returns
    -------
    pd.DataFrame
        Aggregated employee termination counts by month and area
    """
    agg = turnover_df.groupby(['year_month', 'EMAILADDRESS_AREA_RES']).agg({
        'TO_DESLIGAMENTO_DESLIGAMENTO-MES-ATUAL': 'sum'
    }).reset_index()
    
    # Also count total employees
    agg_count = turnover_df.groupby(['year_month', 'EMAILADDRESS_AREA_RES']).size().reset_index(name='n_employees')
    
    # Merge
    agg = agg.merge(agg_count, on=['year_month', 'EMAILADDRESS_AREA_RES'])
    agg.columns = ['year_month', 'area_res', 'n_terminated', 'n_employees']
    return agg


def aggregate_fala_by_month_area(fala_df: pd.DataFrame, area_level: str = 'n2_owner_area') -> pd.DataFrame:
    """
    Aggregate Fala AI voluntary turnover (employee counts) by month and area level.
    
    Counts absolute number of unique employees (by pseudo_person_id + month) 
    predicted/actual to leave.
    
    Parameters
    ----------
    fala_df : pd.DataFrame
        Fala AI dataset with year_month, pseudo_person_id, and area columns
    area_level : str, default='n2_owner_area'
        Area hierarchy level (n2_owner_area, n3_owner_area, or n4_owner_area)
        
    Returns
    -------
    pd.DataFrame
        Aggregated employee counts (turnover and total) by month and area
    """
    # Deduplicate by unique pseudo_person_id + year_month (keep first occurrence)
    fala_df_dedup = fala_df.drop_duplicates(subset=['pseudo_person_id', 'year_month'], keep='first')
    
    agg = fala_df_dedup.groupby(['year_month', area_level]).agg({
        'voluntary_turnover_one_month_flag': ['sum', 'count']
    }).reset_index()
    
    agg.columns = ['year_month', 'area', 'n_voluntary_leave', 'n_employees']
    
    return agg


def compare_predictions(turnover_df: pd.DataFrame, fala_df: pd.DataFrame, 
                       out_dir: str = 'reports/comparison_detailed') -> dict:
    """
    Compare aggregated turnover and Fala AI predictions by month and area.
    
    Produces aggregated data files and visualization comparing both datasets across
    temporal and spatial dimensions.
    
    Parameters
    ----------
    turnover_df : pd.DataFrame
        Turnover dataset with year_month column
    fala_df : pd.DataFrame
        Fala AI dataset with year_month column
    out_dir : str, default='reports/comparison_detailed'
        Output directory for results
        
    Returns
    -------
    dict
        Summary statistics for both datasets
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Aggregate both datasets
    turnover_agg = aggregate_turnover_by_month_area(turnover_df)
    fala_n2 = aggregate_fala_by_month_area(fala_df, 'n2_owner_area')
    fala_n3 = aggregate_fala_by_month_area(fala_df, 'n3_owner_area')
    fala_n4 = aggregate_fala_by_month_area(fala_df, 'n4_owner_area')
    
    # Save aggregations
    turnover_agg.to_csv(os.path.join(out_dir, 'turnover_agg_by_month_area.csv'), index=False)
    fala_n2.to_csv(os.path.join(out_dir, 'fala_agg_by_month_n2.csv'), index=False)
    fala_n3.to_csv(os.path.join(out_dir, 'fala_agg_by_month_n3.csv'), index=False)
    fala_n4.to_csv(os.path.join(out_dir, 'fala_agg_by_month_n4.csv'), index=False)
    
    # Compute summary statistics
    comparison_summary = {
        'turnover_only': {
            'n_month_area_records': len(turnover_agg),
            'total_employees': int(turnover_agg['n_employees'].sum()),
            'total_terminated': int(turnover_agg['n_terminated'].sum()),
        }
    }
    
    for fala_agg, key in [(fala_n2, 'fala_ai_n2'), (fala_n3, 'fala_ai_n3'), (fala_n4, 'fala_ai_n4')]:
        comparison_summary[key] = {
            'n_month_area_records': len(fala_agg),
            'total_employees': int(fala_agg['n_employees'].sum()),
            'total_voluntary_leave': int(fala_agg['n_voluntary_leave'].sum()),
        }
    
    with open(os.path.join(out_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    # Visualization 1: Turnover rate over time
    monthly_terminated = turnover_agg.groupby('year_month')['n_terminated'].sum()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_terminated.index.astype(str), monthly_terminated.values, marker='o', linewidth=2)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Employees Terminated')
    ax.set_title('Employee Terminations Over Time (Turnover Dataset)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'turnover_rate_by_month.png'), dpi=300)
    plt.close()
    
    # Visualization 2: Fala AI voluntary turnover over time
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for fala_agg, ax, title in [(fala_n2, axes[0], 'N2 Areas'), 
                                 (fala_n3, axes[1], 'N3 Areas'), 
                                 (fala_n4, axes[2], 'N4 Areas')]:
        monthly_vol = fala_agg.groupby('year_month')['n_voluntary_leave'].sum()
        ax.plot(monthly_vol.index.astype(str), monthly_vol.values, marker='o', linewidth=2, color='green')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Employees')
        ax.set_title(f'Fala AI Voluntary Turnover - {title}', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fala_ai_voluntary_turnover_by_month.png'), dpi=300)
    plt.close()
    
    # Visualization 3: Distribution comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = [
        turnover_agg['n_terminated'].dropna(),
        fala_n2['n_voluntary_leave'].dropna(),
        fala_n3['n_voluntary_leave'].dropna(),
        fala_n4['n_voluntary_leave'].dropna(),
    ]
    labels_to_plot = ['Turnover\nDataset', 'Fala AI\n(N2)', 'Fala AI\n(N3)', 'Fala AI\n(N4)']
    
    ax.boxplot(data_to_plot, labels=labels_to_plot)
    ax.set_ylabel('Number of Employees')
    ax.set_title('Distribution of Employee Terminations: Comparison', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'turnover_distribution_comparison.png'), dpi=300)
    plt.close()
    
    return comparison_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detailed comparison of turnover and Fala AI data by month and area.'
    )
    parser.add_argument('--turnover-csv', type=str, 
                        default='data/juliana_alves_turnover_with_label.csv',
                        help='Path to turnover dataset')
    parser.add_argument('--fala-csv', type=str, 
                        default='data/juliana_alves_turnover_and_fala_ai_annon_with_label.csv',
                        help='Path to Fala AI dataset')
    parser.add_argument('--out-dir', type=str, default='reports/comparison_detailed',
                        help='Output directory')
    args = parser.parse_args()
    
    turnover_df, fala_df = load_and_prepare_data(args.turnover_csv, args.fala_csv)
    compare_predictions(turnover_df, fala_df, args.out_dir)
