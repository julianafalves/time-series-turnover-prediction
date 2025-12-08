"""
Comparison analysis: Fala AI vs Turnover prediction pipelines by area.

Aggregates Fala AI results and turnover metrics by organizational area hierarchy.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score


def analyze_fala_ai_by_area(fala_csv: str, out_dir: str = 'reports/comparison') -> dict:
    """
    Aggregate Fala AI pulse survey data by organizational area hierarchy.
    
    Computes turnover proportion and sample counts for each area at n2, n3, n4 levels.

    Parameters
    ----------
    fala_csv : str
        Path to Fala AI CSV file
    out_dir : str, default='reports/comparison'
        Output directory for results
        
    Returns
    -------
    dict
        Dictionary with area-level aggregations (n2_owner_area, n3_owner_area, n4_owner_area)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(fala_csv, low_memory=False)
    
    if 'voluntary_turnover_one_month_flag' not in df.columns:
        raise KeyError('voluntary_turnover_one_month_flag column not found')
    
    # Aggregate by area hierarchy
    results_by_area = {}
    
    for area_level in ['n2_owner_area', 'n3_owner_area', 'n4_owner_area']:
        if area_level not in df.columns:
            continue
        
        area_performance = {}
        
        grouped = df.groupby(area_level, dropna=False)
        for area, group_data in grouped:
            if pd.isna(area) or len(group_data) < 5:
                continue
            
            turnover_pct = group_data['voluntary_turnover_one_month_flag'].mean() * 100
            
            area_performance[str(area)] = {
                'turnover_rate': float(turnover_pct),
                'n_samples': int(len(group_data))
            }
        
        results_by_area[area_level] = area_performance
    
    return results_by_area


def analyze_turnover_by_area_res(turnover_csv: str, out_dir: str = 'reports/comparison') -> pd.DataFrame:
    """
    Aggregate turnover data by area responsible (EMAILADDRESS_AREA_RES).
    
    Parameters
    ----------
    turnover_csv : str
        Path to turnover CSV file
    out_dir : str, default='reports/comparison'
        Output directory for results
        
    Returns
    -------
    pd.DataFrame
        DataFrame with turnover statistics by area responsible email
    """
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(turnover_csv, low_memory=False)
    
    if 'TO_TURNOVER_TO-GERAL' not in df.columns or 'EMAILADDRESS_AREA_RES' not in df.columns:
        raise KeyError('Required columns TO_TURNOVER_TO-GERAL or EMAILADDRESS_AREA_RES not found')
    
    # Group by area responsible and compute statistics
    grouped = df.groupby('EMAILADDRESS_AREA_RES', dropna=True).agg({
        'TO_TURNOVER_TO-GERAL': ['mean', 'std', 'count']
    }).round(4)
    
    grouped.columns = ['avg_turnover_rate', 'std_turnover', 'n_records']
    
    return grouped


def compare_pipelines(fala_csv: str, turnover_csv: str, out_dir: str = 'reports/comparison') -> dict:
    """
    Compare Fala AI and turnover datasets at area level.
    
    Aggregates both datasets by area hierarchy and produces summary statistics and visualizations.

    Parameters
    ----------
    fala_csv : str
        Path to Fala AI pulse survey CSV
    turnover_csv : str
        Path to turnover dataset CSV
    out_dir : str, default='reports/comparison'
        Output directory for results
        
    Returns
    -------
    dict
        Summary statistics for both pipelines
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Analyze Fala AI
    fala_results = analyze_fala_ai_by_area(fala_csv, out_dir)
    
    # Analyze Turnover
    turnover_results = analyze_turnover_by_area_res(turnover_csv, out_dir)
    
    # Save results
    with open(os.path.join(out_dir, 'fala_ai_by_area.json'), 'w') as f:
        json.dump(fala_results, f, indent=2)
    
    turnover_results.to_csv(os.path.join(out_dir, 'turnover_by_area_res.csv'))
    
    # Generate summary statistics
    summary = {
        'fala_ai': {},
        'turnover': {
            'n_area_responsible': int(len(turnover_results)),
            'avg_turnover_rate': float(turnover_results['avg_turnover_rate'].mean()),
            'max_turnover_rate': float(turnover_results['avg_turnover_rate'].max()),
            'min_turnover_rate': float(turnover_results['avg_turnover_rate'].min()),
        }
    }
    
    for area_level in fala_results:
        valid_rates = [v['turnover_rate'] for v in fala_results[area_level].values() if isinstance(v, dict)]
        summary['fala_ai'][area_level] = {
            'n_areas': len(fala_results[area_level]),
            'avg_turnover_rate': float(np.mean(valid_rates)) if valid_rates else 0.0,
            'max_turnover_rate': float(np.max(valid_rates)) if valid_rates else 0.0,
        }
    
    with open(os.path.join(out_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Visualize Fala AI turnover by area level
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (area_level, ax) in enumerate(zip(['n2_owner_area', 'n3_owner_area', 'n4_owner_area'], axes)):
        area_data = fala_results.get(area_level, {})
        valid_data = {k: v['turnover_rate'] for k, v in area_data.items() if isinstance(v, dict)}
        
        if valid_data:
            # Top 10 areas for readability
            sorted_areas = sorted(valid_data.items(), key=lambda x: x[1], reverse=True)[:10]
            areas, rates = zip(*sorted_areas)
            
            ax.barh(range(len(rates)), rates, color='steelblue')
            ax.set_yticks(range(len(rates)))
            ax.set_yticklabels([str(a)[:20] for a in areas], fontsize=9)
            ax.set_xlabel('Turnover Rate (%)')
            ax.set_title(f'Top 10 Areas: {area_level}', fontsize=11)
            ax.set_xlim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fala_ai_turnover_by_area.png'), dpi=300)
    plt.close()
    
    # Visualize Turnover rate by area responsible (top 15)
    fig, ax = plt.subplots(figsize=(12, 8))
    top_areas = turnover_results.nlargest(15, 'avg_turnover_rate')
    
    ax.barh(range(len(top_areas)), top_areas['avg_turnover_rate'].values, color='coral')
    ax.set_yticks(range(len(top_areas)))
    ax.set_yticklabels([str(idx)[:35] for idx in top_areas.index], fontsize=10)
    ax.set_xlabel('Turnover Rate')
    ax.set_title('Top 15 Area Responsible: Turnover Rate', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'turnover_rate_by_area_res.png'), dpi=300)
    plt.close()
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Fala AI and turnover pipelines at area level.'
    )
    parser.add_argument('--fala-csv', type=str, 
                        default='data/juliana_alves_turnover_and_fala_ai_annon_with_label.csv',
                        help='Path to Fala AI CSV')
    parser.add_argument('--turnover-csv', type=str, 
                        default='data/juliana_alves_turnover_with_label.csv',
                        help='Path to turnover CSV')
    parser.add_argument('--out-dir', type=str, default='reports/comparison',
                        help='Output directory')
    args = parser.parse_args()
    
    compare_pipelines(args.fala_csv, args.turnover_csv, args.out_dir)
