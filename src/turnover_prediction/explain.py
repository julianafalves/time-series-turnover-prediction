"""
Produce SHAP explanations for the trained model.
"""
import argparse
import joblib
import numpy as np

# Workaround for shap compatibility with numpy >= 1.24
if not hasattr(np, 'int'):
    setattr(np, 'int', int)

import shap
import matplotlib.pyplot as plt
import pandas as pd


def explain(model_path: str, prepared_path: str, out_dir: str = 'reports'):
    """
    Generate SHAP explanations for a trained model.
    
    Creates summary plot showing feature importance and force plot for first test sample.
    Handles both classification and regression tasks automatically.

    Parameters
    ----------
    model_path : str
        Path to trained model (.joblib)
    prepared_path : str
        Path to prepared.joblib containing X_test and col_names
    out_dir : str, default='reports'
        Output directory for SHAP plots
    """
    model = joblib.load(model_path)
    data = joblib.load(prepared_path)
    X_test = data['X_test']
    
    # Retrieve column names from prepared data (required for SHAP feature names)
    col_names = data.get('col_names', None)
    if col_names is None:
        raise ValueError("col_names not found in prepared data. Ensure preprocessing includes col_names.")

    # Create SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle classifier (list of SHAP values per class) vs regressor (single array)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Binary classification: use positive class
    else:
        shap_vals = shap_values

    # Summary plot
    shap.summary_plot(shap_vals, features=X_test, feature_names=col_names, show=False)
    plt.title('SHAP Feature Importance Summary')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_summary.png", dpi=300)
    plt.close()

    # Force plot for first sample
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple)):
        expected_val = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    else:
        expected_val = expected_value

    shap.force_plot(expected_val, shap_vals[0, :], X_test.iloc[0, :] if hasattr(X_test, 'iloc') else X_test[0], 
                    feature_names=col_names, matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_force_sample0.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate SHAP explanations for a trained model.'
    )
    parser.add_argument('--model', type=str, default='models/rf_turnover.joblib',
                        help='Path to trained model')
    parser.add_argument('--prepared', type=str, default='data/prepared.joblib',
                        help='Path to prepared.joblib file')
    parser.add_argument('--out-dir', type=str, default='reports',
                        help='Output directory for SHAP plots')
    args = parser.parse_args()
    explain(args.model, args.prepared, args.out_dir)
