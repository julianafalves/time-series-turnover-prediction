"""
Produce SHAP explanations for the trained model.
"""
import argparse
import joblib
import numpy as np
# Workaround for shap compatibility with numpy >= 1.24 (shap may reference deprecated np.int)
if not hasattr(np, 'int'):
    setattr(np, 'int', int)
import shap
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np


def explain(model_path, prepared_path, out_dir='reports', task='classification'):
    model = joblib.load(model_path)
    data = joblib.load(prepared_path)
    X_test = data['X_test']

    # Load column names if available in prepared joblib else fallback
    if 'col_names' in data:
        col_names = data['col_names']
    else:
        with open('data/col_names.pkl', 'rb') as f:
            col_names = pickle.load(f)

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # For classifiers, shap_values may be a list (class probabilities); select the positive class. For regressors, it's a numpy array.
    if isinstance(shap_values, list):
        # choose index 1 or last.
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Summary plot
    shap.summary_plot(shap_vals, features=X_test, feature_names=col_names, show=False)
    plt.title('SHAP summary')
    plt.savefig(f"{out_dir}/shap_summary.png")
    plt.close()

    # Force plot example (single observation)
    idx = 0
    try:
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            expected_val = expected[1] if len(expected) > 1 else expected[0]
        else:
            expected_val = expected
    except Exception:
        expected_val = None

    if expected_val is not None and isinstance(shap_vals, np.ndarray):
        # Try force_plot (matplotlib version)
        shap.force_plot(expected_val, shap_vals[idx, :], X_test[idx], feature_names=col_names, matplotlib=True, show=False)
    # Make sure we capture the force plot as an image
    plt.savefig(f"{out_dir}/shap_force_sample0.png")
    plt.close()

    print('Saved SHAP plots')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/rf_turnover.joblib')
    parser.add_argument('--prepared', type=str, default='data/prepared.joblib')
    parser.add_argument('--out-dir', type=str, default='reports')
    parser.add_argument('--task', type=str, default='classification', choices=['classification','ts'])
    args = parser.parse_args()
    explain(args.model, args.prepared, args.out_dir, task=args.task)
