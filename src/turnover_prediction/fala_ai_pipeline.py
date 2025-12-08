"""
Employee pulse analysis pipeline using Fala AI survey data.

Trains RandomForestClassifier to predict voluntary turnover from pulse survey responses.
"""
import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


def run_fala_pipeline(input_csv: str, out_dir: str, model_output: str, 
                      n_estimators: int = 100, test_size: float = 0.2, 
                      random_state: int = 42) -> tuple:
    """
    Train RandomForest classifier on Fala AI pulse survey data.
    
    Predicts voluntary_turnover_one_month_flag from survey responses. Applies mean imputation
    to numeric features and outputs model, metrics, and confusion matrix visualization.

    Parameters
    ----------
    input_csv : str
        Path to input CSV (must contain 'voluntary_turnover_one_month_flag' column)
    out_dir : str
        Output directory for metrics JSON and confusion matrix plot
    model_output : str
        Path where trained model will be saved
    n_estimators : int, default=100
        Number of trees in RandomForest
    test_size : float, default=0.2
        Proportion of data for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (trained_model, metrics_dict)
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_output) or '.', exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv, low_memory=False)

    if 'voluntary_turnover_one_month_flag' not in df.columns:
        raise KeyError('Column voluntary_turnover_one_month_flag not found in input CSV')

    # Split features and target
    y = df['voluntary_turnover_one_month_flag']
    X = df.drop(['voluntary_turnover_one_month_flag'], axis=1)

    # Select numeric features for modeling (ignore ID/date columns)
    X_numeric = X.select_dtypes(include=[np.number])

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_numeric)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=random_state
    )

    # Train RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    # Save artifacts
    joblib.dump(clf, model_output)
    with open(os.path.join(out_dir, 'fala_ai_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix visualization
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix: Voluntary Turnover Prediction', fontsize=12)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Turnover', 'Turnover'])
    ax.set_yticklabels(['No Turnover', 'Turnover'])
    
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    return clf, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train RandomForest classifier on Fala AI employee pulse data.'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--out-dir', type=str, default='reports/fala_ai',
                        help='Output directory for metrics and plots')
    parser.add_argument('--model-output', type=str, default='models/fala_rf.joblib',
                        help='Path where trained model will be saved')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in RandomForest')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for testing')
    args = parser.parse_args()

    run_fala_pipeline(args.input, args.out_dir, args.model_output, 
                      n_estimators=args.n_estimators, test_size=args.test_size)
