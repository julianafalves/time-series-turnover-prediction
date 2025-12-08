"""
Train a tree-based model to predict turnover and save the model and metrics.
"""
import argparse
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def train_model(prepared_path: str, model_output: str, task: str = 'classification', 
                n_estimators: int = 100, random_state: int = 42, params: dict = None):
    """
    Train a tree-based model on prepared data and save artifacts.
    
    Trains either a RandomForestClassifier (for binary classification) or XGBRegressor (for 
    time-series regression) based on the task type. Saves the trained model and metrics to disk.

    Parameters
    ----------
    prepared_path : str
        Path to prepared.joblib containing train/test splits with keys: X_train, y_train, X_test, y_test
    model_output : str
        Path where the trained model will be saved as .joblib
    task : str, default='classification'
        Task type: 'classification' for RandomForestClassifier or 'ts' for XGBRegressor
    n_estimators : int, default=100
        Number of trees (used if not in params)
    random_state : int, default=42
        Random seed for reproducibility
    params : dict, optional
        Additional hyperparameters to pass to model constructor
        
    Returns
    -------
    model : estimator
        Trained sklearn/xgboost model
    metrics : dict
        Dictionary of evaluation metrics (classification: accuracy, precision, recall, f1, roc_auc;
        regression: mae, rmse, r2)
    """
    data = joblib.load(prepared_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    if params is None:
        params = {}

    if task == 'classification':
        # RandomForestClassifier for binary classification
        clf_params = params.copy()
        if 'n_estimators' not in clf_params:
            clf_params['n_estimators'] = n_estimators
        clf_params['random_state'] = random_state
        
        clf = RandomForestClassifier(**clf_params)
        clf.fit(X_train, y_train.astype(int))

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, probs)),
        }
        model = clf
        
    else:
        # XGBRegressor for time-series regression
        xgb_params = params.copy()
        if 'n_estimators' not in xgb_params:
            xgb_params['n_estimators'] = n_estimators
        xgb_params['random_state'] = random_state
        xgb_params['objective'] = 'reg:squarederror'
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train.astype(float))
        preds = model.predict(X_test)

        metrics = {
            'mae': float(mean_absolute_error(y_test, preds)),
            'rmse': float(mean_squared_error(y_test, preds, squared=False)),
            'r2': float(r2_score(y_test, preds)),
        }

    joblib.dump(model, model_output)
    metrics_output = model_output + '.metrics.json'
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a turnover prediction model and save artifacts.'
    )
    parser.add_argument('--prepared', type=str, default='data/prepared.joblib',
                        help='Path to prepared.joblib file')
    parser.add_argument('--model-output', type=str, default='models/rf_turnover.joblib',
                        help='Path where trained model will be saved')
    parser.add_argument('--task', type=str, default='classification', 
                        choices=['classification', 'ts'],
                        help='Task type: classification or time-series regression')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees')
    parser.add_argument('--params', type=str, default=None,
                        help='Path to JSON file with hyperparameters')
    args = parser.parse_args()

    params = None
    if args.params:
        with open(args.params, 'r') as f:
            params = json.load(f)

    train_model(args.prepared, args.model_output, task=args.task,
                n_estimators=args.n_estimators, params=params)
