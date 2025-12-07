"""
Train a tree-based model to predict turnover and save the model and metrics.
"""
import argparse
import joblib
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def train_model(prepared_path, model_output, task='classification', n_estimators=100, random_state=42):
    data = joblib.load(prepared_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    if task == 'classification':
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
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
    else:
        # time series regression using XGBoost
        reg = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state, objective='reg:squarederror')
        reg.fit(X_train, y_train.astype(float))
        preds = reg.predict(X_test)

        metrics = {
            'mae': float(mean_absolute_error(y_test, preds)),
            'rmse': float(mean_squared_error(y_test, preds, squared=False)),
            'r2': float(r2_score(y_test, preds)),
        }
        clf = reg

    joblib.dump(clf, model_output)
    metrics_output = model_output + '.metrics.json'
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'Model saved to {model_output}')
    print('Metrics:', metrics)
    return clf, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=str, default='data/prepared.joblib')
    parser.add_argument('--model-output', type=str, default='models/rf_turnover.joblib')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'ts'])
    parser.add_argument('--n-estimators', type=int, default=100)
    args = parser.parse_args()

    train_model(args.prepared, args.model_output, task=args.task, n_estimators=args.n_estimators)
