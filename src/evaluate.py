"""
Evaluate the saved model and produce plots and metrics.
"""
import argparse
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate(prepared_path, model_path, report_dir='reports', task='classification'):
    data = joblib.load(prepared_path)
    X_test = data['X_test']
    y_test = data['y_test']

    model = joblib.load(model_path)

    preds = model.predict(X_test)
    if task == 'classification':
        probs = model.predict_proba(X_test)[:, 1]

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.savefig(f"{report_dir}/confusion_matrix.png")
        plt.close()

        # ROC
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        plt.title('ROC Curve')
        plt.savefig(f"{report_dir}/roc_curve.png")
        plt.close()

        metrics = {
            'roc_auc': float(roc_auc)
        }
    else:
        # Regression metrics for time series
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(mean_squared_error(y_test, preds, squared=False))
        r2 = float(r2_score(y_test, preds))

        metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

        if 'test_dates' in data:
            dates = data['test_dates']
            dates = [np.datetime64(d) for d in dates]
            plt.figure(figsize=(10, 4))
            plt.plot(dates, y_test, label='actual')
            plt.plot(dates, preds, label='predicted')
            plt.legend()
            plt.title('Actual vs Predicted (test set)')
            plt.savefig(f"{report_dir}/ts_actual_vs_pred.png")
            plt.close()
    with open(f"{report_dir}/eval_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved evaluation metrics and plots')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=str, default='data/prepared.joblib')
    parser.add_argument('--model', type=str, default='models/rf_turnover.joblib')
    parser.add_argument('--report-dir', type=str, default='reports')
    parser.add_argument('--task', type=str, default='classification', choices=['classification','ts'])
    args = parser.parse_args()

    evaluate(args.prepared, args.model, args.report_dir, task=args.task)
