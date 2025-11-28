import os
import subprocess
import sys
from pathlib import Path


def test_end_to_end(tmp_path):
    # use the user-provided CSV; fallback to Time_Series___Turnover (1).csv if necessary
    user_csv = Path('data/time_series_turnover.csv')
    fallback_csv = Path('data/Time_Series___Turnover (1).csv')
    if user_csv.exists():
        input_csv = user_csv
    else:
        input_csv = fallback_csv
    assert input_csv.exists(), 'No CSV dataset found in data/; please add data/time_series_turnover.csv'

    # run preprocessing
    prepared = tmp_path / 'prepared.joblib'
    cmd = [sys.executable, 'src/preprocessing.py', '--input', str(input_csv), '--out-dir', str(tmp_path), '--mode', 'ts']
    subprocess.run(cmd, check=True)
    assert prepared.exists()

    # train
    model = tmp_path / 'rf.joblib'
    cmd = [sys.executable, 'src/train.py', '--prepared', str(prepared), '--model-output', str(model), '--task', 'ts']
    subprocess.run(cmd, check=True)
    assert model.exists()

    # explain
    reports_dir = tmp_path / 'reports'
    reports_dir.mkdir(exist_ok=True)
    cmd = [sys.executable, 'src/explain.py', '--model', str(model), '--prepared', str(prepared), '--out-dir', str(reports_dir), '--task','ts']
    subprocess.run(cmd, check=True)

    # Check at least one SHAP plot exists
    assert (reports_dir / 'shap_summary.png').exists()


def test_time_series_pipeline(tmp_path):
    # use user CSV
    user_csv = Path('data/time_series_turnover.csv')
    fallback_csv = Path('data/Time_Series___Turnover (1).csv')
    if user_csv.exists():
        input_csv = user_csv
    else:
        input_csv = fallback_csv
    assert input_csv.exists(), 'No CSV dataset found in data/; please add data/time_series_turnover.csv'

    # preprocess ts into a temporary out dir
    prepared = tmp_path / 'prepared_ts.joblib'
    cmd = [sys.executable, 'src/preprocessing.py', '--input', str(input_csv), '--out-dir', str(tmp_path), '--mode', 'ts', '--n-lags', '6']
    subprocess.run(cmd, check=True)
    assert prepared.exists()

    # train ts
    model = tmp_path / 'xgb.joblib'
    cmd = [sys.executable, 'src/train.py', '--prepared', str(prepared), '--model-output', str(model), '--task', 'ts', '--n-estimators', '50']
    subprocess.run(cmd, check=True)
    assert model.exists()

    # explain ts
    reports_dir = tmp_path / 'reports_ts'
    reports_dir.mkdir(exist_ok=True)
    cmd = [sys.executable, 'src/explain.py', '--model', str(model), '--prepared', str(prepared), '--out-dir', str(reports_dir), '--task', 'ts']
    subprocess.run(cmd, check=True)
    assert (reports_dir / 'shap_summary.png').exists()
