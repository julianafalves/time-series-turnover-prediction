import sys
import subprocess
from pathlib import Path


def test_benchmark(tmp_path: Path):
    # Use the user-provided CSV file
    user_csv = Path('data/time_series_turnover.csv')
    fallback_csv = Path('data/Time_Series___Turnover (1).csv')
    if user_csv.exists():
        out_csv = user_csv
    else:
        out_csv = fallback_csv
    assert out_csv.exists(), 'No CSV dataset found in data/; please add data/time_series_turnover.csv'

    # Run benchmark
    out_dir = tmp_path / 'benchmark_reports'
    cmd = [sys.executable, 'src/benchmark.py', '--input', str(out_csv), '--out-dir', str(out_dir), '--n-lags', '6']
    subprocess.run(cmd, check=True)

    assert (out_dir / 'predictions.csv').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'benchmark_metrics.png').exists()
