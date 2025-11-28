# Explainable Turnover Prediction

Projeto: "Explainable prediction of turnover in a company" — preditor de turnover explicável para análise e interpretação empresarial.

## Estrutura do projeto

- `data/` — conjuntos de dados brutos e processados.
- `src/` — scripts e módulos principais: geração de dados, preprocessamento, treinamento, avaliação e explicabilidade.
- `models/` — modelos treinados (.joblib, .pkl).
- `notebooks/` — notebooks de exploração e demonstração.
- `reports/` — relatórios e figuras.
- `tests/` — testes automatizados (Pytest).

## Objetivo
Criar um pipeline simples que processa um dataset de turnover, treina um modelo de previsão (XGBoost para TS), avalia desempenho e fornece explicações com SHAP para a tomada de decisão.

## Como usar
1. Instalar dependências:

```powershell
python -m pip install -r requirements.txt
```
Por favor, assegure-se de que seu ambiente Python utiliza versões de pacotes compatíveis. Recomendamos o uso de um ambiente virtual fresco com venv. O arquivo `requirements.txt` fixa versões conhecidas como compatíveis.

Para criar e usar um `venv` (Windows PowerShell):
```powershell
# criar
python -m venv .venv
# ativar
. .\.venv\Scripts\Activate
# atualizar pip e instalar dependências
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ou execute o script de conveniência (embelezado):
```powershell
.\scripts\setup_venv.ps1
```

Se preferir usar `conda`, siga os passos padrão do conda para criar e ativar um ambiente.

2. Providenciar o dataset (CSV):

Coloque o seu CSV de séries temporais em `data/time_series_turnover.csv` (por exemplo, com colunas mensais agregadas contendo: `date`, `turnover_rate`, e colunas exógenas opcionais como `avg_salary`, `economy`, `num_employees`). Se o ficheiro tiver outro nome, renomeie ou coloque a cópia em `data/time_series_turnover.csv`.

3. Treinar e avaliar o modelo (supondo que já preprocesseu os dados):

```powershell
python src/preprocessing.py --input data/time_series_turnover.csv --out-dir data
python src/train.py --prepared data/prepared.joblib --model-output models/rf_turnover.joblib
```

Time series usage (assumes you placed `data/time_series_turnover.csv`):
1. Preprocess the time-series (lag features, scaling):
```powershell
python src/preprocessing.py --input data/time_series_turnover.csv --mode ts --n-lags 12 --out-dir data
```
3. Train the time-series forecast model:
```powershell
python src/train.py --prepared data/prepared.joblib --model-output models/xgb_turnover.joblib --task ts --n-estimators 200
```
4. Evaluate & explain (regression/forecast):
```powershell
python src/evaluate.py --prepared data/prepared.joblib --model models/xgb_turnover.joblib --report-dir reports --task ts
python src/explain.py --model models/xgb_turnover.joblib --prepared data/prepared.joblib --out-dir reports --task ts
```
```

4. Gerar explicações SHAP:

```powershell
python src/explain.py --model models/rf_turnover.joblib --data data/time_series_turnover.csv --out reports/shap_summary.png
```

### Executar o pipeline completo (PowerShell)
Se quiser executar o pipeline completo com venv (criação do ambiente + instalação de dependências + todos os passos), use o `run.ps1`:
```powershell
.\run.ps1
```

## Notebooks
- `notebooks/analysis.ipynb` — exploração, treinamento interativo e visualização SHAP.

## Benchmarking (time series)
You can run a benchmark that compares naive/moving average, SARIMAX, RandomForest and XGBoost on your time series dataset:

```powershell
python src/benchmark.py --input data/time_series_turnover.csv --out-dir reports/benchmark --n-lags 12
```

Outputs are saved in `reports/benchmark` and include `predictions.csv`, `metrics.json` and plots of predictions and metrics.
If you have synthetic files in `data/` you want to remove, run the cleanup script:
```powershell
.\scripts\cleanup_synthetic.ps1
```

What the benchmark compares:
- naive_last: persistence (last value)
- moving_avg: short moving average of last k observations
- seasonal_naive: last year same month value (season period 12)
- SARIMAX: statsmodels seasonal ARIMA with exogenous signals if available (optional)
- RandomForestRegressor: tree-based regressor using lag features and exogenous variables
- XGBoost: gradient boosting regressor using lag features and exogenous variables

The report contains per-method metrics (MAE, RMSE, MAPE) and plots comparing actual vs predicted on the rolling-origin test set. Use these metrics to decide which model family to focus on for production or further tuning.

Notes and next steps:
- For multi-step forecasts, adapt `rolling_origin_evaluation` horizon > 1 and aggregate metrics appropriately.
- Hyperparameter tuning: wrap XGB/RF in `GridSearchCV` or `RandomizedSearchCV` adapted to time-series CV.
- For widely deployed models, consider using `pmdarima` / `sktime` / `prophet` / deep learning models (LSTM/TFT) and include them as candidate methods.

## Testes
Rodar os testes com `pytest` requer que você ative o venv criado e tenha instalado as dependências.
No PowerShell:
```powershell
# ativar venv
. .\.venv\Scripts\Activate
# executar testes
pytest
```

## Troubleshooting: setuptools.build_meta import error
If you see an error like:

```
BackendUnavailable: Cannot import 'setuptools.build_meta'
```

It means the venv is missing `setuptools`/`wheel` required by a package's build backend. Fix it by activating the `venv` and installing or upgrading `setuptools` & `wheel` and then reinstalling requirements:

```powershell
. .\.venv\Scripts\Activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If that still doesn't work, try installing `setuptools_scm` or `build`:

```powershell
python -m pip install setuptools_scm build
```

If you installed packages globally before venv, it's best to remove them from global paths to avoid conflicts and create a fresh venv instead.

If the issue persists during `numpy` or similar (many of these are compiled C extensions), consider these options:

- Prefer pre-built wheels (we try `--prefer-binary` in the setup script), but if pip still attempts to compile from source you can either:
	- Install the Microsoft Visual C++ Build Tools (msvc) for Windows. Follow: https://visualstudio.microsoft.com/visual-cpp-build-tools/ (choose "C++ build tools"). After installing, reopen terminal and retry `pip install`.
	- Alternatively, install precompiled wheels directly (some users use unofficial wheels from Christoph Gohlke's site — only if you understand the trust and license implications): https://www.lfd.uci.edu/~gohlke/pythonlibs/
	- Or fall back to `conda` for installing heavy compiled packages (e.g., `conda install numpy scikit-learn`), then return to venv for the rest of the pure-Python packages.

Example: force wheel preference for a specific package:
```powershell
. .\.venv\Scripts\Activate
python -m pip install --prefer-binary numpy==1.24.3
```

## Estrutura mínima de scripts
- `src/preprocessing.py`: limpa, codifica e divide dados em treino/teste (llready handles TS mode with lags).
- `src/train.py`: treina modelo, salva modelo e métricas.

Time-series model details:
- Implemented using XGBoost regressor on feature-engineered lag inputs.
- The `preprocessing.py` script will create lag features (lag_1..lag_n) and rolling means, then scale numeric features.
- `train.py` uses `XGBRegressor` for time-series forecasting and logs MAE/RMSE/R2 metrics in `reports/eval_metrics.json`.
- `src/evaluate.py`: calcula métricas e produz relatórios.
- `src/explain.py`: aplica SHAP e gera gráficos de explicação.

## Licença
Projeto para fins académicos - cite conforme a necessidade.
