# MMLR / WMMLR From Scratch

Implementation of three regression approaches in pure Python/Numpy:
- Single Linear Regression (Ridge)
- MMLR (Multi-Model Linear Regression)
- WMMLR (Weighted Multi-Model Linear Regression)

The project trains and compares these models on three real datasets:
- Energy Efficiency (`data/ENB2012_data.csv`)
- Bike Sharing (`data/hour.csv`)
- Air Quality (`data/AirQualityUCI.csv`)

## Reference Paper

This project is based on the following paper:

- **Title:** An Efficient Data Analysis Method for Big Data Using Multiple-Model Linear Regression (MMLR)
- **Authors:** Bohan Lyu, Jianzhong Li
- **Year:** 2023
- **Source:** arXiv preprint, arXiv:2308.12691
- **Link:** https://arxiv.org/abs/2308.12691

## Observed behavior from the latest run:
- MMLR consistently outperformed Single LR.
- WMMLR underperformed MMLR across all three datasets in this configuration.
- WMMLR beat Single LR only on Bike at some `k` values.

Best RMSE snapshot from that run:

| Dataset | Single LR | Best MMLR | Best WMMLR |
|---|---:|---:|---:|
| Energy | 2.7894 | 1.6475 (k=5) | 4.8584 (k=7) |
| Bike | 140.6294 | 132.2745 (k=7) | 136.3697 (k=7) |
| AirQuality | 0.5711 | 0.5540 (k=7) | 0.6939 (k=7) |

## What This Project Does

- Loads and cleans each dataset
- Standardizes features
- Splits data into train/validation/test
- Trains:
  - one global ridge regressor
  - clustered local regressors (MMLR)
  - weighted soft-expert ensemble (WMMLR)
- Evaluates with RMSE, MAE, R2, and additional metrics
- Saves plots and CSV summaries in timestamped results folders

## Project Structure

- `main.py`
- `models/`
  - `linear_regression.py`
  - `kmeans.py`
  - `mmlr_model.py`
  - `wmmlr_model.py`
- `utils/`
  - `data_loader.py`
  - `preprocessing.py`
  - `metrics.py`
  - `visualizations.py`
- `tests/`
  - `test1_data_loading.py`
  - `test2_split.py`
  - `test3_linear_regression.py`
  - `test4_kmeans.py`
  - `test5_mmlr.py`
  - `test6_wmmlr.py`
- `data/`
  - `ENB2012_data.csv`
  - `hour.csv`
  - `AirQualityUCI.csv`

## Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

On Windows, use the Python.org installer (or `py` launcher) if your default `python`
command points to MSYS2 and causes pip/SSL issues.

## Getting Started

Clone the repository and run the full experiment pipeline.

```powershell
git clone https://github.com/asadabdullahaa1/multi-model-regression-from-scratch.git
cd multi-model-regression-from-scratch
```

Create a virtual environment and install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the full experiment:

```powershell
python main.py
```

## What This Produces

Each run creates a timestamped directory:

```text
results_run_YYYY-MM-DD_HH-MM-SS/
```

Inside each dataset folder, you will find:
- `summary_across_k.csv`
- `cluster_analysis_k*.csv`
- `clusters_k*.png`
- `weights_k*.png`
- `pred_single_k*.png`
- `pred_mmlr_k*.png`
- `pred_wmmlr_k*.png`
- `rmse_compare_k*.png`
- `performance_vs_k.png`

## Running Component Checks

Use these scripts to verify each part of the project:

```powershell
python tests/test1_data_loading.py
python tests/test2_split.py
python tests/test3_linear_regression.py
python tests/test4_kmeans.py
python tests/test5_mmlr.py
python tests/test6_wmmlr.py
```

## Using the Models in Your Own Code

You can import the models directly and run them on your own arrays:

```python
from models.mmlr_model import MMLR
from models.wmmlr_model import WMMLR

# X_train, y_train, X_val, y_val, X_test should be numpy arrays
mmlr = MMLR(k=5, lambda_reg=0.01, random_state=42)
mmlr.fit(X_train, y_train)
y_pred_mmlr = mmlr.predict(X_test)

wmmlr = WMMLR(k=5, lambda_reg=0.01, random_state=42)
wmmlr.fit(X_train, y_train, X_val, y_val)
y_pred_wmmlr = wmmlr.predict(X_test)
```

## Notes

- During long runs, matplotlib may print warnings such as:
  - `FigureCanvasAgg is non-interactive`
  - `More than 20 figures have been opened`
- These warnings do not stop execution. Plots are still saved to disk.

## Reproducibility

Random seeds are fixed in key components (default `random_state=42`) for stable clustering and model behavior.

## Next Work Items

- Improve WMMLR weighting so it is competitive with MMLR across all datasets.
- Convert script-style checks in `tests/` into assertion-based unit tests.
