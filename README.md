# Factor Model

This repository contains the final factor model notebook, model code, and input data needed to run the portfolio backtest.

## Project Layout

- `notebooks/Factor_Model.ipynb` - main notebook to run the model and view results.
- `bayesian_factor_inputs2.py` - builds monthly factor inputs and ETF excess returns.
- `hierarchical_bayesian_etf_model.py` - hierarchical Bayesian factor model.
- `hierarchical_portfolio_backtest.py` - portfolio construction, backtest, diagnostics, and plots.
- `Data/` - macro workbook and ETF metadata.
- `Ready_data/` - prepared model support files, including rebalance eligibility and cached Fama-French factors.

## Setup

Create and activate a virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name factor-model --display-name "Python (factor-model)"
```

Then open `notebooks/Factor_Model.ipynb` in VS Code and select the `Python (factor-model)` kernel.

## Run

Run the notebook from top to bottom. The notebook discovers the project root automatically, so it should work after cloning without editing local paths.

The model builds monthly excess returns from:

- `notebooks/adjusted_close_prices.csv`
- Fama-French `RF`, downloaded online or loaded from `Ready_data/ff5_plus_momentum_monthly_2016_2025.csv`

It also uses:

- `Ready_data/rebalance_universe_matrix.csv`
- `Data/Portfolio Macro Data.xlsx`
- `Data/Meta.csv`

## GitHub Notes

Some data files are larger than 50 MB. GitHub allows files up to 100 MB, but may warn on these files. If the repository becomes slow to clone, use Git LFS for the large CSV files.

Do not commit virtual environments, Python cache files, or notebook checkpoints. They are excluded in `.gitignore`.
