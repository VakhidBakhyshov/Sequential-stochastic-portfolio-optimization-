# Companion code for "Sequential Stochastic Optimization of ETF Portfolios with Horizon-Diversified Risk Control"

This package contains the code and data behind every table and figure of the paper: the monthly
allocation policy (scenario-based CVaR allocation with adaptive parameters, partial execution and the
two-signal exposure overlay), the analysis scripts that produce the reported statistics, the datasets,
and the companion engine used for the CVaR–CDaR comparison, the known-law checks and the theorem-to-computation
bridge. Everything runs from the package root with plain Python.

## Contents

```
README.md, LICENSE, requirements.txt, paper_map.md
datasets/csv/        raw panels: NewClosePrice.csv (adjusted daily closes), OpenPrice.csv, Volume.csv (share
                     volumes) for about 3,100 US-listed ETFs, 2016-01-04 to 2025-12-31; Meta.csv (fund
                     metadata: ISIN, name, provider, asset class, sector, geography, expense ratio, exchange,
                     replication method); vixcls.csv (FRED VIXCLS series)
datasets/excel/      built from the raw panels by the two preparation scripts and shipped for convenience:
                     last_filtered_weights.xlsx (point-in-time eligibility, one sheet per rebalance date),
                     business_dates.xlsx (first trading day of each month), new_etf_returns.csv and
                     new_etf_ewma.csv (daily log returns and their EWMA); rebalance_universe_matrix1.csv and
                     matrix2.csv are the two alternative universes of Section 7.6
scripts/             the engine: dataloader/, models/ (scenario law), optimizers/ (CVaR linear program,
                     adaptive parameters), executions/ (partial execution), calculations/, results/,
                     backtesting/, runs/run.py (the monthly loop), configs/ (one YAML per experiment arm)
*.py at the root     analysis scripts (tables, figures, controls, robustness), see paper_map.md
build_universe.py    builds the eligibility workbook from the raw panels (the liquidity screen of Section 3.2)
prepare_data.py      builds the derived return panels from the raw closes
run_arms.py          runs the experiment arms; reproduce.py runs the analysis chain
companion/           the companion engine (CDaR allocation, multi-signal layer, nested selection, known-law and
                     theorem-bridge scripts) with the run folders the paper cites, its analysis scripts, notebooks,
                     tests and requirements; see companion/README.md
reproduce_companion.py   runs the companion analysis chain (Sections 7.9-7.12, Appendices E, F, I)
extensions/          calibration/ (the multi-objective search of Section 7.12: search code, the 60 trial
                     configurations, trial table, Pareto set, importances, selected configurations and the Optuna
                     study database), universe_comparison/
                     (filtration comparison of Table L.1 and Figures L.3-L.4), synthetic_engine/ (the main policy
                     on synthetic markets with a known law; not used in the paper)
reference/           the tables (CSV/MD) and figures (PNG) of the main engine as they appear in the paper, and the
                     known-law diagnostics of Appendices E, F and I
results/             created by the runs of the main engine
```

## Requirements

Python 3.11 or later. Install the dependencies with

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   or   source .venv/bin/activate
pip install -r requirements.txt
```

The reported results of the main engine were produced with Python 3.14, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1
(HiGHS solves the linear programs), matplotlib 3.11.1 and openpyxl 3.1.5, and reproduce to 1e-9 with numpy 2.4.6,
pandas 3.0.3 and scipy 1.17.1. The companion engine has its own `companion/requirements.txt` (adds statsmodels
and plotly); its run folders were produced with that environment.

## The monthly loop

`scripts/runs/run.py` implements one rebalance per month. The allocator solves the CVaR-constrained program
for the target composition, measuring turnover against its own previous target (`turnover_reference: target`,
the default; `executed` anchors it on the held book instead, which is the mechanism experiment of
Section 7.7). Execution then moves the held book a fraction of the way toward the target. Transaction costs
are 10 basis points per unit of wealth bought or sold (`c_bps: 0.001`), charged on the trades actually made
from the held book, and the return of every held position is booked, including funds that have left the
eligible set. Each run writes `pnl.csv` (balance, return, cost per month), `weights.xlsx` (target and executed
weights per rebalance), `forecast_risk.csv` (the internal risk state, exposure fraction, target and executed
turnover, holdings) and `dynamic_parameter_history.csv`.

The exposure overlay is applied by the analysis scripts on top of the run: `overlay_arms.overlay` returns the
net monthly return of the held book at exposure k, the cash slice at the risk-free rate, less the cost of the
exposure move at the same 10 basis points per unit of wealth moved. Every overlay arm, benchmark and control
uses this function, so the exposure channel and the composition channel are costed at the same rate.

## Reproducing the paper

All commands are run from the package root.

1. Data. The files in `datasets/excel/` are shipped, and each is rebuilt exactly from the raw panels by
   `python build_universe.py` (the eligibility workbook, about half a minute) and `python prepare_data.py`
   (`new_etf_returns.csv`, `new_etf_ewma.csv`, `business_dates.xlsx`, a few minutes). Running both is
   optional; the rebuilt files are identical to the shipped ones.

2. Experiment arms. `python run_arms.py` runs the ten configurations of the paper in sequence
   (`main_dyn_strong` is the policy of the paper; the others are the static, generator, universe, state and
   execution variants of Sections 7.5–7.7). One arm takes about three minutes; each writes
   `results/<name>/`. `python run_arms.py --grid` adds the scenario-count, history-rule and seed grid of the
   trial ledger (about one hour). A single arm: `python run_arms.py main_dyn_strong`.

3. Analysis. `python reproduce.py` runs the analysis scripts in dependency order (overlays, verification
   and bootstraps, benchmarks, tail and daily-path checks, controls, robustness tables, figures) and writes
   the tables and figures next to the scripts; console output goes to `logs/`. `universe_compare.py` takes
   about 20 minutes; skip it with `--skip universe_compare`. Individual scripts can be run directly, for
   example `python verify_all.py` (Table 5) or `python overlay_arms.py main_dyn_strong`.

4. Companion engine (Sections 7.9-7.12, Appendices E, F, I). `python reproduce_companion.py` runs, on the
   shipped run folders, the signal research of the multi-signal arm (Figures 9-12), the matched CVaR-CDaR
   comparison (Table 7), the companion figures (Figures 6-8, 14, L.6-L.8), the calibration tables and figures
   from the trial table (Section 7.12, Figures 17-19) and the extension comparison (Figure 13).
   `python reproduce_companion.py --run-arms` first reruns the four companion arms (the two CDaR arms take
   about 2.5 hours each; all four run in parallel). The known-law scripts are listed in `companion/README.md`.
   The calibration search itself (60 nested walk-forward configurations, about 7.5 hours in one process) is run
   separately with `python -m scripts.optimization.optuna_calibration --config scripts/configs/optuna_calibration.yaml`
   from the `companion` folder (needs `optuna`); its outputs are shipped under `extensions/calibration/results`.

Every script sets its own random seed (42, re-seeded at every rebalance date) so that repeated runs are
identical on the same library versions.

## Notes on the data

Closes are adjusted for distributions and splits; delisted funds remain in the panel. The liquidity
screen (`scripts/dataloader/liquidity_screen.py`) computes each fund's daily traded value as the mid price
times share volume, the 36-month rolling share of aggregate traded value, and marks a fund eligible when
its lagged share exceeds 0.5 x 10^-4 and it has traded in each of the preceding 36 months. The experiments
read the resulting workbook. The two additional rules of the eligibility condition
(an observed close on the last trading day before the rebalance, at least 60 months of observed closes
with funds present since the panel start exempt) are applied by `scripts/runs/run.py` from the
configuration keys `require_recent_price_days` and `require_history_days`.

## Licence and citation

MIT licence (see LICENSE). Please cite the paper when using this code:

Sequential Stochastic Optimization of ETF Portfolios with
Horizon-Diversified Risk Control. [Journal, volume, pages].
