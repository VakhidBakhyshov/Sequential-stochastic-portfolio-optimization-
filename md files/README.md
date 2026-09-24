# Companion code for "Sequential Stochastic Optimization of ETF Portfolios with Horizon-Diversified Risk Control"

This package contains the code and data behind every table and figure of the paper: the monthly
allocation policy (scenario-based CVaR allocation with adaptive parameters, partial execution and the
two-signal exposure overlay), the analysis scripts that produce the reported statistics, the datasets,
and the companion engine used for Appendices A and B and for the known-law identification of
Sections 3.9–3.10 and Appendix C. Everything runs from the package root with plain Python.

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
                     matrix2.csv are the two alternative universes of Section 7.9
scripts/             the engine: dataloader/, models/ (scenario law), optimizers/ (CVaR linear program,
                     adaptive parameters), executions/ (partial execution), calculations/, results/,
                     backtesting/, runs/run.py (the monthly loop), configs/ (one YAML per experiment arm)
*.py at the root     analysis scripts (tables, figures, controls, robustness), see paper_map.md
build_universe.py    builds the eligibility workbook from the raw panels (the liquidity screen of Section 2.2)
prepare_data.py      builds the derived return panels from the raw closes
run_arms.py          runs the experiment arms; reproduce.py runs the analysis chain
companion/           the colleague's engine (multi-signal system of Appendix A, drawdown-constrained
                     allocation of Appendix B, known-law and theorem-bridge scripts) with the result
                     folders the paper cites and its own requirements.txt
extensions/          material not used in the paper: synthetic_engine/ (the production policy run on
                     synthetic markets with a known law), universe_comparison/ and calibration/ (from the
                     companion package; the calibration code is shipped as is and is not run)
reference/           the tables (CSV/MD) and figures (PNG) as they appear in the paper, and the known-law
                     diagnostics of Sections 3.9–3.10 and Appendix B
results/             created by the runs
```

## Requirements

Python 3.11 or later. Install the dependencies with

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   or   source .venv/bin/activate
pip install -r requirements.txt
```

The reported results were produced with Python 3.14, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1 (HiGHS solves
the linear programs), matplotlib 3.11.1 and openpyxl 3.1.5. The companion engine has its own
`companion/requirements.txt`.

## Reproducing the paper

All commands are run from the package root.

1. Data. The files in `datasets/excel/` are shipped, and each is rebuilt exactly from the raw panels by
   `python build_universe.py` (the eligibility workbook, about half a minute) and `python prepare_data.py`
   (`new_etf_returns.csv`, `new_etf_ewma.csv`, `business_dates.xlsx`, a few minutes). Running both is
   optional; the rebuilt files are identical to the shipped ones.

2. Experiment arms. `python run_arms.py` runs the ten configurations of the paper in sequence
   (`main_dyn_strong` is the policy of the paper; the others are the static, generator, universe, state and
   execution variants of Sections 7.5 and 7.9). One arm takes about three minutes; each writes
   `results/<name>/` with `pnl.csv`, `weights.xlsx`, `forecast_risk.csv` and `dynamic_parameter_history.csv`.
   `python run_arms.py --grid` adds the scenario-count, history-rule and seed grid of Table 12 (about
   one hour). A single arm: `python run_arms.py main_dyn_strong`.

3. Analysis. `python reproduce.py` runs the analysis scripts in dependency order (overlays, verification
   and bootstraps, benchmarks, controls, robustness tables, figures) and writes the tables and figures
   next to the scripts; console output goes to `logs/`. `universe_compare.py` takes about 25 minutes;
   skip it with `--skip universe_compare`. Individual scripts can be run directly, for example
   `python verify_all.py` (Tables 7 and 8) or `python overlay_arms.py main_dyn_strong`.

4. Companion engine (Appendices A and B, Sections 3.9–3.10, Appendix C). From `companion/`:
   `python -m scripts.runs.sequential_diffusion_analysis --output results/sequential_diffusion_analysis`
   and `python -m scripts.runs.analytical_benchmark --output results/analytical_benchmark --execution-eta 0.5`
   reproduce Tables 2 and 3; `python -m scripts.runs.cdar_research_validation --output-dir results/cdar_validation`
   reproduces Table 19; `python -m scripts.runs.theorem_empirical_bridge` reproduces Table 20;
   `python -m scripts.runs.compare_cvar_cdar` reproduces the comparison of Appendix B from the shipped
   walk-forward outputs. The walk-forward arms of the companion engine take several hours each; their
   outputs are shipped under `companion/results/`. To re-run them, copy `datasets/` into
   `companion/datasets/` (the engine resolves data relative to its working directory).

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

Vukovic, D., Bakhishov, V., Zinovyev, V., Dalal, A. (2026). Sequential Stochastic Optimization of ETF Portfolios with
Horizon-Diversified Risk Control. [Journal, volume, pages].
