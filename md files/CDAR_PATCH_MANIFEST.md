# CDaR Research Completion Manifest

## Scope
This patch completes and verifies the CDaR research arm of the sequential stochastic portfolio framework without manufacturing historical ETF results that cannot be regenerated from the uploaded archive. The archive does not contain the immutable point-in-time ETF price/volume vintage required for a new historical CVaR-versus-CDaR backtest.

## Main implementation corrections

1. **Outer calibration now supports CDaR as a first-class tail-risk objective.** `scripts/optimization/optuna_calibration.py` and `analysis.py` recognize `CDaR (95%)`, `cdar_95`, and `abs_cdar_95`, and accept `tail_risk_objective: cdar`.
2. **Nested calibration parameter routing is corrected.** Trial-specific optimizer values are synchronized into `optimizer_param_grid` before the nested walk-forward engine is run. Previously, fixed values in that grid could override Optuna's trial-specific optimizer parameters. This affected the scientific interpretation of optimizer calibration generally, not only CDaR.
3. **True two-signal research arms are explicit.** `include_conviction_timer: false` makes the paired exposure engine exactly backward realized-volatility + one forward risk coordinate (CVaR or CDaR), instead of implicitly retaining the conviction timer.
4. **CDaR comparison inference is dependence-aware.** `compare_cvar_cdar.py` uses paired circular block bootstrap by default, adds constraint/slack/shadow-price mechanism summaries, forward-risk correlation, and optional target/executed-weight L1 distances.
5. **The exact-law CDaR validation layer is strengthened.** For zero-drift Brownian motion, terminal drawdown has the half-normal law. The code now checks closed-form VaR/CDaR against Monte Carlo, tests a CVaR-vs-CDaR ranking reversal, and measures finite-scenario convergence.
6. **A deterministic known-DGP CDaR calibration experiment is supplied.** `cdar_synthetic_calibration.py` provides a reproducible calibration/validation workflow when the historical point-in-time data vintage is unavailable.

## Modified files

- `scripts/optimization/optuna_calibration.py`
- `scripts/optimization/analysis.py`
- `scripts/optimization/plots.py`
- `scripts/risk_controls/smart_signals.py`
- `scripts/theory/cdar_benchmark.py`
- `scripts/runs/cdar_research_validation.py`
- `scripts/runs/compare_cvar_cdar.py`
- `tests/test_optuna_calibration.py`
- `tests/test_cdar_research.py`

## New files

- `scripts/configs/optuna_calibration_cdar.yaml`
- `scripts/configs/optuna_calibration_cvar_paired.yaml`
- `scripts/configs/bayessian_cdar_two_signal.yaml`
- `scripts/configs/bayessian_cvar_two_signal.yaml`
- `scripts/runs/cdar_synthetic_calibration.py`
- `CDAR_research_analysis.ipynb`
- `results/cdar_validation/*`
- `results/cdar_synthetic_calibration/*`
- `results/cdar_delivery/*`

## Executed validation

Repository regression suite: **61 passed**.

The sandbox did not contain the declared `beartype` package and package download is blocked. To exercise the repository logic without changing the deliverable dependency graph, the tests were run with a temporary no-op `beartype` compatibility shim outside the repository. That shim is **not** included in this patch.

### Exact-law / controlled CDaR diagnostics

- Zero-drift Brownian terminal CDaR, alpha=0.95: Monte Carlo `0.4636266`; exact half-normal value `0.4675606`; absolute error `0.0039340`.
- Zero-drift Brownian terminal VaR, alpha=0.95: Monte Carlo `0.3898301`; exact `0.3919928`; absolute error `0.0021627`.
- Same-terminal-return path example: CDaR difference `0.18`.
- Explicit risk-geometry ranking reversal: terminal-loss criterion prefers path A while CDaR prefers path B.
- Controlled CDaR LP: budget `0.0563400453`; optimized CDaR `0.0563400453`; zero numerical budget violation; fully invested; economic CDaR shadow price `0.304266`.

### Controlled sequential calibration

Risk-control selection:
- alpha `0.95`
- CDaR budget multiplier `1.00`
- turnover penalty `0.006`
- 18/18 sequential decisions solved
- annualized test return `0.4227%`
- annualized test volatility `1.2897%`
- zero-risk-free Sharpe `0.3278`
- mean out-of-sample CDaR95 `4.7061%`
- max out-of-sample CDaR95 `8.7533%`
- mean executed L1 turnover `0.1041`

Max-return selection:
- alpha `0.90`
- CDaR budget multiplier `1.15`
- turnover penalty `0.003`
- annualized test return `0.5089%`
- annualized test volatility `1.4165%`
- zero-risk-free Sharpe `0.3593`
- mean out-of-sample CDaR95 `5.2046%`
- max out-of-sample CDaR95 `9.2451%`
- mean executed L1 turnover `0.3159`

These calibration figures are controlled known-DGP evidence, **not historical ETF performance**.

## Historical paired rerun commands

Once the immutable ETF vintage is available in the expected data paths:

```bash
python3 -m scripts.runs.run --config-path bayessian_cvar_paired.yaml
python3 -m scripts.runs.run --config-path bayessian_cdar.yaml
python3 -m scripts.runs.compare_cvar_cdar \
  --cvar-run <CVaR_RUN_DIR> \
  --cdar-run <CDaR_RUN_DIR> \
  --output-dir results/cvar_cdar_comparison \
  --block-length 6
python3 -m scripts.runs.cdar_research_validation
python3 -m scripts.runs.cdar_synthetic_calibration
python3 -m tests.test_cdar_research

```

For matched true two-signal arms use `bayessian_cvar_two_signal.yaml` and `bayessian_cdar_two_signal.yaml`.

For outer calibration use `optuna_calibration_cvar_paired.yaml` and `optuna_calibration_cdar.yaml` with the project's existing calibration entry point.

## Claim boundary

The patch supports the mathematical formulation, sparse LP, exact-law validation, scenario convergence, calibrated known-DGP experiment, paired configuration/control design, nested-path causality, two-signal and multi-signal wiring, reporting, and statistical comparison infrastructure for CDaR. It does **not** claim a historical CDaR Sharpe ratio, maximum drawdown, selected ETF parameter vector, or superiority over CVaR until the frozen point-in-time ETF data are rerun.
