"""Analytical and known-DGP validation tools for the sequential portfolio policy.

The modules in this package are deliberately separated from the empirical backtest.
They provide closed-form BM/GBM benchmarks and deterministic Monte Carlo theorem
verification.  Outputs from this package must be labelled synthetic/analytical and
must not be mixed with the historical ETF evidence.
"""

from .diffusion_benchmark import (
    DiffusionSpec,
    centered_normal_cvar,
    correlated_gbm_paths,
    ema_impulse_response,
    gaussian_reversal_probability,
    gbm_log_return_moments,
    gbm_simple_loss_var_cvar,
    normal_loss_var_cvar,
    partial_execution,
    portfolio_log_return_moments,
    solve_return_cvar_lp,
    topk_margin_certificate,
)

__all__ = [
    "DiffusionSpec",
    "centered_normal_cvar",
    "correlated_gbm_paths",
    "ema_impulse_response",
    "gaussian_reversal_probability",
    "gbm_log_return_moments",
    "gbm_simple_loss_var_cvar",
    "normal_loss_var_cvar",
    "partial_execution",
    "portfolio_log_return_moments",
    "solve_return_cvar_lp",
    "topk_margin_certificate",
]
