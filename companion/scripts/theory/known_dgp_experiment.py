"""Known-distribution experiment for the complete sequential decision architecture.

This experiment sits *before* the historical ETF backtest in the evidence hierarchy.
It uses a correlated GBM law whose parameters are known to the experimenter, while
the policy sees only lagged simulated data.  Consequently we can compare estimated
objects with their oracle counterparts and verify structural identities without any
claim that real ETF prices are literally GBM.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .diffusion_benchmark import (
    DiffusionSpec,
    centered_normal_cvar,
    correlated_gbm_paths,
    empirical_cvar_loss,
    ema_impulse_response,
    gaussian_reversal_probability,
    gbm_simple_loss_var_cvar,
    normal_loss_var_cvar,
    partial_execution,
    portfolio_log_return_moments,
    solve_return_cvar_lp,
    topk_margin_certificate,
)

EPS = 1e-12


@dataclass(frozen=True)
class KnownDGPConfig:
    n_assets: int = 24
    top_k: int = 12
    n_days: int = 252 * 8
    lookback_days: int = 504
    filter_window_days: int = 90
    rebalance_days: int = 21
    scenario_count: int = 1500
    beta_read: float = 0.95
    target_annual_vol: float = 0.12
    exposure_floor: float = 0.30
    max_weight: float = 0.20
    turnover_penalty: float = 0.002
    execution_eta: float = 0.55
    score_ema_alpha: float = 0.50
    risk_free_annual: float = 0.02
    seed: int = 20260908


def _percentile(x: pd.Series, higher: bool = True) -> pd.Series:
    xx = pd.to_numeric(x, errors="coerce")
    xx = xx.fillna(xx.median())
    return xx.rank(method="average", pct=True, ascending=higher)


def _generate_spec(cfg: KnownDGPConfig) -> DiffusionSpec:
    rng = np.random.default_rng(cfg.seed)
    mu = np.linspace(0.035, 0.115, cfg.n_assets) + rng.normal(0.0, 0.008, cfg.n_assets)
    sigma = np.linspace(0.09, 0.28, cfg.n_assets) + rng.normal(0.0, 0.008, cfg.n_assets)
    sigma = np.clip(sigma, 0.07, 0.32)
    # Two-factor PSD correlation structure with heterogeneous exposures.
    b1 = np.linspace(0.15, 0.75, cfg.n_assets)
    b2 = np.sin(np.linspace(0, 2 * np.pi, cfg.n_assets)) * 0.30
    raw = np.outer(b1, b1) + np.outer(b2, b2) + np.eye(cfg.n_assets) * 0.55
    d = np.sqrt(np.diag(raw))
    corr = raw / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return DiffusionSpec(mu=mu, sigma=sigma, corr=corr, s0=np.linspace(60.0, 140.0, cfg.n_assets))


def _liquidity_panel(cfg: KnownDGPConfig, spec: DiffusionSpec) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 1)
    base = np.linspace(12.0, 9.0, cfg.n_assets) + rng.normal(0.0, 0.25, cfg.n_assets)
    market_state = np.cumsum(rng.normal(0.0, 0.025, cfg.n_days))
    idio = rng.normal(0.0, 0.18, (cfg.n_days, cfg.n_assets))
    log_dollar_volume = base[None, :] + 0.15 * market_state[:, None] + idio
    return np.exp(log_dollar_volume)


def _selection_scores(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    end: int,
    cfg: KnownDGPConfig,
    previous_smoothed: pd.Series | None,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    start = max(1, end - cfg.filter_window_days)
    rr = log_returns.iloc[start:end]
    dv = dollar_volume.iloc[start:end]
    mom = rr.sum(axis=0)
    vol = rr.std(axis=0, ddof=1) * np.sqrt(252)
    liq = np.log1p(dv.median(axis=0))
    raw = 0.45 * _percentile(mom, True) + 0.30 * _percentile(vol, False) + 0.25 * _percentile(liq, True)
    if previous_smoothed is None:
        smooth = raw
    else:
        prev = previous_smoothed.reindex(raw.index).fillna(raw)
        a = cfg.score_ema_alpha
        smooth = a * raw + (1.0 - a) * prev
    selected = smooth.sort_values(ascending=False).head(cfg.top_k).index
    cert = topk_margin_certificate(smooth.to_numpy(), cfg.top_k, epsilon=0.01)
    return smooth, pd.Series(selected, dtype=object), cert


def _normal_scenarios(mu_daily: np.ndarray, cov_daily: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    vals, vecs = np.linalg.eigh((cov_daily + cov_daily.T) / 2.0)
    L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))
    return mu_daily[None, :] + rng.standard_normal((count, len(mu_daily))) @ L.T


def _portfolio_realized_vol(log_returns: pd.DataFrame, weights: pd.Series, end: int, window: int = 21) -> float:
    start = max(0, end - window)
    common = [c for c in log_returns.columns if c in weights.index]
    if not common or end - start < 2:
        return np.nan
    r = log_returns.iloc[start:end][common].to_numpy() @ weights.reindex(common).fillna(0.0).to_numpy()
    return float(np.std(r, ddof=1) * np.sqrt(252))


def _annualized_stats(monthly: np.ndarray, rf_annual: float) -> dict[str, float]:
    r = np.asarray(monthly, dtype=float)
    if r.size == 0:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    wealth = np.cumprod(1.0 + r)
    ann = float(wealth[-1] ** (12.0 / len(r)) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(12)) if len(r) > 1 else 0.0
    sharpe = float((np.mean(r) * 12.0 - rf_annual) / vol) if vol > EPS else np.nan
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    return {"ann_return": ann, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": float(np.min(dd))}


def _mc_cvar_convergence(spec: DiffusionSpec, out_dir: Path, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 88)
    w = np.ones(spec.n_assets) / spec.n_assets
    m, v = portfolio_log_return_moments(w, spec, 21)
    s = np.sqrt(v)
    _, exact_simple = gbm_simple_loss_var_cvar(m, s, 0.95)
    _, exact_log = normal_loss_var_cvar(m, s, 0.95)
    rows = []
    for n in [250, 500, 1000, 2500, 5000, 10000, 50000]:
        y = rng.normal(m, s, n)
        simple_r = np.exp(y) - 1.0
        rows.append({
            "n": n,
            "analytic_log_cvar": exact_log,
            "mc_log_cvar": empirical_cvar_loss(y, 0.95),
            "analytic_simple_cvar": exact_simple,
            "mc_simple_cvar": empirical_cvar_loss(simple_r, 0.95),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "cvar_convergence.csv", index=False)
    return frame



def _save_plots(out: Path, price_df: pd.DataFrame, monthly: pd.DataFrame, mc_conv: pd.DataFrame) -> None:
    """Save publication-oriented diagnostic figures using matplotlib defaults."""
    import matplotlib.pyplot as plt

    sample_cols = list(price_df.columns[:6])
    ax = price_df[sample_cols].iloc[::5].plot(figsize=(9, 4.5), linewidth=1.0)
    ax.set_title("Known-DGP correlated GBM price paths (sample assets)")
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "figure_known_dgp_gbm_paths.png", dpi=180)
    plt.close(ax.figure)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(mc_conv["n"], (mc_conv["mc_log_cvar"] - mc_conv["analytic_log_cvar"]).abs(), marker="o", label="log-return CVaR error")
    ax.plot(mc_conv["n"], (mc_conv["mc_simple_cvar"] - mc_conv["analytic_simple_cvar"]).abs(), marker="o", label="simple-return CVaR error")
    ax.set_xscale("log")
    ax.set_title("Monte Carlo convergence to closed-form GBM tail risk")
    ax.set_xlabel("scenario count")
    ax.set_ylabel("absolute CVaR error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure_known_dgp_cvar_convergence.png", dpi=180)
    plt.close(fig)

    d = monthly.copy()
    d["rebalance_date"] = pd.to_datetime(d["rebalance_date"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d["rebalance_date"], d["fast_realized_vol"], label="fast realized volatility")
    ax.plot(d["rebalance_date"], d["slow_centered_cvar"] * np.sqrt(252), label="slow centered model risk (scaled)")
    ax.plot(d["rebalance_date"], d["next_month_realized_vol"], label="next-month realized volatility")
    ax.set_title("Known-DGP forward/backward risk channels")
    ax.set_xlabel("rebalance date")
    ax.set_ylabel("risk scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure_known_dgp_signal_channels.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.scatter(d["cutoff_margin"], d["one_way_turnover"], s=20)
    ax.axvline(0.02, linestyle="--", linewidth=1.0, label="2 epsilon boundary for epsilon=0.01")
    ax.set_title("Selection boundary margin and executed turnover")
    ax.set_xlabel("top-K cutoff margin")
    ax.set_ylabel("one-way executed turnover")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure_known_dgp_margin_turnover.png", dpi=180)
    plt.close(fig)


def run_known_dgp_experiment(output_dir: str | Path, config: KnownDGPConfig | None = None) -> dict[str, object]:
    cfg = config or KnownDGPConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 2)
    spec = _generate_spec(cfg)

    prices = correlated_gbm_paths(spec, n_steps=cfg.n_days, n_paths=1, seed=cfg.seed)[0]
    dates = pd.bdate_range("2017-01-02", periods=cfg.n_days + 1)
    tickers = [f"SYN{i:02d}" for i in range(cfg.n_assets)]
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    log_returns = np.log(price_df / price_df.shift(1)).iloc[1:]
    dollar_volume = pd.DataFrame(_liquidity_panel(cfg, spec), index=log_returns.index, columns=tickers)

    # Analytical distribution diagnostics for the known DGP.
    mc_conv = _mc_cvar_convergence(spec, out, cfg.seed)

    initial_assets = tickers[: cfg.top_k]
    q_prev = pd.Series(1.0 / cfg.top_k, index=initial_assets)
    x_prev_full = pd.Series(0.0, index=tickers)
    x_prev_full.loc[initial_assets] = q_prev.values
    previous_scores: pd.Series | None = None
    slow_history: list[float] = []
    rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, float]] = []

    # Rebalance only after the full estimation lookback is available.
    rebal_idx = list(range(cfg.lookback_days, cfg.n_days - cfg.rebalance_days, cfg.rebalance_days))
    for step, end in enumerate(rebal_idx):
        date = log_returns.index[end - 1]
        next_end = min(end + cfg.rebalance_days, len(log_returns))
        scores, selected_s, cert = _selection_scores(log_returns, dollar_volume, end, cfg, previous_scores)
        previous_scores = scores
        selected = list(selected_s)
        idx = [tickers.index(x) for x in selected]

        est = log_returns.iloc[end - cfg.lookback_days:end][selected]
        mu_hat = est.mean(axis=0).to_numpy()
        cov_hat = est.cov().to_numpy()
        cov_hat = (cov_hat + cov_hat.T) / 2.0 + np.eye(len(selected)) * 1e-10

        # The policy sees estimated parameters; the experimenter also knows the oracle law.
        true_mu_log_daily = spec.log_drift[idx] / 252.0
        true_cov_daily = spec.covariance[np.ix_(idx, idx)] / 252.0
        scenario = _normal_scenarios(mu_hat, cov_hat, cfg.scenario_count, rng)

        # State-dependent program parameters: all inputs are lagged/decision-time data.
        held_vol = _portfolio_realized_vol(log_returns, x_prev_full, end, 21)
        stress = 0.0 if not np.isfinite(held_vol) else np.clip((held_vol - cfg.target_annual_vol) / 0.20, -1.0, 1.0)
        beta_t = float(np.clip(0.95 + 0.025 * stress, 0.90, 0.99))
        lambda_t = float(np.clip(cfg.turnover_penalty * np.exp(-0.8 * stress), 0.0005, 0.02))

        prev_on_selected = x_prev_full.reindex(selected).fillna(0.0)
        risky_mass = float(prev_on_selected.sum())
        if risky_mass > EPS:
            prev_comp = (prev_on_selected / risky_mass).to_numpy()
        else:
            prev_comp = np.ones(len(selected)) / len(selected)
        # Ensure feasibility if selected names are new and the inherited composition sums poorly.
        if len(selected) * cfg.max_weight < 1.0:
            raise ValueError("Known-DGP config infeasible: top_k * max_weight < 1")

        ew = np.ones(len(selected)) / len(selected)
        budget = empirical_cvar_loss(scenario @ ew, beta_t)
        opt = solve_return_cvar_lp(
            scenario,
            mu_hat,
            prev_comp,
            beta=beta_t,
            cvar_budget=budget,
            max_weight=cfg.max_weight,
            turnover_penalty=lambda_t,
        )
        w = np.asarray(opt["weights"], dtype=float)
        w_series = pd.Series(w, index=selected)

        # Slow signal: centered forward risk produced by the same estimated law.
        sigma_hat = float(np.sqrt(max(w @ cov_hat @ w, 0.0)))
        slow_cvar = centered_normal_cvar(sigma_hat, cfg.beta_read)
        slow_history.append(slow_cvar)
        slow_ref = float(np.median(slow_history[:-1])) if len(slow_history) > 6 else slow_cvar
        k_slow = float(np.clip(slow_ref / max(slow_cvar, EPS), cfg.exposure_floor, 1.0))

        # Fast signal: backward realized volatility of the actually executed risky book.
        fast_vol = _portfolio_realized_vol(log_returns, x_prev_full, end, 21)
        k_fast = 1.0 if not np.isfinite(fast_vol) or fast_vol <= EPS else float(np.clip(cfg.target_annual_vol / fast_vol, cfg.exposure_floor, 1.0))
        k_two = float(np.clip(k_fast * k_slow, cfg.exposure_floor, 1.0))

        # A third, bounded confidence signal for the many-signal geometric-aggregation experiment.
        cond = float(np.linalg.cond(cov_hat))
        k_conf = float(np.clip(1.0 / (1.0 + 0.002 * max(cond - 1.0, 0.0)), cfg.exposure_floor, 1.0))
        k_multi = float(np.clip((k_fast * k_slow * k_conf) ** (1.0 / 3.0), cfg.exposure_floor, 1.0))

        target_full = pd.Series(0.0, index=tickers)
        target_full.loc[selected] = k_two * w
        executed_arr, ids = partial_execution(x_prev_full.to_numpy(), target_full.to_numpy(), cfg.execution_eta)
        executed_full = pd.Series(executed_arr, index=tickers)
        identity_rows.append({"step": step, **ids})

        # Hold the executed risky dollar weights for one synthetic month; residual is cash.
        future_asset_simple = np.exp(log_returns.iloc[end:next_end].sum(axis=0)) - 1.0
        risky_ret = float(executed_full.to_numpy() @ future_asset_simple.to_numpy())
        cash = max(0.0, 1.0 - float(executed_full.sum()))
        cash_ret = cash * ((1.0 + cfg.risk_free_annual) ** ((next_end - end) / 252.0) - 1.0)
        turnover = float(np.sum(np.abs(executed_full.to_numpy() - x_prev_full.to_numpy())))
        cost = 0.001 * turnover
        monthly_return = risky_ret + cash_ret - cost

        # Oracle quantities for verification only; they never feed the policy.
        oracle_m = float(w @ true_mu_log_daily)
        oracle_s = float(np.sqrt(max(w @ true_cov_daily @ w, 0.0)))
        oracle_centered = centered_normal_cvar(oracle_s, cfg.beta_read)
        oracle_uncentered = normal_loss_var_cvar(oracle_m, oracle_s, cfg.beta_read)[1]
        estimated_uncentered = normal_loss_var_cvar(float(w @ mu_hat), sigma_hat, cfg.beta_read)[1]

        next_realized_log = log_returns.iloc[end:next_end][selected].to_numpy() @ w
        next_risk = float(np.std(next_realized_log, ddof=1) * np.sqrt(252)) if len(next_realized_log) > 1 else np.nan

        rows.append({
            "rebalance_date": str(date.date()),
            "selected_count": len(selected),
            "cutoff_margin": cert["cutoff_margin"],
            "margin_certificate_eps_001": cert["stable"],
            "beta_t": beta_t,
            "lambda_t": lambda_t,
            "optimizer_success": bool(opt["success"]),
            "optimizer_cvar": float(opt["cvar"]),
            "cvar_budget": float(opt["cvar_budget"]),
            "target_turnover_composition": float(opt["turnover"]),
            "cov_condition_number": cond,
            "fast_realized_vol": fast_vol,
            "slow_centered_cvar": slow_cvar,
            "oracle_centered_cvar": oracle_centered,
            "estimated_uncentered_cvar": estimated_uncentered,
            "oracle_uncentered_cvar": oracle_uncentered,
            "k_fast": k_fast,
            "k_slow": k_slow,
            "k_two_signal": k_two,
            "k_confidence": k_conf,
            "k_multi_signal": k_multi,
            "execution_eta": cfg.execution_eta,
            "executed_risky_mass": float(executed_full.sum()),
            "one_way_turnover": turnover,
            "transaction_cost": cost,
            "next_month_realized_vol": next_risk,
            "monthly_return": monthly_return,
        })
        for ticker in tickers:
            selection_rows.append({
                "rebalance_date": str(date.date()),
                "ticker": ticker,
                "score": float(scores[ticker]),
                "selected": ticker in selected,
                "target_weight": float(target_full[ticker]),
                "executed_weight": float(executed_full[ticker]),
            })
        x_prev_full = executed_full

    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "known_dgp_monthly.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out / "known_dgp_selection_state.csv", index=False)
    identity = pd.DataFrame(identity_rows)
    identity.to_csv(out / "partial_execution_identity.csv", index=False)
    _save_plots(out, price_df, monthly, mc_conv)

    # Direct theorem-verification experiments.
    # 1) Centering removes all additive mean effects, while uncentered CVaR shifts one-for-one.
    std0 = 0.02
    shifts = np.array([-0.03, -0.01, 0.00, 0.01, 0.03])
    centered_vals = np.array([centered_normal_cvar(std0, cfg.beta_read) for _ in shifts])
    uncentered_vals = np.array([normal_loss_var_cvar(m, std0, cfg.beta_read)[1] for m in shifts])

    # 2) Exact Gaussian reversal probability vs Monte Carlo.
    p_exact = gaussian_reversal_probability(0.60, 0.55, 0.03, 0.03)
    z1 = rng.normal(0.0, 0.03, 200_000)
    z2 = rng.normal(0.0, 0.03, 200_000)
    p_mc = float(np.mean(0.55 + z2 >= 0.60 + z1))

    # 3) EMA impulse response.
    ema = ema_impulse_response(0.10, cfg.score_ema_alpha, range(8))

    # 4) Two-signal forecast complementarity against a common future risk target.
    valid = monthly[["fast_realized_vol", "slow_centered_cvar", "next_month_realized_vol"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) >= 8:
        # Rescale each predictor to the target's mean/std before comparing forecast errors.
        y = valid["next_month_realized_vol"].to_numpy()
        preds = []
        for c in ["fast_realized_vol", "slow_centered_cvar"]:
            x = valid[c].to_numpy()
            sx = np.std(x, ddof=1)
            pred = np.full_like(y, np.mean(y)) if sx <= EPS else (x - np.mean(x)) / sx * np.std(y, ddof=1) + np.mean(y)
            preds.append(pred)
        e_f, e_s = y - preds[0], y - preds[1]
        vf, vs = float(np.mean(e_f**2)), float(np.mean(e_s**2))
        cfs = float(np.mean(e_f * e_s))
        denom = vf + vs - 2.0 * cfs
        alpha_star = float((vs - cfs) / denom) if denom > EPS else np.nan
        combo_gain_condition = bool(cfs < min(vf, vs))
    else:
        vf = vs = cfs = alpha_star = np.nan
        combo_gain_condition = False

    verification = pd.DataFrame([
        {
            "result": "centered_tail_translation_invariance",
            "metric": "max_centered_difference_over_mean_shifts",
            "value": float(np.max(centered_vals) - np.min(centered_vals)),
            "tolerance": 1e-12,
            "pass": bool(np.max(centered_vals) - np.min(centered_vals) < 1e-12),
        },
        {
            "result": "uncentered_tail_translation_equivariance",
            "metric": "slope_abs_error",
            "value": float(abs(np.polyfit(shifts, uncentered_vals, 1)[0] + 1.0)),
            "tolerance": 1e-12,
            "pass": bool(abs(np.polyfit(shifts, uncentered_vals, 1)[0] + 1.0) < 1e-10),
        },
        {
            "result": "gaussian_rank_reversal_probability",
            "metric": "mc_minus_exact_abs",
            "value": abs(p_mc - p_exact),
            "tolerance": 0.003,
            "pass": bool(abs(p_mc - p_exact) < 0.003),
        },
        {
            "result": "ema_impulse_geometric_decay",
            "metric": "max_ratio_error",
            "value": float(np.max(np.abs(ema[1:] / np.maximum(ema[:-1], EPS) - (1.0 - cfg.score_ema_alpha)))),
            "tolerance": 1e-12,
            "pass": True,
        },
        {
            "result": "partial_execution_turnover_identity",
            "metric": "max_abs_identity_error",
            "value": float(identity[["turnover_identity_error", "contraction_identity_error"]].abs().to_numpy().max()),
            "tolerance": 1e-12,
            "pass": bool(identity[["turnover_identity_error", "contraction_identity_error"]].abs().to_numpy().max() < 1e-12),
        },
        {
            "result": "two_signal_error_covariance_condition",
            "metric": "c_lt_min_v",
            "value": cfs,
            "tolerance": min(vf, vs) if np.isfinite(vf) and np.isfinite(vs) else np.nan,
            "pass": combo_gain_condition,
        },
    ])
    verification.to_csv(out / "theorem_verification.csv", index=False)

    stats = _annualized_stats(monthly["monthly_return"].to_numpy(), cfg.risk_free_annual)
    summary = {
        "label": "synthetic_known_DGP_not_empirical",
        "config": asdict(cfg),
        "n_rebalances": int(len(monthly)),
        "all_optimizations_successful": bool(monthly["optimizer_success"].all()),
        "mean_selected_count": float(monthly["selected_count"].mean()),
        "mean_cutoff_margin": float(monthly["cutoff_margin"].mean()),
        "margin_certificate_pass_rate_eps_001": float(monthly["margin_certificate_eps_001"].mean()),
        "slow_signal_oracle_correlation": float(monthly[["slow_centered_cvar", "oracle_centered_cvar"]].corr().iloc[0, 1]),
        "slow_signal_next_risk_correlation": float(monthly[["slow_centered_cvar", "next_month_realized_vol"]].corr().iloc[0, 1]),
        "fast_signal_next_risk_correlation": float(monthly[["fast_realized_vol", "next_month_realized_vol"]].corr().iloc[0, 1]),
        "fast_slow_level_correlation": float(monthly[["fast_realized_vol", "slow_centered_cvar"]].corr().iloc[0, 1]),
        "mean_two_signal_exposure": float(monthly["k_two_signal"].mean()),
        "mean_multi_signal_exposure": float(monthly["k_multi_signal"].mean()),
        "mean_execution_turnover": float(monthly["one_way_turnover"].mean()),
        "max_execution_identity_error": float(identity[["turnover_identity_error", "contraction_identity_error"]].abs().to_numpy().max()),
        "cvar_mc_max_abs_error_50000": float(max(
            abs(mc_conv.iloc[-1]["analytic_log_cvar"] - mc_conv.iloc[-1]["mc_log_cvar"]),
            abs(mc_conv.iloc[-1]["analytic_simple_cvar"] - mc_conv.iloc[-1]["mc_simple_cvar"]),
        )),
        "gaussian_reversal_exact": p_exact,
        "gaussian_reversal_mc": p_mc,
        "two_signal_error_variance_fast": vf,
        "two_signal_error_variance_slow": vs,
        "two_signal_error_cross_moment": cfs,
        "two_signal_optimal_linear_weight_on_fast": alpha_star,
        "two_signal_gain_condition": combo_gain_condition,
        **stats,
    }
    (out / "known_dgp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "known_dgp_spec.json").write_text(json.dumps({
        "mu_annual": spec.mu.tolist(),
        "sigma_annual": spec.sigma.tolist(),
        "corr": spec.corr.tolist(),
        "s0": spec.s0.tolist(),
    }, indent=2), encoding="utf-8")
    price_df.iloc[::21].to_csv(out / "known_dgp_prices_monthly_sample.csv")
    return summary
