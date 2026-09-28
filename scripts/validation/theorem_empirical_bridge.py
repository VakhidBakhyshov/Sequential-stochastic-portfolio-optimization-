"""Theorem-to-data validation bridge for the sequential ETF portfolio framework.

The manuscript contains exact structural statements (propositions/lemmas), standard
analytical benchmark laws (BM/GBM), and empirical claims.  This module keeps those
layers separate and makes the mapping from mathematics -> code -> synthetic evidence
-> historical evidence machine-readable.

Two entry points are intentionally distinct:

``run_synthetic_theorem_audit``
    Executes deterministic known-DGP and controlled-misspecification experiments.
    It verifies algebraic identities, comparative statics, convergence signatures and
    counterexamples.  These outputs are *not* investment-performance evidence.

``run_historical_theorem_audit``
    Consumes artifacts emitted by ``scripts/runs/run.py`` and measures the real-data
    counterparts that are identifiable from a completed point-in-time ETF run.  If an
    artifact is missing, the audit writes ``not_available`` instead of inventing a
    result.

The module deliberately does not promote conjectures/open problems to theorems.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from scripts.optimizers.dynamic_params import DynamicParameterController
from scripts.risk_controls.smart_signals import apply_smart_position_sizing
from scripts.theory.diffusion_benchmark import (
    ema_impulse_response,
    empirical_cvar_loss,
    gaussian_reversal_probability,
    partial_execution,
    solve_return_cvar_lp,
)
from scripts.theory.sequential_diffusion_analysis import gaussian_loss_cvar

EPS = 1e-12


def _r2(y: Iterable[float], x: Iterable[float]) -> float:
    y = np.asarray(list(y), dtype=float)
    x = np.asarray(list(x), dtype=float)
    good = np.isfinite(y) & np.isfinite(x)
    y, x = y[good], x[good]
    if len(y) < 3:
        return float("nan")
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ b
    den = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - fit) ** 2) / den) if den > EPS else float("nan")


def _max_drawdown(r: Iterable[float]) -> float:
    r = np.asarray(list(r), dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / np.maximum(peak, EPS) - 1.0))


def _sharpe(r: Iterable[float], periods: int = 12) -> float:
    r = np.asarray(list(r), dtype=float)
    s = r.std(ddof=1)
    return float(np.sqrt(periods) * r.mean() / s) if s > EPS else float("nan")


def _equicorr(n: int, rho: float) -> np.ndarray:
    lo = -1.0 / max(n - 1, 1) + 1e-8
    if rho <= lo or rho >= 1.0:
        raise ValueError("rho outside positive-definite equicorrelation range")
    a = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(a, 1.0)
    return a


def _nearest_psd(a: np.ndarray) -> np.ndarray:
    a = (np.asarray(a, float) + np.asarray(a, float).T) / 2.0
    val, vec = np.linalg.eigh(a)
    return (vec * np.clip(val, 1e-10, None)) @ vec.T


def _effective_rank(cov: np.ndarray) -> float:
    eig = np.clip(np.linalg.eigvalsh(np.asarray(cov, float)), 0.0, None)
    s = eig.sum()
    if s <= EPS:
        return 0.0
    p = eig / s
    ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return float(np.exp(ent))


def _jaccard(a: set[int], b: set[int]) -> float:
    u = len(a | b)
    return float(len(a & b) / u) if u else 1.0


def _empirical_cvar_risk(x: np.ndarray, beta: float = 0.95) -> float:
    return empirical_cvar_loss(np.asarray(x, float), beta=beta)


def _coverage_rows() -> list[dict[str, str]]:
    """Curated manuscript result -> computational burden register.

    ``historical_requirement`` is deliberately explicit.  Exact identities need not
    be "proved by data"; a historical test is only listed when economic materiality
    or an implementation-specific consequence can be measured.
    """
    rows = [
        ("Prop1", "Internal-tail spanning under location-scale generators", "Gaussian CVaR-vs-vol spanning; mixture contrast", "R2/shape-ratio distribution from forecast_risk"),
        ("Cor1", "Generator determines zero-cost availability vs tail content", "Gaussian vs regime-mixture shape-ratio dispersion", "Generator-family comparison from historical/saved scenario runs"),
        ("Prop2", "Budget degeneracy when reading a binding constraint at same level", "Binding/slack experiment plus signal-budget gap", "LP binding flag, slack, beta* and beta from run artifacts"),
        ("Prop3", "Mean domination grows with horizon", "H-grid exact Gaussian decomposition", "One-day vs holding-horizon centered/raw signal comparison"),
        ("Lemma1", "Translation invariance of centered tail operator", "Mean-shift identity to machine precision", "Centered = raw + scenario mean residual"),
        ("Prop4", "Forecast-error combination condition", "Error-correlation/variance-ratio surface", "Fast/slow forecast errors against realized future-risk target"),
        ("Prop5", "Sharpe scale invariance and first-order DD component", "Constant-exposure grid; null-alpha Monte Carlo", "Matched-exposure/random-timing controls from completed run"),
        ("Lemma2", "Causality of lagged eligibility", "Prefix-invariance/threshold perturbation checks", "Point-in-time timestamp/vintage audit"),
        ("Prop6", "Selected dimension controls covariance estimation error", "N/T covariance + precision operator-error surface", "N/T, condition number and effective rank per rebalance"),
        ("Lemma3", "Threshold-margin stability", "Threshold noise experiment", "Margin-band counts around production cutoff"),
        ("Lemma4", "Partial-execution contraction and turnover identities", "Random state/target/eta identities", "Target vs executed weights and cost decomposition"),
        ("Lemma5", "Turnover penalty/constraint comparative statics", "Lambda turnover sweep; beta feasible-fraction nesting", "Lambda/beta paths with turnover and binding diagnostics"),
        ("Remark12", "Adaptive parameters stay in configured box", "Controller score grid", "Observed beta/lambda ranges"),
        ("Property1", "Decision-time adaptedness of complete pipeline", "Future-perturbation prefix-invariance for controller", "Nested-window/timestamp audit; vintage still external requirement"),
        ("Lemma6", "EMA score boundedness and impulse decay", "Exact impulse response vs recursion", "Score/EMA path diagnostics when exported"),
        ("Prop7", "Top-K rank-margin stability", "Noise/smoothing membership-change surface", "Cutoff margins and observed overlap"),
        ("Remark13", "Gaussian/sub-Gaussian reversal probability falls with margin", "Exact Gaussian reversal vs Monte Carlo", "Bootstrap/estimated score-noise calibration"),
        ("Lemma7", "Replacement-Jaccard identity", "Random equal-size set identity", "Universe replacement and Jaccard series"),
        ("Prop8", "Existence, non-uniqueness, regularized uniqueness", "Identical-asset degeneracy and ridge tie-break", "Solver feasibility; uniqueness is design-dependent"),
        ("Prop9", "CVaR-budget shadow price equals local value sensitivity", "LP shadow price vs finite-difference value derivative", "Exported shadow price/slack per rebalance"),
        ("Prop10", "No-trade region induced by L1 penalty", "Lambda/no-trade frequency sweep", "Zero-turnover frequency vs lambda/state"),
        ("Prop11", "Lipschitz robustness of geometric aggregation", "Random bounded perturbation stress test", "Signal perturbation/overlay sensitivity"),
        ("Remark14", "Boundedness does not imply convergence", "Alternating-score two-cycle", "Parameter recurrence/state stratification"),
        ("Lemma8", "Coherence preserved by convex horizon aggregation", "Numerical axiom residuals for convex CVaR aggregation", "Not required: background mathematical property"),
        ("Prop12", "Selection optimism from best-of-M search", "Pure-noise max experiment", "Complete trial ledger/DSR"),
        ("Prop13", "Finite-recipe uniform deviation scale", "M x T Monte Carlo scale/heatmap", "Diagnostic only; adaptive search violates theorem assumptions"),
        ("Prop14", "Causality of nested selection", "Ordered-window invariant recorded in coverage register", "selection_audit window ordering and strict nested-engine coverage"),
        ("BenchmarkProp1-5", "Correlated BM/GBM laws and closed-form risk/ranking", "diffusion_benchmark + known_dgp_experiment tests", "Not an ETF model-fit claim"),
        ("BenchmarkProp6-8", "Exact Gaussian CVaR gradient/correlation/oracle geometry", "sequential_diffusion_analysis", "Not an ETF model-fit claim"),
        ("PropA1", "Asset tilts preserve risky exposure", "Actual smart-position-sizing implementation invariant", "Smart rebalance audit exposure sums"),
        ("PropA2", "Geometric aggregation anti-collapse", "M-signal equal-state and random-state checks", "Overlay floor/combination audit"),
        ("PropA3", "Incremental signal population MSE reduction", "FWL residual experiment with redundant/incremental signals", "Walk-forward incremental OOS MSE/HAC test"),
        ("Counterexample3", "Unregularized composition map can jump", "Two-identical-asset epsilon perturbation", "Not a performance claim; motivates regularization/controller separation"),
        # Remarks and construction rules are included explicitly so the audit covers
        # the manuscript's formal narrative, not only numbered propositions.  Where
        # a remark is a consequence/interpretation of a parent theorem, its numeric
        # burden deliberately points to the parent experiment rather than inventing
        # a separate pseudo-test.
        ("Remark1", "Aggregation washout under independent/local-scale tails", "Prop1 Gaussian-vs-mixture shape-ratio experiment", "Generator-family shape-ratio dispersion"),
        ("Remark2", "Reading level should remain separated from adaptive constraint level", "Prop2 beta*/beta and binding experiment", "Historical beta*==beta plus binding frequency"),
        ("Remark3", "Centering removes return-location contamination", "Lemma1 mean-shift identity", "Centered = raw + scenario mean residual"),
        ("Remark4", "Finite-scenario CVaR has a nonzero sampling-noise floor", "Scenario-count convergence in sequential_diffusion_analysis", "Scenario-count/seed sensitivity when rerun"),
        ("Remark5", "Combination assumptions concern errors, clipping and state dependence", "Prop4 controlled error-covariance surface", "Fold-estimated error covariance against future risk"),
        ("Remark6", "Fixed product avoids estimated-weight noise and is conservative", "Product/geometric aggregation response checks", "Matched-exposure combination-form comparison"),
        ("Remark7", "Matched-exposure and timing-destroyed controls are implied by attribution", "Prop5 exposure grid", "Historical matched-exposure/random-timing controls"),
        ("Remark8", "Product-rule turnover must be measured rather than assumed bounded", "Signal-aggregation perturbation stress test", "Historical |delta k| and break-even-cost audit"),
        ("Remark9", "Predictable exposure cannot manufacture alpha under conditional-zero-alpha", "Predictable-volatility-scaling null Monte Carlo", "Historical attribution only; theorem is a null result"),
        ("Remark10", "Selection stability does not imply return superiority", "Prop6 dimension test plus Counterexample1 logic", "Equal-weight universe ablation and conditioning metrics"),
        ("Remark11", "Long-only capped simplex feasible iff N*u_bar >= 1", "Feasibility checked by oracle/scenario programs", "Historical selected N versus position cap"),
        ("Remark13", "Ranking reversal probability decays with score margin", "Prop7 exact Gaussian reversal calibration", "Bootstrap/estimated score-noise calibration"),
        ("Rule1", "Name the internal signal according to generator content", "Prop1/Cor1 generator comparison", "Historical spanning and generator-family diagnostics"),
        ("Rule2", "Read the internal signal away from the adaptive constraint level", "Prop2 degeneracy experiment", "Historical beta*/beta plus binding diagnostic"),
        ("Rule3", "Center scenarios before reading forward risk", "Lemma1/Prop3 translation and horizon tests", "Historical centered/raw path comparison"),
        ("Counterexample1", "No causal screen guarantees ex-post return", "Logical/tie-rule counterexample represented in coverage register", "Historical equal-weight ablation, not a theorem of returns"),
        ("Counterexample2", "Identical scenario columns create non-unique compositions", "Prop8 identical-asset degeneracy", "Solver tie-breaking only; not a performance test"),
        ("FalsifiedB1", "Controller convergence is not guaranteed", "Remark14 alternating-score two-cycle", "Historical recurrence is descriptive only"),
        ("FalsifiedB2", "Covariance cleaning cannot dominate pointwise for every sample", "Logical exact-sample counterexample; no universal superiority claim", "Any cleaning comparison must be model/sample specific"),
        ("FalsifiedB3", "BH FDR control is not valid under arbitrary dependence", "Theory/literature boundary; BY remains dependence-robust", "Report BH and BY separately"),
        ("FalsifiedB4", "Internal CVaR need not contain tail information beyond second moments", "Prop1/Cor1 spanning plus mixture contrast", "Historical generator comparison"),
        ("ConjectureB1", "Adaptive margin-aware universe stabilization", "OPEN: not promoted to theorem", "Requires new predeclared experiment"),
        ("ConjectureB2", "Shadow-price-driven state control", "OPEN: Prop9 supplies observable state, not optimal update", "Requires frozen out-of-sample policy comparison"),
        ("ConjectureB3", "Error-covariance weighting may help many-signal aggregation", "OPEN: Prop4/A3 diagnose information only", "Requires nested walk-forward comparison"),
        ("ConjectureB4", "Joint selection-optimization stability under strong regularization", "OPEN: Counterexample3 blocks unregularized version", "Requires theorem plus regularized implementation"),
        ("OpenProblemB1-B4", "Online/dynamic regret, state recurrence and broader tail estimators", "OPEN PROBLEMS: deliberately not validated", "Future work; no current empirical claim"),
    ]
    return [
        {
            "result": r,
            "statement": s,
            "synthetic_or_unit_test": t,
            "historical_requirement": h,
            "synthetic_status": (
                "open_not_claimed" if r.startswith("Conjecture") or r.startswith("OpenProblem")
                else "theory_boundary" if r == "FalsifiedB3"
                else "implemented_and_run_or_covered_by_parent"
            ),
            "historical_status": (
                "not_applicable_open_problem" if r.startswith("Conjecture") or r.startswith("OpenProblem")
                else "auto_after_complete_run" if not h.startswith("Not required") and not h.startswith("Not an")
                else "not_required_for_identity"
            ),
        }
        for r, s, t, h in rows
    ]


def _lp_value(fit: dict[str, Any]) -> float:
    return float(fit.get("objective", np.nan))


def run_synthetic_theorem_audit(output_dir: str | Path, seed: int = 20260910) -> dict[str, Any]:
    """Run deterministic theorem/lemma diagnostics under known and misspecified laws."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    beta = 0.95
    n = 16
    sigma = np.linspace(0.10, 0.28, n)
    corr = _equicorr(n, 0.30)
    cov_d = np.outer(sigma, sigma) * corr / 252.0
    mu_d = np.linspace(0.02, 0.10, n) / 252.0
    prev = np.ones(n) / n

    # ------------------------------------------------------------------ Prop 1 / Corollary 1
    scenarios_g = rng.multivariate_normal(mu_d, cov_d, size=20000)
    z = rng.random(20000) < 0.10
    scenarios_mix = rng.multivariate_normal(mu_d, cov_d, size=20000)
    scenarios_mix[z] = rng.multivariate_normal(mu_d * -2.0, cov_d * 5.0, size=int(z.sum()))
    rows: list[dict[str, Any]] = []
    for _ in range(100):
        w = rng.dirichlet(np.ones(n))
        m = float(w @ mu_d)
        s = float(np.sqrt(w @ cov_d @ w))
        rg = scenarios_g @ w
        rm = scenarios_mix @ w
        rows.append({
            "mean": m,
            "std": s,
            "gaussian_cvar": empirical_cvar_loss(rg, beta),
            "gaussian_centered_cvar": empirical_cvar_loss(rg - rg.mean(), beta),
            "mixture_cvar": empirical_cvar_loss(rm, beta),
            "mixture_centered_cvar": empirical_cvar_loss(rm - rm.mean(), beta),
        })
    span = pd.DataFrame(rows)
    span["gaussian_shape_ratio"] = span.gaussian_centered_cvar / span["std"]
    span["mixture_shape_ratio"] = span.mixture_centered_cvar / span["std"]
    span.to_csv(out / "prop1_spanning_generator_comparison.csv", index=False)

    # ------------------------------------------------------------------ Prop 2 + Prop 9: binding + shadow prices
    # Scenario set held fixed so local value derivatives can be compared to the LP dual.
    sc_bind = rng.multivariate_normal(mu_d, cov_d, size=2500)
    ew_emp_budget = empirical_cvar_loss(sc_bind @ prev, beta)
    deg_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    for mult in np.linspace(0.90, 1.30, 17):
        b = float(mult * ew_emp_budget)
        fit = solve_return_cvar_lp(sc_bind, mu_d, prev, beta=beta, cvar_budget=b, max_weight=0.12, turnover_penalty=0.0)
        w = np.asarray(fit["weights"], float)
        empirical = empirical_cvar_loss(sc_bind @ w, beta)
        slack = float(fit.get("cvar_constraint_slack", b - empirical))
        shadow = float(fit.get("cvar_budget_shadow_price_max", np.nan))
        deg_rows.append({
            "budget_multiplier": mult,
            "signal": empirical,
            "budget": b,
            "slack": slack,
            "binding": bool(fit.get("cvar_constraint_binding", abs(slack) < 1e-7)),
            "abs_signal_minus_budget": abs(empirical - b),
            "shadow_price": shadow,
        })
        # One-sided derivative is more stable near an LP basis boundary.
        h = max(1e-7, abs(b) * 2e-4)
        f2 = solve_return_cvar_lp(sc_bind, mu_d, prev, beta=beta, cvar_budget=b + h, max_weight=0.12, turnover_penalty=0.0)
        fd = (_lp_value(f2) - _lp_value(fit)) / h
        shadow_rows.append({
            "budget_multiplier": mult,
            "budget": b,
            "slack": slack,
            "binding": bool(fit.get("cvar_constraint_binding", False)),
            "shadow_price": shadow,
            "finite_difference_dV_dc": fd,
            "abs_dual_fd_error": abs(shadow - fd) if np.isfinite(shadow) and np.isfinite(fd) else np.nan,
        })
    degeneracy = pd.DataFrame(deg_rows)
    shadow_df = pd.DataFrame(shadow_rows)
    degeneracy.to_csv(out / "prop2_budget_degeneracy.csv", index=False)
    shadow_df.to_csv(out / "prop9_shadow_price_envelope.csv", index=False)

    # ------------------------------------------------------------------ Prop 3 / Lemma 1: horizon and centering
    h_rows = []
    w_ew = prev.copy()
    for H in [1, 5, 21, 63, 126, 252]:
        mh = mu_d * H
        ch = cov_d * H
        raw = gaussian_loss_cvar(w_ew, mh, ch, beta, centered=False)
        cen = gaussian_loss_cvar(w_ew, mh, ch, beta, centered=True)
        h_rows.append({
            "horizon_days": H,
            "mean_term": float(w_ew @ mh),
            "vol_term": float(np.sqrt(w_ew @ ch @ w_ew)),
            "uncentered_cvar": raw,
            "centered_cvar": cen,
            "mean_to_vol_ratio": float((w_ew @ mh) / np.sqrt(w_ew @ ch @ w_ew)),
        })
    horizon = pd.DataFrame(h_rows)
    horizon.to_csv(out / "prop3_horizon_mean_domination.csv", index=False)
    shift_rows = []
    for shift in np.linspace(-0.02, 0.02, 17):
        shift_rows.append({
            "common_mean_shift": shift,
            "centered_cvar": gaussian_loss_cvar(w_ew, mu_d + shift, cov_d, beta, centered=True),
            "uncentered_cvar": gaussian_loss_cvar(w_ew, mu_d + shift, cov_d, beta, centered=False),
        })
    pd.DataFrame(shift_rows).to_csv(out / "lemma1_translation_invariance.csv", index=False)

    # ------------------------------------------------------------------ Prop 4: error covariance surface / heatmap
    combo_rows = []
    for sd_ratio in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for rho in np.linspace(-0.8, 0.95, 36):
            sd_f, sd_s = 1.0, float(sd_ratio)
            cov_e = np.array([[sd_f**2, rho * sd_f * sd_s], [rho * sd_f * sd_s, sd_s**2]])
            e = rng.multivariate_normal([0.0, 0.0], cov_e, size=10000)
            vF, vS = float(np.mean(e[:, 0] ** 2)), float(np.mean(e[:, 1] ** 2))
            c = float(np.mean(e[:, 0] * e[:, 1]))
            den = vF + vS - 2.0 * c
            alpha = (vS - c) / den if den > EPS else 0.5
            ec = alpha * e[:, 0] + (1.0 - alpha) * e[:, 1]
            mse_c = float(np.mean(ec**2))
            combo_rows.append({
                "slow_to_fast_error_sd_ratio": sd_ratio,
                "error_corr": float(np.corrcoef(e.T)[0, 1]),
                "alpha_star": float(alpha),
                "mse_fast": vF,
                "mse_slow": vS,
                "mse_combined": mse_c,
                "combined_to_best_mse": mse_c / min(vF, vS),
                "strict_gain": bool(mse_c < min(vF, vS)),
            })
    combo = pd.DataFrame(combo_rows)
    combo.to_csv(out / "prop4_combination_error_covariance.csv", index=False)

    # ------------------------------------------------------------------ Prop 5 / Remark 9: scale attribution + no-alpha null
    monthly = rng.normal(0.008, 0.045, 240)
    exp_rows = []
    base_dd = _max_drawdown(monthly)
    for k in [0.25, 0.40, 0.60, 0.80, 1.00]:
        rr = k * monthly
        exp_rows.append({
            "k": k,
            "sharpe": _sharpe(rr),
            "max_drawdown": _max_drawdown(rr),
            "first_order_dd": k * base_dd,
            "dd_approx_error": _max_drawdown(rr) - k * base_dd,
        })
    exposure = pd.DataFrame(exp_rows)
    exposure.to_csv(out / "prop5_exposure_invariance_drawdown.csv", index=False)
    # Under conditional-zero-alpha, predictable k_t depending on past volatility cannot create expected excess return.
    null_rows = []
    for rep in range(500):
        r = rng.normal(0.0, 0.04, 180)
        vol = pd.Series(r).rolling(12, min_periods=3).std().shift(1).fillna(0.04).to_numpy()
        k = np.clip(0.04 / np.maximum(vol, 1e-6), 0.30, 1.0)
        null_rows.append({"rep": rep, "base_mean": float(r.mean()), "scaled_mean": float((k * r).mean())})
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(out / "remark9_no_conditional_alpha_mc.csv", index=False)

    # ------------------------------------------------------------------ Prop 6: covariance + precision error vs N/T
    dim_rows = []
    for T in [126, 252, 504, 756]:
        for N in [8, 16, 32, 64, 96]:
            if N >= T:
                continue
            cov_errs, prec_errs, conds = [], [], []
            for _ in range(10):
                B = rng.normal(size=(N, max(2, N // 8)))
                pop = _nearest_psd(B @ B.T / max(B.shape[1], 1) + np.eye(N))
                x = rng.multivariate_normal(np.zeros(N), pop, size=T)
                sh = _nearest_psd(np.cov(x, rowvar=False))
                cov_errs.append(np.linalg.norm(sh - pop, ord=2) / max(np.linalg.norm(pop, ord=2), EPS))
                try:
                    prec_errs.append(np.linalg.norm(np.linalg.inv(sh) - np.linalg.inv(pop), ord=2) / max(np.linalg.norm(np.linalg.inv(pop), ord=2), EPS))
                except np.linalg.LinAlgError:
                    prec_errs.append(np.nan)
                conds.append(np.linalg.cond(sh))
            dim_rows.append({
                "T": T,
                "N": N,
                "N_over_T": N / T,
                "mean_relative_cov_op_error": float(np.nanmean(cov_errs)),
                "mean_relative_precision_op_error": float(np.nanmean(prec_errs)),
                "median_condition_number": float(np.nanmedian(conds)),
            })
    dimension = pd.DataFrame(dim_rows)
    dimension.to_csv(out / "prop6_dimension_covariance_error.csv", index=False)

    # ------------------------------------------------------------------ Lemma 3 + Lemma 6 + Prop 7 + Remark 13
    # Threshold screen exact stability.
    threshold_rows = []
    theta = 0.50
    for margin in [0.005, 0.01, 0.02, 0.04, 0.08]:
        for eps in [0.0025, 0.005, 0.01, 0.02, 0.04]:
            true = np.array([theta - margin, theta + margin])
            changes = 0
            for _ in range(2000):
                noise = rng.uniform(-eps, eps, size=2)
                changes += int(np.any((true + noise >= theta) != (true >= theta)))
            threshold_rows.append({
                "margin": margin,
                "epsilon_bound": eps,
                "certificate_stable": margin > eps,
                "membership_change_probability_mc": changes / 2000.0,
            })
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(out / "lemma3_threshold_margin_stability.csv", index=False)

    # EMA impulse identity.
    ema_rows = []
    for alpha in [0.2, 0.5, 0.8]:
        state = 0.0
        delta = 0.7
        for h in range(9):
            if h == 0:
                state = alpha * delta
            else:
                state = (1 - alpha) * state
            exact = float(ema_impulse_response(delta, alpha, [h])[0])
            ema_rows.append({"alpha": alpha, "horizon": h, "recursive_effect": state, "exact_effect": exact, "residual": state - exact})
    ema_df = pd.DataFrame(ema_rows)
    ema_df.to_csv(out / "lemma6_ema_impulse_decay.csv", index=False)

    # Top-K score stability with and without smoothing; exact pair reversal probability.
    rank_rows = []
    K = 20
    base_scores = np.sort(rng.normal(size=60))[::-1]
    margin = float(base_scores[K - 1] - base_scores[K])
    for alpha in [0.25, 0.50, 1.00]:
        for tau in [0.01, 0.03, 0.05, 0.10, 0.20, 0.35]:
            changed = 0
            effective_tau = alpha * tau
            for _ in range(1200):
                est = base_scores + rng.normal(scale=effective_tau, size=base_scores.size)
                top = set(np.argsort(est)[-K:])
                changed += int(top != set(range(K)))
            pair_p = gaussian_reversal_probability(base_scores[K - 1], base_scores[K], effective_tau, effective_tau)
            rank_rows.append({
                "alpha_s": alpha,
                "raw_score_noise_sd": tau,
                "effective_one_step_noise_sd": effective_tau,
                "cutoff_margin": margin,
                "certificate_2_alpha_eps_using_eps=tau": bool(margin > 2 * alpha * tau),
                "selection_change_probability_mc": changed / 1200.0,
                "boundary_pair_reversal_probability_exact": pair_p,
            })
    rank = pd.DataFrame(rank_rows)
    rank.to_csv(out / "prop7_rank_margin_reversal.csv", index=False)

    # Lemma 7 exact set identity U = (1-J)/(1+J).
    j_rows = []
    universe = np.arange(200)
    for repl in [0, 5, 10, 20, 40, 60]:
        A = set(range(80))
        common = 80 - repl
        B = set(list(range(common)) + list(range(80, 80 + repl)))
        I = len(A & B)
        U = 1 - I / 80.0
        J = _jaccard(A, B)
        U_from_J = (1 - J) / (1 + J)
        j_rows.append({"replacements": repl, "U": U, "J": J, "U_from_J": U_from_J, "identity_residual": U - U_from_J})
    jacc = pd.DataFrame(j_rows)
    jacc.to_csv(out / "lemma7_turnover_jaccard_identity.csv", index=False)

    # ------------------------------------------------------------------ Lemma 4: partial execution
    pe_rows = []
    for eta in [0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        max_contract = max_turn = 0.0
        mean_cost_factor = []
        for _ in range(300):
            a = rng.dirichlet(np.ones(n))
            b = rng.dirichlet(np.ones(n))
            _, ids = partial_execution(a, b, eta)
            max_contract = max(max_contract, abs(ids["contraction_identity_error"]))
            max_turn = max(max_turn, abs(ids["turnover_identity_error"]))
            if ids["target_distance_l1"] > EPS:
                mean_cost_factor.append(ids["executed_turnover_l1"] / ids["target_distance_l1"])
        pe_rows.append({
            "eta": eta,
            "max_contraction_identity_residual": max_contract,
            "max_turnover_identity_residual": max_turn,
            "mean_executed_to_target_turnover_ratio": float(np.mean(mean_cost_factor)) if mean_cost_factor else 0.0,
        })
    partial_df = pd.DataFrame(pe_rows)
    partial_df.to_csv(out / "lemma4_partial_execution_distribution.csv", index=False)

    # ------------------------------------------------------------------ Lemma 5 / Prop 10 + beta nesting
    sc = rng.multivariate_normal(mu_d, cov_d, size=1800)
    budget = empirical_cvar_loss(sc @ prev, beta=beta)
    lam_rows = []
    mean_perturbations = [rng.normal(scale=0.00015, size=n) for _ in range(20)]
    for lam in [0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]:
        turnovers, no_trade = [], []
        for perturb in mean_perturbations:
            muv = mu_d + perturb
            fit = solve_return_cvar_lp(sc, muv, prev, beta=beta, cvar_budget=budget, max_weight=0.12, turnover_penalty=lam)
            tv = float(np.sum(np.abs(np.asarray(fit["weights"]) - prev)))
            turnovers.append(tv)
            no_trade.append(tv < 1e-7)
        lam_rows.append({"lambda": lam, "mean_turnover": float(np.mean(turnovers)), "no_trade_frequency": float(np.mean(no_trade))})
    lamdf = pd.DataFrame(lam_rows)
    lamdf.to_csv(out / "lemma5_prop10_turnover_penalty_no_trade.csv", index=False)

    # Feasible fractions for fixed candidate portfolios must be nonincreasing as beta deepens.
    candidate_w = rng.dirichlet(np.ones(n), size=500)
    beta_rows = []
    fixed_budget = float(np.quantile([empirical_cvar_loss(sc @ w, 0.90) for w in candidate_w], 0.55))
    for blevel in [0.90, 0.925, 0.95, 0.975, 0.99]:
        risks = np.array([empirical_cvar_loss(sc @ w, blevel) for w in candidate_w])
        beta_rows.append({"beta": blevel, "feasible_fraction": float(np.mean(risks <= fixed_budget)), "fixed_budget": fixed_budget})
    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(out / "lemma5_beta_feasible_set_nesting.csv", index=False)

    # ------------------------------------------------------------------ Remark 12 / 14 + Property 1 prefix-invariance
    cfg = {
        "confidence_level": 0.95,
        "turnover_penalty": 0.01,
        "dynamic_parameters": {
            "enabled": True,
            "confidence_min": 0.90,
            "confidence_max": 0.995,
            "turnover_min": 0.005,
            "turnover_max": 0.05,
            "confidence_step": 0.05,
            "turnover_step": 0.55,
            "confidence_direction": -1.0,
            "turnover_direction": 1.0,
            "performance_weight": 1.0,
            "regime_weight": 0.0,
        },
    }
    score_rows = []
    for score in np.linspace(-1, 1, 41):
        # Use the same transformation as DynamicParameterController.next_optimizer_config.
        dyn = cfg["dynamic_parameters"]
        conf = np.clip(cfg["confidence_level"] + dyn["confidence_direction"] * dyn["confidence_step"] * score,
                       dyn["confidence_min"], dyn["confidence_max"])
        turn = np.clip(cfg["turnover_penalty"] * np.exp(dyn["turnover_direction"] * dyn["turnover_step"] * score),
                       dyn["turnover_min"], dyn["turnover_max"])
        score_rows.append({"score": score, "beta": conf, "lambda": turn})
    controller_map = pd.DataFrame(score_rows)
    controller_map.to_csv(out / "remark12_bounded_parameter_map.csv", index=False)

    # Alternating state produces bounded non-convergence (two-cycle after warmup).
    alt_rows = []
    for t in range(20):
        score = -1.0 if t % 2 == 0 else 1.0
        row = controller_map.iloc[(controller_map["score"] - score).abs().argmin()]
        alt_rows.append({"t": t, "score": score, "beta": float(row.beta), "lambda": float(row["lambda"])})
    alt_df = pd.DataFrame(alt_rows)
    alt_df.to_csv(out / "remark14_alternating_state_nonconvergence.csv", index=False)

    # Prefix-invariance of the actual dynamic controller: future observations cannot affect decision at t.
    hist = rng.normal(0.0002, 0.01, size=(300, 8))
    altered = hist.copy()
    altered[250:] += 1.0  # impossible future shock, deliberately huge
    c1 = DynamicParameterController.from_optimizer_config(cfg)
    c2 = DynamicParameterController.from_optimizer_config(cfg)
    for r in [0.01, -0.005, 0.002]:
        c1.update_after_realized_return(r)
        c2.update_after_realized_return(r)
    o1, i1 = c1.next_optimizer_config(cfg, hist[:250], date="t")
    o2, i2 = c2.next_optimizer_config(cfg, altered[:250], date="t")
    pd.DataFrame([{
        "beta_original_prefix": o1["confidence_level"],
        "beta_future_perturbed_prefix": o2["confidence_level"],
        "lambda_original_prefix": o1["turnover_penalty"],
        "lambda_future_perturbed_prefix": o2["turnover_penalty"],
        "max_decision_difference": max(abs(o1["confidence_level"] - o2["confidence_level"]), abs(o1["turnover_penalty"] - o2["turnover_penalty"])),
    }]).to_csv(out / "property1_prefix_invariance.csv", index=False)

    # ------------------------------------------------------------------ Prop 8 + Counterexample 3: nonuniqueness / regularized uniqueness / discontinuity
    # Two identical scenario columns: risk is independent of how weight is split.
    base_col = rng.normal(0.0003, 0.01, size=3000)
    R2 = np.column_stack([base_col, base_col])
    prev2 = np.array([0.5, 0.5])
    deg_opt_rows = []
    for eps in [-1e-8, 0.0, 1e-8]:
        mu2 = np.array([0.0003 + eps, 0.0003 - eps])
        fit = solve_return_cvar_lp(R2, mu2, prev2, beta=0.95, cvar_budget=empirical_cvar_loss(base_col, 0.95) + 1e-6,
                                   max_weight=1.0, turnover_penalty=0.0)
        w2 = np.asarray(fit["weights"], float)
        deg_opt_rows.append({"mean_tilt_epsilon": eps, "w1": w2[0], "w2": w2[1], "objective": fit["objective"]})
    # Strict concave tie-break: maximize common mean minus kappa/2 ||w||^2 -> equal split uniquely.
    kappa = 0.01
    res_ridge = minimize(lambda w: -(0.0003 * np.sum(w) - 0.5 * kappa * np.sum(w**2)), np.array([0.2, 0.8]),
                         method="SLSQP", bounds=[(0, 1), (0, 1)], constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}])
    deg_opt_rows.append({"mean_tilt_epsilon": np.nan, "w1": float(res_ridge.x[0]), "w2": float(res_ridge.x[1]), "objective": float(-res_ridge.fun), "case": "strictly_concave_regularized"})
    optdeg = pd.DataFrame(deg_opt_rows)
    optdeg.to_csv(out / "prop8_counterexample3_nonuniqueness_discontinuity.csv", index=False)

    # ------------------------------------------------------------------ Prop 11 + A2: geometric aggregation robustness/anti-collapse
    lip_rows = []
    kmin = 0.30
    a = np.array([0.5, 0.3, 0.2])
    ratios = []
    for _ in range(6000):
        x = rng.uniform(kmin, 1, size=3)
        y = np.clip(x + rng.normal(0, 0.03, size=3), kmin, 1)
        Kx = float(np.exp(a @ np.log(x)))
        Ky = float(np.exp(a @ np.log(y)))
        den = np.sum(a * np.abs(x - y))
        ratios.append(abs(Kx - Ky) / max(den, EPS))
    max_ratio = float(np.max(ratios))
    for M in [2, 3, 5, 10]:
        common = 0.60
        raw_product = common**M
        equal_geomean = common
        lip_rows.append({
            "M": M,
            "k_min": kmin,
            "common_signal": common,
            "raw_product": raw_product,
            "equal_weight_geometric_mean": equal_geomean,
            "empirical_max_lipschitz_ratio": max_ratio,
            "theoretical_lipschitz_bound": 1 / kmin,
            "bound_holds": max_ratio <= 1 / kmin + 1e-9,
        })
    lip_df = pd.DataFrame(lip_rows)
    lip_df.to_csv(out / "prop11_propA2_geometric_aggregation.csv", index=False)

    # ------------------------------------------------------------------ Lemma 8: coherence checks for convex horizon CVaR aggregation
    coh_rows = []
    horizon_weights = np.array([0.2, 0.3, 0.5])
    for rep in range(300):
        X = rng.normal(0.0, 0.02, size=(3000, 3))
        Y = rng.normal(0.0, 0.025, size=(3000, 3))
        risks_x = np.array([_empirical_cvar_risk(X[:, h], beta) for h in range(3)])
        risks_y = np.array([_empirical_cvar_risk(Y[:, h], beta) for h in range(3)])
        risks_xy = np.array([_empirical_cvar_risk(X[:, h] + Y[:, h], beta) for h in range(3)])
        rho_x = float(horizon_weights @ risks_x)
        rho_y = float(horizon_weights @ risks_y)
        rho_xy = float(horizon_weights @ risks_xy)
        a_scale = 1.7
        rho_ax = float(horizon_weights @ np.array([_empirical_cvar_risk(a_scale * X[:, h], beta) for h in range(3)]))
        c_shift = 0.01
        rho_shift = float(horizon_weights @ np.array([_empirical_cvar_risk(X[:, h] + c_shift, beta) for h in range(3)]))
        coh_rows.append({
            "rep": rep,
            "subadditivity_residual": rho_xy - (rho_x + rho_y),
            "positive_homogeneity_residual": rho_ax - a_scale * rho_x,
            "translation_equivariance_residual": rho_shift - (rho_x - c_shift),
        })
    coherence = pd.DataFrame(coh_rows)
    coherence.to_csv(out / "lemma8_coherent_horizon_aggregation.csv", index=False)

    # ------------------------------------------------------------------ Props 12/13: selection optimism and uniform-deviation complexity
    sel_rows = []
    for T in [42, 83, 166, 332]:
        for M in [1, 2, 5, 10, 20, 40, 80, 160, 320]:
            maxima = np.max(rng.normal(0, 1 / np.sqrt(T), size=(1200, M)), axis=1)
            sel_rows.append({
                "T_eff": T,
                "M": M,
                "mean_selected_noise": float(np.mean(maxima)),
                "p95_selected_noise": float(np.quantile(maxima, 0.95)),
                "sqrt_logM_over_T": float(np.sqrt(np.log(max(M, 2)) / T)),
            })
    selection = pd.DataFrame(sel_rows)
    selection.to_csv(out / "prop12_13_selection_optimism_uniform_scale.csv", index=False)

    # ------------------------------------------------------------------ Prop A1 actual implementation invariant
    a1_rows = []
    for rep in range(200):
        tickers = [f"A{i}" for i in range(12)]
        incoming_exposure = float(rng.uniform(0.3, 1.0))
        base = pd.Series(rng.dirichlet(np.ones(len(tickers))) * incoming_exposure, index=tickers)
        table = pd.DataFrame({
            "signal_quality": rng.uniform(0.05, 0.95, len(tickers)),
            "kelly_fraction": rng.uniform(0.0, 0.25, len(tickers)),
            "risk_reward_ratio": rng.uniform(0.2, 3.0, len(tickers)),
        }, index=tickers)
        tilted = apply_smart_position_sizing(asset_weights=base, signal_table=table, config={"max_weight": None, "min_active_weight": 0.0})
        a1_rows.append({"rep": rep, "incoming_exposure": incoming_exposure, "outgoing_exposure": float(tilted.sum()), "exposure_residual": float(tilted.sum() - incoming_exposure)})
    a1 = pd.DataFrame(a1_rows)
    a1.to_csv(out / "propA1_exposure_preserving_role_separation.csv", index=False)

    # ------------------------------------------------------------------ Prop A3 FWL incremental MSE identity + redundancy contrast
    a3_rows = []
    for rep in range(300):
        m = 800
        latent = rng.normal(size=m)
        x1 = latent + rng.normal(scale=0.7, size=m)
        x_redundant = x1 + rng.normal(scale=0.02, size=m)
        x_incremental = rng.normal(size=m)
        y = 0.8 * latent + 0.45 * x_incremental + rng.normal(scale=0.8, size=m)
        Xs = np.column_stack([np.ones(m), x1])
        b_y, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        ytil = y - Xs @ b_y
        for name, xj in [("redundant", x_redundant), ("incremental", x_incremental)]:
            b_x, *_ = np.linalg.lstsq(Xs, xj, rcond=None)
            xtil = xj - Xs @ b_x
            cov = float(np.mean((ytil - ytil.mean()) * (xtil - xtil.mean())))
            varx = float(np.var(xtil))
            formula_gain = cov**2 / max(varx, EPS)
            gamma = cov / max(varx, EPS)
            mse0 = float(np.mean(ytil**2))
            mse1 = float(np.mean((ytil - gamma * xtil) ** 2))
            a3_rows.append({"rep": rep, "candidate": name, "formula_mse_reduction": formula_gain, "measured_mse_reduction": mse0 - mse1, "identity_residual": (mse0 - mse1) - formula_gain})
    a3 = pd.DataFrame(a3_rows)
    a3.to_csv(out / "propA3_incremental_signal_mse.csv", index=False)

    # ------------------------------------------------------------------ Coverage + summary
    coverage = pd.DataFrame(_coverage_rows())
    coverage.to_csv(out / "theorem_coverage_register.csv", index=False)

    binding_subset = shadow_df[shadow_df["binding"] & np.isfinite(shadow_df["shadow_price"]) & np.isfinite(shadow_df["finite_difference_dV_dc"])]
    summary_rows = [
        ("Prop1/Cor1", "Gaussian location-scale spanning", _r2(span.gaussian_centered_cvar, span["std"]), "R2(centered CVaR ~ sigma)"),
        ("Prop1 contrast", "Regime-mixture tail shape not constant", float(span.mixture_shape_ratio.std()), "std(CVaR/sigma)"),
        ("Prop2", "Binding-level budget degeneracy", float(degeneracy.loc[degeneracy.binding, "abs_signal_minus_budget"].median()) if degeneracy.binding.any() else float("nan"), "median |signal-budget| when binding"),
        ("Prop3/Lemma1", "Horizon mean domination", float(horizon.mean_to_vol_ratio.iloc[-1] / horizon.mean_to_vol_ratio.iloc[0]), "252d/1d mean-vol multiplier"),
        ("Prop4", "Combination strict-gain region", float(combo.strict_gain.mean()), "share error surface with strict gain"),
        ("Prop5", "Constant-scale Sharpe invariance", float(exposure.sharpe.max() - exposure.sharpe.min()), "Sharpe range across k"),
        ("Remark9", "Predictable scaling no-alpha null", float(null_df.scaled_mean.mean()), "mean scaled excess return over null MC"),
        ("Prop6", "Dimension worsens covariance estimation", float(dimension.corr(numeric_only=True).loc["N_over_T", "mean_relative_cov_op_error"]), "corr(N/T,cov error)"),
        ("Lemma3", "Threshold certificate", float(threshold_df.loc[threshold_df.certificate_stable, "membership_change_probability_mc"].max()), "max flip probability inside certified region"),
        ("Lemma6", "EMA impulse identity", float(ema_df.residual.abs().max()), "max impulse residual"),
        ("Prop7", "Rank instability rises with score noise", float(rank[rank.alpha_s == 1.0].corr(numeric_only=True).loc["raw_score_noise_sd", "selection_change_probability_mc"]), "corr(noise, membership change)"),
        ("Lemma7", "Turnover-Jaccard identity", float(jacc.identity_residual.abs().max()), "max identity residual"),
        ("Lemma4", "Partial execution identities", float(max(partial_df.max_contraction_identity_residual.max(), partial_df.max_turnover_identity_residual.max())), "max numerical residual"),
        ("Lemma5/Prop10", "Turnover penalty reduces turnover", float(lamdf.corr(numeric_only=True).loc["lambda", "mean_turnover"]), "corr(lambda,turnover)"),
        ("Prop9", "Shadow-price envelope", float(binding_subset.abs_dual_fd_error.median()) if len(binding_subset) else float("nan"), "median |dual - dV/dc| on binding cells"),
        ("Remark12", "Parameter box", float(max(controller_map.beta.max() - cfg["dynamic_parameters"]["confidence_max"], cfg["dynamic_parameters"]["confidence_min"] - controller_map.beta.min(), 0.0)), "beta bound violation"),
        ("Property1", "Prefix invariance", float(abs(o1["confidence_level"] - o2["confidence_level"]) + abs(o1["turnover_penalty"] - o2["turnover_penalty"])), "decision difference after future-only perturbation"),
        ("Prop11/A2", "Geometric aggregation Lipschitz", max_ratio, "empirical max ratio"),
        ("Lemma8", "Coherent aggregation residual", float(max(coherence.subadditivity_residual.max(), coherence.positive_homogeneity_residual.abs().max(), coherence.translation_equivariance_residual.abs().max())), "max axiom residual (positive means violation for subadditivity)"),
        ("Prop12/13", "Search creates selection optimism", float(selection[(selection.T_eff == 83) & (selection.M == 320)].mean_selected_noise.iloc[0] - selection[(selection.T_eff == 83) & (selection.M == 1)].mean_selected_noise.iloc[0]), "optimism M=320 minus M=1 at T=83"),
        ("PropA1", "Exposure-preserving tilts", float(a1.exposure_residual.abs().max()), "max exposure residual"),
        ("PropA3", "FWL incremental MSE identity", float(a3.identity_residual.abs().max()), "max formula-vs-measured residual"),
        ("Counterexample3", "Composition discontinuity", float(abs(optdeg.iloc[0].w1 - optdeg.iloc[2].w1)), "weight jump under +/- epsilon mean tilt"),
    ]
    summary = pd.DataFrame(summary_rows, columns=["result", "check", "value", "metric"])
    summary.to_csv(out / "theorem_to_synthetic_summary.csv", index=False)

    _plot_synthetic(
        out, span, degeneracy, shadow_df, horizon, combo, exposure, dimension,
        threshold_df, rank, lamdf, controller_map, selection, a3,
    )
    payload = {
        "mode": "synthetic_known_dgp_and_controlled_misspecification",
        "seed": seed,
        "n_coverage_objects": int(len(coverage)),
        "checks": summary.to_dict(orient="records"),
        "historical_note": "Historical ETF audit is generated only from a completed point-in-time run; this synthetic audit does not replace it.",
    }
    (out / "theorem_empirical_bridge_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _plot_synthetic(
    out: Path,
    span: pd.DataFrame,
    degeneracy: pd.DataFrame,
    shadow: pd.DataFrame,
    horizon: pd.DataFrame,
    combo: pd.DataFrame,
    exposure: pd.DataFrame,
    dimension: pd.DataFrame,
    threshold: pd.DataFrame,
    rank: pd.DataFrame,
    lamdf: pd.DataFrame,
    controller: pd.DataFrame,
    selection: pd.DataFrame,
    a3: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    plots = [
        ("figure_prop1_spanning.png", span["std"], span.gaussian_centered_cvar, "portfolio sigma", "centered CVaR", "Proposition 1: Gaussian spanning"),
        ("figure_prop2_budget_degeneracy.png", degeneracy.slack, degeneracy.abs_signal_minus_budget, "CVaR budget slack", "|signal-budget|", "Proposition 2: binding-budget degeneracy"),
        ("figure_prop3_horizon_mean_domination.png", horizon.horizon_days, horizon.mean_to_vol_ratio, "horizon (days)", "mean / volatility", "Proposition 3: horizon mean domination"),
        ("figure_prop5_exposure_attribution.png", exposure.k, exposure.max_drawdown, "constant risky exposure k", "maximum drawdown", "Proposition 5: drawdown under constant exposure"),
        ("figure_prop6_dimension_error.png", dimension.N_over_T, dimension.mean_relative_cov_op_error, "N/T", "relative covariance operator error", "Proposition 6: dimension channel"),
        ("figure_prop10_no_trade.png", lamdf["lambda"], lamdf.mean_turnover, "turnover penalty lambda", "mean L1 turnover", "Lemma 5 / Proposition 10: no-trade channel"),
        ("figure_prop12_selection_optimism.png", selection[selection.T_eff == 83].M, selection[selection.T_eff == 83].mean_selected_noise, "number of tried recipes M", "expected selected estimation noise", "Proposition 12: selection optimism"),
    ]
    for fn, x, y, xl, yl, title in plots:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.plot(x, y, marker="o")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out / fn, dpi=180)
        plt.close(fig)

    # Shadow-price plot.
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(shadow.budget_multiplier, shadow.shadow_price, marker="o", label="LP shadow price")
    ax.plot(shadow.budget_multiplier, shadow.finite_difference_dV_dc, marker="x", label="finite difference dV/dc")
    ax.set_xlabel("CVaR budget multiplier")
    ax.set_ylabel("marginal objective value")
    ax.set_title("Proposition 9: shadow price versus local value sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure_prop9_shadow_price.png", dpi=180)
    plt.close(fig)

    # Combination heatmap.
    piv = combo.pivot_table(index="slow_to_fast_error_sd_ratio", columns="error_corr", values="combined_to_best_mse", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
    xt = np.linspace(0, len(piv.columns) - 1, 7).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{piv.columns[i]:.2f}" for i in xt])
    ax.set_xlabel("forecast-error correlation")
    ax.set_ylabel("slow/fast error SD ratio")
    ax.set_title("Proposition 4: combined MSE / better single-signal MSE")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "figure_prop4_combination_gain_heatmap.png", dpi=180)
    plt.close(fig)

    # Dimension heatmap.
    piv = dimension.pivot(index="N", columns="T", values="mean_relative_cov_op_error")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(np.arange(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("T observations"); ax.set_ylabel("N selected assets")
    ax.set_title("Proposition 6: covariance error across N/T")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out / "figure_prop6_dimension_error_heatmap.png", dpi=180); plt.close(fig)

    # Rank stability heatmap.
    piv = rank.pivot(index="alpha_s", columns="raw_score_noise_sd", values="selection_change_probability_mc")
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(piv.columns))); ax.set_xticklabels([f"{x:.2f}" for x in piv.columns])
    ax.set_yticks(np.arange(len(piv.index))); ax.set_yticklabels([f"{x:.2f}" for x in piv.index])
    ax.set_xlabel("raw score-noise SD"); ax.set_ylabel("EMA alpha")
    ax.set_title("Proposition 7 / Lemma 6: selection-change probability")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out / "figure_prop7_rank_stability_heatmap.png", dpi=180); plt.close(fig)

    # Controller parameter map.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(controller.score, controller.beta, marker="o", label="beta")
    ax2 = ax.twinx()
    ax2.plot(controller.score, controller["lambda"], marker="x", label="lambda")
    ax.set_xlabel("bounded control score")
    ax.set_ylabel("CVaR level beta")
    ax2.set_ylabel("turnover penalty lambda")
    ax.set_title("Remark 12: bounded adaptive-parameter map")
    fig.tight_layout(); fig.savefig(out / "figure_remark12_parameter_map.png", dpi=180); plt.close(fig)

    # A3 redundancy vs incremental signal.
    g = a3.groupby("candidate")["measured_mse_reduction"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.bar(g.index, g.values)
    ax.set_ylabel("mean population-projection MSE reduction")
    ax.set_title("Proposition A3: redundant vs incremental signal")
    fig.tight_layout(); fig.savefig(out / "figure_propA3_incremental_signal.png", dpi=180); plt.close(fig)


def _append_check(checks: list[dict[str, Any]], result: str, metric: str, value: Any, status: str = "measured") -> None:
    checks.append({"result": result, "metric": metric, "value": value, "status": status})


def run_historical_theorem_audit(result_folder: str | Path) -> dict[str, Any]:
    """Measure empirical counterparts from a completed real-data ``run.py`` folder.

    The function is fail-soft by design.  Missing run artifacts are reported as
    unavailable.  This is essential for publication integrity: a theorem is not turned
    into a historical fact merely because the code knows how to test it.
    """
    folder = Path(result_folder)
    out = folder / "theorem_empirical_audit"
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    fr = folder / "forecast_risk.csv"
    if fr.exists():
        d = pd.read_csv(fr)
        numeric_candidates = [
            "cvar_model_centered", "cvar_model_raw", "vol_model", "portfolio_scenario_mean",
            "backward_vol_timer", "forward_cvar_timer", "forecast_cvar_level",
            "optimizer_cvar_level", "optimizer_cvar_constraint_slack", "optimizer_cvar_budget_shadow_price_max",
            "optimizer_cvar_budget", "optimizer_cvar_value", "scenario_pit_realized_target",
            "realized_target_risky_return", "scenario_var95_loss", "scenario_es95_loss",
        ]
        for c in numeric_candidates:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        if {"cvar_model_centered", "vol_model"} <= set(d):
            z = d[["cvar_model_centered", "vol_model"]].dropna()
            _append_check(checks, "Prop1/Cor1", "historical R2 centered CVaR ~ model vol", _r2(z.iloc[:, 0], z.iloc[:, 1]))
            if len(z):
                ratio = z.iloc[:, 0] / z.iloc[:, 1].replace(0, np.nan)
                _append_check(checks, "Prop1/Cor1", "historical std(CVaR/sigma)", float(ratio.std()))
        if {"cvar_model_centered", "cvar_model_raw", "portfolio_scenario_mean"} <= set(d):
            z = d[["cvar_model_centered", "cvar_model_raw", "portfolio_scenario_mean"]].dropna()
            if len(z):
                err = np.abs(z.cvar_model_centered - (z.cvar_model_raw + z.portfolio_scenario_mean))
                _append_check(checks, "Lemma1/Rule3", "max centering identity residual", float(err.max()))
        if {"backward_vol_timer", "forward_cvar_timer"} <= set(d):
            z = d[["backward_vol_timer", "forward_cvar_timer"]].dropna()
            _append_check(checks, "Prop4", "historical timer correlation", float(z.corr().iloc[0, 1]) if len(z) > 2 else np.nan)
        if {"forecast_cvar_level", "optimizer_cvar_level"} <= set(d):
            z = d[["forecast_cvar_level", "optimizer_cvar_level"]].dropna()
            if len(z):
                _append_check(checks, "Prop2", "historical frequency beta* == beta", float(np.mean(np.isclose(z.iloc[:, 0], z.iloc[:, 1], atol=1e-12))))
        if "optimizer_cvar_constraint_binding" in d:
            bind = d["optimizer_cvar_constraint_binding"].astype(str).str.lower().isin(["true", "1", "yes"])
            _append_check(checks, "Prop2/Prop9", "historical CVaR-constraint binding frequency", float(bind.mean()))
        if "optimizer_cvar_budget_shadow_price_max" in d:
            x = pd.to_numeric(d["optimizer_cvar_budget_shadow_price_max"], errors="coerce").dropna()
            if len(x):
                _append_check(checks, "Prop9", "historical mean CVaR shadow price", float(x.mean()))
                _append_check(checks, "Prop9", "historical positive-shadow frequency", float(np.mean(x > 1e-10)))
        if "scenario_pit_realized_target" in d:
            pit = d["scenario_pit_realized_target"].dropna()
            if len(pit):
                _append_check(checks, "Predictive calibration", "PIT mean", float(pit.mean()))
                _append_check(checks, "Predictive calibration", "PIT variance", float(pit.var(ddof=1)))
                # Save distribution for plotting/review.
                pit.to_frame("pit").to_csv(out / "historical_pit_values.csv", index=False)

    dyn = folder / "dynamic_parameter_history.csv"
    if dyn.exists():
        d = pd.read_csv(dyn)
        # Actual column names in the current controller are confidence_level and turnover_penalty.
        for beta_name in ["confidence_level", "beta"]:
            if beta_name in d:
                x = pd.to_numeric(d[beta_name], errors="coerce").dropna()
                if len(x):
                    _append_check(checks, "Remark12", f"historical {beta_name} range", float(x.max() - x.min()))
                break
        for lam_name in ["turnover_penalty", "lambda"]:
            if lam_name in d:
                x = pd.to_numeric(d[lam_name], errors="coerce").dropna()
                if len(x):
                    _append_check(checks, "Remark12/Prop10", f"historical {lam_name} range", float(x.max() - x.min()))
                break

    sa = folder / "selection_audit.csv"
    if sa.exists():
        d = pd.read_csv(sa)
        if "optimizer_cvar_constraint_binding" in d:
            bind = d["optimizer_cvar_constraint_binding"].astype(str).str.lower().isin(["true", "1", "yes"])
            _append_check(checks, "Prop2/Prop9", "selection-audit binding frequency", float(bind.mean()))
        if "optimizer_cvar_constraint_slack" in d:
            x = pd.to_numeric(d["optimizer_cvar_constraint_slack"], errors="coerce").dropna()
            if len(x): _append_check(checks, "Prop2/Prop9", "median CVaR slack", float(x.median()))
        if "optimizer_cvar_budget_shadow_price_max" in d:
            x = pd.to_numeric(d["optimizer_cvar_budget_shadow_price_max"], errors="coerce").dropna()
            if len(x): _append_check(checks, "Prop9", "median shadow price", float(x.median()))
        if "engine_used" in d:
            used = d["engine_used"].astype(str).str.lower().isin(["true", "1", "yes"])
            _append_check(checks, "Prop14", "nested-engine coverage", float(used.mean()))

    # Risk matrix diagnostics: N/T, condition number, effective rank, MP location.
    risks = folder / "risk_matrices.json"
    risk_rows: list[dict[str, Any]] = []
    if risks.exists():
        try:
            obj = json.loads(risks.read_text(encoding="utf-8"))
            target_exec_gap = []
            for row in obj:
                cov = row.get("covariance_daily")
                tickers = row.get("tickers") or []
                T = row.get("estimation_observations")
                if cov is not None:
                    C = np.asarray(cov, float)
                    if C.ndim == 2 and C.shape[0] == C.shape[1] and C.size:
                        eig = np.clip(np.linalg.eigvalsh((C + C.T) / 2), 0, None)
                        N = int(C.shape[0])
                        q = (N / float(T)) if T not in (None, 0) else np.nan
                        sigma2 = float(np.mean(np.diag(C)))
                        mp_plus = sigma2 * (1 + np.sqrt(q))**2 if np.isfinite(q) else np.nan
                        risk_rows.append({
                            "date": row.get("date"), "N": N, "T": T, "N_over_T": q,
                            "condition_number": float(np.linalg.cond(C)),
                            "effective_rank": _effective_rank(C),
                            "mp_upper_edge": mp_plus,
                            "fraction_eigenvalues_above_mp_edge": float(np.mean(eig > mp_plus)) if np.isfinite(mp_plus) else np.nan,
                        })
                t, e = row.get("target_weights"), row.get("executed_weights")
                if isinstance(t, dict) and isinstance(e, dict):
                    keys = set(t) & set(e)
                    target_exec_gap.append(sum(abs(float(t[k]) - float(e[k])) for k in keys))
            if risk_rows:
                rdf = pd.DataFrame(risk_rows)
                rdf.to_csv(out / "historical_covariance_geometry.csv", index=False)
                _append_check(checks, "Prop6", "mean historical N/T", float(pd.to_numeric(rdf.N_over_T, errors="coerce").mean()))
                _append_check(checks, "Prop6", "median covariance condition number", float(rdf.condition_number.median()))
                _append_check(checks, "Prop6", "median covariance effective rank", float(rdf.effective_rank.median()))
            if target_exec_gap:
                _append_check(checks, "Lemma4", "mean target-executed L1 gap", float(np.mean(target_exec_gap)))
        except Exception as exc:
            _append_check(checks, "risk_matrices", "parse error", str(exc), status="error")

    pnl = folder / "pnl.csv"
    if pnl.exists():
        d = pd.read_csv(pnl, index_col=0)
        candidates = [c for c in d.columns if c.lower() in {"returns", "return", "portfolio_return"}]
        if candidates:
            r = pd.to_numeric(d[candidates[0]], errors="coerce").dropna().values
            if len(r) > 3:
                _append_check(checks, "Prop5", "historical base Sharpe", _sharpe(r))
                _append_check(checks, "Prop5", "historical base max drawdown", _max_drawdown(r))

    if not checks:
        checks = [{"result": "historical bridge", "metric": "required run artifacts", "value": None, "status": "not_available"}]

    df = pd.DataFrame(checks)
    df.to_csv(out / "historical_theorem_audit.csv", index=False)
    payload = {"mode": "historical_real_data", "result_folder": str(folder), "checks": checks}
    (out / "historical_theorem_audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _plot_historical_if_available(out, fr if fr.exists() else None, risk_rows)
    return payload


def _plot_historical_if_available(out: Path, forecast_path: Path | None, risk_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    if forecast_path is not None and forecast_path.exists():
        try:
            d = pd.read_csv(forecast_path)
            if {"cvar_model_centered", "vol_model"} <= set(d):
                x = pd.to_numeric(d["vol_model"], errors="coerce")
                y = pd.to_numeric(d["cvar_model_centered"], errors="coerce")
                good = x.notna() & y.notna()
                if good.sum() > 2:
                    fig, ax = plt.subplots(figsize=(6.8, 4.5)); ax.scatter(x[good], y[good]); ax.set_xlabel("model volatility"); ax.set_ylabel("centered model CVaR"); ax.set_title("Historical Proposition 1 diagnostic"); fig.tight_layout(); fig.savefig(out / "historical_prop1_cvar_vs_vol.png", dpi=180); plt.close(fig)
            if "scenario_pit_realized_target" in d:
                pit = pd.to_numeric(d["scenario_pit_realized_target"], errors="coerce").dropna()
                if len(pit):
                    fig, ax = plt.subplots(figsize=(6.8, 4.4)); ax.hist(pit, bins=10, density=True); ax.axhline(1.0, linestyle="--"); ax.set_xlabel("scenario PIT of realized target return"); ax.set_ylabel("density"); ax.set_title("Historical predictive-distribution calibration"); fig.tight_layout(); fig.savefig(out / "historical_pit_histogram.png", dpi=180); plt.close(fig)
        except Exception:
            pass

    if risk_rows:
        try:
            d = pd.DataFrame(risk_rows)
            fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.scatter(d.N_over_T, d.condition_number); ax.set_xlabel("N/T"); ax.set_ylabel("covariance condition number"); ax.set_title("Historical dimension/conditioning diagnostic"); fig.tight_layout(); fig.savefig(out / "historical_prop6_dimension_conditioning.png", dpi=180); plt.close(fig)
        except Exception:
            pass
