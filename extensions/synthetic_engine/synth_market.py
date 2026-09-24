# -*- coding: utf-8 -*-
"""Synthetic markets with a KNOWN law for the identification layer of the paper (Section 3.9-3.10).

Two laws, one calendar, one asset cross-section:
  * "gbm"    -- correlated geometric Brownian motion with constant drift, volatility and correlation
                (Layer 1: identities, estimation risk, dial specificity; nothing to time).
  * "regime" -- the same diffusion modulated by a two-state Markov chain (calm / stress) that scales
                volatility, raises correlation and lowers drift in stress (Layer 2: risk moves at
                dates the experimenter knows, so the dials can be scored against the truth).

The panel mimics the production inputs: daily closes, share volume, a first-business-day calendar,
daily log returns and their EWMA (halflife 30, as parse_close.py), and the eligibility rule of the
paper (36-month traded-value share above a threshold, observed close on the last trading day before
the rebalance, 60 months of observed closes with funds present since the panel start exempt).
Everything the policy is allowed to see is lagged; the truth (regime path, true drift, volatility and
correlation) is written separately and never enters the decision.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

TRADING_DAYS = 252
EWMA_HALFLIFE = 30          # parse_close.py: ewm(halflife=WINDOW, adjust=False)


@dataclass(frozen=True)
class MarketSpec:
    law: str = "gbm"                 # "gbm" | "regime"
    seed: int = 20260913
    n_assets: int = 40
    start: str = "2016-01-04"
    end: str = "2025-12-31"
    n_late: int = 5                  # late listings (first observed 2020-07-01) exercise the history rule
    late_start: str = "2020-07-01"
    # regime law
    stress_vol_mult: float = 2.2
    stress_drift_shift: float = -0.20   # annual, added to every drift in stress
    stress_corr_weight: float = 0.5     # corr_stress = (1-w) corr_calm + w * J(0.8)
    stress_equicorr: float = 0.80
    mean_calm_months: float = 12.0
    mean_stress_months: float = 3.0
    # persistent component of the regime law: a slow log-volatility cycle (Ornstein-Uhlenbeck, daily) that the
    # 3-year estimator can track and the 21-day estimator sees only with noise; zero under the constant law
    slow_cycle_sd: float = 0.25          # stationary standard deviation of log volatility multiplier
    slow_cycle_halflife_days: int = 252  # mean-reversion half-life
    regime_random: bool = False      # False: the designed schedule below (dates known to the experimenter)
    stress_episodes: tuple = (("2017-03-01", "2017-05-31"),   # inside the estimation history only
                              ("2019-10-01", "2019-11-30"),
                              ("2020-08-03", "2020-11-30"),
                              ("2021-11-01", "2022-01-31"),
                              ("2023-03-01", "2023-08-31"),
                              ("2024-09-02", "2024-10-31"))
    # eligibility rule
    share_threshold: float = 0.015      # theta: 36-month traded-value share
    history_days: int = 1260            # 60 months of observed closes
    recent_days: int = 1


def _two_factor_corr(n: int) -> np.ndarray:
    b1 = np.linspace(0.30, 0.80, n)
    b2 = 0.35 * np.sin(np.linspace(0.0, 2.0 * np.pi, n))
    raw = np.outer(b1, b1) + np.outer(b2, b2) + 0.45 * np.eye(n)
    d = np.sqrt(np.diag(raw))
    c = raw / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return c


def _chol_psd(c: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((c + c.T) / 2.0)
    return vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))


def make_market(spec: MarketSpec) -> dict:
    """Return the synthetic panel and its truth. Prices S0 on the first date; log returns from the second."""
    rng = np.random.default_rng(spec.seed)
    dates = pd.bdate_range(spec.start, spec.end)
    T, N = len(dates), spec.n_assets
    tickers = [f"SYN{i:02d}" for i in range(N)]
    mu = np.linspace(0.03, 0.11, N) + rng.normal(0.0, 0.010, N)                 # annual arithmetic drift
    sig = np.clip(np.linspace(0.10, 0.30, N) + rng.normal(0.0, 0.020, N), 0.08, 0.36)
    corr_calm = _two_factor_corr(N)
    J = np.full((N, N), spec.stress_equicorr); np.fill_diagonal(J, 1.0)
    corr_stress = (1.0 - spec.stress_corr_weight) * corr_calm + spec.stress_corr_weight * J
    # regime path (daily two-state Markov chain), calm at the start
    state = np.zeros(T, dtype=int)
    if spec.law == "regime" and spec.regime_random:
        p01 = 1.0 / (21.0 * spec.mean_calm_months); p10 = 1.0 / (21.0 * spec.mean_stress_months)
        u = rng.random(T)
        for t in range(1, T):
            if state[t - 1] == 0: state[t] = 1 if u[t] < p01 else 0
            else: state[t] = 0 if u[t] < p10 else 1
    elif spec.law == "regime":
        for a, b in spec.stress_episodes:
            state[(dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))] = 1
    _ = rng.random(T)   # keep the draw sequence identical across the two regime options
    cycle = np.zeros(T)
    if spec.law == "regime" and spec.slow_cycle_sd > 0:
        phi = 0.5 ** (1.0 / spec.slow_cycle_halflife_days); innov_sd = spec.slow_cycle_sd * np.sqrt(1.0 - phi ** 2)
        x = np.zeros(T); e_c = rng.normal(0.0, innov_sd, T); x[0] = rng.normal(0.0, spec.slow_cycle_sd)
        for t in range(1, T): x[t] = phi * x[t - 1] + e_c[t]
        cycle = x
    vol_mult = np.where(state == 1, spec.stress_vol_mult, 1.0) * np.exp(cycle)
    drift_shift = np.where(state == 1, spec.stress_drift_shift, 0.0)
    Lc, Ls = _chol_psd(corr_calm), _chol_psd(corr_stress)
    z = rng.standard_normal((T, N))
    eps = np.where((state == 1)[:, None], z @ Ls.T, z @ Lc.T)
    sig_t = sig[None, :] * vol_mult[:, None]                                       # annual vol by day
    mu_t = mu[None, :] + drift_shift[:, None]
    logret = (mu_t - 0.5 * sig_t ** 2) / TRADING_DAYS + sig_t / np.sqrt(TRADING_DAYS) * eps
    logret[0, :] = 0.0
    s0 = np.linspace(50.0, 150.0, N)
    prices = s0[None, :] * np.exp(np.cumsum(logret, axis=0))
    # share volume: persistent cross-section (so the liquidity share screen bites), market factor, AR(1) noise
    base = np.linspace(11.8, 9.4, N) + rng.normal(0.0, 0.20, N)
    mkt = np.cumsum(rng.normal(0.0, 0.02, T)); mkt -= mkt.mean()
    idio = np.zeros((T, N)); e = rng.normal(0.0, 0.25, (T, N))
    for t in range(1, T): idio[t] = 0.6 * idio[t - 1] + e[t]
    volume = np.exp(base[None, :] + 0.25 * mkt[:, None] + idio)
    # late listings: no observation before late_start
    late = list(range(N - spec.n_late, N)) if spec.n_late > 0 else []
    mask_late = dates < pd.Timestamp(spec.late_start)
    prices_obs = prices.copy(); volume_obs = volume.copy()
    for i in late:
        prices_obs[mask_late, i] = np.nan; volume_obs[mask_late, i] = np.nan
    truth = dict(mu_annual=mu.tolist(), sigma_annual=sig.tolist(), corr_calm=corr_calm.tolist(), corr_stress=corr_stress.tolist(),
                 stress_vol_mult=spec.stress_vol_mult, stress_drift_shift=spec.stress_drift_shift, late_tickers=[tickers[i] for i in late],
                 vol_mult_daily=vol_mult.tolist(), cycle_log_daily=cycle.tolist())
    return dict(dates=dates, tickers=tickers, prices=pd.DataFrame(prices_obs, index=dates, columns=tickers),
                volume=pd.DataFrame(volume_obs, index=dates, columns=tickers), state=pd.Series(state, index=dates),
                truth=truth, sig=sig, mu=mu, corr_calm=corr_calm, corr_stress=corr_stress, vol_mult=vol_mult, drift_shift=drift_shift)


def true_daily_cov(m: dict, day_index: int) -> np.ndarray:
    """True conditional daily covariance on a given day (regime-dependent)."""
    s = m["sig"] * m["vol_mult"][day_index]
    c = m["corr_stress"] if m["state"].iloc[day_index] == 1 else m["corr_calm"]
    return np.outer(s, s) * c / TRADING_DAYS


def engine_inputs(m: dict) -> dict:
    """Frames in the exact shape run.model_computation() consumes."""
    dates = m["dates"]; prices = m["prices"]
    logret = np.log(prices / prices.shift(1))                           # NaN on the first day and before listing
    returns_all = logret.iloc[1:].copy()
    ewma = returns_all.fillna(0.0).ewm(halflife=EWMA_HALFLIFE, adjust=False).mean()
    returns_all = returns_all.fillna(0.0)                               # the production loader fills NaN with 0
    returns_all.insert(0, "Date", returns_all.index); returns_all = returns_all.reset_index(drop=True)
    ewma.insert(0, "Date", ewma.index); ewma = ewma.reset_index(drop=True)
    mc = (prices * m["volume"]).copy()
    mc.insert(0, "Date", mc.index); mc = mc.reset_index(drop=True)
    fbd = pd.Series(dates.to_series().groupby([dates.year, dates.month]).min().values, name="Values")
    fbd = pd.Series(pd.to_datetime(fbd.values), name="Values")          # RangeIndex, like business_dates.xlsx
    return dict(returns_all=returns_all, ewma_returns_all=ewma, market_cap=mc, first_business_days=fbd)


def eligibility(m: dict, spec: MarketSpec, first_business_days: pd.Series) -> tuple[dict, pd.DataFrame]:
    """portfolios dict {date_str: DataFrame[Key, Value]} under the paper's rule, plus an audit frame."""
    prices, volume, dates = m["prices"], m["volume"], m["dates"]
    traded = (prices * volume)                                          # daily traded value (open ~ close)
    monthly = traded.groupby([dates.year, dates.month]).sum(min_count=1)
    monthly.index = [pd.Timestamp(y, mo, 1) for (y, mo) in monthly.index]
    obs = prices.notna(); cum_obs = obs.cumsum()
    first_obs = prices.apply(lambda s: s.first_valid_index())
    exempt = set(first_obs[first_obs == dates.min()].index)
    portfolios = {}; audit = []
    for d in first_business_days.iloc[1:]:
        m0 = pd.Timestamp(d.year, d.month, 1)
        hist = monthly.loc[monthly.index < m0].tail(36)                # months m-36 ... m-1
        share = hist.sum(axis=0, min_count=1) / float(np.nansum(hist.values))
        liquid = share > spec.share_threshold
        recent = prices.loc[dates < d].tail(spec.recent_days).notna().any(axis=0)
        before = cum_obs.loc[dates < d]
        cnt = before.iloc[-1] if not before.empty else pd.Series(0, index=prices.columns)
        history_ok = pd.Series([(t in exempt) or int(cnt[t]) >= spec.history_days for t in prices.columns], index=prices.columns)
        elig = (liquid & recent & history_ok).astype(int)
        portfolios[d.strftime("%Y-%m-%d")] = pd.DataFrame({"Key": list(prices.columns), "Value": elig.values})
        audit.append({"date": d, "liquid": int(liquid.sum()), "recent": int(recent.sum()), "history_ok": int(history_ok.sum()), "eligible": int(elig.sum())})
    return portfolios, pd.DataFrame(audit)


def write_inputs(m: dict, spec: MarketSpec, out: Path, audit: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    m["prices"].to_csv(out / "synthetic_close.csv"); m["volume"].to_csv(out / "synthetic_volume.csv")
    logret = np.log(m["prices"] / m["prices"].shift(1)).iloc[1:]
    daily = logret.copy(); daily.insert(0, "Date", daily.index); daily.reset_index(drop=True).to_csv(out / "synthetic_daily_returns.csv")
    m["state"].rename("state").to_csv(out / "regime_state.csv")
    (out / "truth.json").write_text(json.dumps({**m["truth"], "spec": asdict(spec)}, indent=2), encoding="utf-8")
    audit.to_csv(out / "eligibility_audit.csv", index=False)
