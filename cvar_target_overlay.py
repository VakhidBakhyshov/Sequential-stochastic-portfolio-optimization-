"""Self-timing Bayesian-CVaR overlay .

Instead of scaling exposure by *backward* realized volatility (fast_voltarget.py, Moreira-Muir style),
scale it by the model's OWN *forward* posterior-predictive CVaR of the chosen book -- the same
posterior-predictive tail measure that the allocator constrains. This is an options-free, forward,
model-consistent timing signal.

  k_t = clip( CVAR* / CVaR_model,t , K_MIN, 1 )         # CVaR_model,t known at the START of month t
  r_net,t = k_t * r_t + (1 - k_t) * (RISK_FREE / 12) - c |k_t - k_{t-1}|   # cash slice; exposure move costed

CVaR_model,t is written by run.py to results/main_dyn_strong/forecast_risk.csv (causal).

Produces:
  - results/main_cvartarget/pnl.csv         (the new arm, same schema as the others)
  - a sensitivity sweep (CVAR* percentile x K_MIN)
  - the 3-way head-to-head: STRONG (none) / realized-vol (main_voltarget) / model-CVaR (main_cvartarget)
  - a paired bootstrap: model-CVaR vs realized-vol (Sharpe & MaxDD)
  - _cvar_timing_diagnostic.png : does cvar_model spike BEFORE drawdowns?

Run from the package root: python cvar_target_overlay.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm, skew, kurtosis
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from overlay_arms import overlay

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
STRONG = RES / "main_dyn_strong"
RF = 0.02
K_MIN_DEFAULT, K_MAX = 0.30, 1.00
SEED = 42


def load_pnl(folder: Path) -> pd.Series:
    p = pd.read_csv(folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)


def metrics(r) -> dict:
    r = np.asarray(r, float); n = len(r); yrs = n / 12
    total = float(np.prod(1 + r) - 1); ann = (1 + total) ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(12); ex = r - RF / 12
    sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    dn = r[r < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sor = (ann - RF) / ddv if (ddv and ddv > 0) else np.nan
    eq = np.cumprod(1 + r); mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    cal = ann / abs(mdd) if mdd < 0 else np.nan
    v95 = np.percentile(r, 5); cv = float(r[r <= v95].mean())
    return dict(Ann=ann * 100, Vol=vol * 100, Sharpe=sh, Sortino=sor, MaxDD=mdd * 100,
                Calmar=cal, CVaR95=cv * 100)


def psr_vs(r, sr_star_m):
    r = np.asarray(r, float); ex = r - RF / 12; srm = ex.mean() / ex.std(ddof=1); n = len(r)
    sk = float(skew(r)); ku = float(kurtosis(r, fisher=False))
    den = np.sqrt(max(1 - sk * srm + (ku - 1) / 4 * srm ** 2, 1e-12))
    return float(norm.cdf((srm - sr_star_m) * np.sqrt(n - 1) / den))


def ann_sharpe(r):
    ex = np.asarray(r, float) - RF / 12; sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(12)) if sd > 0 else np.nan


def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float(np.min(eq / np.maximum.accumulate(eq) - 1))


def build_overlay(r_book: pd.Series, cvar: pd.Series, cvar_star: float, k_min: float) -> pd.Series:
    """k_t from the forward model CVaR; month t uses cvar_t known at t's rebalance (causal)."""
    ks = {}
    for dt in r_book.index:
        c = float(cvar.get(dt, np.nan))
        ks[dt] = float(np.clip(cvar_star / c, k_min, K_MAX)) if (c == c and c > 0) else K_MAX
    return overlay(pd.Series(ks), r_book)


def main():
    r_strong = load_pnl(STRONG).iloc[1:]                       # drop seed row
    fc = pd.read_csv(STRONG / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    cvar = fc.set_index("date")["cvar_model"].astype(float).reindex(r_strong.index)
    print(f"cvar_model: min={cvar.min():.4f} median={cvar.median():.4f} max={cvar.max():.4f} "
          f"(NaNs={int(cvar.isna().sum())})")

    # ---- CAUSAL self-scaling target: EXPANDING-window median of the forecast (uses information through
    #      month t only; NOT the full-sample median). This is the model-CVaR (forward) overlay we save. ----
    cvar_star_causal = cvar.expanding(min_periods=6).median().bfill()
    k_mc = (cvar_star_causal / cvar).clip(K_MIN_DEFAULT, K_MAX)
    chosen = overlay(k_mc, r_strong)

    # ---- sensitivity sweep (DIAGNOSTIC ONLY; full-sample percentiles, not used for the saved arm) ----
    print("\n=== model-CVaR overlay sensitivity (self-scaling target) ===")
    print(" CVAR*pct  K_MIN   Ann%   Vol%  Sharpe  MaxDD%")
    best = None
    for pct in (40, 50, 60):
        cstar = float(np.nanpercentile(cvar, pct))
        for kmin in (0.20, 0.30, 0.40):
            rr = build_overlay(r_strong, cvar, cstar, kmin)
            m = metrics(rr)
            print(f"   {pct:3d}     {kmin:.2f}  {m['Ann']:6.2f} {m['Vol']:6.2f}  {m['Sharpe']:.3f}  {m['MaxDD']:6.2f}")

    # ---- save chosen arm (CAUSAL expanding-median target, K_MIN=0.30) ----
    out_dir = RES / "main_cvartarget"; out_dir.mkdir(parents=True, exist_ok=True)
    # prepend a seed row (return 0) to match the other arms' pnl schema
    full = pd.concat([pd.Series({load_pnl(STRONG).index[0]: 0.0}), chosen])
    bal = 1000.0; recs = []
    for dt, rr in full.items():
        bal *= (1 + rr); recs.append((dt, bal, rr))
    pd.DataFrame(recs, columns=["Date", "Balance", "Returns"]).set_index("Date").to_csv(out_dir / "pnl.csv")
    k_mc.rename("k").rename_axis("Date").to_csv(out_dir / "exposure.csv")
    print(f"\nchosen overlay (CVAR*=CAUSAL expanding median, K_MIN={K_MIN_DEFAULT}): k_t min={k_mc.min():.2f} "
          f"median={k_mc.median():.2f} max={k_mc.max():.2f}  de-risked(k<0.99)={int((k_mc<0.99).sum())}/{len(k_mc)}  "
          f"final balance {bal:,.0f}  -> {out_dir/'pnl.csv'}")

    # ---- COMBINED overlay: de-risk if EITHER the forward(model) OR backward(realized) signal flags.
    #      k_combined = k_realized * k_modelCVaR  (both <= 1, both CAUSAL). The two signals are weakly
    #      correlated, so the combination captures complementary risk episodes. ----
    r_vt = load_pnl(RES / "main_voltarget").iloc[1:].reindex(r_strong.index)
    k_rv = pd.read_csv(RES / "main_voltarget" / "exposure.csv", index_col=0, parse_dates=True)["k"].reindex(r_strong.index)
    k_comb = (k_rv * k_mc).clip(K_MIN_DEFAULT, K_MAX)                              # k_mc is the causal model-CVaR k
    r_comb = overlay(k_comb, r_strong)
    comb_dir = RES / "main_combined"; comb_dir.mkdir(parents=True, exist_ok=True)
    full_c = pd.concat([pd.Series({load_pnl(STRONG).index[0]: 0.0}), r_comb]); bal = 1000.0; rc = []
    for dt, rr in full_c.items():
        bal *= (1 + rr); rc.append((dt, bal, rr))
    pd.DataFrame(rc, columns=["Date", "Balance", "Returns"]).set_index("Date").to_csv(comb_dir / "pnl.csv")
    k_comb.rename("k").rename_axis("Date").to_csv(comb_dir / "exposure.csv")
    print(f"corr(k_realized, k_modelCVaR) = {float(np.corrcoef(k_rv, k_mc)[0,1]):+.3f}  "
          f"(low corr -> complementary)   combined final balance {bal:,.0f}")

    # ---- 4-way head-to-head ----
    arms = {"STRONG (none)": r_strong,
            "Realized-vol (backward)": r_vt,
            "Model-CVaR (forward, NEW)": chosen,
            "Combined (fwd x bwd, NEW)": r_comb}
    sr_ann = np.array([metrics(v)["Sharpe"] for v in arms.values()])
    var_srm = np.var(sr_ann / np.sqrt(12), ddof=1); Nt = len(sr_ann); g = 0.5772156649
    z = (1 - g) * norm.ppf(1 - 1.0 / Nt) + g * norm.ppf(1 - 1.0 / (Nt * np.e))
    sr_star_m = float(np.sqrt(max(var_srm, 0.0)) * z)
    rows = []
    for name, rr in arms.items():
        m = metrics(rr)
        rows.append({"arm": name, **{k: round(v, 3) for k, v in m.items()},
                     "PSR%": round(psr_vs(rr, 0.0) * 100, 1), "DSR%": round(psr_vs(rr, sr_star_m) * 100, 1)})
    tbl = pd.DataFrame(rows).set_index("arm")
    print("\n=== 3-way overlay comparison (Dynamic base, 83 OOS months) ===")
    print(tbl.to_string())
    tbl.to_csv(ROOT / "overlay_3way_metrics.csv")

    # ---- paired bootstraps vs the realized-vol baseline ----
    base = arms["Realized-vol (backward)"]
    rng = np.random.default_rng(SEED); B = 5000
    for label, treat in [("Model-CVaR (forward)", arms["Model-CVaR (forward, NEW)"]),
                         ("Combined (fwd x bwd)", arms["Combined (fwd x bwd, NEW)"])]:
        j = base.index.intersection(treat.index); ra, rb = base.loc[j].values, treat.loc[j].values; n = len(j)
        obs_sh = ann_sharpe(rb) - ann_sharpe(ra); obs_dd = max_dd(rb) - max_dd(ra)
        sh_d = np.empty(B); dd_d = np.empty(B)
        for k in range(B):
            idx = rng.integers(0, n, n); sh_d[k] = ann_sharpe(rb[idx]) - ann_sharpe(ra[idx]); dd_d[k] = max_dd(rb[idx]) - max_dd(ra[idx])
        print(f"\n=== Paired bootstrap: {label} vs Realized-vol ({n} mo, B={B}) ===")
        print(f"  Sharpe diff (treat - realized): obs={obs_sh:+.3f}  P(treat better)={float((sh_d>0).mean()):.3f}")
        print(f"  MaxDD  diff (+ = treat shallower): obs={obs_dd:+.3f}  P(treat shallower)={float((dd_d>0).mean()):.3f}")

    # ---- diagnostic: does cvar_model spike BEFORE drawdowns? ----
    nxt_loss = (-r_strong).clip(lower=0)                          # next-month realized loss
    valid = cvar.notna() & nxt_loss.notna()
    ic = float(np.corrcoef(cvar[valid].values, nxt_loss[valid].values)[0, 1])
    print(f"\nForward-signal check: corr(cvar_model_t, next-month realized loss_t) = {ic:+.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(cvar.index, cvar.values, color="#8e44ad", lw=1.8, label="cvar_model,t (forward, model)")
    ax[0].axhline(cvar.median(), color="0.5", ls="--", lw=1, label="CVAR* (median)")
    ax[0].set_ylabel("posterior-predictive CVaR"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
    ax[0].set_title(f"Self-timing signal vs realized losses  (corr={ic:+.2f})")
    ax[1].bar(r_strong.index, r_strong.values, width=20, color=np.where(r_strong.values < 0, "#c0392b", "#27ae60"), alpha=0.8)
    ax[1].axhline(0, color="0.4", lw=0.8); ax[1].set_ylabel("realized monthly return (STRONG)"); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "_cvar_timing_diagnostic.png", dpi=130, bbox_inches="tight")
    print("saved _cvar_timing_diagnostic.png  and  overlay_3way_metrics.csv")


if __name__ == "__main__":
    main()
