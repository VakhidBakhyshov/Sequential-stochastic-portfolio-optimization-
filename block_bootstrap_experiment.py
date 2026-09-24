"""Block-bootstrap experiment — can within-month volatility clustering preserve tail shape at H=21?

Motivation. The FHS experiment showed genuine tail-shape variation at H=1 (c = 2.42 +/- 0.15) but
~75% of it vanished at H=21, because 21 INDEPENDENT days sum toward Gaussian by the CLT. Real months
are fat-tailed precisely because bad days CLUSTER: a month is not a sum of iid days. This experiment
replaces iid day-resampling with contiguous 21-day block resampling, which preserves within-month
volatility clustering and crash momentum, and asks whether the monthly shape then (i) varies over
time and (ii) predicts anything beyond volatility.

Variants at H = 21 (all monthly scenarios, all causal):
  gauss      : sum of 21 iid N(0, Sigma_hat) draws              [control; theory says c = const]
  fhs_iid    : sum of 21 iid resampled standardized days        [previous result]
  fhs_block  : ONE contiguous 21-day block of standardized days [residual clustering preserved]
  raw_block  : ONE contiguous 21-day block of RAW returns       [full vol clustering preserved]

c = CVaR95 / sigma is computed on de-meaned scenario portfolio returns, so it is pure SHAPE and is
comparable across variants regardless of level.

Run from the package root: python block_bootstrap_experiment.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "main_dyn_strong"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
S = 30000; H = 21; LB = 756; EWMA_LAMBDA = 0.94; SEED = 42
MIN_OBS = 650
GAUSS_C = 2.0627


def cvar_over_sigma(x):
    x = x - x.mean(); sd = x.std(ddof=1)
    loss = -x; v = np.quantile(loss, 0.95); tail = loss[loss >= v]
    return float(tail.mean() / sd) if sd > 0 else np.nan


def load_weights():
    xl = pd.ExcelFile(RES / "weights.xlsx"); out = {}
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        if "weights" not in d.columns: continue
        w = pd.to_numeric(d["weights"], errors="coerce").fillna(0.0)
        s = pd.Series(w.values, index=d[d.columns[0]].astype(str).values); s = s[s > 1e-9]
        if not s.empty:
            try: out[pd.to_datetime(sh)] = s / s.sum()
            except Exception: pass
    return dict(sorted(out.items()))


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DAILY)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    R = df.apply(pd.to_numeric, errors="coerce")

    W = load_weights()
    dates = [d for d in W.keys() if d >= pd.Timestamp("2019-01-01")]
    all_dates = sorted(W.keys())

    rows = []
    for t in dates:
        w_ser = W[t]
        tick = [c for c in w_ser.index if c in R.columns]
        win_all = R.loc[R.index < t, tick].tail(LB)
        ok = [c for c in tick
              if (win_all[c].fillna(0.0) != 0.0).sum() >= MIN_OBS and win_all[c].std() > 1e-8]
        if len(ok) < 3: continue
        X = win_all[ok].fillna(0.0).values
        w = w_ser.reindex(ok).fillna(0.0).values
        if w.sum() <= 0: continue
        w = w / w.sum()
        T, N = X.shape
        full_std = X.std(axis=0, ddof=1)

        # causal EWMA devolatilisation with a hard floor (same as FHS v2)
        sig2 = np.zeros_like(X); sig2[0] = np.var(X[:60], axis=0) + 1e-10
        for d in range(1, T):
            sig2[d] = EWMA_LAMBDA * sig2[d - 1] + (1 - EWMA_LAMBDA) * X[d - 1] ** 2
        sig = np.maximum(np.sqrt(sig2), 0.3 * full_std[None, :])
        Z = X / sig
        sig_now = sig[-1]

        nstart = T - H                      # valid contiguous block starts
        starts = rng.integers(0, nstart, S)
        blk = starts[:, None] + np.arange(H)[None, :]        # (S, H) index matrix

        # --- variants ---
        idxH = rng.integers(0, T, (S, H))
        scen = {}
        scen["fhs_iid"]   = (Z[idxH] * sig_now[None, None, :]).sum(axis=1)
        scen["fhs_block"] = (Z[blk] * sig_now[None, None, :]).sum(axis=1)
        scen["raw_block"] = X[blk].sum(axis=1)

        C1 = np.cov(scen["fhs_iid"] / np.sqrt(H), rowvar=False)   # per-day cov implied by iid variant
        L = np.linalg.cholesky(C1 + 1e-12 * np.eye(N))
        scen["gauss"] = (rng.standard_normal((S, H, N)) @ L.T).sum(axis=1)

        # concentrated + equal-weight comparison portfolios
        top3 = np.argsort(w)[-3:]; w_c = np.zeros(N); w_c[top3] = 1 / 3
        w_eq = np.ones(N) / N

        rec = {"date": t, "n_assets": N}
        for vname, sc in scen.items():
            rec[f"c_{vname}"] = cvar_over_sigma(sc @ w)
            rec[f"ceq_{vname}"] = cvar_over_sigma(sc @ w_eq)
            rec[f"cconc_{vname}"] = cvar_over_sigma(sc @ w_c)
        # conditional vol forecast (annualised) from the iid FHS variant, as the control
        rec["sigma_held"] = float((scen["fhs_iid"] @ w).std(ddof=1) / np.sqrt(H) * np.sqrt(252))

        # --- realised next-month outcomes ---
        pos = all_dates.index(t)
        if pos + 1 < len(all_dates):
            t1 = all_dates[pos + 1]
            nxt = R.loc[(R.index >= t) & (R.index < t1), ok].fillna(0.0).values
            if nxt.shape[0] >= 10:
                pr = nxt @ w
                rec["real_vol"] = float(pr.std(ddof=1) * np.sqrt(252))
                rec["real_worst"] = float(-pr.min())
                rec["real_shape"] = float(-pr.min() / max(pr.std(ddof=1), 1e-12))
        rows.append(rec)

    D = pd.DataFrame(rows).set_index("date")
    D.to_csv(ROOT / "block_bootstrap_results.csv")

    mc = 2.0 / np.sqrt(S * 0.05)   # rough MC sd of a 5%-tail mean ratio
    print("=" * 100)
    print(f"STAGE 1 - monthly (H={H}) shape ratio c = CVaR95/sigma.  Gaussian theory = {GAUSS_C:.3f}")
    print(f"n={len(D)} dates, S={S}, MC noise floor on c ~ {mc:.3f}")
    print("=" * 100)
    print(f"{'variant':14s}{'mean':>8s}{'sd(time)':>10s}{'min':>8s}{'max':>8s}{'x-port spread':>15s}")
    for v in ["gauss", "fhs_iid", "fhs_block", "raw_block"]:
        s = D[f"c_{v}"].dropna()
        spread = (D[[f"c_{v}", f"ceq_{v}", f"cconc_{v}"]].max(axis=1)
                  - D[[f"c_{v}", f"ceq_{v}", f"cconc_{v}"]].min(axis=1)).mean()
        print(f"  {v:12s}{s.mean():8.3f}{s.std():10.3f}{s.min():8.3f}{s.max():8.3f}{spread:15.3f}")
    print()
    print("time-variation gain vs iid:  fhs_block %.2fx | raw_block %.2fx"
          % (D['c_fhs_block'].std() / D['c_fhs_iid'].std(),
             D['c_raw_block'].std() / D['c_fhs_iid'].std()))
    print("stress-window c (2020-03..2020-07), raw_block:",
          D.loc['2020-03':'2020-07', 'c_raw_block'].round(3).tolist())
    print("top-5 fattest raw_block dates:", D['c_raw_block'].nlargest(5).index.strftime('%Y-%m').tolist())

    print()
    print("=" * 100)
    print("STAGE 2 - does monthly shape predict next-month outcomes BEYOND volatility?")
    print("=" * 100)
    M = D.dropna(subset=["real_vol", "real_shape"])
    n = len(M); thr = 1.96 / np.sqrt(n - 3)

    def partial(y, x, ctrl):
        rx = x - np.polyval(np.polyfit(ctrl, x, 1), ctrl)
        ry = y - np.polyval(np.polyfit(ctrl, y, 1), ctrl)
        return float(np.corrcoef(rx, ry)[0, 1])

    print(f"n={n}; |r| needed for ~95% significance = {thr:.3f}")
    print(f"(control: corr(sigma_model, real_vol) = {np.corrcoef(M['sigma_held'], M['real_vol'])[0,1]:+.3f})")
    print()
    print(f"{'variant':12s}{'target':12s}{'corr':>8s}{'partial|sigma':>15s}{'   verdict'}")
    any_hit = False
    for v in ["fhs_iid", "fhs_block", "raw_block"]:
        for tgt in ["real_vol", "real_worst", "real_shape"]:
            r_c = float(np.corrcoef(M[f"c_{v}"], M[tgt])[0, 1])
            r_p = partial(M[tgt].values, M[f"c_{v}"].values, M["sigma_held"].values)
            hit = abs(r_p) > thr
            any_hit = any_hit or hit
            print(f"  {v:10s}{tgt:12s}{r_c:+8.3f}{r_p:+15.3f}{'   <-- SIGNIFICANT' if hit else ''}")
    print()
    print("VERDICT:", "at least one significant incremental signal" if any_hit
          else "NO incremental predictive content beyond volatility for any variant")
    print("\nsaved block_bootstrap_results.csv")


if __name__ == "__main__":
    main()
