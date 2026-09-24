"""FHS experiment v2 — does tail DEPENDENCE create a genuine endogenous tail signal?

v2 fixes: strict validity filter (>=650/756 non-zero obs per asset), per-asset sigma floor
(0.3 x window std) so the EWMA filter cannot collapse on quiet/zero stretches, structured test
portfolios (defensive 1/vol vs cyclical vol tilts), and a sharper Stage-2 target built from
next-month DAILY book returns (realized level AND realized tail shape).
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "main_dyn_strong"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
S = 40000; H = 21; LB = 756; EWMA_LAMBDA = 0.94; SEED = 42
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

        # causal EWMA devolatilisation with a hard floor
        sig2 = np.zeros_like(X); sig2[0] = np.var(X[:60], axis=0) + 1e-10
        for d in range(1, T):
            sig2[d] = EWMA_LAMBDA * sig2[d-1] + (1 - EWMA_LAMBDA) * X[d-1]**2
        sig = np.maximum(np.sqrt(sig2), 0.3 * full_std[None, :])
        Z = X / sig
        sig_now = sig[-1]

        idx1 = rng.integers(0, T, S)
        scen_f1 = Z[idx1] * sig_now[None, :]
        idxH = rng.integers(0, T, (S, H))
        scen_fH = (Z[idxH] * sig_now[None, None, :]).sum(axis=1)

        C1 = np.cov(scen_f1, rowvar=False)
        L = np.linalg.cholesky(C1 + 1e-12 * np.eye(N))
        scen_g1 = rng.standard_normal((S, N)) @ L.T

        v = full_std.copy()
        w_def = (1 / v); w_def /= w_def.sum()          # defensive tilt
        w_cyc = v / v.sum()                             # cyclical tilt
        top3 = np.argsort(w)[-3:]; w_c = np.zeros(N); w_c[top3] = 1/3
        rec = {"date": t, "n_assets": N}
        for pname, pw in [("held", w), ("def", w_def), ("cyc", w_cyc), ("conc", w_c)]:
            rec[f"c_f1_{pname}"] = cvar_over_sigma(scen_f1 @ pw)
            rec[f"c_fH_{pname}"] = cvar_over_sigma(scen_fH @ pw)
        rec["c_g1_held"] = cvar_over_sigma(scen_g1 @ w)
        rec["sigma_held"] = float((scen_f1 @ w).std(ddof=1) * np.sqrt(252))

        # --- Stage-2 targets: next-month realized daily book returns ---
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
    D.to_csv(ROOT / "fhs_experiment_results.csv")

    print("=" * 100)
    print(f"STAGE 1 (clean) - shape ratio c = CVaR95/sigma. Gaussian constant {GAUSS_C:.3f}. n={len(D)} dates")
    print("=" * 100)
    print(f"{'':24s}{'mean':>8s}{'sd(time)':>10s}{'min':>8s}{'max':>8s}")
    for col, lab in [("c_g1_held", "Gaussian H=1 held"),
                     ("c_f1_held", "FHS H=1  held"), ("c_fH_held", "FHS H=21 held"),
                     ("c_f1_def", "FHS H=1  defensive"), ("c_f1_cyc", "FHS H=1  cyclical"),
                     ("c_f1_conc", "FHS H=1  conc-3")]:
        s = D[col].dropna()
        print(f"  {lab:22s}{s.mean():8.3f}{s.std():10.3f}{s.min():8.3f}{s.max():8.3f}")
    sp1 = D[["c_f1_held", "c_f1_def", "c_f1_cyc"]].max(axis=1) - D[["c_f1_held", "c_f1_def", "c_f1_cyc"]].min(axis=1)
    spH = D[["c_fH_held", "c_fH_def", "c_fH_cyc"]].max(axis=1) - D[["c_fH_held", "c_fH_def", "c_fH_cyc"]].min(axis=1)
    print(f"\ncross-portfolio spread within a date: FHS H=1 mean {sp1.mean():.3f} | FHS H=21 mean {spH.mean():.3f} | MC floor ~0.012")
    print("top-5 fattest-tail dates (c_f1_held):", D['c_f1_held'].nlargest(5).index.strftime('%Y-%m').tolist())
    print("2020-04..2020-06 (post-crash windows):", D.loc['2020-04':'2020-07','c_f1_held'].round(3).tolist())

    print()
    print("=" * 100)
    print("STAGE 2 - incremental prediction of next-month realized outcomes (n = usable months)")
    print("=" * 100)
    M = D.dropna(subset=["real_vol", "real_shape"])
    def partial(y, x, ctrl):
        rx = x - np.polyval(np.polyfit(ctrl, x, 1), ctrl)
        ry = y - np.polyval(np.polyfit(ctrl, y, 1), ctrl)
        return float(np.corrcoef(rx, ry)[0, 1])
    n = len(M); se = 1 / np.sqrt(n - 3)
    print(f"n={n}, |r| needed for ~95% significance ≈ {1.96*se:.3f}")
    for sig_col, c_col in [("sigma_held", "c_f1_held"), ("sigma_held", "c_fH_held")]:
        for tgt in ["real_vol", "real_worst", "real_shape"]:
            r_c = float(np.corrcoef(M[c_col], M[tgt])[0, 1])
            r_p = partial(M[tgt].values, M[c_col].values, M[sig_col].values)
            print(f"  {c_col:10s} -> {tgt:10s}: corr={r_c:+.3f}  partial|sigma={r_p:+.3f}")
        r_s = float(np.corrcoef(M[sig_col], M['real_vol'])[0, 1])
        print(f"  (control: corr(sigma_model, real_vol) = {r_s:+.3f})")
    print("\nsaved fhs_experiment_results.csv")


if __name__ == "__main__":
    main()
