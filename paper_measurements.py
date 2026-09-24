"""Final measurements for the paper's Results section.

(a) dial correlation in stress subsamples (H2 pre-commitment)
(b) fraction of dates with beta* = beta_t (H5 pre-commitment)
(c) shuffled-k placebo: distribution-matched control (H4, Remark 7)
(d) overlay turnover |dk| per dial + break-even incremental cost (H7)
(e) exposure-paths figure in house style
(f) Prop 5(b) check: predicted mechanical MDD = kbar * MDD(1) vs measured constant-k MDD
(g) sign violations of the uncentred signal in our pipeline (H5)
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import report_style as rs; rs.apply()

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "main_dyn_strong"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RF = 0.02; KMIN, KMAX = 0.30, 1.00; SPAN, TV = 21, 0.12; SEED = 42


def load_pnl():
    p = pd.read_csv(RES / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)


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


def ann_sharpe(r):
    r = np.asarray(r, float); ex = r - RF / 12; return ex.mean() / ex.std(ddof=1) * np.sqrt(12)


def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    df = pd.read_csv(DAILY)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    DLY = np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0

    W = load_weights(); r_full = load_pnl(); idx = r_full.index; rs_ = r_full.iloc[1:]
    reb = list(W.keys()); parts = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]; w = W[d0]; cc = [t for t in w.index if t in DLY.columns]
        win = DLY.loc[(DLY.index > d0) & (DLY.index <= d1), cc].fillna(0.0)
        if not win.empty: parts.append(pd.Series(win.values @ w.reindex(cc).values, index=win.index))
    book = pd.concat(parts).sort_index()
    va = book.rolling(SPAN, min_periods=10).std() * np.sqrt(252)
    k_rv = pd.Series({idx[j]: (float(np.clip(TV / va.asof(idx[j - 1]), KMIN, KMAX))
                      if (va.asof(idx[j - 1]) == va.asof(idx[j - 1]) and va.asof(idx[j - 1]) > 0) else KMAX)
                      for j in range(1, len(idx))}).reindex(rs_.index)
    fc = pd.read_csv(RES / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    cv = fc.set_index("date")["cvar_model"].astype(float).reindex(rs_.index)
    k_mc = (cv.expanding(min_periods=6).median().bfill() / cv).clip(KMIN, KMAX)
    k_cb = (k_rv * k_mc).clip(KMIN, KMAX)

    print("=" * 96)
    print("(a) DIAL CORRELATION corr(k_fast, k_slow) BY SUBSAMPLE  [H2]")
    print("=" * 96)
    K = pd.DataFrame({"rv": k_rv, "mc": k_mc}).dropna()
    print(f"  full sample (n={len(K)})            : {K['rv'].corr(K['mc']):+.3f}")
    for y in [2020, 2022]:
        Ky = K[K.index.year == y]
        print(f"  {y} (n={len(Ky)})                    : {Ky['rv'].corr(Ky['mc']):+.3f}")
    stress = K[(K.index.year == 2020) | (K.index.year == 2022)]
    calm = K[~((K.index.year == 2020) | (K.index.year == 2022))]
    print(f"  stress pooled 2020+2022 (n={len(stress)}) : {stress['rv'].corr(stress['mc']):+.3f}")
    print(f"  calm remainder (n={len(calm)})           : {calm['rv'].corr(calm['mc']):+.3f}")

    print(); print("=" * 96)
    print("(b) FRACTION OF DATES WITH beta* = beta_t  [H5]  (beta*=0.95 fixed; beta_t adaptive)")
    print("=" * 96)
    h = pd.read_csv(RES / "dynamic_parameter_history.csv"); h["date"] = pd.to_datetime(h["date"])
    bt = h.set_index("date")["confidence_level"].astype(float)
    frac_eq = float((np.abs(bt - 0.95) < 0.005).mean())
    print(f"  months: {len(bt)} | beta_t range [{bt.min():.3f}, {bt.max():.3f}]")
    print(f"  fraction with |beta_t - 0.95| < 0.005 : {frac_eq:.3f}")
    print(f"  fraction beta_t > 0.95 (deeper than beta*): {(bt > 0.955).mean():.3f} | "
          f"beta_t < 0.95: {(bt < 0.945).mean():.3f}")

    print(); print("=" * 96)
    print("(c) SHUFFLED-k PLACEBO  [H4]  (distribution-matched, timing destroyed; 2000 permutations)")
    print("=" * 96)
    rng = np.random.default_rng(SEED); P = 2000
    kv = k_cb.values; rv_arr = rs_.values
    r_dyn = kv * rv_arr + (1 - kv) * RF / 12
    dyn_dd, dyn_sh = max_dd(r_dyn), ann_sharpe(r_dyn)
    dds = np.empty(P); shs = np.empty(P)
    for p in range(P):
        kp = rng.permutation(kv)
        rp = kp * rv_arr + (1 - kp) * RF / 12
        dds[p] = max_dd(rp); shs[p] = ann_sharpe(rp)
    print(f"  dynamic:              MaxDD {dyn_dd*100:6.2f}%   Sharpe {dyn_sh:.3f}")
    print(f"  placebo distribution: MaxDD mean {dds.mean()*100:6.2f}% [5th {np.percentile(dds,5)*100:.2f}, 95th {np.percentile(dds,95)*100:.2f}]"
          f"   Sharpe mean {shs.mean():.3f}")
    print(f"  P(dynamic MDD shallower than placebo) = {float((dyn_dd > dds).mean()):.3f}")
    print(f"  P(dynamic Sharpe > placebo)           = {float((dyn_sh > shs).mean()):.3f}")

    print(); print("=" * 96)
    print("(d) OVERLAY TURNOVER AND BREAK-EVEN COST  [H7]")
    print("=" * 96)
    for nm, k in [("fast (k_rv)", k_rv), ("slow (k_mc)", k_mc), ("combined", k_cb)]:
        dk = k.diff().abs().dropna()
        print(f"  {nm:14s} mean|dk| = {dk.mean():.4f}/mo   annual one-way overlay turnover = {dk.mean()*12:.2f}x")
    dkc = k_cb.diff().abs().fillna(0).values
    base_sh = ann_sharpe(rv_arr)
    lo, hi = 0.0, 1.0
    def sh_net(bps):
        return ann_sharpe(kv * rv_arr + (1 - kv) * RF / 12 - bps * dkc)
    if sh_net(0) > base_sh:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if sh_net(mid) > base_sh: lo = mid
            else: hi = mid
        print(f"  break-even INCREMENTAL overlay cost vs unmanaged base: {0.5*(lo+hi)*1e4:.0f} bps per unit turnover")
    else:
        print("  combined does not exceed base Sharpe at zero incremental cost")

    print(); print("=" * 96)
    print("(f) PROP 5(b) CHECK: predicted mechanical MDD = kbar x MDD(1)")
    print("=" * 96)
    kbar = float(k_cb.mean()); mdd1 = max_dd(rv_arr)
    r_const = kbar * rv_arr + (1 - kbar) * RF / 12
    print(f"  kbar = {kbar:.3f}   MDD(1) = {mdd1*100:.2f}%")
    print(f"  predicted mechanical MDD = kbar*MDD(1) = {kbar*mdd1*100:.2f}%")
    print(f"  measured constant-k MDD                = {max_dd(r_const)*100:.2f}%")
    print(f"  dynamic MDD                            = {dyn_dd*100:.2f}%")

    print(); print("=" * 96)
    print("(g) SIGN VIOLATIONS OF THE UNCENTRED SIGNAL (our one-day-scale pipeline)  [H5]")
    print("=" * 96)
    print(f"  months with cvar_model <= 0 : {int((cv <= 0).sum())} of {cv.notna().sum()}")
    print(f"  cvar_model: mean {cv.mean():.4f}  min {cv.min():.4f}")

    # (e) exposure paths figure
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(k_rv.index, k_rv.values, color=rs.BLUE, lw=1.6, label="fast dial $k^{F}$ (21-day realized vol)")
    ax.plot(k_mc.index, k_mc.values, color=rs.ORANGE, lw=1.6, label="slow dial $k^{S}$ (model-implied)")
    ax.plot(k_cb.index, k_cb.values, color=rs.NAVY, lw=2.2, label="combined $k = k^{F}k^{S}$")
    ax.fill_between(k_cb.index, KMIN, k_cb.values, color=rs.NAVY, alpha=0.06)
    ax.axhline(1.0, color=rs.GRID, lw=1); ax.axhline(KMIN, color=rs.GRID, lw=1, ls="--")
    ax.set_ylabel("risky exposure $k_t$"); ax.set_ylim(0.25, 1.05); ax.grid(axis="y")
    ax.legend(loc="lower left", fontsize=8.5, ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    rs.title(ax, "The two exposure dials and their combination",
             "Fast and slow dials fire in different episodes; the product de-risks when either fires")
    fig.tight_layout(); fig.savefig(ROOT / "fig_exposure_paths.png", dpi=150, bbox_inches="tight")
    print("\nsaved fig_exposure_paths.png")


if __name__ == "__main__":
    main()


def subperiod_table():
    """(h) Sub-period robustness: split at 2022-07; Sharpe and within-half MDD per arm."""
    import numpy as np, pandas as pd
    from pathlib import Path
    RES = Path(__file__).resolve().parent / "results"; RF = 0.02
    def lp(f):
        p = pd.read_csv(RES / f / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
        return p.set_index(c)["Returns"].astype(float).iloc[1:]
    def met(r):
        rv = np.asarray(r.dropna(), float); ex = rv - RF / 12
        sh = ex.mean() / ex.std(ddof=1) * (12 ** 0.5)
        eq = np.cumprod(1 + rv); mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        return sh, mdd * 100
    split = pd.Timestamp("2022-07-01")
    print("(h) SUB-PERIOD ROBUSTNESS (split 2022-07; MDD within each half)")
    for k, f in [("Dynamic base", "main_dyn_strong"), ("Fast dial", "main_voltarget"),
                 ("HDRC combined", "main_combined")]:
        r = lp(f)
        s1, m1 = met(r[r.index < split]); s2, m2 = met(r[r.index >= split])
        print(f"  {k:14s} H1 {s1:.3f}/{m1:.2f}%   H2 {s2:.3f}/{m2:.2f}%")


if __name__ == "__main__" and "--subperiod" in __import__("sys").argv:
    subperiod_table()
