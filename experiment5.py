"""Experiment 5 — causally optimize the forward/backward combination weight.

The Primary combines the two exposures with a FIXED product k = k_realized * k_modelCVaR
("de-risk if either flags"). Here we ask whether a tuned / causally-adaptive combination beats it.
Pure post-processing on saved exposures (no backtest re-run). Two cuts:

  A) STATIC robustness surface (in-sample): is the fixed product near the optimum, or cherry-picked?
     - exponent family  k = clip(k_rv^a * k_mc^b)         -> the product is (a,b)=(1,1)
     - convex blend     k = clip(w*k_rv + (1-w)*k_mc)     -> 50/50 is w=0.5
  B) CAUSAL adaptive (out-of-sample): tilt the multiplicative combination toward whichever signal has
     the better TRAILING performance: k = k_rv^(1+d_t) * k_mc^(1-d_t), d_t from trailing Sharpe diff
     (uses only past months). Compare to the fixed product.

Run from the package root: python experiment5.py
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
RF = 0.02; KMIN, KMAX = 0.30, 1.00; SEED = 42


def load_pnl(folder):
    p = pd.read_csv(RES / folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)


def metrics(r):
    r = np.asarray(r, float); n = len(r); tot = float(np.prod(1 + r) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12); ex = r - RF / 12; sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    dn = r[r < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sor = (ann - RF) / ddv if (ddv and ddv > 0) else np.nan
    eq = np.cumprod(1 + r); mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    cal = ann / abs(mdd) if mdd < 0 else np.nan; v95 = np.percentile(r, 5); cv = float(r[r <= v95].mean())
    return dict(Cumul=tot*100, Ann=ann*100, Vol=vol*100, Sharpe=sh, Sortino=sor, MaxDD=mdd*100, Calmar=cal, CVaR95=cv*100)


def psr_vs(r, s):
    r = np.asarray(r, float); ex = r - RF/12; srm = ex.mean()/ex.std(ddof=1); n = len(r)
    sk = float(skew(r)); ku = float(kurtosis(r, fisher=False)); den = np.sqrt(max(1 - sk*srm + (ku-1)/4*srm**2, 1e-12))
    return float(norm.cdf((srm - s) * np.sqrt(n - 1) / den))


def ann_sharpe(r):
    ex = np.asarray(r, float) - RF/12; sd = ex.std(ddof=1); return float(ex.mean()/sd*np.sqrt(12)) if sd > 0 else np.nan
def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float(np.min(eq/np.maximum.accumulate(eq) - 1))
def apply_k(k, r_book):
    return overlay(k, r_book)


def main():
    r_book = load_pnl("main_dyn_strong").iloc[1:]
    fc = pd.read_csv(RES / "main_dyn_strong" / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    cv = fc.set_index("date")["cvar_model"].astype(float).reindex(r_book.index)

    # realized-vol exposure (backward), the path saved by fast_voltarget.py
    k_rv = pd.read_csv(RES / "main_voltarget" / "exposure.csv", index_col=0, parse_dates=True)["k"].reindex(r_book.index)
    # model-CVaR exposure (forward); CAUSAL normalizer = expanding median of the signal's own past
    # (min 6 months, first months back-filled) — same convention as verify_all.py / overlay_arms.py.
    k_mc = (cv.expanding(min_periods=6).median().bfill() / cv).clip(KMIN, KMAX)

    r_realized = apply_k(k_rv, r_book)
    r_model = apply_k(k_mc, r_book)
    r_product = apply_k((k_rv * k_mc).clip(KMIN, KMAX), r_book)       # the Primary

    # ---------- A) STATIC robustness surfaces ----------
    print("=== A1) exponent family  k = clip(k_rv^a * k_mc^b)  [product = (1,1)] ===")
    grid = [0.0, 0.5, 1.0, 1.5, 2.0]; best = (-9, None)
    surf = pd.DataFrame(index=[f"a={a}" for a in grid], columns=[f"b={b}" for b in grid], dtype=float)
    for a in grid:
        for b in grid:
            k = (np.power(k_rv, a) * np.power(k_mc, b)).clip(KMIN, KMAX)
            s = metrics(apply_k(k, r_book))["Sharpe"]; surf.loc[f"a={a}", f"b={b}"] = round(s, 3)
            if s > best[0]: best = (s, (a, b))
    print(surf.to_string())
    print(f"  product (1,1) Sharpe = {metrics(r_product)['Sharpe']:.3f}   |   grid-best {best[1]} Sharpe = {best[0]:.3f}")

    print("\n=== A2) convex blend  k = clip(w*k_rv + (1-w)*k_mc) ===")
    ws = np.round(np.arange(0, 1.01, 0.1), 2); blend = {}
    for w in ws:
        k = (w * k_rv + (1 - w) * k_mc).clip(KMIN, KMAX); blend[w] = metrics(apply_k(k, r_book))["Sharpe"]
    bw = max(blend, key=blend.get)
    print("  " + "  ".join(f"w={w}:{blend[w]:.3f}" for w in ws))
    print(f"  best convex blend w={bw} Sharpe={blend[bw]:.3f}  (vs product {metrics(r_product)['Sharpe']:.3f}, realized=w1.0 {blend[1.0]:.3f})")

    # ---------- A3) MATCHED-AGGRESSIVENESS: isolate the combination FORM (equal average exposure) ----------
    # The exponent grid is confounded: bigger (a,b) just de-risk more. Rescale every candidate to the
    # product's mean exposure, then the only thing that differs is HOW the two signals are mixed.
    kp = (k_rv * k_mc).clip(KMIN, KMAX); target = kp.mean()
    print(f"\n=== A3) combination FORM at matched mean exposure (= product's {target:.3f}) ===")
    for lbl, k in [("product  k_rv*k_mc", kp), ("min(k_rv,k_mc)", np.minimum(k_rv, k_mc)),
                   ("convex 0.5/0.5", 0.5 * k_rv + 0.5 * k_mc), ("realized only", k_rv), ("model only", k_mc)]:
        k = k.clip(KMIN, KMAX); k2 = (k * (target / k.mean())).clip(KMIN, KMAX)
        m = metrics(apply_k(k2, r_book))
        print(f"  {lbl:20s} Sharpe={m['Sharpe']:.3f}  Ann%={m['Ann']:.2f}  MaxDD%={m['MaxDD']:.2f}")

    # ---------- B) CAUSAL adaptive tilt around the product ----------
    L = 12; GAMMA = 1.0
    d = pd.Series(0.0, index=r_book.index)
    idx = list(r_book.index)
    for i in range(len(idx)):
        if i < L:
            continue
        past = idx[i - L:i]                                # strictly past months (causal)
        sr_rv = ann_sharpe(r_realized.loc[past].values)
        sr_mc = ann_sharpe(r_model.loc[past].values)
        scale = abs(sr_rv) + abs(sr_mc) + 1e-6
        d.iloc[i] = float(np.clip(GAMMA * (sr_rv - sr_mc) / scale, -1.0, 1.0))
    k_adapt = (np.power(k_rv, 1.0 + d) * np.power(k_mc, 1.0 - d)).clip(KMIN, KMAX)
    r_adapt = apply_k(k_adapt, r_book)

    arms = {"Realized-vol only": r_realized, "Model-CVaR only": r_model,
            "FIXED product (Primary)": r_product, "CAUSAL adaptive tilt": r_adapt}
    sr = np.array([metrics(v)["Sharpe"] for v in arms.values()])
    var_srm = np.var(sr/np.sqrt(12), ddof=1); Nt = len(sr); g = 0.5772156649
    z = (1-g)*norm.ppf(1-1.0/Nt) + g*norm.ppf(1-1.0/(Nt*np.e)); sr_star = float(np.sqrt(max(var_srm,0))*z)
    rows = []
    for name, v in arms.items():
        m = metrics(v); rows.append({"arm": name, **{k: round(val,3) for k,val in m.items()}, "DSR%": round(psr_vs(v, sr_star)*100,1)})
    tbl = pd.DataFrame(rows).set_index("arm")
    print("\n=== B) Causal adaptive vs fixed product (Dynamic book, 83 mo) ===")
    print(tbl.to_string()); tbl.to_csv(ROOT / "experiment5_metrics.csv")

    rng = np.random.default_rng(SEED); B = 5000
    def boot(a, b, lbl):
        j = a.index.intersection(b.index); ra, rb = a.loc[j].values, b.loc[j].values; n = len(j)
        shd = np.empty(B); ddd = np.empty(B)
        for k in range(B):
            ix = rng.integers(0, n, n); shd[k] = ann_sharpe(rb[ix]) - ann_sharpe(ra[ix]); ddd[k] = max_dd(rb[ix]) - max_dd(ra[ix])
        print(f"  {lbl}: dSharpe={ann_sharpe(rb)-ann_sharpe(ra):+.3f} P(better)={float((shd>0).mean()):.3f} | "
              f"dMaxDD={max_dd(rb)-max_dd(ra):+.3f} P(shallower)={float((ddd>0).mean()):.3f}")
    print("\n=== Paired bootstraps (5000) ===")
    boot(arms["FIXED product (Primary)"], arms["CAUSAL adaptive tilt"], "adaptive vs product")

    # ---------- plots ----------
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    ax[0].plot(ws, [blend[w] for w in ws], "-o", color="#2471a3", label="convex blend k=w·k_rv+(1-w)·k_mc")
    ax[0].axhline(metrics(r_product)["Sharpe"], color="#c0392b", ls="--", label=f"FIXED product ({metrics(r_product)['Sharpe']:.3f})")
    ax[0].set_xlabel("w  (weight on realized-vol)"); ax[0].set_ylabel("Sharpe"); ax[0].grid(alpha=0.25); ax[0].legend(fontsize=8)
    ax[0].set_title("Static combination surface — product beats any convex blend")
    ax[1].plot(d.index, d.values, color="#16a085", lw=1.6); ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_ylabel("tilt d_t  (+ = lean realized, − = lean model)"); ax[1].grid(alpha=0.25)
    ax[1].set_title("Causal adaptive tilt over time (0 = product)")
    fig.tight_layout(); fig.savefig(ROOT / "_experiment5_combination.png", dpi=130, bbox_inches="tight")
    print("\nsaved experiment5_metrics.csv + _experiment5_combination.png")


if __name__ == "__main__":
    main()
