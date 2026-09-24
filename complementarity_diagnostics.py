"""The four complementarity diagnostics of Section 7.1, computed on the canonical run
for the two core dials k^F (21-day realized volatility, read as of the previous rebalance) and
k^S (centred model-implied CVaR of the chosen book, expanding-median normalizer):

  1. lead-lag structure          : cross-correlogram corr(k^F_t, k^S_{t+l}), l = -6..6, and the lag of the
                                   strongest absolute correlation for every pair in a small signal set
  2. defensive-event overlap     : Jaccard overlap of the months each dial classifies as defensive
                                   (bottom-quintile months, q = 0.20) + the event timeline
  3. conditional future risk     : mean next-month realized volatility of the book by tercile of k^F x tercile of k^S
  4. response surface            : combined exposure over the (k^F, k^S) square under the product rule with floor
                                   (the policy) and under the equal-weight geometric mean (the companion's rule)

Row convention: every monthly row is dated by the month-end (next rebalance) date d1; the dials and the
model forecast were formed at the previous rebalance d0, the realized quantities cover (d0, d1].
Outputs: fig_leadlag.png, fig_event_overlap.png, fig_conditional_risk.png, fig_response_surface.png,
complementarity_diagnostics.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import journal_style as rs; rs.apply()
from overlay_arms import build, RES

ROOT = Path(__file__).resolve().parent
RUN = RES / "main_dyn_strong"; Q = 0.20; LAGS = range(-6, 7)


def load():
    arms, ks = build(RUN)
    _, ks_raw = build(RUN, "weights", "cvar_model_raw")
    fc = pd.read_csv(RUN / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"]); fc = fc.set_index("date")
    mo = pd.read_csv(ROOT / "signal_validity_monthly.csv"); mo["date"] = pd.to_datetime(mo["date"]); mo = mo.set_index("date")
    idx = ks["fast"].index.intersection(mo.index)
    D = pd.DataFrame({
        "k_F": ks["fast"].reindex(idx), "k_S": ks["slow"].reindex(idx), "k_S_raw": ks_raw["slow"].reindex(idx),
        "sigma_model": fc["vol_model"].reindex(idx) * np.sqrt(252),
        "rvol_next": mo["rvol"].reindex(idx), "abs_ret_next": mo["ret"].abs().reindex(idx), "ret_next": mo["ret"].reindex(idx),
        "mdd_next": mo["mdd"].reindex(idx),
    })
    # the fast dial's own input: annualized 21-day realized volatility of the book as of d0 (implied by k_F where unclipped)
    return D.dropna(subset=["k_F", "k_S"])


def xcorr(x, y, lag):
    """corr(x_t, y_{t+lag}); lag > 0 means y follows x (x leads)."""
    return float(x.corr(y.shift(-lag)))


def fig_leadlag(D, out):
    lags = list(LAGS); c = [xcorr(D.k_F, D.k_S, l) for l in lags]
    lstar = lags[int(np.argmax(np.abs(c)))]
    names = ["k_F", "k_S", "k_S_raw", "sigma_model", "rvol_next", "abs_ret_next"]
    labels = {"k_F": "fast dial k^F", "k_S": "slow dial k^S", "k_S_raw": "slow dial, raw", "sigma_model": "model sigma",
              "rvol_next": "realized vol (month)", "abs_ret_next": "|return| (month)"}
    M = pd.DataFrame(index=names, columns=names, dtype=float); C0 = M.copy()
    for a in names:
        for b in names:
            cc = [xcorr(D[a], D[b], l) for l in lags]; M.loc[a, b] = lags[int(np.argmax(np.abs(cc)))]; C0.loc[a, b] = cc[lags.index(0)]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    cols = [rs.ORANGE if l == lstar else rs.BLUE for l in lags]
    ax[0].bar(lags, c, color=cols, width=0.7); ax[0].axhline(0, color="0.5", lw=0.6)
    ax[0].set_xlabel("lag l (months): corr(k^F_t, k^S_{t+l}); l > 0 = fast dial leads"); ax[0].set_ylabel("cross-correlation")
    ax[0].set_xticks(lags); ax[0].text(0.02, 0.95, f"strongest |corr| at l = {lstar:+d} ({c[lags.index(lstar)]:+.2f}); l = 0: {c[lags.index(0)]:+.2f}",
                                       transform=ax[0].transAxes, fontsize=8, va="top")
    im = ax[1].imshow(M.values.astype(float), cmap="coolwarm", vmin=-6, vmax=6)
    ax[1].set_xticks(range(len(names))); ax[1].set_yticks(range(len(names)))
    ax[1].set_xticklabels([labels[n] for n in names], rotation=45, ha="right", fontsize=7); ax[1].set_yticklabels([labels[n] for n in names], fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax[1].text(j, i, f"{int(M.iloc[i, j]):+d}", ha="center", va="center", fontsize=7, color="black")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03); cb.set_label("lag of strongest |corr| (months)", fontsize=7); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return dict(leadlag_lstar=lstar, corr_at_lstar=c[lags.index(lstar)], corr_at_0=c[lags.index(0)],
                corr_lag_m1=c[lags.index(-1)], corr_lag_p1=c[lags.index(1)]), M, C0


def fig_events(D, out):
    sig = {"fast dial k^F": D.k_F, "slow dial k^S": D.k_S, "slow dial, raw": D.k_S_raw, "model sigma (high)": -D.sigma_model}
    ev = {k: (v <= v.quantile(Q)) for k, v in sig.items()}
    names = list(ev); J = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            A, B = ev[a], ev[b]; J.loc[a, b] = float((A & B).sum() / max(1, (A | B).sum()))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
    im = ax[0].imshow(J.values.astype(float), cmap="Blues", vmin=0, vmax=1)
    ax[0].set_xticks(range(len(names))); ax[0].set_yticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7); ax[0].set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax[0].text(j, i, f"{J.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if J.iloc[i, j] > 0.6 else "black")
    ax[0].set_title(f"Jaccard overlap of defensive months (bottom {int(Q*100)}%)", fontsize=8, loc="left")
    # timeline strip: which months each core dial is defensive
    F, S = ev["fast dial k^F"], ev["slow dial k^S"]
    for y, (lab, e, col) in enumerate([("fast dial", F, rs.BLUE), ("slow dial", S, rs.ORANGE), ("both", F & S, rs.NAVY)]):
        d = e[e].index; ax[1].scatter(d, [y] * len(d), marker="s", s=28, color=col, linewidths=0)
    ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(["fast dial", "slow dial", "both"], fontsize=8); ax[1].set_ylim(-0.7, 2.7)
    ax[1].xaxis.set_major_locator(mdates.YearLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax[1].set_title("defensive months of the two core dials", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    both = int((F & S).sum()); either = int((F | S).sum())
    first = {}
    for yr in (2020, 2022):
        f = F[F & (F.index.year == yr)].index.min(); s = S[S & (S.index.year == yr)].index.min()
        first[yr] = (f.strftime("%Y-%m") if pd.notna(f) else None, s.strftime("%Y-%m") if pd.notna(s) else None)
    return dict(jaccard_fast_slow=float(J.loc["fast dial k^F", "slow dial k^S"]), n_fast=int(F.sum()), n_slow=int(S.sum()),
                n_both=both, n_either=either, first_fast_2020=first[2020][0], first_slow_2020=first[2020][1],
                first_fast_2022=first[2022][0], first_slow_2022=first[2022][1]), J


def terciles(s):
    r = s.rank(method="first"); return pd.qcut(r, 3, labels=["low", "mid", "high"])


def fig_condrisk(D, out):
    tF, tS = terciles(D.k_F), terciles(D.k_S)   # low k = most defensive state
    T = D.assign(tF=tF, tS=tS)
    lev = ["low", "mid", "high"]
    V = T.pivot_table(index="tS", columns="tF", values="rvol_next", aggfunc="mean").reindex(index=lev[::-1], columns=lev) * 100
    A = T.pivot_table(index="tS", columns="tF", values="abs_ret_next", aggfunc="mean").reindex(index=lev[::-1], columns=lev) * 100
    N = T.pivot_table(index="tS", columns="tF", values="rvol_next", aggfunc="count").reindex(index=lev[::-1], columns=lev)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for a, (M, ttl, unit) in zip(ax, [(V, "next-month realized volatility of the book (ann. %)", "%"), (A, "next-month absolute return (%)", "%")]):
        im = a.imshow(M.values.astype(float), cmap="YlOrRd")
        a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels([f"k^F {l}" for l in lev], fontsize=8); a.set_yticklabels([f"k^S {l}" for l in lev[::-1]], fontsize=8)
        for i in range(3):
            for j in range(3):
                v = M.iloc[i, j]; n = N.iloc[i, j]
                a.text(j, i, f"{v:.1f}\n(n={int(n) if pd.notna(n) else 0})", ha="center", va="center", fontsize=7.5)
        a.set_title(ttl, fontsize=8, loc="left"); a.set_xlabel("fast-dial state (low = defensive)", fontsize=8); a.set_ylabel("slow-dial state", fontsize=8)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    # marginal gradients: does each dial sort future risk holding the other's state fixed?
    rowF = T.groupby("tF", observed=True)["rvol_next"].mean() * 100; rowS = T.groupby("tS", observed=True)["rvol_next"].mean() * 100
    return dict(rvol_by_kF_low_mid_high=[round(float(rowF[l]), 2) for l in lev], rvol_by_kS_low_mid_high=[round(float(rowS[l]), 2) for l in lev],
                corner_low_low=float(V.loc["low", "low"]), corner_high_high=float(V.loc["high", "high"])), V, N


def fig_surface(D, out):
    g = np.linspace(0.30, 1.0, 141); X, Y = np.meshgrid(g, g)
    P = np.clip(X * Y, 0.30, 1.0); G = np.clip(np.sqrt(X * Y), 0.30, 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for a, (Z, ttl) in zip(ax, [(P, "product rule k = clip(k^F k^S, 0.30, 1)  (the policy)"), (G, "geometric mean clip((k^F k^S)^{1/2}, 0.30, 1)  (companion rule)")]):
        im = a.imshow(Z, origin="lower", extent=[0.3, 1, 0.3, 1], cmap="viridis", vmin=0.3, vmax=1, aspect="auto")
        cs = a.contour(X, Y, Z, levels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], colors="white", linewidths=0.5); a.clabel(cs, fontsize=6, fmt="%.1f")
        a.scatter(D.k_F, D.k_S, s=9, color=rs.ORANGE, edgecolors="black", linewidths=0.2, zorder=3, label="realized monthly pairs")
        a.set_xlabel("fast dial k^F"); a.set_ylabel("slow dial k^S"); a.set_title(ttl, fontsize=8, loc="left")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).set_label("combined exposure k", fontsize=7)
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    kp = np.clip(D.k_F * D.k_S, 0.3, 1); kg = np.clip(np.sqrt(D.k_F * D.k_S), 0.3, 1)
    return dict(mean_k_product=float(kp.mean()), mean_k_geometric=float(kg.mean()), months_at_floor_product=int((kp <= 0.3 + 1e-9).sum()),
                months_both_dials_below_0_8=int(((D.k_F < 0.8) & (D.k_S < 0.8)).sum()), months_only_one_below_0_8=int(((D.k_F < 0.8) ^ (D.k_S < 0.8)).sum()))


def main():
    D = load(); print(f"months: {len(D)}  corr(k_F,k_S) = {D.k_F.corr(D.k_S):+.3f}")
    r1, M, C0 = fig_leadlag(D, ROOT / "fig_leadlag.png"); print("lead-lag:", r1); print(M.to_string())
    r2, J = fig_events(D, ROOT / "fig_event_overlap.png"); print("events:", r2)
    r3, V, N = fig_condrisk(D, ROOT / "fig_conditional_risk.png"); print("conditional risk:", r3); print(V.round(1).to_string()); print(N.to_string())
    r4 = fig_surface(D, ROOT / "fig_response_surface.png"); print("surface:", r4)
    rows = [{"diagnostic": k, "value": v} for r in (r1, r2, r3, r4) for k, v in r.items()]
    pd.DataFrame(rows).to_csv(ROOT / "complementarity_diagnostics.csv", index=False)
    M.to_csv(ROOT / "complementarity_leadlag_matrix.csv"); J.to_csv(ROOT / "complementarity_jaccard.csv"); V.round(2).to_csv(ROOT / "complementarity_condrisk.csv")
    print("saved fig_leadlag.png, fig_event_overlap.png, fig_conditional_risk.png, fig_response_surface.png, complementarity_diagnostics.csv")


if __name__ == "__main__":
    main()
