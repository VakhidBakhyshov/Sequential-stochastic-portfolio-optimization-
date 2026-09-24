"""Regenerate _universe_eligible_count.png and signal_decorrelation.png (style via report_style API)."""
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

# ---------------- universe eligible count ----------------
# Source: the canonical run's own weights workbook (Value = 1 marks the names eligible that month AFTER the
# screen, the data-availability guard and the history rule), not the raw screen workbook.
xl = pd.ExcelFile(RES / "weights.xlsx")
cnt = {}
for sh in xl.sheet_names:
    try:
        d = pd.to_datetime(sh)
    except Exception:
        continue
    if d < pd.Timestamp("2019-01-01"):
        continue
    df = xl.parse(sh)
    if "Value" in df.columns:
        cnt[d] = int((pd.to_numeric(df["Value"], errors="coerce") == 1).sum())
s = pd.Series(cnt).sort_index()
fig, ax = plt.subplots(figsize=(7.0, 2.9))
ax.plot(s.index, s.values, color=rs.NAVY, lw=1.4)
ax.scatter([s.index[0]], [s.iloc[0]], color=rs.ORANGE, zorder=3, s=22)
pk = s.idxmax()
ax.scatter([pk], [s.max()], color=rs.GREEN, zorder=3, s=22)
ax.scatter([s.index[-1]], [s.iloc[-1]], color=rs.NAVY, zorder=3, s=22)
ax.annotate(f"{s.iloc[0]}", (s.index[0], s.iloc[0]), textcoords="offset points",
            xytext=(4, -11), fontsize=8, color=rs.ORANGE)
ax.annotate(f"max {s.max()}", (pk, s.max()), textcoords="offset points",
            xytext=(4, 4), fontsize=8, color=rs.GREEN)
ax.set_ylabel("eligible funds")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
rs.title(ax, "Eligible ETFs per month", f"median {int(s.median())}, range {s.min()}-{s.max()}")
fig.tight_layout()
fig.savefig(ROOT / "_universe_eligible_count.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"universe chart: n={len(s)} months, median {int(s.median())}, range [{s.min()}, {s.max()}]")

# ---------------- signal decorrelation ----------------
def load_pnl():
    p = pd.read_csv(RES / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)

def load_weights():
    x = pd.ExcelFile(RES / "weights.xlsx"); out = {}
    for sh in x.sheet_names:
        d = x.parse(sh)
        if "weights" not in d.columns: continue
        w = pd.to_numeric(d["weights"], errors="coerce").fillna(0.0)
        ser = pd.Series(w.values, index=d[d.columns[0]].astype(str).values); ser = ser[ser > 1e-9]
        if not ser.empty:
            try: out[pd.to_datetime(sh)] = ser / ser.sum()
            except Exception: pass
    return dict(sorted(out.items()))

df = pd.read_csv(ROOT / "datasets" / "excel" / "new_etf_returns.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
DLY = np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0

W = load_weights(); r_full = load_pnl(); rs_ = r_full.iloc[1:]
reb = list(W.keys()); parts = []
for i in range(len(reb) - 1):
    d0, d1 = reb[i], reb[i + 1]; w = W[d0]; cc = [t for t in w.index if t in DLY.columns]
    win = DLY.loc[(DLY.index > d0) & (DLY.index <= d1), cc].fillna(0.0)
    if not win.empty: parts.append(pd.Series(win.values @ w.reindex(cc).values, index=win.index))
book = pd.concat(parts).sort_index()
v21 = book.rolling(21, min_periods=10).std() * np.sqrt(252)
realized = pd.Series({d: v21.asof(d) for d in rs_.index})
fc = pd.read_csv(RES / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
cv = fc.set_index("date")["cvar_model"].astype(float).reindex(rs_.index)

z = lambda x: (x - x.mean()) / x.std()
zr, zc = z(realized), z(cv)
ok = zr.notna() & zc.notna()
rho = float(np.corrcoef(zr[ok], zc[ok])[0, 1])

fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [2.1, 1]})
axes[0].axhline(0, color=rs.GRID, lw=0.8)
axes[0].plot(zr.index, zr.values, color=rs.BLUE, lw=1.2, label="fast: realized 21-day volatility")
axes[0].plot(zc.index, zc.values, color=rs.ORANGE, lw=1.2, label="slow: model-implied risk")
axes[0].set_ylabel("standardized signal")
axes[0].legend(loc="upper right", fontsize=7.5)
axes[0].xaxis.set_major_locator(mdates.YearLocator(2))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].scatter(zr[ok], zc[ok], s=12, color=rs.NAVY, alpha=0.55, edgecolor="none")
b = np.polyfit(zr[ok], zc[ok], 1); xs = np.array([zr[ok].min(), zr[ok].max()])
axes[1].plot(xs, b[0] * xs + b[1], color=rs.RED, lw=1.0, ls="--")
axes[1].set_xlabel("fast signal (z)"); axes[1].set_ylabel("slow signal (z)")
axes[1].annotate(f"$\\rho$ = {rho:+.2f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8.5)
rs.title(axes[0], "Signals fire at different times", f"corr = {rho:+.2f}")
fig.tight_layout()
fig.savefig(ROOT / "signal_decorrelation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"decorrelation chart: rho = {rho:+.3f}, n = {int(ok.sum())}")
