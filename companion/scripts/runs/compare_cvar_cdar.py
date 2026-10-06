"""Paired historical CVaR-vs-CDaR comparison for frozen run artifacts.

The script is deliberately post-processing only: it never feeds realized holdout data
back into either optimizer.  Run the two policies on the same immutable point-in-time
vintage first, then compare their saved artifact folders here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.results.stats import metric_summary


def _read_pnl(folder: Path) -> pd.DataFrame:
    p = pd.read_csv(folder / "pnl.csv")
    date_col = "Date" if "Date" in p.columns else p.columns[0]
    p[date_col] = pd.to_datetime(p[date_col], errors="coerce")
    p = p.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if "Returns" not in p or "Balance" not in p:
        raise ValueError(f"{folder/'pnl.csv'} must contain Returns and Balance")
    return p


def _aligned(cvar_dir: Path, cdar_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = _read_pnl(cvar_dir), _read_pnl(cdar_dir)
    idx = a.index.intersection(b.index)
    if len(idx) < 6:
        raise ValueError("Need at least six common dated observations for a paired comparison")
    return a.loc[idx].copy(), b.loc[idx].copy()


def _max_drawdown_from_returns(r: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(r, dtype=float))
    if wealth.size == 0:
        return np.nan
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def _sharpe(r: np.ndarray, rf_annual: float = 0.02, ppy: int = 12) -> float:
    x = np.asarray(r, dtype=float) - rf_annual / ppy
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return float(np.mean(x) / sd * np.sqrt(ppy)) if sd > 0 else np.nan


def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Paired circular block bootstrap: contiguous blocks of `block_length` months, wrapped at the sample end."""
    k = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=k)
    idx = np.concatenate([(s + np.arange(block_length)) % n for s in starts])
    return idx[:n]


def paired_bootstrap(a: pd.Series, b: pd.Series, *, draws: int = 5000, seed: int = 20260911,
                     block_length: int = 6) -> pd.DataFrame:
    """Paired block bootstrap of the CDaR minus CVaR Sharpe and drawdown differences. Both arms are resampled on
    the same month indices, so market episodes stay aligned; blocks preserve the serial dependence of monthly
    returns. block_length = 1 is the iid paired bootstrap."""
    x = pd.concat([a.rename("cvar"), b.rename("cdar")], axis=1).dropna()
    arr = x.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    ds, dd = [], []
    n = len(arr); L = max(1, int(block_length))
    for _ in range(int(draws)):
        idx = _block_indices(n, L, rng)
        aa, bb = arr[idx, 0], arr[idx, 1]
        ds.append(_sharpe(bb) - _sharpe(aa))
        # Positive means CDaR is shallower / better.
        dd.append(_max_drawdown_from_returns(bb) - _max_drawdown_from_returns(aa))
    ds = np.asarray(ds, dtype=float); dd = np.asarray(dd, dtype=float)
    kind = "paired_circular_block" if L > 1 else "paired_iid"
    return pd.DataFrame([
        {
            "comparison": "CDaR minus CVaR Sharpe",
            "estimate": _sharpe(arr[:, 1]) - _sharpe(arr[:, 0]),
            "p_treatment_better": float(np.mean(ds > 0)),
            "ci_2_5": float(np.nanquantile(ds, 0.025)),
            "ci_97_5": float(np.nanquantile(ds, 0.975)),
            "bootstrap_type": kind, "block_length": L, "draws": int(draws),
        },
        {
            "comparison": "CDaR minus CVaR max drawdown (positive=shallower)",
            "estimate": _max_drawdown_from_returns(arr[:, 1]) - _max_drawdown_from_returns(arr[:, 0]),
            "p_treatment_better": float(np.mean(dd > 0)),
            "ci_2_5": float(np.nanquantile(dd, 0.025)),
            "ci_97_5": float(np.nanquantile(dd, 0.975)),
            "bootstrap_type": kind, "block_length": L, "draws": int(draws),
        },
    ])


def _forecast_comparison(cvar_dir: Path, cdar_dir: Path) -> pd.DataFrame:
    rows = []
    for label, folder in (("CVaR", cvar_dir), ("CDaR", cdar_dir)):
        path = folder / "forecast_risk.csv"
        if not path.exists():
            continue
        f = pd.read_csv(path)
        numeric = f.apply(pd.to_numeric, errors="coerce")
        rec = {"arm": label, "months": int(len(f))}
        for col in (
            "cvar_model_centered", "cdar_model_centered", "target_risky_exposure",
            "forward_cvar_timer", "forward_cdar_timer", "forward_risk_timer", "overlay_fraction",
        ):
            if col in numeric:
                rec[f"mean_{col}"] = float(numeric[col].mean())
                rec[f"std_{col}"] = float(numeric[col].std(ddof=1))
        rows.append(rec)
    return pd.DataFrame(rows)


def run(cvar_dir: Path, cdar_dir: Path, out: Path, bootstrap_draws: int = 5000, block_length: int = 6) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cvar, cdar = _aligned(cvar_dir, cdar_dir)
    metrics = []
    for name, df in (("CVaR", cvar), ("CDaR", cdar)):
        m = metric_summary(df, risk_free_rate=0.02, periods_per_year=12)
        metrics.append({"arm": name, **m})
    pd.DataFrame(metrics).to_csv(out / "cvar_cdar_metrics.csv", index=False)
    paired_bootstrap(cvar["Returns"], cdar["Returns"], draws=bootstrap_draws, block_length=block_length).to_csv(
        out / "cvar_cdar_paired_bootstrap.csv", index=False
    )
    _forecast_comparison(cvar_dir, cdar_dir).to_csv(out / "cvar_cdar_forward_risk_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for name, df in (("CVaR", cvar), ("CDaR", cdar)):
        wealth = (1.0 + df["Returns"].astype(float)).cumprod()
        ax.plot(wealth.index, wealth.values, label=name)
    ax.set_title("Frozen paired comparison: cumulative wealth")
    ax.set_ylabel("growth of 1"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "figure_cvar_cdar_cumulative.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for name, df in (("CVaR", cvar), ("CDaR", cdar)):
        wealth = (1.0 + df["Returns"].astype(float)).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        ax.plot(drawdown.index, drawdown.values, label=name)
    ax.set_title("Frozen paired comparison: drawdown paths")
    ax.set_ylabel("drawdown"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "figure_cvar_cdar_drawdowns.png", dpi=180); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvar-dir", required=True)
    ap.add_argument("--cdar-dir", required=True)
    ap.add_argument("--output-dir", default="results/cvar_cdar_comparison")
    ap.add_argument("--bootstrap-draws", type=int, default=5000)
    ap.add_argument("--block-length", type=int, default=6, help="months per bootstrap block; 1 gives the iid paired bootstrap")
    args = ap.parse_args()
    run(Path(args.cvar_dir), Path(args.cdar_dir), Path(args.output_dir), args.bootstrap_draws, args.block_length)


if __name__ == "__main__":
    main()
