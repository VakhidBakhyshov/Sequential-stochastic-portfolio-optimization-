"""
signal_relationship_audit.py

NEW analysis only. Put this file in the repository root next to overlay_arms.py.

Purpose
-------
Reproduce/audit the presentation result for the two exposure timers using the
EXACT archived framework construction in overlay_arms.build():

    fast = k^F = backward 21-day realized-vol exposure timer
    slow = k^S = forward model-CVaR exposure timer

The canonical presentation analysis is based on:
    results/main_dyn_strong

IMPORTANT
---------
The script NEVER changes a correlation to make it equal to 0.17.
If a different run (e.g. main_dyn_on) produces 0.33, that is reported as-is.

Outputs
-------
- signal_relationship_summary.csv
- signal_performance_by_period.csv
- signal_monthly.csv
- signal_variant_audit.csv
- presentation_timer_timeline.png
- presentation_timer_scatter.png
- presentation_timer_stress_scatter.png
- presentation_timer_rolling_corr.png

Examples
--------
Canonical presentation run:
    python3 signal_relationship_audit.py

Explicit run:
    python3 signal_relationship_audit.py --run main_dyn_strong

Test another result:
    python3 signal_relationship_audit.py --run main_dyn_on

Audit available signal-construction variants in one run:
    python3 signal_relationship_audit.py --run main_dyn_strong --audit-variants

Scan all result folders that contain pnl.csv, weights.xlsx and forecast_risk.csv:
    python3 signal_relationship_audit.py --scan-runs
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, spearmanr

import overlay_arms as oa


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

CANONICAL_RUN = "main_dyn_strong"
PRESENTATION_RHO = 0.17
STRESS_YEARS = (2020, 2022)


def resolve_run(run_arg: str) -> Path:
    p = Path(run_arg).expanduser()

    if p.exists():
        return p.resolve()

    candidate = RESULTS / run_arg
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Result folder not found: {run_arg}\n"
        f"Tried:\n  {p}\n  {candidate}"
    )


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson"):
    """
    Correlation without ConstantInputWarning / divide-by-zero warnings.
    Returns (correlation, p-value). p-value is NaN for Spearman if undefined.
    """
    d = pd.concat(
        [pd.to_numeric(x, errors="coerce"),
         pd.to_numeric(y, errors="coerce")],
        axis=1
    ).dropna()

    if len(d) < 3:
        return np.nan, np.nan

    a = d.iloc[:, 0]
    b = d.iloc[:, 1]

    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            r = pearsonr(a.to_numpy(), b.to_numpy())
            return float(r.statistic), float(r.pvalue)

        if method == "spearman":
            r = spearmanr(a.to_numpy(), b.to_numpy())
            return float(r.statistic), float(r.pvalue)

    raise ValueError(method)


def fisher_ci(r: float, n: int):
    if n < 4 or not np.isfinite(r):
        return np.nan, np.nan

    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3.0)
    zcrit = 1.959963984540054

    return (
        float(np.tanh(z - zcrit * se)),
        float(np.tanh(z + zcrit * se)),
    )


def build_exact_signals(
    run: Path,
    weights_col: str = "weights",
    cvar_col: str = "cvar_model",
):
    """
    Exact framework construction. No month remapping and no alternative formula.

    This is deliberately a thin wrapper around overlay_arms.build(), because the
    presentation/archive should be audited with the same code path that generated
    the research results.
    """
    arms, ks = oa.build(
        run,
        weights_col=weights_col,
        cvar_col=cvar_col,
    )

    K = pd.DataFrame(
        {
            "fast": ks["fast"],
            "slow": ks["slow"],
            "combined": ks["combined"],
        }
    ).dropna()

    if K.empty:
        raise ValueError(
            "overlay_arms.build() produced zero overlapping fast/slow rows.\n"
            "Do NOT silently month-align this when trying to reproduce the archived "
            "presentation result. Check the dates in pnl.csv and forecast_risk.csv."
        )

    return arms, K.sort_index()


def get_periods(K: pd.DataFrame):
    masks = {
        "full_sample": pd.Series(True, index=K.index),
    }

    for year in STRESS_YEARS:
        masks[str(year)] = pd.Series(K.index.year == year, index=K.index)

    stress = np.isin(K.index.year, STRESS_YEARS)
    masks["stress_2020_2022"] = pd.Series(stress, index=K.index)
    masks["calm_remainder"] = pd.Series(~stress, index=K.index)

    return {name: K.loc[mask.values].copy() for name, mask in masks.items()}


def relationship_row(period: str, d: pd.DataFrame):
    x = d["fast"]
    y = d["slow"]

    pr, pp = safe_corr(x, y, "pearson")
    sr, sp = safe_corr(x, y, "spearman")
    lo, hi = fisher_ci(pr, len(d))

    if len(d) >= 2 and x.nunique() >= 2 and y.nunique() >= 2:
        slope, intercept = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    else:
        slope, intercept = np.nan, np.nan

    return {
        "period": period,
        "n": len(d),
        "pearson_r": pr,
        "pearson_p": pp,
        "pearson_ci_low": lo,
        "pearson_ci_high": hi,
        "spearman_rho": sr,
        "spearman_p": sp,
        "ols_slope_slow_on_fast": slope,
        "ols_intercept": intercept,
        "mean_fast_exposure": x.mean(),
        "mean_slow_exposure": y.mean(),
        "mean_combined_exposure": d["combined"].mean(),
        "share_any_timer_derisks": (
            ((x < 1.0 - 1e-12) | (y < 1.0 - 1e-12)).mean()
            if len(d) else np.nan
        ),
        "share_both_timers_derisk": (
            ((x < 1.0 - 1e-12) & (y < 1.0 - 1e-12)).mean()
            if len(d) else np.nan
        ),
    }


def relationship_table(K: pd.DataFrame):
    return pd.DataFrame(
        relationship_row(name, d)
        for name, d in get_periods(K).items()
    )


def max_drawdown(r: pd.Series):
    r = pd.Series(r).dropna()
    if r.empty:
        return np.nan
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def perf(r: pd.Series):
    r = pd.Series(r).dropna()
    n = len(r)

    if n == 0:
        return {
            "n_months": 0,
            "total_return_pct": np.nan,
            "annual_return_pct": np.nan,
            "annual_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
        }

    total = float((1.0 + r).prod() - 1.0)
    ann = float((1.0 + total) ** (12.0 / n) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(12.0)) if n > 1 else np.nan

    ex = r - oa.RF / 12.0
    sd = ex.std(ddof=1)
    sharpe = float(ex.mean() / sd * np.sqrt(12.0)) if n > 1 and sd > 0 else np.nan

    return {
        "n_months": n,
        "total_return_pct": 100.0 * total,
        "annual_return_pct": 100.0 * ann,
        "annual_vol_pct": 100.0 * vol if np.isfinite(vol) else np.nan,
        "sharpe": sharpe,
        "max_drawdown_pct": 100.0 * max_drawdown(r),
    }


def performance_table(arms: dict[str, pd.Series], K: pd.DataFrame):
    rows = []

    for period, d in get_periods(K).items():
        idx = d.index

        for arm_name in ("base", "fast", "slow", "combined"):
            row = {
                "period": period,
                "arm": arm_name,
            }
            row.update(perf(arms[arm_name].reindex(idx)))
            rows.append(row)

    return pd.DataFrame(rows)


def plot_timeline(K: pd.DataFrame, out: Path):
    """
    Presentation layout:
      forward/model-CVaR de-risk amount above zero
      backward/realized-vol de-risk amount below zero
    """
    forward_derisk = 1.0 - K["slow"]
    backward_derisk = -(1.0 - K["fast"])

    fig, ax = plt.subplots(figsize=(11.5, 4.8))

    ax.bar(K.index, forward_derisk, width=22, label="Forward · model-CVaR (tail)")
    ax.bar(K.index, backward_derisk, width=22, label="Backward · realized-vol (variance)")

    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel("De-risk amount  (1 - exposure)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def scatter_panel(ax, d: pd.DataFrame, title: str):
    ax.scatter(d["fast"], d["slow"], s=42, alpha=0.75)

    r, _ = safe_corr(d["fast"], d["slow"], "pearson")

    if (
        len(d) >= 2
        and d["fast"].nunique() >= 2
        and d["slow"].nunique() >= 2
    ):
        slope, intercept = np.polyfit(
            d["fast"].to_numpy(),
            d["slow"].to_numpy(),
            1,
        )
        xx = np.linspace(
            float(d["fast"].min()),
            float(d["fast"].max()),
            100,
        )
        ax.plot(xx, slope * xx + intercept, linestyle="--", linewidth=1.5)

    label = "undefined" if not np.isfinite(r) else f"{r:+.2f}"
    ax.text(
        0.05, 0.94,
        f"ρ = {label}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_title(title, loc="left")
    ax.set_xlabel("Backward exposure (realized-vol)")
    ax.set_ylabel("Forward exposure (model-CVaR)")
    ax.set_xlim(0.27, 1.04)
    ax.set_ylim(0.27, 1.04)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_full_scatter(K: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    scatter_panel(ax, K, "Fast vs slow exposure — full sample")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stress_scatter(K: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.2, 4.8),
        sharex=True,
        sharey=True,
    )

    for ax, year in zip(axes, STRESS_YEARS):
        d = K[K.index.year == year]
        scatter_panel(ax, d, f"{year} stress subsample")

    fig.suptitle(
        "Fast/slow exposure relationship in the pre-specified stress years",
        x=0.02,
        ha="left",
    )

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_corr(K: pd.DataFrame, out: Path, window: int = 12):
    rolling = (
        K["fast"]
        .rolling(window=window, min_periods=max(6, window // 2))
        .corr(K["slow"])
    )

    full_r, _ = safe_corr(K["fast"], K["slow"], "pearson")

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.plot(rolling.index, rolling.values, linewidth=1.7, label=f"{window}-month rolling corr")
    ax.axhline(0.0, linewidth=0.8)
    if np.isfinite(full_r):
        ax.axhline(full_r, linestyle="--", linewidth=1.0, label=f"Full-sample ρ={full_r:+.2f}")

    for year in STRESS_YEARS:
        ax.axvspan(
            pd.Timestamp(f"{year}-01-01"),
            pd.Timestamp(f"{year}-12-31"),
            alpha=0.08,
        )

    ax.set_ylabel(f"{window}-month correlation")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Rolling fast/slow exposure correlation", loc="left")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return rolling


def available_weight_columns(run: Path):
    path = run / "weights.xlsx"
    if not path.exists():
        return []

    xl = pd.ExcelFile(path)
    found = set()

    for sh in xl.sheet_names[: min(5, len(xl.sheet_names))]:
        try:
            d = xl.parse(sh, nrows=5)
            for c in ("weights", "executed_weights"):
                if c in d.columns:
                    found.add(c)
        except Exception:
            pass

    return sorted(found)


def available_cvar_columns(run: Path):
    path = run / "forecast_risk.csv"
    if not path.exists():
        return []

    cols = pd.read_csv(path, nrows=2).columns
    return [c for c in ("cvar_model", "cvar_model_raw") if c in cols]


def audit_variants(run: Path):
    """
    Diagnostic only. This does NOT select the variant nearest 0.17.

    It lets you see whether the archived statistic came from:
      - target weights vs executed weights
      - centered cvar_model vs cvar_model_raw
    """
    weights_cols = available_weight_columns(run) or ["weights"]
    cvar_cols = available_cvar_columns(run) or ["cvar_model"]

    rows = []

    for wc in weights_cols:
        for cc in cvar_cols:
            try:
                _, K = build_exact_signals(run, wc, cc)
                T = relationship_table(K)

                row = {
                    "run": run.name,
                    "weights_col": wc,
                    "cvar_col": cc,
                    "n": len(K),
                }

                for period in (
                    "full_sample",
                    "2020",
                    "2022",
                    "stress_2020_2022",
                    "calm_remainder",
                ):
                    z = T.loc[T["period"] == period].iloc[0]
                    row[f"corr_{period}"] = z["pearson_r"]

                row["abs_gap_to_presentation_0_17"] = (
                    abs(row["corr_full_sample"] - PRESENTATION_RHO)
                    if np.isfinite(row["corr_full_sample"])
                    else np.nan
                )

                rows.append(row)

            except Exception as e:
                rows.append(
                    {
                        "run": run.name,
                        "weights_col": wc,
                        "cvar_col": cc,
                        "error": str(e),
                    }
                )

    return pd.DataFrame(rows)


def scan_runs():
    rows = []

    if not RESULTS.exists():
        return pd.DataFrame()

    for run in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        required = [
            run / "pnl.csv",
            run / "weights.xlsx",
            run / "forecast_risk.csv",
        ]

        if not all(p.exists() for p in required):
            continue

        try:
            _, K = build_exact_signals(run, "weights", "cvar_model")
            T = relationship_table(K)

            row = {
                "run": run.name,
                "n": len(K),
            }

            for period in (
                "full_sample",
                "2020",
                "2022",
                "stress_2020_2022",
                "calm_remainder",
            ):
                z = T.loc[T["period"] == period].iloc[0]
                row[f"corr_{period}"] = z["pearson_r"]

            row["abs_gap_to_presentation_0_17"] = (
                abs(row["corr_full_sample"] - PRESENTATION_RHO)
                if np.isfinite(row["corr_full_sample"])
                else np.nan
            )

            rows.append(row)

        except Exception as e:
            rows.append(
                {
                    "run": run.name,
                    "error": str(e),
                }
            )

    return pd.DataFrame(rows)


def print_interpretation(run: Path, summary: pd.DataFrame):
    full = summary.loc[
        summary["period"] == "full_sample",
        "pearson_r"
    ].iloc[0]

    r2020 = summary.loc[
        summary["period"] == "2020",
        "pearson_r"
    ].iloc[0]

    r2022 = summary.loc[
        summary["period"] == "2022",
        "pearson_r"
    ].iloc[0]

    pooled = summary.loc[
        summary["period"] == "stress_2020_2022",
        "pearson_r"
    ].iloc[0]

    print("\n=== PRESENTATION REPRODUCTION CHECK ===")
    print(f"run                         : {run.name}")
    print(f"canonical presentation run  : {CANONICAL_RUN}")
    print(f"archived presentation rho   : ~{PRESENTATION_RHO:.2f}")
    print(f"calculated full-sample rho  : {full:+.4f}")

    if run.name != CANONICAL_RUN:
        print(
            "\nNOTE: this is NOT the canonical presentation run. "
            "A different correlation is expected and should not be treated as an error."
        )

    if np.isfinite(full):
        print(
            f"gap from presentation rho   : "
            f"{abs(full - PRESENTATION_RHO):.4f}"
        )

    print("\nStress relationships:")
    print(f"  2020                 : {r2020:+.4f}" if np.isfinite(r2020) else "  2020                 : undefined")
    print(f"  2022                 : {r2022:+.4f}" if np.isfinite(r2022) else "  2022                 : undefined")
    print(f"  pooled 2020 + 2022   : {pooled:+.4f}" if np.isfinite(pooled) else "  pooled 2020 + 2022   : undefined")

    if np.isfinite(r2020) and np.isfinite(r2022):
        if r2020 < 0 and r2022 < 0:
            print("  result               : negative in BOTH stress years")
        else:
            print(
                "  result               : the data do NOT show a negative "
                "relationship in both stress years for this run"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        default=CANONICAL_RUN,
        help=(
            "Result folder name under results/ or direct path. "
            f"Default: {CANONICAL_RUN}"
        ),
    )

    parser.add_argument(
        "--weights-col",
        default="weights",
    )

    parser.add_argument(
        "--cvar-col",
        default="cvar_model",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--out-dir",
        default=None,
    )

    parser.add_argument(
        "--audit-variants",
        action="store_true",
        help="Audit weights/executed_weights and cvar_model/cvar_model_raw if available.",
    )

    parser.add_argument(
        "--scan-runs",
        action="store_true",
        help="Scan all valid folders under results/ and compare their correlations.",
    )

    args = parser.parse_args()

    if args.scan_runs:
        scan = scan_runs()
        out = ROOT / "signal_run_scan.csv"
        scan.to_csv(out, index=False)

        print("\n=== RUN SCAN ===")
        if scan.empty:
            print("No valid result folders found.")
        else:
            show = scan.copy()
            numeric = show.select_dtypes(include=[np.number]).columns
            show[numeric] = show[numeric].round(4)
            print(show.to_string(index=False))

        print(f"\nSaved: {out}")
        return

    run = resolve_run(args.run)

    out = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else run / "signal_relationship_audit"
    )
    out.mkdir(parents=True, exist_ok=True)

    arms, K = build_exact_signals(
        run,
        weights_col=args.weights_col,
        cvar_col=args.cvar_col,
    )

    summary = relationship_table(K)
    performance = performance_table(arms, K)

    rolling = plot_rolling_corr(
        K,
        out / "presentation_timer_rolling_corr.png",
        window=args.rolling_window,
    )

    monthly = K.copy()
    monthly["fast_derisk"] = 1.0 - monthly["fast"]
    monthly["slow_derisk"] = 1.0 - monthly["slow"]
    monthly[f"rolling_corr_{args.rolling_window}m"] = rolling

    monthly.to_csv(out / "signal_monthly.csv")
    summary.to_csv(out / "signal_relationship_summary.csv", index=False)
    performance.to_csv(out / "signal_performance_by_period.csv", index=False)

    plot_timeline(
        K,
        out / "presentation_timer_timeline.png",
    )

    plot_full_scatter(
        K,
        out / "presentation_timer_scatter.png",
    )

    plot_stress_scatter(
        K,
        out / "presentation_timer_stress_scatter.png",
    )

    print("\n=== EXACT FRAMEWORK SIGNAL CONSTRUCTION ===")
    print(f"run         : {run}")
    print(f"weights col : {args.weights_col}")
    print(f"CVaR col    : {args.cvar_col}")
    print(f"rows        : {len(K)}")
    print(f"range       : {K.index.min()} -> {K.index.max()}")

    print("\n=== FAST/SLOW RELATIONSHIP ===")
    show = summary.copy()
    numeric = show.select_dtypes(include=[np.number]).columns
    show[numeric] = show[numeric].round(4)
    print(show.to_string(index=False))

    print_interpretation(run, summary)

    print("\n=== PERFORMANCE BY PERIOD ===")
    pshow = performance.copy()
    numeric = pshow.select_dtypes(include=[np.number]).columns
    pshow[numeric] = pshow[numeric].round(4)
    print(pshow.to_string(index=False))

    if args.audit_variants:
        audit = audit_variants(run)
        audit.to_csv(out / "signal_variant_audit.csv", index=False)

        print("\n=== VARIANT AUDIT ===")
        ashow = audit.copy()
        numeric = ashow.select_dtypes(include=[np.number]).columns
        ashow[numeric] = ashow[numeric].round(4)
        print(ashow.to_string(index=False))

        print(
            "\nVariant audit is diagnostic only. "
            "Do not choose a construction merely because it is closest to 0.17."
        )

    print(f"\nSaved new outputs to:\n{out}")


if __name__ == "__main__":
    main()
