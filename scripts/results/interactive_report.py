"""Self-contained Plotly research dashboard for monthly ETF rebalancing."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from scripts.results.stats import (
    annual_return_table,
    calculate_train_test_metrics,
    metric_summary,
    monthly_return_matrix,
)
from scripts.validation.false_strategy import false_strategy_density_surface, exact_expected_max_sharpe_gaussian

EPS = 1e-12


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    first = df.columns[0]
    name = str(first).strip().lower()
    sample = df[first].dropna().astype(str).head(20)
    looks_dated = any(token in name for token in ("date", "time", "month", "rebalance"))
    if not looks_dated and not sample.empty:
        looks_dated = sample.str.match(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?").mean() >= 0.8
    if looks_dated:
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().sum() >= max(1, len(df) // 2):
            df[first] = parsed
            df = df.set_index(first).sort_index()
    return df


def _num(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default




def _load_risk_matrices_for_report(folder: Path, max_full_json_mb: float = 32.0) -> list[dict[str, Any]]:
    """Load only dashboard-sized risk matrices.

    Full ``risk_matrices.json`` is a research artifact and can be hundreds of MB.
    Loading it wholesale and then embedding every snapshot in Plotly can multiply
    peak RAM.  Prefer the compact cache created by ``save_risk_matrices`` and skip
    the full file when it exceeds a conservative threshold.
    """
    compact = folder / "risk_matrices_report.json"
    if compact.exists():
        data = _load_json(compact, [])
        return data if isinstance(data, list) else []
    full = folder / "risk_matrices.json"
    if not full.exists():
        return []
    try:
        if full.stat().st_size > float(max_full_json_mb) * 1024.0 * 1024.0:
            return []
    except OSError:
        return []
    data = _load_json(full, [])
    return data if isinstance(data, list) else []


def _fig_html(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    fig.update_layout(margin=dict(l=50, r=30, t=70, b=45))
    return pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"responsive": True, "displaylogo": False},
    )


def _weights(path: Path) -> dict[str, pd.DataFrame]:
    try:
        return {str(k): v for k, v in pd.read_excel(path, sheet_name=None).items()} if path.exists() else {}
    except Exception:
        return {}


def _benchmark_from_real(real: pd.DataFrame, return_type: str) -> pd.Series | None:
    if real.empty:
        return None
    x = _num(real)
    if x.empty:
        return None
    col = next((c for c in x.columns if str(c).upper() == "SPY"), None)
    r = x[col] if col is not None else x.mean(axis=1)
    return np.expm1(r.astype(float)) if return_type == "log-returns" else r.astype(float)


def _validation_row(validation: pd.DataFrame) -> dict[str, Any]:
    return validation.iloc[0].to_dict() if not validation.empty else {}


def _kpis(pnl: pd.DataFrame, validation: pd.DataFrame) -> str:
    if pnl.empty:
        return ""
    vr = _validation_row(validation)
    m = metric_summary(
        pnl,
        num_trials=float(vr.get("effective_number_of_trials", 1) or 1),
        trial_sharpe_mean=float(vr.get("trial_sharpe_mean", 0) or 0),
        trial_sharpe_std=float(vr.get("trial_sharpe_std", 0) or 0),
    )
    cards = [
        ("CAGR", m.get("Annualized Return"), True),
        ("Sharpe", m.get("Sharpe Ratio"), False),
        ("Sortino", m.get("Sortino Ratio"), False),
        ("Max drawdown", m.get("Max Drawdown"), True),
        ("CVaR 95%", m.get("CVaR (95%)"), True),
        ("CDaR 95%", m.get("CDaR (95%)"), True),
        ("PSR", vr.get("psr", m.get("PSR")), True),
        ("DSR", vr.get("dsr", m.get("DSR")), True),
        ("Effective trials", vr.get("effective_number_of_trials", 1), False),
    ]
    out = ["<div class='cards'>"]
    for label, value, pct in cards:
        try:
            value = float(value)
        except Exception:
            value = np.nan
        if not np.isfinite(value):
            text = "-"
        else:
            text = f"{value:.1%}" if pct else f"{value:.2f}"
        out.append(f"<div class='card'><span>{html.escape(label)}</span><strong>{text}</strong></div>")
    return "".join(out) + "</div>"


def _validation_html(validation: pd.DataFrame) -> str:
    if validation.empty:
        return "<p>Research validation artifacts are not available for this run.</p>"
    row = validation.iloc[0]
    checks = [
        ("PSR threshold", bool(row.get("psr_pass", False))),
        ("DSR threshold", bool(row.get("dsr_pass", False))),
        ("Track record length", bool(row.get("track_record_length_sufficient", False))),
    ]
    if pd.notna(row.get("causality_pass_rate", np.nan)):
        checks.append(("Point-in-time causality", float(row.get("causality_pass_rate", 0.0)) >= 0.999))
    badges = "".join(
        f"<span class='badge {'pass' if passed else 'warn'}'>{html.escape(label)}: {'PASS' if passed else 'REVIEW'}</span>"
        for label, passed in checks
    )
    fields = [
        "observed_annualized_sharpe", "dsr_benchmark_expected_max_sharpe",
        "raw_number_of_trials", "effective_number_of_trials", "trial_sharpe_std",
        "minimum_track_record_length_for_dsr", "nested_pbo_proxy",
        "causality_pass_rate", "full_process_pass_rate", "fdr_by_discoveries",
        "fdr_tested_recipes",
    ]
    table = pd.DataFrame({"Diagnostic": fields, "Value": [row.get(f, np.nan) for f in fields]})
    return badges + table.to_html(index=False, classes="dataframe")


def _performance(pnl: pd.DataFrame, benchmark: pd.Series | None) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.68, 0.32],
        subplot_titles=("Growth of capital", "Drawdown"),
    )
    if pnl.empty:
        return fig
    r = pd.to_numeric(pnl["Returns"], errors="coerce").fillna(0.0)
    growth = pd.to_numeric(pnl.get("Balance"), errors="coerce")
    if growth.isna().all():
        growth = (1.0 + r).cumprod()
    else:
        growth = growth / max(float(growth.dropna().iloc[0]), EPS)
    dd = growth / growth.cummax() - 1.0
    fig.add_trace(go.Scatter(x=growth.index, y=growth, name="Strategy", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Strategy drawdown", fill="tozeroy", mode="lines"), row=2, col=1)
    if benchmark is not None:
        br = benchmark.reindex(pnl.index).fillna(0.0)
        bg = (1.0 + br).cumprod()
        bdd = bg / bg.cummax() - 1.0
        fig.add_trace(go.Scatter(x=bg.index, y=bg, name="SPY", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bdd.index, y=bdd, name="SPY drawdown", mode="lines"), row=2, col=1)
    fig.update_layout(height=700, template="plotly_dark", hovermode="x unified")
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all"),
        ]),
        row=1,
        col=1,
    )
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    return fig


def _monthly_returns_heatmap(pnl: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if pnl.empty:
        return fig
    matrix = monthly_return_matrix(pd.to_numeric(pnl["Returns"], errors="coerce"))
    text = matrix.map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    fig.add_trace(go.Heatmap(
        z=matrix.to_numpy(dtype=float),
        x=[str(c) for c in matrix.columns],
        y=matrix.index.tolist(),
        colorscale="RdYlGn",
        zmid=0.0,
        text=text.to_numpy(),
        texttemplate="%{text}",
        hovertemplate="Year %{x}<br>Month %{y}<br>Return %{z:.2%}<extra></extra>",
        colorbar=dict(title="Return"),
    ))
    fig.update_layout(height=600, template="plotly_dark", title="Monthly strategy returns", xaxis_title="Year", yaxis_title="Month")
    return fig


def _annual_returns_bar(pnl: pd.DataFrame, benchmark: pd.Series | None) -> go.Figure:
    fig = go.Figure()
    if pnl.empty:
        return fig
    table = annual_return_table(pd.to_numeric(pnl["Returns"], errors="coerce"), benchmark, "SPY")
    for col in table.columns:
        fig.add_trace(go.Bar(x=table.index.astype(str), y=table[col], name=str(col), text=[f"{v:.1%}" for v in table[col]]))
    fig.update_layout(
        height=500,
        template="plotly_dark",
        title="Annual returns: strategy vs SPY",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="Return",
        yaxis_tickformat=".1%",
        hovermode="x unified",
    )
    return fig


def _false_strategy_heatmap(surface: pd.DataFrame) -> go.Figure:
    if surface.empty:
        surface = false_strategy_density_surface(sharpe_mean=0.0, sharpe_std=1.0)
    mean_sr = float(pd.to_numeric(surface.get("null_sharpe_mean", pd.Series([0.0])), errors="coerce").dropna().iloc[0]) if not surface.empty else 0.0
    std_sr = float(pd.to_numeric(surface.get("null_sharpe_std", pd.Series([1.0])), errors="coerce").dropna().iloc[0]) if not surface.empty else 1.0
    pivot = surface.pivot_table(index="max_sharpe", columns="number_of_trials", values="relative_density", aggfunc="mean")
    expected = surface.groupby("number_of_trials")["expected_max_sharpe"].first().sort_index()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=pivot.to_numpy(dtype=float),
        x=pivot.columns.to_numpy(dtype=float),
        y=pivot.index.to_numpy(dtype=float),
        colorscale="Inferno",
        colorbar=dict(title="Relative density"),
        hovertemplate="Trials %{x:.0f}<br>max(SR) %{y:.2f}<br>Relative density %{z:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=expected.index.to_numpy(dtype=float),
        y=expected.to_numpy(dtype=float),
        mode="lines",
        line=dict(dash="dash", width=3),
        name="E[max(SR)] FST approximation",
    ))
    if len(expected):
        exact_idx = np.unique(np.linspace(0, len(expected) - 1, min(14, len(expected))).astype(int))
        exact_x = expected.index.to_numpy(dtype=float)[exact_idx]
        exact_y = [exact_expected_max_sharpe_gaussian(k, sharpe_mean=mean_sr, sharpe_std=std_sr) for k in exact_x]
        fig.add_trace(go.Scatter(
            x=exact_x, y=exact_y, mode="lines+markers",
            line=dict(width=1), marker=dict(size=5),
            name="Exact iid-Gaussian E[max(SR)]",
        ))
    fig.update_layout(
        height=650,
        template="plotly_dark",
        title=(
            "False-strategy maximum under iid Gaussian no-skill Sharpe estimates "
            f"(mean={mean_sr:.3g}, std={std_sr:.3g}; not Uniform)"
        ),
        xaxis_type="log",
        xaxis_title="Number of trials K",
        yaxis_title="Maximum Sharpe among K null strategies",
    )
    return fig


def _pbo_plot(pbo: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if pbo.empty:
        return fig
    y = pd.to_numeric(pbo.get("internal_test_rank_percentile"), errors="coerce")
    fig.add_trace(go.Bar(x=pbo.index, y=y, name="Internal-test rank percentile"))
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="Lower-half boundary")
    fig.update_layout(
        height=420,
        template="plotly_dark",
        title="Nested selection overfitting diagnostic",
        xaxis_title="Rebalance date",
        yaxis_title="Test rank percentile",
        yaxis_tickformat=".0%",
    )
    return fig


def _causality_plot(causality: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if causality.empty:
        return fig
    cols = [c for c in [
        "history_before_decision", "train_before_decision", "validation_before_decision",
        "internal_test_before_decision", "nested_engine_used", "no_fallback",
    ] if c in causality.columns]
    if not cols:
        return fig
    values = causality[cols].map(lambda x: 1.0 if str(x).lower() in {"true", "1", "1.0"} or x is True else 0.0)
    fig.add_trace(go.Heatmap(
        z=values.to_numpy(dtype=float).T,
        x=[str(x) for x in causality.index],
        y=cols,
        colorscale=[[0.0, "#7f1d1d"], [0.499, "#7f1d1d"], [0.5, "#14532d"], [1.0, "#14532d"]],
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate="Rebalance %{x}<br>Check %{y}<br>Pass %{z}<extra></extra>",
    ))
    fig.update_layout(height=430, template="plotly_dark", title="Point-in-time causality and process audit")
    return fig


def _rolling(rolling: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in rolling.columns:
        fig.add_trace(go.Scatter(
            x=rolling.index,
            y=pd.to_numeric(rolling[c], errors="coerce"),
            mode="lines",
            name=str(c),
            visible=True if "12" in str(c) else "legendonly",
        ))
    fig.update_layout(height=450, template="plotly_dark", title="Rolling Sharpe and Sortino", hovermode="x unified")
    return fig


def _histogram(pnl: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if pnl.empty:
        return fig
    r = pd.to_numeric(pnl["Returns"], errors="coerce").dropna()
    windows = [("All", r), ("Last 12 months", r.tail(12)), ("Last 36 months", r.tail(36))]
    for i, (name, x) in enumerate(windows):
        fig.add_trace(go.Histogram(x=x, nbinsx=35, name=name, visible=(i == 0)))
    buttons = [
        dict(
            label=name,
            method="update",
            args=[{"visible": [j == i for j in range(len(windows))]}, {"title": f"Return distribution: {name}"}],
        )
        for i, (name, _) in enumerate(windows)
    ]
    fig.update_layout(
        height=430,
        template="plotly_dark",
        title="Return distribution: All",
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0, y=1.13)],
        bargap=0.05,
    )
    return fig


def _turnover(weights: dict[str, pd.DataFrame]) -> go.Figure:
    dates: list[pd.Timestamp] = []
    turns: list[float] = []
    exposure: list[float] = []
    prev = None
    for date, df in sorted(weights.items()):
        key = "Key" if "Key" in df else df.columns[0]
        col = "executed_weights" if "executed_weights" in df else "weights"
        if col not in df:
            continue
        w = pd.Series(pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(), index=df[key].astype(str))
        if prev is not None:
            idx = prev.index.union(w.index)
            turns.append(float((w.reindex(idx, fill_value=0) - prev.reindex(idx, fill_value=0)).abs().sum()))
            dates.append(pd.to_datetime(date, errors="coerce"))
            exposure.append(float(w.sum()))
        prev = w
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=dates, y=turns, name="Turnover"), secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=exposure, name="Risky exposure", mode="lines+markers"), secondary_y=True)
    fig.update_layout(height=440, template="plotly_dark", title="Executed turnover and risky exposure", hovermode="x unified")
    fig.update_yaxes(tickformat=".1%", secondary_y=False)
    fig.update_yaxes(tickformat=".1%", secondary_y=True)
    return fig


def _eligibility_heatmap(weights: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    if not weights:
        return fig
    counts: list[str] = []
    for _, df in weights.items():
        key = "Key" if "Key" in df else df.columns[0]
        counts.extend(df[key].astype(str).tolist())
    tickers = pd.Series(counts).value_counts().head(120).index.tolist()
    dates: list[str] = []
    elig: list[np.ndarray] = []
    sel: list[np.ndarray] = []
    for date, df in sorted(weights.items()):
        key = "Key" if "Key" in df else df.columns[0]
        x = df.copy()
        x.index = x[key].astype(str)
        e = pd.to_numeric(x["Value"], errors="coerce") if "Value" in x else pd.Series(1.0, index=x.index)
        wc = "executed_weights" if "executed_weights" in x else "weights"
        w = pd.to_numeric(x[wc], errors="coerce") if wc in x else pd.Series(0.0, index=x.index)
        elig.append(e.reindex(tickers).fillna(0).to_numpy())
        sel.append((w.reindex(tickers).fillna(0) > 1e-6).astype(float).to_numpy())
        dates.append(date)
    fig.add_trace(go.Heatmap(z=np.asarray(elig).T, x=dates, y=tickers, colorscale="Blues", visible=True, name="Eligible"))
    fig.add_trace(go.Heatmap(z=np.asarray(sel).T, x=dates, y=tickers, colorscale="Greens", visible=False, name="Selected"))
    fig.update_layout(
        height=900,
        template="plotly_dark",
        title="Eligible ETF universe",
        updatemenus=[dict(type="dropdown", buttons=[
            dict(label="Eligible", method="update", args=[{"visible": [True, False]}, {"title": "Eligible ETF universe"}]),
            dict(label="Selected non-zero", method="update", args=[{"visible": [False, True]}, {"title": "Selected non-zero ETFs"}]),
        ])],
    )
    return fig


def _risk_matrix(snapshots: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not snapshots:
        fig.update_layout(template="plotly_dark", title="Actual model risk matrices were not exported")
        return fig
    traces: list[go.Heatmap] = []
    buttons: list[dict[str, Any]] = []
    for snap_i, snapshot in enumerate(snapshots):
        labels = list(snapshot.get("tickers", []))[:80]
        for kind, label in [("correlation", "Correlation"), ("covariance_horizon", "Holding-period covariance")]:
            z = np.asarray(snapshot.get(kind, []), dtype=float)
            if z.ndim != 2 or z.size == 0:
                continue
            z = z[:len(labels), :len(labels)]
            visible = snap_i == len(snapshots) - 1 and kind == "correlation"
            traces.append(go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale="RdBu" if kind == "correlation" else "Viridis",
                zmid=0 if kind == "correlation" else None,
                visible=visible,
                name=f"{snapshot.get('date')} {label}",
            ))
    for trace in traces:
        fig.add_trace(trace)
    for i, trace in enumerate(traces):
        buttons.append(dict(
            label=trace.name,
            method="update",
            args=[{"visible": [j == i for j in range(len(traces))]}, {"title": trace.name}],
        ))
    title = traces[-2].name if len(traces) >= 2 else "Model risk matrix"
    fig.update_layout(height=820, template="plotly_dark", title=title, updatemenus=[dict(type="dropdown", buttons=buttons, x=0, y=1.08)])
    return fig


def _line_panel(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for c in df.columns:
        y = pd.to_numeric(df[c], errors="coerce")
        if y.notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=y, mode="lines", name=str(c)))
    fig.update_layout(height=450, template="plotly_dark", title=title, hovermode="x unified")
    return fig


def _metrics_html(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "<p>No train/test metrics.</p>"
    df = metrics.copy()
    pct_terms = ("Return", "Volatility", "Drawdown", "VaR", "CVaR", "CDaR", "Deviation", "Win Rate", "Expectancy", "PSR", "DSR")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if any(term.lower() in str(c).lower() for term in pct_terms):
                df[c] = df[c].map(lambda v: f"{v:.2%}" if pd.notna(v) else "")
            else:
                df[c] = df[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    return df.to_html(index=False, classes="dataframe")


def _monthly_tables(weights: dict[str, pd.DataFrame], real: pd.DataFrame) -> str:
    if not weights:
        return "<p>No monthly weights.</p>"
    chunks: list[str] = []
    real_num = _num(real)
    for n, (date, df) in enumerate(sorted(weights.items(), reverse=True)):
        d = df.copy()
        key = "Key" if "Key" in d else d.columns[0]
        d[key] = d[key].astype(str)
        wc = "executed_weights" if "executed_weights" in d else "weights"
        if wc not in d:
            d[wc] = 0.0
        d["Weights"] = pd.to_numeric(d[wc], errors="coerce").fillna(0.0)
        match = real_num.loc[real_num.index.astype(str) == str(date)] if not real_num.empty else pd.DataFrame()
        d["Realized holding-period return"] = d[key].map(match.iloc[0].to_dict()) if not match.empty else np.nan
        d = d[[key, "Weights", "Realized holding-period return"]].sort_values("Weights", ascending=False)
        tid = f"month_{n}"
        controls = (
            f"<label>Min weight <input id='{tid}_min' type='number' step='0.001' value='0'></label>"
            f"<label>Max weight <input id='{tid}_max' type='number' step='0.001' value='1'></label>"
            f"<button onclick=\"filterWeights('{tid}')\">Apply</button>"
        )
        table = d.to_html(index=False, table_id=tid, classes="dataframe", float_format=lambda x: f"{x:.6f}")
        chunks.append(f"<details {'open' if n == 0 else ''}><summary>{html.escape(str(date))}</summary>{controls}{table}</details>")
    return "".join(chunks)


def build_interactive_report(
    result_folder: str | Path,
    *,
    output_html: str | Path | None = None,
    benchmark_returns: pd.Series | None = None,
    title: str = "ETF Portfolio Research Dashboard",
) -> Path:
    folder = Path(result_folder)
    folder.mkdir(parents=True, exist_ok=True)
    pnl = _read_csv(folder / "pnl.csv")
    real = _read_csv(folder / "real.csv")
    rolling = _read_csv(folder / "rolling_metrics.csv")
    forecast = _read_csv(folder / "forecast_risk.csv")
    dynamic = _read_csv(folder / "dynamic_parameter_history.csv")
    smart = _read_csv(folder / "smart_rebalance_audit.csv")
    audit = _read_csv(folder / "selection_audit.csv")
    validation = _read_csv(folder / "research_validation_summary.csv")
    multiple = _read_csv(folder / "multiple_testing_results.csv")
    pbo = _read_csv(folder / "nested_pbo_proxy.csv")
    causality = _read_csv(folder / "causality_audit.csv")
    surface = _read_csv(folder / "false_strategy_surface_calibrated.csv")
    if surface.empty:
        surface = _read_csv(folder / "false_strategy_surface.csv")
    weights = _weights(folder / "weights.xlsx")
    matrices = _load_risk_matrices_for_report(folder)
    meta = _load_json(folder / "run_metadata.json", {})

    if benchmark_returns is None:
        benchmark_returns = _benchmark_from_real(real, str(meta.get("return_type", "log-returns")))
    metrics = _read_csv(folder / "train_test_metrics.csv")
    if metrics.empty and not pnl.empty:
        vr = _validation_row(validation)
        metrics = calculate_train_test_metrics(
            pnl,
            num_trials=float(vr.get("effective_number_of_trials", 1) or 1),
            trial_sharpe_mean=float(vr.get("trial_sharpe_mean", 0) or 0),
            trial_sharpe_std=float(vr.get("trial_sharpe_std", 0) or 0),
        )
        metrics.to_csv(folder / "train_test_metrics.csv", index=False)

    sections = [
        _kpis(pnl, validation),
        _fig_html(_performance(pnl, benchmark_returns), True),
        _fig_html(_monthly_returns_heatmap(pnl)),
        _fig_html(_annual_returns_bar(pnl, benchmark_returns)),
        "<h2>False-strategy and multiple-testing validation</h2>" + _validation_html(validation),
        _fig_html(_false_strategy_heatmap(surface)),
    ]
    if not pbo.empty:
        sections.append(_fig_html(_pbo_plot(pbo)))
    if not causality.empty:
        sections.append(_fig_html(_causality_plot(causality)))
    if not multiple.empty:
        keep = [c for c in [
            "recipe_id", "source", "method", "n_rebalances", "mean_validation_sharpe",
            "effective_rebalances", "raw_pvalue", "p_sidak", "p_holm", "p_hochberg", "p_fdr_bh", "p_fdr_by", "reject_fdr_by",
        ] if c in multiple.columns]
        sections.append("<h2>Recipe-level multiple-testing corrections</h2>" + multiple[keep].head(200).to_html(index=False, classes="dataframe"))
    sections.append("<h2>Overall / train / test metrics</h2>" + _metrics_html(metrics))
    if not rolling.empty:
        sections.append(_fig_html(_rolling(rolling)))
    sections += [
        _fig_html(_histogram(pnl)),
        _fig_html(_turnover(weights)),
        _fig_html(_eligibility_heatmap(weights)),
        _fig_html(_risk_matrix(matrices)),
    ]
    if not forecast.empty:
        sections.append(_fig_html(_line_panel(forecast, "Forward risk and exposure signals")))
    if not dynamic.empty:
        sections.append(_fig_html(_line_panel(dynamic, "Dynamic optimizer controls")))
    if not smart.empty:
        smart_view = smart.tail(120).drop(columns=["smart_signal_table"], errors="ignore")
        sections.append("<h2>Smart signal audit</h2>" + smart_view.to_html(classes="dataframe"))
    if not audit.empty:
        sections.append("<h2>Nested selection audit</h2>" + audit.tail(120).to_html(classes="dataframe"))
    sections.append("<h2>Monthly ETF weights and returns</h2>" + _monthly_tables(weights, real))

    css = """<style>
body{margin:0;background:#0f172a;color:#e5e7eb;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}
.wrap{max-width:1500px;margin:auto;padding:24px}h1{font-size:34px;margin-bottom:4px}h2{margin-top:34px}.sub{color:#94a3b8;margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0 28px}.card{background:#111827;border:1px solid #334155;border-radius:12px;padding:14px}.card span{display:block;color:#94a3b8;font-size:12px}.card strong{font-size:24px}
.badge{display:inline-block;padding:7px 10px;margin:4px;border-radius:999px;font-size:12px;font-weight:700}.badge.pass{background:#14532d;color:#dcfce7}.badge.warn{background:#7f1d1d;color:#fee2e2}
table.dataframe{border-collapse:collapse;width:100%;margin:12px 0 24px;font-size:12px}table.dataframe th,table.dataframe td{border:1px solid #334155;padding:6px 8px;text-align:right}table.dataframe th{background:#1e293b;position:sticky;top:0}table.dataframe td:first-child,table.dataframe th:first-child{text-align:left}
details{background:#111827;border:1px solid #334155;border-radius:10px;padding:10px 14px;margin:10px 0}summary{cursor:pointer;font-weight:650}input,button{background:#0b1220;color:#e5e7eb;border:1px solid #475569;padding:6px;border-radius:6px;margin:10px 6px 10px 0}
</style>"""
    js = """<script>
function filterWeights(id){const t=document.getElementById(id);if(!t)return;const mn=parseFloat(document.getElementById(id+'_min').value||0),mx=parseFloat(document.getElementById(id+'_max').value||1);const hs=Array.from(t.querySelectorAll('thead th')).map(x=>x.innerText.toLowerCase());const i=hs.indexOf('weights');Array.from(t.querySelectorAll('tbody tr')).forEach(r=>{const v=parseFloat(r.children[i].innerText);r.style.display=(!isNaN(v)&&v>=mn&&v<=mx)?'':'none'})}
</script>"""
    out = Path(output_html) if output_html else folder / "interactive_report.html"
    out.write_text(
        f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>{css}{js}</head>"
        f"<body><div class='wrap'><h1>{html.escape(title)}</h1><div class='sub'>Audit folder: {html.escape(str(folder))}</div>{''.join(sections)}</div></body></html>",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-folder", required=True)
    parser.add_argument("--output-html")
    args = parser.parse_args()
    print(build_interactive_report(args.result_folder, output_html=args.output_html))


if __name__ == "__main__":
    main()
