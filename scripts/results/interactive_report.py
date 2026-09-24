"""
Plotly HTML report for monthly ETF rebalancing runs.

The report is designed to be called automatically from scripts/runs/run.py or manually:
    python -m scripts.results.interactive_report --result-folder results/advanced_bayesian_portfolio

It uses only files already produced by the pipeline when possible:
    pnl.csv, real.csv, preds.csv, model.csv, weights.xlsx, selection_audit.csv,
    dynamic_parameter_history.csv, forecast_risk.csv, smart_rebalance_audit.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

EPS = 1e-12


def _read_indexed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    first = df.columns[0]
    try:
        df[first] = pd.to_datetime(df[first], errors="coerce")
        if df[first].notna().sum() >= max(1, len(df) // 2):
            df = df.set_index(first)
    except Exception:
        pass
    return df


def _load_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return _read_indexed_csv(path)
    return pd.DataFrame()


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def drawdown_from_balance(balance: pd.Series) -> pd.Series:
    b = balance.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return b / b.cummax() - 1.0


def strategy_metrics(returns: pd.Series, balance: pd.Series | None = None, periods_per_year: int = 12, rf: float = 0.02) -> dict[str, float]:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}
    if balance is None or len(balance) == 0:
        b = (1.0 + r).cumprod()
    else:
        b = pd.Series(balance).astype(float).reindex(r.index).ffill().dropna()
        if b.empty:
            b = (1.0 + r).cumprod()
    years = max(len(r) / periods_per_year, EPS)
    total = float((1.0 + r).prod() - 1.0)
    ann = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0
    excess = r - rf / periods_per_year
    ex_sd = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    sharpe = float(excess.mean() / ex_sd * np.sqrt(periods_per_year)) if ex_sd > EPS else np.nan
    downside = r[r < 0]
    downside_dev = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(periods_per_year))
    sortino = float((ann - rf) / downside_dev) if downside_dev > EPS else np.nan
    dd = drawdown_from_balance(b)
    mdd = float(dd.min()) if not dd.empty else 0.0
    calmar = float(ann / abs(mdd)) if mdd < -EPS else np.nan
    var95 = float(np.nanpercentile(r, 5))
    cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else var95
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / len(r)) if len(r) else np.nan
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > EPS else np.inf
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    underwater = dd[dd < 0]
    avg_drawdown = float(underwater.mean()) if not underwater.empty else 0.0
    return {
        "total_return": total,
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "average_drawdown": avg_drawdown,
        "calmar_ratio": calmar,
        "var_95": var95,
        "cvar_95": cvar95,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def train_test_metrics(pnl: pd.DataFrame, train_fraction: float = 0.70) -> pd.DataFrame:
    if pnl.empty or "Returns" not in pnl.columns:
        return pd.DataFrame()
    n = len(pnl)
    split = max(1, min(n - 1, int(round(n * train_fraction)))) if n > 2 else n
    pieces = {
        "overall": pnl,
        "train": pnl.iloc[:split],
        "test": pnl.iloc[split:],
    }
    rows = []
    for name, df in pieces.items():
        if df.empty:
            continue
        m = strategy_metrics(df["Returns"], df["Balance"] if "Balance" in df else None)
        rows.append({"sample": name, "start": str(df.index.min()), "end": str(df.index.max()), "n_periods": len(df), **m})
    return pd.DataFrame(rows)


def rolling_ratio(returns: pd.Series, window: int = 12, ratio: str = "sharpe", rf: float = 0.02) -> pd.Series:
    r = pd.Series(returns).astype(float)
    out = []
    idx = []
    for i in range(window, len(r) + 1):
        x = r.iloc[i - window:i]
        ex = x - rf / 12
        if ratio == "sortino":
            dd = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
            val = (x.mean() * 12 - rf) / (dd * np.sqrt(12)) if dd > EPS else np.nan
        else:
            sd = ex.std(ddof=1)
            val = ex.mean() / sd * np.sqrt(12) if sd > EPS else np.nan
        out.append(val)
        idx.append(r.index[i - 1])
    return pd.Series(out, index=idx, name=f"rolling_{ratio}_{window}")


def _figure_html(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False, config={"displaylogo": False, "responsive": True})


def _plot_performance(pnl: pd.DataFrame, benchmark: pd.Series | None = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=("Growth of capital", "Drawdown"))
    if pnl.empty or "Returns" not in pnl.columns:
        return fig
    if "Balance" in pnl.columns:
        growth = pnl["Balance"].astype(float) / float(pnl["Balance"].dropna().iloc[0])
    else:
        growth = (1.0 + pnl["Returns"].astype(float)).cumprod()
    dd = growth / growth.cummax() - 1.0
    fig.add_trace(go.Scatter(x=growth.index, y=growth, mode="lines", name="Strategy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="Strategy DD"), row=2, col=1)
    if benchmark is not None and not benchmark.empty:
        br = benchmark.reindex(pnl.index).fillna(0.0).astype(float)
        bg = (1.0 + br).cumprod()
        bdd = bg / bg.cummax() - 1.0
        fig.add_trace(go.Scatter(x=bg.index, y=bg, mode="lines", name="SPY / benchmark"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bdd.index, y=bdd, mode="lines", name="Benchmark DD"), row=2, col=1)
    fig.update_layout(height=680, template="plotly_dark", hovermode="x unified")
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    return fig


def _plot_rolling(pnl: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if pnl.empty or "Returns" not in pnl.columns:
        return fig
    r = pnl["Returns"].astype(float)
    for window in (6, 12, 24):
        if len(r) >= window:
            fig.add_trace(go.Scatter(x=rolling_ratio(r, window, "sharpe").index, y=rolling_ratio(r, window, "sharpe").values, mode="lines", name=f"Sharpe {window}m"))
            fig.add_trace(go.Scatter(x=rolling_ratio(r, window, "sortino").index, y=rolling_ratio(r, window, "sortino").values, mode="lines", name=f"Sortino {window}m", visible="legendonly"))
    fig.update_layout(height=450, template="plotly_dark", title="Rolling Sharpe / Sortino", hovermode="x unified")
    return fig


def _plot_histogram(pnl: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if pnl.empty or "Returns" not in pnl.columns:
        return fig
    r = pnl["Returns"].astype(float)
    fig.add_trace(go.Histogram(x=r, nbinsx=40, name="All months"))
    if len(r) >= 12:
        fig.add_trace(go.Histogram(x=r.iloc[-12:], nbinsx=25, name="Last 12 months", visible="legendonly"))
    if len(r) >= 36:
        fig.add_trace(go.Histogram(x=r.iloc[-36:], nbinsx=30, name="Last 36 months", visible="legendonly"))
    fig.update_layout(height=420, template="plotly_dark", barmode="overlay", title="Return histogram with legend-selectable windows")
    fig.update_traces(opacity=0.72)
    return fig


def _weights_from_excel(weights_path: Path) -> dict[str, pd.DataFrame]:
    if not weights_path.exists():
        return {}
    try:
        xls = pd.read_excel(weights_path, sheet_name=None)
        return {str(k): v for k, v in xls.items()}
    except Exception:
        return {}


def _plot_turnover(weights_by_month: dict[str, pd.DataFrame]) -> go.Figure:
    dates, vals = [], []
    prev = None
    for date, df in sorted(weights_by_month.items()):
        if "weights" not in df.columns:
            continue
        key_col = "Key" if "Key" in df.columns else df.columns[0]
        w = pd.Series(pd.to_numeric(df["weights"], errors="coerce").fillna(0.0).values, index=df[key_col].astype(str))
        if prev is not None:
            common = sorted(set(prev.index) | set(w.index))
            vals.append(float((w.reindex(common).fillna(0.0) - prev.reindex(common).fillna(0.0)).abs().sum()))
            dates.append(pd.to_datetime(date, errors="coerce"))
        prev = w
    fig = go.Figure(go.Bar(x=dates, y=vals, name="Turnover"))
    fig.update_layout(height=420, template="plotly_dark", title="Monthly turnover", yaxis_tickformat=".1%")
    return fig


def _monthly_tables_html(weights_by_month: dict[str, pd.DataFrame], real: pd.DataFrame) -> str:
    if not weights_by_month:
        return "<p>No weights.xlsx was found, so monthly weight tables are unavailable.</p>"
    months_html = []
    for date, df in sorted(weights_by_month.items())[-24:]:  # keep HTML manageable; xlsx still contains all months
        dfx = df.copy()
        key_col = "Key" if "Key" in dfx.columns else dfx.columns[0]
        dfx[key_col] = dfx[key_col].astype(str)
        if "weights" in dfx.columns:
            dfx["weights"] = pd.to_numeric(dfx["weights"], errors="coerce").fillna(0.0)
        else:
            dfx["weights"] = 0.0
        try:
            rrow = real.loc[pd.to_datetime(date)] if not real.empty and isinstance(real.index, pd.DatetimeIndex) and pd.to_datetime(date) in real.index else None
            if rrow is not None:
                ret_map = rrow.apply(pd.to_numeric, errors="coerce")
                dfx["month_return"] = dfx[key_col].map(ret_map.to_dict())
        except Exception:
            pass
        view = dfx[[c for c in [key_col, "Value", "weights", "month_return"] if c in dfx.columns]].copy()
        view = view.sort_values("weights", ascending=False).head(80)
        table_id = f"tbl_{str(date).replace('-', '_').replace(':', '_')}"
        months_html.append(f"<details><summary>{date}: top non-zero / eligible ETF weights</summary>" +
                           f"<input type='number' step='0.001' value='0' oninput=\"filterWeights('{table_id}', this.value)\"> min weight" +
                           view.to_html(index=False, table_id=table_id, classes="dataframe monthly-table", float_format=lambda x: f"{x:.6f}") +
                           "</details>")
    return "\n".join(months_html)


def _matrix_heatmap_from_weights(weights_by_month: dict[str, pd.DataFrame], real: pd.DataFrame) -> go.Figure:
    # Best-effort heatmap: latest selected ETFs' realized-return covariance/correlation.
    fig = go.Figure()
    if not weights_by_month or real.empty:
        fig.update_layout(template="plotly_dark", title="Covariance / correlation matrix unavailable")
        return fig
    latest_date, latest_df = sorted(weights_by_month.items())[-1]
    key_col = "Key" if "Key" in latest_df.columns else latest_df.columns[0]
    if "weights" in latest_df.columns:
        selected = latest_df.loc[pd.to_numeric(latest_df["weights"], errors="coerce").fillna(0.0) > 1e-6, key_col].astype(str).tolist()
    else:
        selected = latest_df.loc[latest_df.get("Value", 0).astype(bool), key_col].astype(str).tolist()
    selected = [c for c in selected if c in real.columns][:60]
    if len(selected) < 2:
        fig.update_layout(template="plotly_dark", title="Need at least two selected ETFs for matrix heatmap")
        return fig
    window = _to_numeric_df(real[selected]).dropna(how="all").tail(36)
    corr = window.corr().fillna(0.0)
    cov = window.cov().fillna(0.0)
    fig.add_trace(go.Heatmap(z=corr.values, x=selected, y=selected, colorscale="RdBu", zmid=0, name="Correlation", visible=True))
    fig.add_trace(go.Heatmap(z=cov.values, x=selected, y=selected, colorscale="Viridis", name="Covariance", visible=False))
    fig.update_layout(
        height=760,
        template="plotly_dark",
        title=f"Latest selected ETF matrix ({latest_date}) - toggle correlation/covariance",
        updatemenus=[dict(type="dropdown", x=0.0, y=1.08, buttons=[
            dict(label="Correlation", method="update", args=[{"visible": [True, False]}, {"title": f"Correlation matrix ({latest_date})"}]),
            dict(label="Covariance", method="update", args=[{"visible": [False, True]}, {"title": f"Covariance matrix ({latest_date})"}]),
        ])],
    )
    return fig


def _heatmap_eligible_vs_selected(weights_by_month: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    if not weights_by_month:
        return fig
    all_keys = []
    rows_elig, rows_sel, dates = [], [], []
    for date, df in sorted(weights_by_month.items()):
        key_col = "Key" if "Key" in df.columns else df.columns[0]
        keys = df[key_col].astype(str).tolist()
        all_keys.extend(keys)
    top_keys = pd.Index(all_keys).value_counts().head(100).index.tolist()
    for date, df in sorted(weights_by_month.items()):
        key_col = "Key" if "Key" in df.columns else df.columns[0]
        x = df.set_index(df[key_col].astype(str))
        elig = pd.to_numeric(x.get("Value", 0), errors="coerce").reindex(top_keys).fillna(0.0)
        # w = pd.to_numeric(x.get("weights", 0), errors="coerce").reindex(top_keys).fillna(0.0)
        w = pd.to_numeric(x.get("weights", pd.Series(0, index=x.index)), errors="coerce").reindex(top_keys).fillna(0.0)
        rows_elig.append(elig.values)
        rows_sel.append((w > 1e-6).astype(float).values)
        dates.append(date)
    fig.add_trace(go.Heatmap(z=np.asarray(rows_elig).T, x=dates, y=top_keys, colorscale="Blues", name="Eligible ETFs", visible=True))
    fig.add_trace(go.Heatmap(z=np.asarray(rows_sel).T, x=dates, y=top_keys, colorscale="Greens", name="Selected non-zero weights", visible=False))
    fig.update_layout(
        height=900,
        template="plotly_dark",
        title="Filtered ETF universe vs selected non-zero weights",
        updatemenus=[dict(type="dropdown", x=0.0, y=1.05, buttons=[
            dict(label="Filtered / eligible", method="update", args=[{"visible": [True, False]}]),
            dict(label="Selected non-zero", method="update", args=[{"visible": [False, True]}]),
        ])],
    )
    return fig


def _metrics_table_html(metrics_df: pd.DataFrame) -> str:
    if metrics_df.empty:
        return "<p>Metrics unavailable.</p>"
    pct_cols = [c for c in metrics_df.columns if any(k in c for k in ["return", "volatility", "drawdown", "var_", "cvar_", "win_rate", "expectancy", "avg_win", "avg_loss"])]
    df = metrics_df.copy()
    for c in pct_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    for c in df.columns:
        if c not in pct_cols and df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return df.to_html(index=False, classes="dataframe metrics-table")


def build_interactive_report(
    result_folder: str | Path,
    *,
    output_html: str | Path | None = None,
    benchmark_returns: pd.Series | None = None,
    title: str = "ETF Portfolio Monthly Rebalancing Dashboard",
) -> Path:
    folder = Path(result_folder)
    folder.mkdir(parents=True, exist_ok=True)
    pnl = _load_optional(folder / "pnl.csv")
    real = _load_optional(folder / "real.csv")
    preds = _load_optional(folder / "preds.csv")
    model = _load_optional(folder / "model.csv")
    forecast = _load_optional(folder / "forecast_risk.csv")
    dynamic = _load_optional(folder / "dynamic_parameter_history.csv")
    audit = _load_optional(folder / "selection_audit.csv")
    smart = _load_optional(folder / "smart_rebalance_audit.csv")
    weights_by_month = _weights_from_excel(folder / "weights.xlsx")

    if benchmark_returns is None and not real.empty:
        # Prefer pure SPY if present; otherwise use equal-weight of columns as a neutral benchmark.
        cols = [c for c in real.columns if str(c).upper() == "SPY"]
        real_num = _to_numeric_df(real)
        if cols:
            benchmark_returns = real_num[cols[0]].astype(float)
        elif not real_num.empty:
            benchmark_returns = real_num.mean(axis=1).astype(float)

    metrics_df = train_test_metrics(pnl)
    if not metrics_df.empty:
        metrics_df.to_csv(folder / "train_test_metrics.csv", index=False)

    sections = []
    sections.append(_figure_html(_plot_performance(pnl, benchmark_returns), include_plotlyjs=True))
    sections.append(_metrics_table_html(metrics_df))
    sections.append(_figure_html(_plot_rolling(pnl)))
    sections.append(_figure_html(_plot_histogram(pnl)))
    sections.append(_figure_html(_plot_turnover(weights_by_month)))
    sections.append(_figure_html(_heatmap_eligible_vs_selected(weights_by_month)))
    sections.append(_figure_html(_matrix_heatmap_from_weights(weights_by_month, real)))

    if not forecast.empty:
        fig = go.Figure()
        for c in forecast.columns:
            if c.lower() != "date":
                fig.add_trace(go.Scatter(x=forecast.index, y=pd.to_numeric(forecast[c], errors="coerce"), mode="lines", name=c))
        fig.update_layout(height=420, template="plotly_dark", title="Forward model-implied risk signals")
        sections.append(_figure_html(fig))

    if not dynamic.empty:
        fig = go.Figure()
        for c in dynamic.columns:
            if c.lower() != "date" and pd.api.types.is_numeric_dtype(pd.to_numeric(dynamic[c], errors="coerce")):
                fig.add_trace(go.Scatter(x=dynamic.index, y=pd.to_numeric(dynamic[c], errors="coerce"), mode="lines", name=c))
        fig.update_layout(height=440, template="plotly_dark", title="Dynamic optimizer parameters / regime scores")
        sections.append(_figure_html(fig))

    if not smart.empty:
        sections.append("<h2>Smart risk-signal audit</h2>" + smart.tail(120).to_html(classes="dataframe", index=True))
    if not audit.empty:
        sections.append("<h2>Nested train/validation/test selection audit</h2>" + audit.tail(120).to_html(classes="dataframe", index=True))
    sections.append("<h2>Monthly return and weight tables</h2>" + _monthly_tables_html(weights_by_month, real))

    style = """
    <style>
    body { margin: 0; background: #0f172a; color: #e5e7eb; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; }
    .wrap { max-width: 1480px; margin: auto; padding: 24px; }
    h1 { font-size: 32px; margin-bottom: 4px; }
    h2 { margin-top: 32px; }
    .sub { color: #94a3b8; margin-bottom: 24px; }
    table.dataframe { border-collapse: collapse; width: 100%; margin: 14px 0 28px 0; font-size: 12px; }
    table.dataframe th, table.dataframe td { border: 1px solid #334155; padding: 6px 8px; text-align: right; }
    table.dataframe th { background: #1e293b; color: #f8fafc; position: sticky; top: 0; }
    table.dataframe td:first-child, table.dataframe th:first-child { text-align: left; }
    details { background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 10px 14px; margin: 10px 0; }
    summary { cursor: pointer; font-weight: 650; }
    input { background: #0b1220; color: #e5e7eb; border: 1px solid #475569; padding: 6px; border-radius: 6px; margin: 10px 6px 10px 0; }
    </style>
    <script>
    function filterWeights(tableId, minVal) {
        const tbl = document.getElementById(tableId); if (!tbl) return;
        const minW = parseFloat(minVal || 0); const headers = Array.from(tbl.querySelectorAll('th')).map(x => x.innerText.toLowerCase());
        const idx = headers.indexOf('weights'); if (idx < 0) return;
        Array.from(tbl.querySelectorAll('tbody tr')).forEach(row => {
            const val = parseFloat(row.children[idx].innerText); row.style.display = (isNaN(val) || val >= minW) ? '' : 'none';
        });
    }
    </script>
    """
    html = f"<html><head><meta charset='utf-8'><title>{title}</title>{style}</head><body><div class='wrap'><h1>{title}</h1><div class='sub'>Generated from {folder}</div>" + "\n".join(sections) + "</div></body></html>"
    out = Path(output_html) if output_html is not None else folder / "interactive_report.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-folder", required=True)
    p.add_argument("--output-html", default=None)
    args = p.parse_args()
    out = build_interactive_report(args.result_folder, output_html=args.output_html)
    print(out)


if __name__ == "__main__":
    main()
