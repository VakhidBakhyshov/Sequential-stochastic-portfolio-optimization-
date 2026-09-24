from __future__ import annotations
import io
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

MACRO_SHEET = "5) Macro and rates predictors"
REGIME_SHEET = "6) Regime and stress indicators"
TCOST_SHEET = "7) Transaction costs and fees "

RF_COL = "RF"
ADJUSTED_CLOSE_FILENAME = "adjusted_close_prices.csv"
META_FILENAME = "Meta.csv"

FRENCH_START = "2016-01-31"
FRENCH_END = "2025-12-31"
FRENCH_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
FRENCH_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)
FRENCH_CACHE_FILENAME = "ff5_plus_momentum_monthly_2016_2025.csv"

EQUITY_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

LEGACY_OBSERVED_FACTOR_NAME_MAP = {
    "Mkt-RF": "Mkt-RF (market)",
    "SMB": "SMB (size)",
    "HML": "HML (value)",
    "RMW": "RMW (profitability)",
    "CMA": "CMA (investment)",
    "Mom": "MOM (momentum)",
}

MACRO_FACTOR_COLS = [
    "inflation",
    "real interest rates",
    "term structure",
    "SOFR O/N",
    "MM_12M",
    "IG OAS",
    "HY OAS",
    "NY Fed recession",
    "VIX",
]

MODEL_MACRO_DIFF_COLS = [
    "inflation",
    "real interest rates",
    "term structure",
    "SOFR O/N",
    "MM_12M",
    "IG OAS",
    "HY OAS",
    "NY Fed recession",
    "VIX",
]

MACRO_PERCENT_COLS = [
    "UST 1m YTM",
    "UST 3m YTM",
    "UST 2Y YTM",
    "UST 5Y YTM",
    "UST 10Y YTM",
    "UST 30Y YTM",
    "T10Y-3M",
    "T10Y-2Y",
    "SOFR O/N",
    "SOFR 1m",
    "SOFR 3m",
    "SOFR 6m",
    "SOFR 12m",
    "EFFR O/N",
    "LIBOR O/N",
    "LIBOR 1m",
    "LIBOR 3m",
    "LIBOR 6m",
    "LIBOR 12m",
    "CPI inflation rate, monthly",
    "5Y BEI",
    "10Y BEI",
    "IG OAS",
    "HY OAS",
    "IG T-Spread",
    "HY T-Spread",
    "NY Fed Recession",
]

ASSET_CLASS_MAP = {
    "Акции": "equity",
    "Облигации": "bond",
}

def _standardize_column_name(column: Any) -> str:
    text = str(column).strip()
    text = (
        text.replace("ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“", "-")
        .replace("Ã¢â‚¬â€œ", "-")
        .replace("â€“", "-")
        .replace("\xa0", " ")
    )
    if text.startswith("T10Y") and "3M" in text:
        return "T10Y-3M"
    if text.startswith("T10Y") and "2Y" in text:
        return "T10Y-2Y"
    return text


def _first_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        col_str = str(col).lower()
        if "date" in col_str or "unnamed" in col_str or col == ",%":
            return col
    return str(df.columns[0])


def _select_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = [col for col in columns if col in df.columns]
    return df[keep].copy()


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {label}: {missing}")


def clean_time_series_sheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_standardize_column_name(col) for col in out.columns]
    date_col = _first_date_column(out)
    out = out.rename(columns={date_col: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    out = (
        out.dropna(subset=["Date"])
        .sort_values("Date")
        .set_index("Date")
    )

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def convert_percent_like_to_decimal(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col] / 100.0
    return out


def align_daily_to_log_returns(df: pd.DataFrame, log_returns_index: pd.Index | None) -> pd.DataFrame:
    if log_returns_index is None:
        return df.sort_index()
    common_dates = df.index.intersection(log_returns_index)
    return df.loc[common_dates].sort_index()


def make_monthly_end_of_month(df_daily: pd.DataFrame) -> pd.DataFrame:
    return df_daily.resample("ME").last()


def _log_diff_positive(series: pd.Series) -> pd.Series:
    positive = series.where(series > 0.0)
    return np.log(positive).diff()


def stitch_spread_adjusted_series(
    df_daily: pd.DataFrame,
    libor_col: str,
    sofr_col: str,
    output_col: str,
) -> tuple[pd.Series, dict[str, Any]]:
    _require_columns(df_daily, [libor_col, sofr_col], label=output_col)

    libor = df_daily[libor_col].copy()
    sofr = df_daily[sofr_col].copy()

    first_sofr_date = sofr.first_valid_index()
    overlap = pd.concat([libor.rename("libor"), sofr.rename("sofr")], axis=1).dropna()

    spread = np.nan
    spread_adjusted = False
    adjusted_libor = libor.copy()
    note = ""

    if not overlap.empty:
        spread = float((overlap["libor"] - overlap["sofr"]).mean())
        adjusted_libor = libor - spread
        spread_adjusted = True
        note = f"Spread-adjusted using mean({libor_col} - {sofr_col}) over overlap."
    else:
        note = (
            f"No overlap between {libor_col} and {sofr_col}; stitched without adjustment. "
            "This may introduce a level break."
        )

    stitched = pd.Series(np.nan, index=df_daily.index, name=output_col, dtype=float)

    if first_sofr_date is None:
        stitched.loc[:] = adjusted_libor
        note = f"{sofr_col} never becomes available; {output_col} stays on the {libor_col} leg."
    else:
        before_sofr_mask = stitched.index < first_sofr_date
        stitched.loc[before_sofr_mask] = adjusted_libor.loc[before_sofr_mask]
        stitched.loc[~before_sofr_mask] = sofr.loc[~before_sofr_mask]

    metadata = {
        "series": output_col,
        "libor_col": libor_col,
        "sofr_col": sofr_col,
        "first_sofr_date": first_sofr_date,
        "overlap_count": int(len(overlap)),
        "spread_adjusted": spread_adjusted,
        "spread": spread,
        "note": note,
    }

    return stitched, metadata


def summarize_missingness(df: pd.DataFrame, group_map: dict[str, str] | None = None) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100.0),
        }
    )

    if group_map is not None:
        summary.insert(0, "group", summary.index.map(group_map))

    return summary.sort_values(["missing_pct", "missing_count"], ascending=False)


def load_french_monthly(url: str, target_cols: list[str]) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("latin1")

    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        parts = [_standardize_column_name(part) for part in line.split(",")]
        if parts[: len(target_cols) + 1] == [""] + target_cols:
            header_idx = i
            break

    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        parts = [_standardize_column_name(part) for part in line.split(",")]
        if parts and re.fullmatch(r"\d{6}", parts[0]):
            data_lines.append(line)
        elif len(data_lines) > 1:
            break

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [_standardize_column_name(c) for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Date"})
    _require_columns(df, ["Date", *target_cols], label=url)

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("Date").apply(pd.to_numeric, errors="coerce").sort_index() / 100.0

    return df.loc[FRENCH_START:FRENCH_END]


def load_fama_french_monthly(
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    project_root = Path(project_root)
    cache_path = project_root / "Ready_data" / FRENCH_CACHE_FILENAME
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ff5 = load_french_monthly(
            FRENCH_FF5_URL,
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        )
        mom = load_french_monthly(
            FRENCH_MOM_URL,
            ["Mom"],
        )

        combined = pd.concat([ff5, mom], axis=1).loc[FRENCH_START:FRENCH_END]
        combined.to_csv(cache_path)
        info = {
            "source": "download",
            "cache_path": str(cache_path),
        }
    except Exception as exc:
        if not cache_path.exists():
            raise RuntimeError(
                "Fama-French download failed and no local cache file is available."
            ) from exc

        combined = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        combined = combined.sort_index().loc[FRENCH_START:FRENCH_END]
        info = {
            "source": f"cache_fallback ({type(exc).__name__})",
            "cache_path": str(cache_path),
        }

    _require_columns(combined, EQUITY_FACTOR_COLS + [RF_COL], label="Fama-French monthly data")

    equity_factors = combined[EQUITY_FACTOR_COLS].copy()
    rf_df = combined[[RF_COL]].copy()
    return equity_factors, rf_df, info


def load_adjusted_close_prices(project_root: str | Path) -> pd.DataFrame:
    project_root = Path(project_root)
    adjusted_close_path = project_root / "notebooks" / ADJUSTED_CLOSE_FILENAME
    adjusted_close = pd.read_csv(adjusted_close_path, index_col=0, parse_dates=True)
    adjusted_close.index = pd.to_datetime(adjusted_close.index)
    adjusted_close = adjusted_close.sort_index().apply(pd.to_numeric, errors="coerce")
    return adjusted_close


def compute_monthly_simple_returns_from_adjusted_close(
    adjusted_close: pd.DataFrame,
    target_columns: list[str] | None = None,
) -> pd.DataFrame:
    if target_columns is not None:
        adjusted_close = adjusted_close.reindex(columns=target_columns)
    monthly_adjusted_close = adjusted_close.resample("ME").last()
    return monthly_adjusted_close.pct_change()


def build_macro_factor_frame(
    macro_monthly: pd.DataFrame,
    regime_monthly: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        macro_monthly,
        [
            "CPI inflation rate, monthly",
            "UST 10Y YTM",
            "10Y BEI",
            "T10Y-3M",
            "SOFR O/N",
            "MM_12M",
            "IG OAS",
            "HY OAS",
            "NY Fed Recession",
        ],
        label="macro_monthly",
    )
    _require_columns(regime_monthly, ["VIX"], label="regime_monthly")

    macro_factors_df = pd.DataFrame(index=macro_monthly.index)
    macro_factors_df["inflation"] = macro_monthly["CPI inflation rate, monthly"]
    macro_factors_df["real interest rates"] = macro_monthly["UST 10Y YTM"] - macro_monthly["10Y BEI"]
    macro_factors_df["term structure"] = macro_monthly["T10Y-3M"]
    macro_factors_df["SOFR O/N"] = macro_monthly["SOFR O/N"]
    macro_factors_df["MM_12M"] = macro_monthly["MM_12M"]
    macro_factors_df["IG OAS"] = macro_monthly["IG OAS"]
    macro_factors_df["HY OAS"] = macro_monthly["HY OAS"]
    macro_factors_df["NY Fed recession"] = macro_monthly["NY Fed Recession"]
    macro_factors_df["VIX"] = regime_monthly["VIX"]

    return macro_factors_df.sort_index()


def build_transformed_macro_factor_frame(
    raw_macro_factors_df: pd.DataFrame,
    diff_columns: list[str] | tuple[str, ...] = MODEL_MACRO_DIFF_COLS,
) -> pd.DataFrame:
    transformed = raw_macro_factors_df.copy()
    diff_columns_set = set(diff_columns)
    for column in transformed.columns:
        if column == "VIX" and column in diff_columns_set:
            transformed[column] = _log_diff_positive(transformed[column])
        elif column in diff_columns_set:
            transformed[column] = transformed[column].diff()
    return transformed.sort_index()


def build_transformation_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "equity_factors",
                "variable": "Mkt-RF, SMB, HML, RMW, CMA, Mom",
                "transform": "No additional transform beyond monthly download and conversion to decimals.",
                "model_note": "Kept in simple-return space for an excess-return model.",
            },
            {
                "dataset": "macro_factors_df",
                "variable": "inflation",
                "transform": "Convert percent-like workbook series to decimals, sample at month-end, then take first differences.",
                "model_note": "Modeled as the monthly change in inflation rather than the inflation rate level.",
            },
            {
                "dataset": "macro_factors_df",
                "variable": "real interest rates, term structure, SOFR O/N, MM_12M, IG OAS, HY OAS, NY Fed recession",
                "transform": "Convert percent-like workbook series to decimals, stitch MM_12M, sample at month-end, then take first differences.",
                "model_note": "Modeled as monthly shocks rather than levels.",
            },
            {
                "dataset": "macro_factors_df",
                "variable": "VIX",
                "transform": "Sample VIX at month-end, then take log differences.",
                "model_note": "Uses volatility shocks in log space rather than end-of-month level differences.",
            },
            {
                "dataset": "monthly_excess_returns",
                "variable": "ETF excess returns",
                "transform": "Read adjusted close prices, compute month-end simple returns with pct_change(), then subtract monthly Fama-French RF.",
                "model_note": "Kept in simple-return space as requested.",
            },
            {
                "dataset": "bond_factors_df",
                "variable": "duration",
                "transform": "Use the negative monthly change in UST 10Y YTM.",
                "model_note": "This serves as the bond duration shock factor.",
            },
        ]
    )

def build_bond_factor_suggestions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "suggested_factor": "duration",
                "proxy_idea": "Use long-duration Treasury excess returns over cash, or a negative monthly change in the 10Y Treasury yield.",
                "why_it_matters": "Duration is usually the first-order style exposure across bond ETFs.",
            },
        ]
    )

def load_asset_class_series(
    project_root: str | Path,
    target_columns: list[str] | None = None,
) -> pd.Series:
    meta_path = Path(project_root) / "Data" / META_FILENAME
    meta = pd.read_csv(meta_path).drop_duplicates(subset="Тикер", keep="first")

    _require_columns(meta, ["Тикер", "Объект инвестирования"], label=str(meta_path))

    asset_class = (
        meta.loc[:, ["Тикер", "Объект инвестирования"]]
        .rename(columns={"Тикер": "ticker", "Объект инвестирования": "raw_asset_class"})
        .assign(asset_class=lambda frame: frame["raw_asset_class"].map(ASSET_CLASS_MAP))
        .dropna(subset=["ticker"])
        .drop_duplicates(subset="ticker", keep="first")
        .set_index("ticker")["asset_class"]
    )

    if target_columns is not None:
        asset_class = asset_class.reindex(target_columns)

    return asset_class


def build_bond_factor_frame(macro_monthly: pd.DataFrame) -> pd.DataFrame:
    _require_columns(macro_monthly, ["UST 10Y YTM"], label="macro_monthly")

    bond_factors_df = pd.DataFrame(index=macro_monthly.index)
    bond_factors_df["duration"] = -macro_monthly["UST 10Y YTM"].diff()

    return bond_factors_df.sort_index()


def build_raw_bond_factor_frame(macro_monthly: pd.DataFrame) -> pd.DataFrame:
    _require_columns(macro_monthly, ["UST 10Y YTM"], label="macro_monthly")

    raw_bond_factors_df = pd.DataFrame(index=macro_monthly.index)
    raw_bond_factors_df["duration"] = macro_monthly["UST 10Y YTM"]

    return raw_bond_factors_df.sort_index()


def build_bayesian_factor_inputs(
    project_root: str | Path,
    log_returns: pd.DataFrame | None = None,
    monthly_log_returns: pd.DataFrame | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    workbook_path = project_root / "Data" / "Portfolio Macro Data.xlsx"

    macro_daily = pd.read_excel(workbook_path, sheet_name=MACRO_SHEET)
    regime_daily = pd.read_excel(workbook_path, sheet_name=REGIME_SHEET)
    tcost_daily = pd.read_excel(workbook_path, sheet_name=TCOST_SHEET)

    macro_daily_clean = clean_time_series_sheet(macro_daily)
    regime_daily_clean = clean_time_series_sheet(regime_daily)
    tcost_daily_clean = clean_time_series_sheet(tcost_daily)

    macro_daily_clean = convert_percent_like_to_decimal(
        macro_daily_clean,
        [col for col in MACRO_PERCENT_COLS if col in macro_daily_clean.columns],
    )

    aligned_log_index = None if log_returns is None else pd.DatetimeIndex(log_returns.index)
    macro_daily_clean = align_daily_to_log_returns(macro_daily_clean, aligned_log_index)
    regime_daily_clean = align_daily_to_log_returns(regime_daily_clean, aligned_log_index)
    tcost_daily_clean = align_daily_to_log_returns(tcost_daily_clean, aligned_log_index)

    mm_12m, mm_12m_meta = stitch_spread_adjusted_series(
        macro_daily_clean,
        libor_col="LIBOR 12m",
        sofr_col="SOFR 12m",
        output_col="MM_12M",
    )
    macro_daily_clean["MM_12M"] = mm_12m

    macro_monthly = make_monthly_end_of_month(macro_daily_clean)
    regime_monthly = make_monthly_end_of_month(regime_daily_clean)
    tcost_monthly = make_monthly_end_of_month(tcost_daily_clean)

    raw_macro_factors_df = build_macro_factor_frame(macro_monthly=macro_monthly, regime_monthly=regime_monthly)
    transformed_macro_factors_df = build_transformed_macro_factor_frame(raw_macro_factors_df=raw_macro_factors_df)
    raw_bond_factors_df = build_raw_bond_factor_frame(macro_monthly=macro_monthly)
    bond_factors_df = build_bond_factor_frame(macro_monthly=macro_monthly)
    equity_factors, rf_df, fama_french_download_info = load_fama_french_monthly(project_root=project_root)
    raw_equity_factors_df = equity_factors.copy()
    transformed_equity_factors_df = equity_factors.copy()
    observed_factor_df = equity_factors.rename(columns=LEGACY_OBSERVED_FACTOR_NAME_MAP)

    adjusted_close_prices = load_adjusted_close_prices(project_root=project_root)
    target_columns = None if monthly_log_returns is None else list(monthly_log_returns.columns)
    monthly_simple_returns = compute_monthly_simple_returns_from_adjusted_close(
        adjusted_close=adjusted_close_prices,
        target_columns=target_columns,
    )
    monthly_excess_returns = monthly_simple_returns.sub(rf_df[RF_COL], axis=0)
    monthly_excess_returns = monthly_excess_returns.dropna(how="all")

    tcost_cols = list(tcost_monthly.columns)
    tcost_df = _select_existing(tcost_monthly, tcost_cols)

    common_months = pd.DatetimeIndex(monthly_excess_returns.index)
    for df in [equity_factors, rf_df, raw_macro_factors_df, bond_factors_df, tcost_df]:
        common_months = common_months.intersection(df.index)
    if monthly_log_returns is not None:
        common_months = common_months.intersection(monthly_log_returns.index)
    common_months = common_months.sort_values()

    monthly_excess_returns = monthly_excess_returns.loc[common_months].sort_index()
    monthly_simple_returns = monthly_simple_returns.loc[common_months].sort_index()
    equity_factors = equity_factors.loc[common_months].sort_index()
    raw_equity_factors_df = raw_equity_factors_df.loc[common_months].sort_index()
    transformed_equity_factors_df = transformed_equity_factors_df.loc[common_months].sort_index()
    observed_factor_df = observed_factor_df.loc[common_months].sort_index()
    rf_df = rf_df.loc[common_months].sort_index()
    raw_macro_factors_df = raw_macro_factors_df.loc[common_months].sort_index()
    transformed_macro_factors_df = transformed_macro_factors_df.loc[common_months].sort_index()
    macro_factors_df = raw_macro_factors_df.copy()
    raw_bond_factors_df = raw_bond_factors_df.loc[common_months].sort_index()
    bond_factors_df = bond_factors_df.loc[common_months].sort_index()
    tcost_df = tcost_df.loc[common_months].sort_index()

    monthly_log_returns_aligned = None
    if monthly_log_returns is not None:
        monthly_log_returns_aligned = monthly_log_returns.loc[common_months].sort_index()

    observed_features_monthly = pd.concat(
        [rf_df, equity_factors, macro_factors_df, bond_factors_df, tcost_df],
        axis=1,
    ).sort_index()

    X_monthly = pd.concat(
        [equity_factors, macro_factors_df, bond_factors_df, tcost_df],
        axis=1,
    ).sort_index()

    raw_factor_blocks_df = pd.concat(
        [raw_equity_factors_df, raw_macro_factors_df, raw_bond_factors_df],
        axis=1,
    ).sort_index()

    transformed_factor_blocks_df = pd.concat(
        [transformed_equity_factors_df, transformed_macro_factors_df, bond_factors_df],
        axis=1,
    ).sort_index()

    group_map = {RF_COL: "rf"}
    group_map.update({col: "equity_factor" for col in equity_factors.columns})
    group_map.update({col: "macro_factor" for col in macro_factors_df.columns})
    group_map.update({col: "bond_factor" for col in bond_factors_df.columns})
    group_map.update({col: "transaction_cost" for col in tcost_df.columns})

    missing_pct_by_factor = summarize_missingness(observed_features_monthly, group_map=group_map)
    money_market_stitch_info = pd.DataFrame([mm_12m_meta]).set_index("series")
    transformation_notes = build_transformation_notes()
    bond_factor_suggestions = build_bond_factor_suggestions()
    asset_class_series = load_asset_class_series(
        project_root=project_root,
        target_columns=None if target_columns is None else list(target_columns),
    )

    return {
        "macro_daily_clean": macro_daily_clean,
        "regime_daily_clean": regime_daily_clean,
        "tcost_daily_clean": tcost_daily_clean,
        "monthly_log_returns": monthly_log_returns_aligned,
        "monthly_simple_returns": monthly_simple_returns,
        "monthly_excess_returns": monthly_excess_returns,
        "rf_df": rf_df,
        "equity_factors": equity_factors,
        "raw_equity_factors_df": raw_equity_factors_df,
        "transformed_equity_factors_df": transformed_equity_factors_df,
        "observed_factor_df": observed_factor_df,
        "macro_factors_df": macro_factors_df,
        "raw_macro_factors_df": raw_macro_factors_df,
        "transformed_macro_factors_df": transformed_macro_factors_df,
        "bond_factors_df": bond_factors_df,
        "raw_bond_factors_df": raw_bond_factors_df,
        "transformed_bond_factors_df": bond_factors_df,
        "tcost_df": tcost_df,
        "observed_features_monthly": observed_features_monthly,
        "X_monthly": X_monthly,
        "raw_factor_blocks_df": raw_factor_blocks_df,
        "transformed_factor_blocks_df": transformed_factor_blocks_df,
        "asset_class_series": asset_class_series,
        "missing_pct_by_factor": missing_pct_by_factor,
        "money_market_stitch_info": money_market_stitch_info,
        "transformation_notes": transformation_notes,
        "bond_factor_suggestions": bond_factor_suggestions,
        "fama_french_download_info": fama_french_download_info,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    ready_dir = project_root / "Ready_data"

    log_returns = pd.read_csv(ready_dir / "etf_log_returns.csv", index_col=0, parse_dates=True)
    eligibility_matrix = pd.read_csv(
        ready_dir / "rebalance_universe_matrix.csv",
        index_col=0,
        parse_dates=True,
    )
    eligibility_matrix = (
        eligibility_matrix
        .replace({"True": True, "False": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )

    eligible_once_tickers = eligibility_matrix.any(axis=0).loc[lambda s: s].index.tolist()
    monthly_log_returns = log_returns.loc[:, eligible_once_tickers].resample("ME").sum(min_count=1)

    outputs = build_bayesian_factor_inputs(
        project_root=project_root,
        log_returns=log_returns,
        monthly_log_returns=monthly_log_returns,
    )
