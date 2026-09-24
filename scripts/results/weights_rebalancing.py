from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from loguru import logger


def main():
    # folder_name = "bayessian_cvar_returns_no_previous"
    # folder_name = "advanced_bayessian_walk_forward"
    # folder_name = "advanced_bayessian_walk_forward_no_max"
    folder_name = "new_liquidity_weighted"
    
    inputs = Path.cwd() / "results" / folder_name
    excel_file = Path(inputs / "weights.xlsx")
    sheets = pd.read_excel(excel_file, sheet_name=None)

    data = []
    for sheet_name, df in sheets.items():
        df.columns = df.columns.str.strip()

        if "Key" not in df.columns or "weights" not in df.columns:
            continue

        temp = df[["Key", "weights"]].dropna()
        temp = temp[temp["weights"] > 0]
        temp["date"] = pd.to_datetime(sheet_name)
        temp = temp.rename(columns={"Key": "ETF", "weights": "weight"})

        data.append(temp[["date", "ETF", "weight"]])

    df = pd.concat(data)

    weights = (
        df.pivot_table(index="date", columns="ETF", values="weight", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )

    # Convert percentages to decimals if needed
    if weights.max().max() > 1.5:
        weights = weights / 100

    weights = weights.div(weights.sum(axis=1), axis=0) # Normalize each month to 100%

    # top_n = 20
    # top_etfs = weights.max().sort_values(ascending=False).head(top_n).index
    # plot_df = weights[top_etfs].copy()
    # plot_df["Other ETFs"] = weights.drop(columns=top_etfs).sum(axis=1)

    # II way - Keep only ETF weights >= min_weight at least once - only etfs weights >= min_weight
    min_weight = 0
    plot_df = weights.loc[:, (weights >= min_weight).any(axis=0)].copy()
    plot_df = plot_df.where(plot_df >= min_weight, 0) # Optional: hide tiny weights in months where ETF < min_weight

    # Plot stacked bars
    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(18, 8),
        width=0.9
    )

    ax.set_title("ETF portfolio composition over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Portfolio weight")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="ETF", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.xticks(
        range(len(plot_df.index)),
        [d.strftime("%Y-%m") for d in plot_df.index],
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    
    output_file = excel_file.parent / "etf_stacked_weights_filtered_dynamic_max.png"
    plt.savefig(output_file, dpi=180, bbox_inches="tight")
    logger.info(f"Saved chart to: {output_file}")



if __name__ == "__main__":
    main()
