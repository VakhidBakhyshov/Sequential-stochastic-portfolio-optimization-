import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Union
from beartype import beartype
from matplotlib.colors import ListedColormap
from loguru import logger


@beartype
def plot_heatmap(sheets: dict[Any, pd.DataFrame], output: Union[Path, None]=None):
    data = {}

    for sheet_name, df in sheets.items():
        df.columns = df.columns.str.strip()

        if "Key" not in df.columns or "Value" not in df.columns:
            continue

        df["Key"] = df["Key"].astype(str).str.strip()
        df["Value"] = df["Value"].fillna(0).astype(int)

        # sheet_name is the date
        data[sheet_name] = df.set_index("Key")["Value"]

    # Combine into one matrix - rows = ETFs, columns = dates
    heatmap_df = pd.DataFrame(data).fillna(0).astype(int)

    # Sort dates
    heatmap_df.columns = pd.to_datetime(heatmap_df.columns)
    heatmap_df = heatmap_df.sort_index(axis=1)

    # Keep only ETFs where Value = 1 at least once
    heatmap_df = heatmap_df[heatmap_df.sum(axis=1) > 0]

    # Optional: sort ETFs by number of selected months
    heatmap_df = heatmap_df.loc[heatmap_df.sum(axis=1).sort_values(ascending=False).index]

    # Two colors: 0 = white, 1 = black
    cmap = ListedColormap(["white", "black"])

    plt.figure(figsize=(18, max(8, len(heatmap_df) * 0.12)))

    plt.imshow(
        heatmap_df.values,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    plt.xticks(
        ticks=range(len(heatmap_df.columns)),
        labels=[d.strftime("%Y-%m-%d") for d in heatmap_df.columns],
        rotation=90,
        fontsize=8
    )

    plt.yticks(
        ticks=range(len(heatmap_df.index)),
        labels=heatmap_df.index,
        fontsize=6
    )

    plt.xlabel("Date")
    plt.ylabel("ETF ticker")
    plt.title("ETF Selection Heatmap: Value = 1")
    plt.tight_layout()
    
    if output is not None:
        plt.savefig(output / "heatmap.png", bbox_inches="tight")
        logger.info(f'Successfully saved heatmap to {output / "heatmap.png"}')


def main():
    filepath = Path.cwd() / "datasets" / "excel"
    first_business_days = pd.read_excel(filepath / "business_dates.xlsx", index_col=0)
    first_business_days = first_business_days['Values'].iloc[1:]
    sheet_names_str = [key.strftime('%Y-%m-%d') for key in first_business_days]

    excel_file = filepath / "filtered_weights_1.xlsx"
    # excel_file = filepath.parent / "monthly_filtration" / "etf_values_by_date.xlsx"
    
    sheets = pd.read_excel(excel_file, sheet_name=sheet_names_str)
    plot_heatmap(sheets, output=Path.cwd() / "plots")



if __name__ == "__main__":
    main()
