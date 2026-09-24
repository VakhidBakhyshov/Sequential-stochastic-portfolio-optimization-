import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from beartype import beartype
from typing import Union
from plotly.subplots import make_subplots

from scripts.past.parser import *
from scripts.past.calculate import *
from scripts.past.portfolio import *


@beartype
def portfolio_heatmap(portfolio: dict[pd.Timestamp, dict[str, int]], save_path: Union[Path, None] = None) -> None:
    df = pd.DataFrame(portfolio).T # Convert to DataFrame: dates as index, tickers as columns
    
    fig = go.Figure(data=go.Heatmap(
        z=df.T.values,
        x=df.index.strftime('%Y-%m-%d'),
        y=df.columns, 
        colorscale=[[0, 'white'], [1, 'black']], showscale=False, hoverongaps=False,
        hovertemplate='Date: %{x}<br>Ticker: %{y}<br>Status: %{z}<extra></extra>',
        zmin=0, zmax=1
    ))

    fig.update_layout(
        title={'text': 'Ticker Availability (0 = Not Rebalancing, 1 = Rebalancing)','x': 0.5,'xanchor': 'center'},
        xaxis={'title': 'Date', 'tickangle': 45, 'nticks': min(20, len(df.index)), 'tickfont': {'size': 10}},
        yaxis={'title': 'Ticker', 'tickfont': {'size': 8}, 'autorange': 'reversed'},
        height=max(800, len(df.columns) * 3),
        width=max(1200, len(df.index) * 10), 
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        minor=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        minor=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    if save_path:
        plots_dir = save_path.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        html_path = plots_dir / "tickers_heatmap.html"
        fig.write_html(html_path)
        logger.info(f"Interactive portfolio heatmap saved to {html_path}")


@beartype
def plot_pie_diagram(portfolios: dict[pd.Timestamp, dict[str, int]], summmary: pd.DataFrame, save_path: Union[Path, None]) -> None:
    rows = len(portfolios)
    specs = []
    for _ in range(rows):
        row_specs = []
        for _ in range(3):
            row_specs.append({'type': 'pie'})
        specs.append(row_specs)
        
    subplot_titles = []
    for date in portfolios.keys():
        for _ in range(3):
            subplot_titles.append(date.strftime('%Y-%m-%d'))
    
    fig = make_subplots(rows=rows, cols=3, subplot_titles=subplot_titles, specs=specs, vertical_spacing=0.0001, horizontal_spacing=0.1)
    
    for ind, ticker_dict in enumerate(portfolios.values()):
        mask = summmary['Тикер'].isin(list(ticker_dict.keys()))
        filtered_df = summmary[mask][['Объект инвестирования', 'Сектор инвестирования', 'География инвестирования', 'Тикер']]
        
        for idx, col_name in enumerate(filtered_df.columns[:-1]):
            series = filtered_df[col_name].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=series.index, 
                    values=series.values,
                    name=col_name,
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3)
                ),
                row=ind+1, col=idx+1
            )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500*rows, width=1800, showlegend=True)
    
    if save_path:
        fig.write_html(save_path / "pies.html")
