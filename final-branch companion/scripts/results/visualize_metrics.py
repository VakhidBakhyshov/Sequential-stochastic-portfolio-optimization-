import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import os
from typing import Union
from pathlib import Path

def plot_animated_timeseries(
    df, 
    time_col, 
    metric_cols, 
    use_subplots=False, 
    frame_step=1, 
    output_mode='both', 
    filename: Union[str, None]=None
):
    """
    Plots multiple time series metrics with synced Play/Pause animation and slider controls.
    Styled with a premium dark theme while preserving distinct neon-accented line data paths.
    """
    # Ensure time column is sorted
    df = df.sort_values(by=time_col).reset_index(drop=True)
    num_metrics = len(metric_cols)
    
    # Define grid specs and trace mapping configurations dynamically
    if use_subplots:
        total_rows = num_metrics + 1
        grid_specs = [[{"type": "xy"}] for _ in range(num_metrics)] + [[{"type": "table"}]]
        subplot_titles = [f"Metric: {col}" for col in metric_cols] + ["Current Period Values Snapshot"]
        trace_row_mapping = {idx: idx + 1 for idx in range(num_metrics)}
        table_row_idx = total_rows
    else:
        total_rows = 2
        grid_specs = [[{"type": "xy"}], [{"type": "table"}]]
        subplot_titles = ["Metrics Trend", "Current Period Values Snapshot"]
        trace_row_mapping = {idx: 1 for idx in range(num_metrics)}
        table_row_idx = 2
        
    # 1. Base Setup: Define Grid Layout First
    fig = make_subplots(
        rows=total_rows, 
        cols=1, 
        shared_xaxes=True if use_subplots else False,
        vertical_spacing=0.06 if use_subplots else 0.15,
        subplot_titles=subplot_titles,
        specs=grid_specs
    )

    # Vivid dark-mode friendly color palette (Distinct neon-leaning hex colors)
    colors = ['#00CC96', '#FF6692', '#636EFA', '#EF553B', '#AB63FA', '#19D3F3', '#E45756']

    # Dark Table Theme Styling Variables
    table_header_fill = '#1A1C23'  # Slate black header
    table_header_font = '#00CC96'  # Mint green accent font
    table_cell_fill = '#242834'    # Deep charcoal body cells
    table_cell_font = '#FFFFFF'    # Crisp white text value cells

    # 2. Add Initial Data (First frame snapshot)
    for idx, col in enumerate(metric_cols):
        color = colors[idx % len(colors)]
        row_target = trace_row_mapping[idx]
        
        fig.add_trace(
            go.Scatter(
                x=df[time_col][:2],
                y=df[col][:2],
                mode='lines+markers',
                name=col,
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
                showlegend=not use_subplots
            ),
            row=row_target, col=1
        )

    # Add the single-row dark table trace placeholder
    headers = [time_col] + metric_cols
    initial_vals = [str(df[time_col].iloc[1])[:10]] + [f"{df[col].iloc[1]:.2f}" for col in metric_cols]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=headers, 
                fill_color=table_header_fill, 
                align='center', 
                font=dict(size=12, weight='bold', color=table_header_font),
                line_color='#111318'
            ),
            cells=dict(
                values=[[v] for v in initial_vals], 
                fill_color=table_cell_fill, 
                align='center', 
                font=dict(size=11, color=table_cell_font),
                line_color='#111318'
            )
        ),
        row=table_row_idx, col=1
    )

    # 3. Construct Synchronized Frames
    frames = []
    for i in range(2, len(df) + 1, frame_step):
        frame_data = []
        
        # Build trending paths up to step index point i
        for idx, col in enumerate(metric_cols):
            color = colors[idx % len(colors)]
            trace = go.Scatter(
                x=df[time_col][:i],
                y=df[col][:i],
                mode='lines+markers',
                name=col,
                line=dict(color=color, width=2.5),
                marker=dict(size=4)
            )
            frame_data.append(trace)
            
        # Extract individual cell strings for this active slice position
        current_date_str = str(df[time_col].iloc[i-1])[:10]
        current_vals = [current_date_str] + [f"{df[col].iloc[i-1]:.2f}" for col in metric_cols]
        
        # Build individual row data dashboard snapshot state with dark palette properties
        table_trace = go.Table(
            header=dict(
                values=[time_col] + metric_cols, 
                fill_color=table_header_fill, 
                align='center', 
                font=dict(weight='bold', color=table_header_font),
                line_color='#111318'
            ),
            cells=dict(
                values=[[v] for v in current_vals], 
                fill_color=table_cell_fill, 
                align='center',
                font=dict(color=table_cell_font),
                line_color='#111318'
            )
        )
        frame_data.append(table_trace)
        
        frames.append(
            go.Frame(
                data=frame_data,
                name=f'frame{i}',
                layout=go.Layout(title_text=f"Timeline Progression — {current_date_str}")
            )
        )
    fig.frames = frames

    # 4. Master Layout Customization (Axis Ranges, Dark Template, Buttons & Sliders)
    x_range = [df[time_col].min(), df[time_col].max()]
    
    # Base layout configs applied via global template engine injection
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111318', # Ultra deep background body tone
        plot_bgcolor='#111318',  # Synchronize inner grids color palette
        title='Animated Metrics Panel',
        height=350 + (220 * num_metrics) if use_subplots else 550,
    )
    
    if use_subplots:
        for idx in range(num_metrics):
            col = metric_cols[idx]
            y_min, y_max = df[col].min(), df[col].max()
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
            
            fig.update_layout({
                f'xaxis{idx+1 if idx > 0 else ""}': dict(range=x_range, autorange=False, gridcolor='#222632'),
                f'yaxis{idx+1 if idx > 0 else ""}': dict(range=[y_min - margin, y_max + margin], autorange=False, gridcolor='#222632')
            })
    else:
        all_y_min = df[metric_cols].min().min()
        all_y_max = df[metric_cols].max().max()
        margin = (all_y_max - all_y_min) * 0.1
        fig.update_layout(
            xaxis=dict(range=x_range, autorange=False, gridcolor='#222632'),
            yaxis=dict(range=[all_y_min - margin, all_y_max + margin], autorange=False, gridcolor='#222632')
        )

    # Setup animation UI menus with high-contrast colors
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'direction': 'left',
            'pad': {'r': 10, 't': 40},
            'showactive': False,
            'x': 0.1, 'y': -0.12,
            'xanchor': 'right', 'yanchor': 'top',
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top', 'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 14, 'color': '#FFFFFF'},
                'prefix': 'Time Step: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 0},
            'pad': {'b': 10, 't': 40},
            'len': 0.9,
            'x': 0.1, 'y': -0.12,
            'steps': [
                {
                    'args': [
                        [f'frame{k}'],
                        {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ],
                    'label': str(df[time_col].iloc[k-1])[:10],
                    'method': 'animate'
                } for k in range(2, len(df) + 1, max(1, frame_step))
            ]
        }]
    )

    # 5. Output Management Execution
    if output_mode in ['html', 'both'] and filename is not None:
        fig.write_html(filename, auto_open=False)
        print(f"Chart saved successfully as: {os.path.abspath(filename)}")
        
    if output_mode in ['browser', 'both']:
        fig.show()



def main():
    filepath = Path.cwd() / "datasets" / "monthly_filtration" / "monthly_etf_scores.csv"
    df = pd.read_csv(filepath)
    market_df = df[df["etf"]=='SPY']
    market_df
    
    metrics_cols = [
        "price_change",
        "volume_change",
        "volatility",
        "positive_days_ratio",
        "max_drawdown",
        "beta_to_benchmark",
        "aggregate_score"
    ]

    # Execution parameters: Change use_subplots=False if you want overlaid lines instead.
    plot_animated_timeseries(
        df=market_df,
        time_col='rebalance_date',
        metric_cols=metrics_cols,
        use_subplots=True,        
        frame_step=1,             
        output_mode='html',       
        filename=filepath.parent / 'market_subplots.html'
    )

    # # This will open a tab in your web browser AND save 'market_overlapped.html' locally
    # plot_animated_timeseries(
    #     df=market_df,
    #     time_col='rebalance_date',
    #     metric_cols=metrics_cols,
    #     use_subplots=False, # Table works perfectly now whether set to True or False
    #     frame_step=1,             
    #     output_mode='html',       
    #     filename=filepath.parent / 'market_overlapped.html'
    # )


if __name__ == "__main__":
    main()
