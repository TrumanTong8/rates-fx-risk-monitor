from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


def make_technical_chart(frame: pd.DataFrame):
    recent = frame.tail(2600).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=recent["date"],
            open=recent["open"],
            high=recent["high"],
            low=recent["low"],
            close=recent["close"],
            name="Price",
        )
    )
    for window, color in [(20, "#2563eb"), (50, "#16a34a"), (100, "#f59e0b"), (200, "#dc2626")]:
        fig.add_trace(
            go.Scatter(
                x=recent["date"],
                y=recent[f"ma_{window}"],
                mode="lines",
                name=f"{window}D MA",
                line={"width": 1.5, "color": color},
            )
        )
    fig.update_layout(
        height=560,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend_orientation="h",
    )
    return fig

