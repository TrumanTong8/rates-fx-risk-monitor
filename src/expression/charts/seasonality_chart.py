from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return hex_color
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _win_rate_color(win_rate: float) -> str:
    value = float(min(max(win_rate, 0.0), 1.0))
    color = px.colors.sample_colorscale("RdYlGn", [value])[0]
    return _hex_to_rgba(color, 0.82)


def make_seasonality_chart(stats: pd.DataFrame) -> go.Figure:
    """Create a side-by-side box-and-whisker dashboard for the 10Y / 20Y / 30Y seasonality windows."""
    if stats.empty:
        return go.Figure()

    lookbacks = [10, 20, 30]
    if "lookback_years" in stats.columns:
        unique_lookbacks = sorted(stats["lookback_years"].dropna().astype(int).unique().tolist())
        lookbacks = [year for year in [10, 20, 30] if year in unique_lookbacks] or [10, 20, 30]

    if not lookbacks:
        return go.Figure()

    text_color = "#1F2937"
    grid_color = "rgba(31, 41, 55, 0.15)"
    zero_line = "rgba(15, 23, 42, 0.7)"
    box_outline = "#0F172A"

    fig = make_subplots(
        rows=1,
        cols=len(lookbacks),
        subplot_titles=[f"{years}Y" for years in lookbacks],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for col_idx, years in enumerate(lookbacks, start=1):
        subset = stats[stats["lookback_years"].eq(years)].sort_values("month_index").reset_index(drop=True)
        if subset.empty:
            continue

        for _, row in subset.iterrows():
            month_name = MONTHS[int(row["month_index"]) - 1]
            q1 = float(row["q1"])
            median = float(row["median_return"])
            q3 = float(row["q3"])
            lower = float(row["lower"])
            upper = float(row["upper"])
            win_rate = float(row["win_rate"])
            avg_return = float(row["avg_return"])
            sample = int(row.get("sample", 0))

            fig.add_trace(
                go.Box(
                    x=[month_name],
                    q1=[q1],
                    median=[median],
                    q3=[q3],
                    lowerfence=[lower],
                    upperfence=[upper],
                    boxpoints=False,
                    fillcolor=_win_rate_color(win_rate),
                    line={"color": box_outline, "width": 2.5},
                    whiskerwidth=0.8,
                    width=0.7,
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Win rate: %{customdata[0]:.1%}<br>"
                        "Avg return: %{customdata[1]:.1%}<br>"
                        "Median: %{customdata[2]:.1%}<br>"
                        "Q1 / Q3: %{customdata[3]:.1%} / %{customdata[4]:.1%}<br>"
                        "Sample: %{customdata[5]}<extra></extra>"
                    ),
                    customdata=[[win_rate, avg_return, median, q1, q3, sample]],
                ),
                row=1,
                col=col_idx,
            )

            y_top = max(upper, q3, median, q1, lower, 0.0) * 1.18
            fig.add_annotation(
                x=month_name,
                y=y_top,
                text=f"{win_rate:.0%}",
                showarrow=False,
                font={"color": text_color, "size": 11},
                bgcolor="rgba(255,255,255,0.65)",
                borderpad=2,
                row=1,
                col=col_idx,
            )

        fig.add_trace(
            go.Scatter(
                x=[MONTHS[int(row["month_index"]) - 1] for _, row in subset.iterrows()],
                y=[float(row["avg_return"]) for _, row in subset.iterrows()],
                mode="lines+markers",
                name="Mean forward return" if col_idx == 1 else None,
                line={"color": "#111827", "width": 2.0, "dash": "dot"},
                marker={"size": 6, "color": "#111827"},
                showlegend=col_idx == 1,
                hovertemplate="<b>%{x}</b><br>Mean forward return: %{y:.2%}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(
            categoryorder="array",
            categoryarray=MONTHS,
            tickfont={"color": text_color},
            title_text="Month",
            title_font={"color": text_color},
            gridcolor=grid_color,
            zerolinecolor=zero_line,
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            tickformat=".0%",
            tickfont={"color": text_color},
            title_text="Forward return" if col_idx == 1 else "",
            title_font={"color": text_color},
            gridcolor=grid_color,
            zerolinecolor=zero_line,
            row=1,
            col=col_idx,
        )

    color_scale = [
        (0.0, "#D64545"),
        (0.25, "#F2C14E"),
        (0.5, "#E7E9A0"),
        (0.75, "#A8D08D"),
        (1.0, "#3FAE66"),
    ]
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=color_scale,
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Win rate", side="right"),
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                    outlinewidth=0,
                    thickness=12,
                    len=0.8,
                    x=1.02,
                    y=0.5,
                ),
                size=0,
            ),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=440,
        margin=dict(l=30, r=80, t=40, b=40),
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="#F5F7FA",
        font={"color": text_color},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.12, "xanchor": "left", "x": 0.0},
    )
    return fig
