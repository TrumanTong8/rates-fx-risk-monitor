from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.commodities.backtest import Z_THRESHOLD, distribution_frame

SEASONAL_RANGE_YEARS = 5
SEASONAL_LONG_RANGE_YEARS = 10
# Accent for "current" lines; readable on both the light and dark Streamlit themes.
CURRENT_COLOR = "#f59e0b"
PRICE_COLOR = "#94a3b8"


def seasonal_profile(series: pd.Series, years: int = SEASONAL_RANGE_YEARS) -> tuple[pd.DataFrame, int]:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.empty:
        return pd.DataFrame(), pd.Timestamp.today().year

    current_year = int(values.index.max().year)
    history_mask = (values.index.year >= current_year - years) & (values.index.year <= current_year - 1)
    history = values[history_mask]
    current = values[values.index.year == current_year]

    hist_df = pd.DataFrame({"value": history})
    hist_df["week"] = hist_df.index.isocalendar().week.astype(int)
    profile = hist_df[hist_df["week"] <= 52].groupby("week")["value"].agg(["min", "max", "mean"]).reindex(range(1, 53))

    current_df = pd.DataFrame({"value": current})
    current_df["week"] = current_df.index.isocalendar().week.astype(int)
    current_profile = current_df[current_df["week"] <= 52].groupby("week")["value"].last()
    profile["current"] = current_profile
    return profile, current_year


def add_seasonal_range(
    fig: go.Figure,
    profile: pd.DataFrame,
    weeks: pd.Index,
    years: int,
    fillcolor: str,
    avg_color: str,
    avg_dash: str | None = None,
    show_range_legend: bool = True,
) -> None:
    fig.add_trace(go.Scatter(
        x=weeks,
        y=profile["max"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=weeks,
        y=profile["min"],
        mode="lines",
        fill="tonexty",
        fillcolor=fillcolor,
        line=dict(width=0),
        name=f"{years}Y range",
        showlegend=show_range_legend,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=weeks,
        y=profile["mean"],
        mode="lines",
        name=f"{years}Y avg",
        line=dict(color=avg_color, width=2, dash=avg_dash),
    ))


def plot_seasonal_inventory(series: pd.Series, title: str, ytitle: str) -> go.Figure:
    profile, current_year = seasonal_profile(series, SEASONAL_RANGE_YEARS)
    long_profile, _ = seasonal_profile(series, SEASONAL_LONG_RANGE_YEARS)
    fig = go.Figure()
    if profile.empty:
        fig.update_layout(title=title, height=420)
        return fig

    weeks = profile.index
    long_profile = long_profile.reindex(weeks)
    range_hover = np.stack([
        profile["min"].to_numpy(),
        profile["max"].to_numpy(),
        profile["mean"].to_numpy(),
        long_profile["min"].to_numpy(),
        long_profile["max"].to_numpy(),
        long_profile["mean"].to_numpy(),
        profile["current"].to_numpy(),
    ], axis=-1)

    if not long_profile[["min", "max", "mean"]].dropna(how="all").empty:
        add_seasonal_range(
            fig,
            long_profile,
            weeks,
            SEASONAL_LONG_RANGE_YEARS,
            "rgba(100, 116, 139, 0.16)",
            "#64748b",
            "dot",
        )
    add_seasonal_range(
        fig,
        profile,
        weeks,
        SEASONAL_RANGE_YEARS,
        "rgba(37, 99, 235, 0.28)",
        "#ef4444",
    )

    fig.add_trace(go.Scatter(
        x=weeks,
        y=profile["current"],
        mode="lines",
        customdata=range_hover,
        name=str(current_year),
        line=dict(color=CURRENT_COLOR, width=3),
        hovertemplate=(
            "Week %{x}<br>"
            f"{SEASONAL_RANGE_YEARS}Y min: %{{customdata[0]:,.2f}}<br>"
            f"{SEASONAL_RANGE_YEARS}Y max: %{{customdata[1]:,.2f}}<br>"
            f"{SEASONAL_RANGE_YEARS}Y avg: %{{customdata[2]:,.2f}}<br>"
            f"{SEASONAL_LONG_RANGE_YEARS}Y min: %{{customdata[3]:,.2f}}<br>"
            f"{SEASONAL_LONG_RANGE_YEARS}Y max: %{{customdata[4]:,.2f}}<br>"
            f"{SEASONAL_LONG_RANGE_YEARS}Y avg: %{{customdata[5]:,.2f}}<br>"
            f"{current_year}: %{{customdata[6]:,.2f}}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text=title, y=0.96, x=0, xanchor="left", font=dict(size=14)),
        height=460,
        margin=dict(l=42, r=24, t=70, b=92),
        xaxis_title="Week of year",
        yaxis_title=ytitle,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0,
            font=dict(size=10),
            traceorder="normal",
        ),
    )
    fig.update_xaxes(range=[1, 52], dtick=4)
    return fig


def plot_inventory_vs_oi(df: pd.DataFrame, name: str, inventory_units: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["inventory"], name="Inventory"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["open_interest"], name="Open interest"), secondary_y=True)
    fig.update_layout(height=430, margin=dict(l=40, r=40, t=45, b=35), title=f"{name}: inventory vs open interest")
    fig.update_yaxes(title_text=inventory_units, secondary_y=False)
    fig.update_yaxes(title_text="Contracts", secondary_y=True)
    return fig


def plot_line(df: pd.DataFrame, columns: list[str], title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig.update_layout(height=380, margin=dict(l=40, r=40, t=45, b=35), title=title, yaxis_title=ytitle)
    return fig


def plot_horizon_bars(summary: pd.DataFrame, metric: str, title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    colors = {"Long (z < -2)": "#2ca02c", "Short (z > +2)": "#d62728"}
    for direction in ["Long (z < -2)", "Short (z > +2)"]:
        subset = summary[summary["Direction"] == direction]
        fig.add_trace(go.Bar(
            x=subset["Horizon"],
            y=subset[metric],
            name=direction.replace(" (z < -2)", "").replace(" (z > +2)", ""),
            marker_color=colors[direction],
        ))
    fig.update_layout(
        title=title,
        yaxis_title=ytitle,
        xaxis_title="Forward horizon",
        barmode="group",
        height=360,
        margin=dict(l=40, r=30, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if metric == "Hit Rate %":
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.6)
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    return fig


def plot_signal_timeline(result: dict, title: str) -> go.Figure:
    price = result["price"]
    z = result["z"].clip(-6, 6)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.04,
        subplot_titles=["Price with inventory shock signals", "Inventory-change z-score"],
    )
    fig.add_trace(go.Scatter(x=price.index, y=price.values, name="Price", line=dict(color=PRICE_COLOR)), row=1, col=1)

    long_px = price.reindex(result["long_dates"]).dropna()
    short_px = price.reindex(result["short_dates"]).dropna()
    if not long_px.empty:
        fig.add_trace(go.Scatter(
            x=long_px.index,
            y=long_px.values,
            mode="markers",
            name="Long signal",
            marker=dict(symbol="triangle-up", size=10, color="#2ca02c", line=dict(color="#14532d", width=1)),
        ), row=1, col=1)
    if not short_px.empty:
        fig.add_trace(go.Scatter(
            x=short_px.index,
            y=short_px.values,
            mode="markers",
            name="Short signal",
            marker=dict(symbol="triangle-down", size=10, color="#d62728", line=dict(color="#7f1d1d", width=1)),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(x=z.index, y=z.values, name="Z-score", line=dict(color="#2563eb")), row=2, col=1)
    fig.add_hrect(y0=Z_THRESHOLD, y1=6, fillcolor="rgba(214,39,40,0.12)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=-6, y1=-Z_THRESHOLD, fillcolor="rgba(44,160,44,0.12)", line_width=0, row=2, col=1)
    fig.add_hline(y=Z_THRESHOLD, line_dash="dot", line_color="#d62728", row=2, col=1)
    fig.add_hline(y=-Z_THRESHOLD, line_dash="dot", line_color="#2ca02c", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.45, row=2, col=1)
    fig.update_layout(
        title=title,
        height=650,
        margin=dict(l=40, r=30, t=75, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", range=[-6.2, 6.2], row=2, col=1)
    return fig


def plot_return_distribution(result: dict, title: str) -> go.Figure:
    dist = distribution_frame(result)
    fig = go.Figure()
    colors = {"Long": "#2ca02c", "Short": "#d62728"}
    for direction in ["Long", "Short"]:
        subset = dist[dist["Direction"] == direction]
        if subset.empty:
            continue
        fig.add_trace(go.Box(
            x=subset["Horizon"],
            y=subset["Forward return %"],
            name=direction,
            marker_color=colors[direction],
            boxpoints="outliers",
        ))
    fig.update_layout(
        title=title,
        yaxis_title="Forward return %",
        xaxis_title="Forward horizon",
        boxmode="group",
        height=420,
        margin=dict(l=40, r=30, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    return fig


def plot_spr_history(series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name="SPR",
        mode="lines",
        line=dict(color=CURRENT_COLOR, width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>SPR: %{y:,.2f} million barrels<extra></extra>",
    ))
    fig.update_layout(
        title="US Strategic Petroleum Reserve",
        height=560,
        margin=dict(l=45, r=30, t=60, b=45),
        xaxis_title="Date",
        yaxis_title="Million barrels",
        hovermode="x unified",
    )
    return fig
