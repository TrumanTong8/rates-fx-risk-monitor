from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TFF_PARTICIPANTS = [
    ("Dealer", "dealer_positions_long_all", "dealer_positions_short_all"),
    ("Asset Manager", "asset_mgr_positions_long", "asset_mgr_positions_short"),
    ("Leveraged Money", "lev_money_positions_long", "lev_money_positions_short"),
    ("Other Reportables", "other_rept_positions_long", "other_rept_positions_short"),
    ("Non-Reportables", "nonrept_positions_long_all", "nonrept_positions_short_all"),
]

DISAGG_PARTICIPANTS = [
    ("Producer / Merchant", "prod_merc_positions_long", "prod_merc_positions_short"),
    ("Swap Dealer", "swap_positions_long_all", "swap__positions_short_all"),
    ("Managed Money", "m_money_positions_long_all", "m_money_positions_short_all"),
    ("Other Reportables", "other_rept_positions_long", "other_rept_positions_short"),
    ("Non-Reportables", "nonrept_positions_long_all", "nonrept_positions_short_all"),
]


def participant_columns(cot: pd.DataFrame) -> list[tuple[str, str, str]]:
    if cot.empty:
        return []
    source = str(cot["source_url"].iloc[-1]) if "source_url" in cot.columns else ""
    candidates = DISAGG_PARTICIPANTS if "kh3c-gbw2" in source else TFF_PARTICIPANTS
    return [(name, long_col, short_col) for name, long_col, short_col in candidates if long_col in cot.columns and short_col in cot.columns]


def headline_spec_columns(cot: pd.DataFrame) -> tuple[str | None, str | None, str]:
    columns = participant_columns(cot)
    for label, long_col, short_col in columns:
        if label in ["Leveraged Money", "Managed Money"]:
            return long_col, short_col, label
    if columns:
        label, long_col, short_col = columns[0]
        return long_col, short_col, label
    return None, None, "Speculative"


def score_heatmap(frame: pd.DataFrame):
    columns = ["Seasonality", "Technicals", "Positioning", "Event Risk", "Regime"]
    data = frame.set_index("Expression")[columns]
    fig = px.imshow(data, text_auto=".0f", aspect="auto", color_continuous_scale="RdYlGn", zmin=0, zmax=100)
    fig.update_layout(height=max(320, 48 * len(data) + 140), margin={"l": 10, "r": 10, "t": 30, "b": 10})
    return fig


def event_timeline(events: pd.DataFrame):
    if events.empty:
        return None
    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["impact_level"] = frame["impact"].map({"High": 2, "Medium": 1}).fillna(1)
    hover = ["title", "forecast", "previous"]
    if "source" in frame.columns:
        hover.append("source")
    fig = px.scatter(frame, x="date", y="impact_level", color="impact", hover_data=hover, labels={"impact_level": "Impact"})
    fig.update_yaxes(tickvals=[1, 2], ticktext=["Medium", "High"])
    fig.update_layout(height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    return fig


def regime_matrix(selected_regime: str, regimes: pd.DataFrame | None = None):
    if regimes is None or regimes.empty or "regime" not in regimes.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical regime data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 30, "b": 10})
        return fig

    counts = regimes["regime"].value_counts().rename_axis("Regime").reset_index(name="Months")
    counts["Selected"] = counts["Regime"].eq(selected_regime)
    fig = px.bar(
        counts.sort_values("Months", ascending=True),
        x="Months",
        y="Regime",
        orientation="h",
        color="Selected",
        color_discrete_map={True: "#16a34a", False: "#94a3b8"},
        hover_data=["Months"],
    )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
        xaxis_title="Historical months",
        yaxis_title="",
    )
    fig.add_annotation(text=f"Selected: {selected_regime}", xref="paper", yref="paper", x=0, y=1.12, showarrow=False)
    return fig


def positioning_chart(cot: pd.DataFrame):
    if cot.empty:
        return None
    long_col, short_col, category = headline_spec_columns(cot)
    if not long_col or not short_col:
        return None

    frame = cot.copy()
    frame["long"] = pd.to_numeric(frame[long_col], errors="coerce")
    frame["short"] = pd.to_numeric(frame[short_col], errors="coerce")
    frame = frame.dropna(subset=["date", "long", "short"]).sort_values("date")
    if frame.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["long"],
            mode="lines",
            name=f"{category} gross long",
            line={"color": "#16a34a", "width": 2.0},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["short"],
            mode="lines",
            name=f"{category} gross short",
            line={"color": "#f59e0b", "width": 2.0},
        )
    )
    fig.update_layout(
        dragmode="pan",
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified",
        legend_orientation="h",
        xaxis={"rangeslider": {"visible": True}, "fixedrange": False},
        yaxis={"title": f"{category} gross contracts", "gridcolor": "#e5e7eb"},
        template="plotly_white",
    )
    return fig


def net_position_chart(cot: pd.DataFrame):
    if cot.empty:
        return None
    long_col, short_col, category = headline_spec_columns(cot)
    if not long_col or not short_col:
        return None

    frame = cot.copy()
    frame["long"] = pd.to_numeric(frame[long_col], errors="coerce")
    frame["short"] = pd.to_numeric(frame[short_col], errors="coerce")
    frame["net_position"] = frame["long"] - frame["short"]
    frame = frame.dropna(subset=["date", "long", "short", "net_position"]).sort_values("date")
    if frame.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["net_position"],
            mode="lines",
            name=f"{category} net position",
            line={"color": "#2563eb", "width": 2.2},
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#475569",
        opacity=0.7,
    )
    fig.update_layout(
        dragmode="pan",
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified",
        xaxis={"rangeslider": {"visible": True}, "fixedrange": False},
        yaxis={"title": f"{category} net contracts", "gridcolor": "#dbeafe"},
        template="plotly_white",
    )
    return fig


def net_percent_open_interest_chart(cot: pd.DataFrame):
    if cot.empty:
        return None
    long_col, short_col, category = headline_spec_columns(cot)
    oi_col = "open_interest_all" if "open_interest_all" in cot.columns else "open_interest" if "open_interest" in cot.columns else None
    if not long_col or not short_col or not oi_col:
        return None

    frame = cot.copy()
    frame["long"] = pd.to_numeric(frame[long_col], errors="coerce")
    frame["short"] = pd.to_numeric(frame[short_col], errors="coerce")
    frame["open_interest"] = pd.to_numeric(frame[oi_col], errors="coerce")
    frame["net_pct_oi"] = (frame["long"] - frame["short"]) / frame["open_interest"].replace(0, pd.NA)
    frame = frame.dropna(subset=["date", "net_pct_oi"]).sort_values("date")
    if frame.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["net_pct_oi"],
            mode="lines",
            name=f"{category} net / OI",
            line={"color": "#7c3aed", "width": 2.2},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.7)
    fig.update_layout(
        dragmode="pan",
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified",
        xaxis={"rangeslider": {"visible": True}, "fixedrange": False},
        yaxis={"title": f"{category} net as % of open interest", "tickformat": ".0%", "gridcolor": "#ede9fe"},
        template="plotly_white",
    )
    return fig


def participant_breakdown_chart(cot: pd.DataFrame):
    if cot.empty:
        return None
    latest = cot.sort_values("date").iloc[-1]
    rows = []
    for participant, long_col, short_col in participant_columns(cot):
        long_value = pd.to_numeric(pd.Series([latest[long_col]]), errors="coerce").iloc[0]
        short_value = pd.to_numeric(pd.Series([latest[short_col]]), errors="coerce").iloc[0]
        if pd.isna(long_value) or pd.isna(short_value):
            continue
        rows.append({"Participant": participant, "Side": "Long", "Contracts": float(long_value)})
        rows.append({"Participant": participant, "Side": "Short", "Contracts": -float(short_value)})
    if not rows:
        return None

    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="Participant",
        y="Contracts",
        color="Side",
        barmode="relative",
        color_discrete_map={"Long": "#16a34a", "Short": "#dc2626"},
    )
    fig.add_hline(y=0, line_color="#475569", opacity=0.7)
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="Latest gross contracts, shorts shown negative",
        xaxis_title="",
        hovermode="x unified",
        legend_orientation="h",
        template="plotly_white",
    )
    return fig


def participant_net_percent_oi_chart(cot: pd.DataFrame):
    if cot.empty:
        return None
    oi_col = "open_interest_all" if "open_interest_all" in cot.columns else "open_interest" if "open_interest" in cot.columns else None
    if not oi_col:
        return None

    frame = cot.copy().sort_values("date")
    frame["open_interest"] = pd.to_numeric(frame[oi_col], errors="coerce")
    rows = []
    for participant, long_col, short_col in participant_columns(cot):
        data = frame[["date", long_col, short_col, "open_interest"]].copy()
        data["long"] = pd.to_numeric(data[long_col], errors="coerce")
        data["short"] = pd.to_numeric(data[short_col], errors="coerce")
        data["Net % OI"] = (data["long"] - data["short"]) / data["open_interest"].replace(0, pd.NA)
        data = data.dropna(subset=["date", "Net % OI"])
        for _, row in data.iterrows():
            rows.append({"Date": row["date"], "Participant": participant, "Net % OI": float(row["Net % OI"])})
    if not rows:
        return None

    plot_frame = pd.DataFrame(rows)
    fig = px.line(
        plot_frame,
        x="Date",
        y="Net % OI",
        color="Participant",
        labels={"Net % OI": "Net position as % of open interest"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.7)
    fig.update_layout(
        dragmode="pan",
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified",
        legend_orientation="h",
        xaxis={"rangeslider": {"visible": True}, "fixedrange": False},
        yaxis={"tickformat": ".0%", "gridcolor": "#e5e7eb"},
        template="plotly_white",
    )
    return fig


def participant_weekly_net_change_chart(cot: pd.DataFrame):
    if cot.empty:
        return None
    frame = cot.copy().sort_values("date")
    if len(frame) < 2:
        return None
    latest = frame.iloc[-1]
    prior = frame.iloc[-2]
    rows = []
    for participant, long_col, short_col in participant_columns(cot):
        latest_long = pd.to_numeric(pd.Series([latest[long_col]]), errors="coerce").iloc[0]
        latest_short = pd.to_numeric(pd.Series([latest[short_col]]), errors="coerce").iloc[0]
        prior_long = pd.to_numeric(pd.Series([prior[long_col]]), errors="coerce").iloc[0]
        prior_short = pd.to_numeric(pd.Series([prior[short_col]]), errors="coerce").iloc[0]
        if pd.isna(latest_long) or pd.isna(latest_short) or pd.isna(prior_long) or pd.isna(prior_short):
            continue
        rows.append(
            {
                "Participant": participant,
                "Weekly Net Change": float((latest_long - latest_short) - (prior_long - prior_short)),
            }
        )
    if not rows:
        return None

    plot_frame = pd.DataFrame(rows)
    plot_frame["Direction"] = plot_frame["Weekly Net Change"].map(lambda value: "Net increased" if value >= 0 else "Net decreased")
    fig = px.bar(
        plot_frame,
        x="Participant",
        y="Weekly Net Change",
        color="Direction",
        color_discrete_map={"Net increased": "#16a34a", "Net decreased": "#dc2626"},
    )
    fig.add_hline(y=0, line_color="#475569", opacity=0.7)
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="1W net contract change",
        xaxis_title="",
        hovermode="x unified",
        showlegend=False,
        template="plotly_white",
    )
    return fig
