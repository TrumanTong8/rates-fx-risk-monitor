# views/commodities/spr.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.commodities.charts import plot_spr_history
from src.commodities.config import SPR_SERIES_ID
from src.data import cache

DEFAULT_START_DATE = pd.Timestamp("2020-01-01").date()

st.title("Strategic Petroleum Reserve")
st.caption("Historical view of US Strategic Petroleum Reserve stocks from EIA.")

try:
    spr = cache.eia_series(SPR_SERIES_ID).sort_index().astype(float) / 1000.0
except Exception as exc:
    st.error(f"Could not load SPR data: {type(exc).__name__}: {exc}")
    st.stop()

if spr.empty:
    st.info("No SPR data is available.")
    st.stop()

min_date = spr.index.min().date()
max_date = spr.index.max().date()
default_start = max(DEFAULT_START_DATE, min_date)

st.session_state.setdefault("energy_spr_start", default_start)
st.session_state.setdefault("energy_spr_end", max_date)
st.session_state.setdefault("energy_spr_range", (st.session_state["energy_spr_start"], st.session_state["energy_spr_end"]))


def sync_from_dates() -> None:
    start = st.session_state["energy_spr_start"]
    end = st.session_state["energy_spr_end"]
    if start > end:
        start, end = end, start
        st.session_state["energy_spr_start"] = start
        st.session_state["energy_spr_end"] = end
    st.session_state["energy_spr_range"] = (start, end)


def sync_from_slider() -> None:
    start, end = st.session_state["energy_spr_range"]
    st.session_state["energy_spr_start"] = start
    st.session_state["energy_spr_end"] = end


controls = st.columns([1, 1, 2])
controls[0].date_input(
    "Start",
    min_value=min_date,
    max_value=max_date,
    key="energy_spr_start",
    on_change=sync_from_dates,
)
controls[1].date_input(
    "End",
    min_value=min_date,
    max_value=max_date,
    key="energy_spr_end",
    on_change=sync_from_dates,
)
controls[2].slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    key="energy_spr_range",
    on_change=sync_from_slider,
)

start = pd.Timestamp(st.session_state["energy_spr_start"])
end = pd.Timestamp(st.session_state["energy_spr_end"])
filtered = spr[(spr.index >= start) & (spr.index <= end)]
if filtered.empty:
    st.warning("No SPR observations in the selected range.")
    st.stop()

latest = filtered.iloc[-1]
first = filtered.iloc[0]
kpi_cols = st.columns(3)
kpi_cols[0].metric("Latest", f"{latest:,.2f} Mbbl")
kpi_cols[1].metric("Change over range", f"{latest - first:+,.2f} Mbbl")
kpi_cols[2].metric("Observations", f"{len(filtered):,}")

st.plotly_chart(plot_spr_history(filtered), width="stretch")
