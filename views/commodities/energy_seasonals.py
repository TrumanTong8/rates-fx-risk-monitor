# views/commodities/energy_seasonals.py
from __future__ import annotations

import streamlit as st

from src.commodities.charts import plot_seasonal_inventory
from src.commodities.config import ENERGY_SEASONAL_SERIES
from src.data import cache

st.title("Energy Seasonals")
st.caption("Current year versus the prior 5-year and 10-year weekly ranges and averages for key EIA inventory series.")
st.info(
    "**How to read these charts:** amber = current year, red = 5Y average, "
    "blue band = 5Y min/max range, gray dotted = 10Y average, gray band = 10Y min/max range. "
    "Current inventory near the bottom of the range is seasonally tight; near the top is seasonally heavy."
)

data = {}
with st.spinner("Loading EIA inventories..."):
    for label, (series_id, unit) in ENERGY_SEASONAL_SERIES.items():
        try:
            data[label] = (cache.eia_series(series_id), unit)
        except Exception as exc:
            st.warning(f"Could not load {label}: {type(exc).__name__}: {exc}")

if not data:
    st.info("No EIA seasonal series could be loaded.")
    st.stop()

items = list(data.items())
for idx in range(0, len(items), 2):
    cols = st.columns(2)
    for col, (label, (series, unit)) in zip(cols, items[idx:idx + 2]):
        col.plotly_chart(plot_seasonal_inventory(series, label, unit), width="stretch")
