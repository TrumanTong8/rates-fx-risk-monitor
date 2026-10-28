# views/rates_fx/overview.py
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.data import cache
from src.workflows.daily_pack import build_daily_pack

st.title("Overview")

CALENDAR_COLUMNS = ["date", "time", "country", "title", "impact", "forecast", "previous", "source"]
CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CNY", "AUD", "CAD", "CHF", "NZD"]
IMPACT_MAP = {
    "High": ("High",),
    "Medium+": ("High", "Medium"),
    "All": None,
}


def load_events(start: date, end: date, currencies: list[str], impacts: tuple[str, ...] | None):
    try:
        return cache.events(start=start, end=end, currencies=tuple(currencies), impacts=impacts)
    except Exception:
        return None


# =========================================================
# 1) Economic Calendar
# =========================================================
st.header("Economic Calendar")

cal_top = st.columns([1, 1, 2, 1])
with cal_top[0]:
    cal_start = st.date_input("Start", value=date.today(), key="cal_start")
with cal_top[1]:
    cal_end = st.date_input("End", value=date.today() + timedelta(days=7), key="cal_end")
with cal_top[2]:
    currencies = st.multiselect("Currencies", CURRENCIES, default=["USD"], key="cal_currencies")
with cal_top[3]:
    importance = st.selectbox("Importance", list(IMPACT_MAP), index=0, key="cal_importance")

impacts = IMPACT_MAP[importance]
cal_df = load_events(cal_start, cal_end, currencies, impacts)

if cal_df is None:
    st.error("Calendar fetch failed. The sources may be rate-limiting; try again in a few minutes.")
elif len(cal_df) == 0:
    st.warning("No events returned for this window / filters.")
else:
    st.dataframe(cal_df[CALENDAR_COLUMNS], use_container_width=True, hide_index=True)
    st.caption(f"Returned {len(cal_df)} events for {cal_start} → {cal_end} (times in SGT)")
st.caption(
    "ForexFactory covers this week and next for all currencies (with forecast/previous). "
    "Beyond that, scheduled US releases come from the FRED release calendar and FOMC dates from the Fed."
)

st.divider()

# =========================================================
# 2) Daily Pack (charts + summary + key events)
# =========================================================
st.header("Daily Pack")

with st.expander("Generate pack (3–5 charts + discussion summary)", expanded=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        asof = st.date_input("As-of date", value=date.today(), key="dp_asof")
    with c2:
        lookback = st.slider("Lookback (days)", 30, 365, 180, key="dp_lookback")
    with c3:
        pack_type = st.selectbox(
            "Pack type",
            ["Rates+FX Core", "Rates Focus", "FX Focus"],
            index=0,
            key="dp_pack_type",
        )
    with c4:
        run = st.button("Generate pack", key="dp_run")

    if run:
        with st.spinner("Building pack..."):
            pack_events = load_events(asof, asof + timedelta(days=2), currencies, impacts or ("High",))
            pack = build_daily_pack(
                asof=asof,
                lookback_days=lookback,
                pack_type=pack_type,
                fred_api_key=cache.get_secret("FRED_API_KEY"),
                events=pack_events,
            )
            st.session_state["daily_pack"] = pack

    pack = st.session_state.get("daily_pack")

    if pack:
        figs = pack.get("figures", [])
        if figs:
            cols = st.columns(2)
            for i, fig in enumerate(figs):
                cols[i % 2].plotly_chart(fig, use_container_width=True)

        summary_md = pack.get("summary_md", "")
        st.subheader("Discussion summary (copy/paste)")
        st.text_area("Summary", summary_md, height=240, key="dp_summary")

        st.download_button(
            "Download summary (.md)",
            data=summary_md.encode("utf-8"),
            file_name=f"daily_pack_{asof}.md",
            mime="text/markdown",
        )

        events_df = pack.get("events")
        if events_df is not None and len(events_df) > 0:
            st.subheader("Next 48h key events (from Daily Pack)")
            st.dataframe(
                events_df[CALENDAR_COLUMNS].head(12),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No events returned (API issue or empty window).")
