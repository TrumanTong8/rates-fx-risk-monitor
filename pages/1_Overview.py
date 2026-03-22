# pages/1_Overview.py
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.workflows.daily_pack import build_daily_pack
from src.data.econ_calendar import fetch_te_calendar

st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")

# --- Trading Economics key (default to guest) ---
TE_API_KEY = st.secrets.get("TE_API_KEY", "guest:guest").strip()

# --- Cached calendar fetch (prevents spamming the API on reruns) ---
@st.cache_data(ttl=1800)
def get_te_calendar_cached(api_key: str, start: date, end: date, country: str, importance: int | None):
    return fetch_te_calendar(
        api_key=api_key,
        start=start,
        end=end,
        tz="Asia/Singapore",
        country=country,
        importance=importance,
    )


# =========================================================
# 1) Economic Calendar
# =========================================================
st.header("Economic Calendar")

cal_top = st.columns([1, 1, 1, 1])
with cal_top[0]:
    cal_start = st.date_input("Start", value=date.today(), key="cal_start")
with cal_top[1]:
    cal_end = st.date_input("End", value=date.today() + timedelta(days=7), key="cal_end")
with cal_top[2]:
    country = st.selectbox(
        "Country",
        ["United States", "Japan", "Euro area", "United Kingdom", "China", "Singapore"],
        index=0,
        key="cal_country",
    )
with cal_top[3]:
    importance = st.selectbox(
        "Importance",
        ["All", "High (3)", "Medium+ (2+)", "Low+ (1+)"],
        index=1,
        key="cal_importance",
    )

importance_map = {
    "All": None,
    "High (3)": 3,
    "Medium+ (2+)": 2,
    "Low+ (1+)": 1,
}
imp_val = importance_map[importance]

if not TE_API_KEY:
    st.info("No TE_API_KEY found. Add it to `.streamlit/secrets.toml` (repo root).")
else:
    try:
        cal_df = get_te_calendar_cached(TE_API_KEY, cal_start, cal_end, country, imp_val)
        if cal_df is None or len(cal_df) == 0:
            st.warning("No events returned for this window / filters. "
                       "Try going back to 2025 as Trading Economic Calendar does not allow recent data to be pulled.")
        else:
            st.dataframe(
                cal_df[["time", "report", "period", "actual", "forecast", "previous"]],
                use_container_width=True,
                hide_index=True,
            )
    except Exception:
        # Keep error messages token-safe and clean
        st.error("Calendar fetch failed. Check TE access / rate limits, and try again later.")

    st.caption(f"Returned {len(cal_df)} events for {cal_start} → {cal_end}")

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
            pack = build_daily_pack(
                asof=asof,
                lookback_days=lookback,
                pack_type=pack_type,
                fred_api_key=st.secrets.get("FRED_API_KEY", ""),  # <-- add this line
                te_api_key=TE_API_KEY if TE_API_KEY else None,
                te_country=country,
                te_importance=3 if imp_val is None else imp_val,
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
                events_df[["time", "report", "period", "actual", "forecast", "previous"]].head(12),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No events returned (missing key, API issue, or empty window).")