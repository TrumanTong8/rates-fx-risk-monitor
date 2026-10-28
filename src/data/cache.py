# src/data/cache.py
"""
Streamlit-cached entry points to the data layer. Pages call these instead of the
source modules so repeated reruns (and other pages) reuse the same in-memory results.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import streamlit as st

from src.data import calendar, cftc, eia, fred, yahoo

HOUR = 60 * 60


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)).strip()
    except FileNotFoundError:
        return default


@st.cache_data(ttl=HOUR, show_spinner=False)
def daily_prices(symbol: str, years: int | None = 10) -> pd.DataFrame:
    return yahoo.fetch_daily_prices(symbol, years=years)


@st.cache_data(ttl=HOUR, show_spinner=False)
def close_series(symbol: str, years: int | None = 10) -> pd.Series:
    return yahoo.fetch_close(symbol, years=years)


@st.cache_data(ttl=6 * HOUR, show_spinner=False)
def fred_series(series_id: str, start: date | None = None, end: date | None = None) -> pd.Series:
    return fred.fetch_fred_series(series_id, start=start, end=end, api_key=get_secret("FRED_API_KEY") or None)


@st.cache_data(ttl=6 * HOUR, show_spinner=False)
def fred_indicator(series_ids: str | tuple[str, ...], transform: str = "level") -> pd.Series:
    ids = series_ids if isinstance(series_ids, str) else list(series_ids)
    return fred.fetch_fred_indicator(ids, transform, api_key=get_secret("FRED_API_KEY") or None)


@st.cache_data(ttl=6 * HOUR, show_spinner=False)
def eia_series(series_id: str) -> pd.Series:
    return eia.fetch_eia_series(series_id, get_secret("EIA_API_KEY"))


@st.cache_data(ttl=6 * HOUR, show_spinner=False)
def cot_market(market: str | None) -> pd.DataFrame:
    return cftc.fetch_cot_market(market)


@st.cache_data(ttl=6 * HOUR, show_spinner=False)
def open_interest(market: str) -> pd.Series:
    return cftc.fetch_open_interest(market)


@st.cache_data(ttl=6 * HOUR, show_spinner=False)
def cot_markets(markets: tuple[str, ...]) -> dict[str, pd.DataFrame | str]:
    """
    Fetch several markets at once; the CFTC API takes ~5s per query regardless of size,
    so running them in parallel is what keeps page loads fast. Failed markets map to
    their error message instead of a frame.
    """
    def fetch(market: str) -> pd.DataFrame | str:
        try:
            return cftc.fetch_cot_market(market)
        except Exception as exc:
            return f"CFTC fetch failed for {market}: {exc}"

    unique = list(dict.fromkeys(m for m in markets if m))
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(unique, pool.map(fetch, unique)))


@st.cache_data(ttl=HOUR, show_spinner=False)
def _all_events(start: date, tz: str) -> pd.DataFrame:
    return calendar.fetch_events(start=start, impacts=None, tz=tz, fred_api_key=get_secret("FRED_API_KEY") or None)


def events(
    start: date | None = None,
    end: date | None = None,
    currencies: tuple[str, ...] | None = None,
    impacts: tuple[str, ...] | None = ("High", "Medium"),
    tz: str = "Asia/Singapore",
) -> pd.DataFrame:
    """All sources are downloaded once per start date; filters are applied per call."""
    return calendar.filter_events(
        _all_events(start or date.today(), tz),
        start=start,
        end=end,
        currencies=list(currencies) if currencies else None,
        impacts=impacts,
    )
