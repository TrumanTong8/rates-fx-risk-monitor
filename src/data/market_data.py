# src/data/market_data.py
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
import requests


def fetch_fred_series(
    api_key: str,
    series_id: str,
    start: date,
    end: date,
) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()

    obs = payload.get("observations", [])
    df = pd.DataFrame(obs)

    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].sort_index()


@lru_cache(maxsize=1)
def frankfurter_latest_date() -> date:
    """
    Frankfurter latest working-day date.
    Uses /v1/latest (NOT /latest).
    """
    url = "https://api.frankfurter.dev/v1/latest"
    params = {"base": "EUR", "symbols": "USD"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    return pd.to_datetime(payload["date"]).date()


def fetch_eurusd_frankfurter(start: date, end: date) -> pd.Series:
    """
    Time series endpoint: /v1/{start}..{end}
    Filter with base=EUR&symbols=USD.
    Clamps end to Frankfurter's latest available date.
    """
    latest_available = frankfurter_latest_date()
    end_clamped = min(end, latest_available)

    if start > end_clamped:
        start = end_clamped - timedelta(days=30)

    url = f"https://api.frankfurter.dev/v1/{start.isoformat()}..{end_clamped.isoformat()}"
    params = {"base": "EUR", "symbols": "USD"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()

    rates = payload.get("rates", {})
    rows = []
    for d, v in rates.items():
        rows.append((pd.to_datetime(d), v.get("USD")))

    s = pd.Series(dict(rows)).sort_index()
    s.name = "EURUSD"
    return s.astype(float)


def build_rates_fx_frame(fred_api_key: str, start: date, end: date) -> pd.DataFrame:
    """
    Columns: UST_2Y, UST_10Y, 2s10s, EURUSD.
    Clamp end to Frankfurter latest so FX never 404s.
    """
    fx_latest = frankfurter_latest_date()
    end_clamped = min(end, fx_latest)

    ust2 = fetch_fred_series(fred_api_key, "DGS2", start, end_clamped).rename("UST_2Y")
    ust10 = fetch_fred_series(fred_api_key, "DGS10", start, end_clamped).rename("UST_10Y")
    eurusd = fetch_eurusd_frankfurter(start, end_clamped)

    df = pd.concat([ust2, ust10, eurusd], axis=1)
    df["2s10s"] = df["UST_10Y"] - df["UST_2Y"]
    return df.dropna(how="all")