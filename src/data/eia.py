# src/data/eia.py
from __future__ import annotations

import pandas as pd
import requests

SERIESID_URL = "https://api.eia.gov/v2/seriesid/{series_id}"


def fetch_eia_series(series_id: str, api_key: str) -> pd.Series:
    """
    EIA Open Data v2 series by legacy v1 id (e.g. "PET.WCESTUS1.W").
    Returns a date-indexed Series named after the series id.
    """
    if not api_key:
        raise ValueError("Missing EIA_API_KEY (free key: https://www.eia.gov/opendata/register.php).")

    r = requests.get(SERIESID_URL.format(series_id=series_id), params={"api_key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json().get("response", {}).get("data", [])
    if not data:
        raise ValueError(f"No EIA data returned for {series_id}")

    df = pd.DataFrame(data)
    values = pd.to_numeric(df["value"], errors="coerce")
    s = pd.Series(values.values, index=pd.to_datetime(df["period"], errors="coerce"), name=series_id)
    return s[s.index.notna()].dropna().sort_index()
