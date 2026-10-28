# src/data/fred.py
from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests

API_URL = "https://api.stlouisfed.org/fred/series/observations"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(
    series_id: str,
    start: date | None = None,
    end: date | None = None,
    api_key: str | None = None,
) -> pd.Series:
    """
    Date-indexed FRED series named after the series id.
    Uses the official API when api_key is given, else the public CSV download.
    """
    if api_key:
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
        if start:
            params["observation_start"] = start.isoformat()
        if end:
            params["observation_end"] = end.isoformat()
        r = requests.get(API_URL, params=params, timeout=20)
        r.raise_for_status()
        df = pd.DataFrame(r.json().get("observations", []), columns=["date", "value"])
    else:
        params = {"id": series_id}
        if start:
            params["cosd"] = start.isoformat()
        if end:
            params["coed"] = end.isoformat()
        r = requests.get(CSV_URL, params=params, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = ["date", "value"]

    if df.empty:
        raise ValueError(f"No FRED data returned for {series_id}")

    values = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    s = pd.Series(values.values, index=pd.to_datetime(df["date"]), name=series_id)
    return s.dropna().sort_index()


# How each macro indicator is derived from its raw FRED series, so values come out in the
# same units the old Bloomberg tickers used (e.g. CPI YOY Index = CPI 12m % change).
TRANSFORMS = {
    "level": lambda s: s,
    "diff": lambda s: s.diff(),
    "mom_pct": lambda s: s.pct_change(fill_method=None) * 100,
    "yoy_pct": lambda s: s.pct_change(12, fill_method=None) * 100,
    "thousands": lambda s: s / 1000,
}


def fetch_fred_indicator(
    series_ids: str | list[str],
    transform: str = "level",
    api_key: str | None = None,
) -> pd.Series:
    """
    Fetch one indicator, splicing several FRED ids end-to-end when given a list
    (later ids win on overlap, e.g. DFEDTAR then DFEDTARU), then apply a transform.
    """
    ids = [series_ids] if isinstance(series_ids, str) else list(series_ids)
    parts = [fetch_fred_series(sid, api_key=api_key) for sid in ids]
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    out = TRANSFORMS[transform](s).dropna()
    out.name = "+".join(ids)
    return out
