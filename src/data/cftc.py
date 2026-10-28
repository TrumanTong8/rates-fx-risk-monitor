# src/data/cftc.py
from __future__ import annotations

import re

import pandas as pd
import requests

BASE_URL = "https://publicreporting.cftc.gov/resource/{dataset}.json"

# Futures-only Commitments of Traders datasets on the CFTC Socrata API.
LEGACY = "6dca-aqww"   # commercial / non-commercial split, all markets
TFF = "gpe5-46if"      # Traders in Financial Futures (equity, rates, FX, VIX)
DISAGG = "kh3c-gbw2"   # Disaggregated (physical commodities)

# CFTC renames contracts over time (e.g. "CRUDE OIL, LIGHT SWEET" -> "WTI-PHYSICAL" in 2022),
# so markets are queried by their stable contract code. Names are kept for old configs.
MARKET_CODES = {
    "E-MINI S&P 500": "13874A",
    "NASDAQ MINI": "209742",
    "RUSSELL 2000 MINI": "239742",
    "JAPANESE YEN": "097741",
    "EURO FX": "099741",
    "AUSTRALIAN DOLLAR": "232741",
    "UST 2Y NOTE": "042601",
    "UST 10Y NOTE": "043602",
    "UST BOND": "020601",
    "VIX FUTURES": "1170E1",
    "GOLD": "088691",
    "SILVER": "084691",
    "COPPER- #1": "085692",
    "CRUDE OIL, LIGHT SWEET-WTI": "067651",
    "WTI CRUDE": "067651",
    "NATURAL GAS": "023651",
    "RBOB GASOLINE": "111659",
    "ULSD": "022651",
    "PROPANE": "06665O",
}

DISAGG_CODES = {"088691", "084691", "085692", "067651", "023651", "111659", "022651", "06665O"}

_CODE_RE = re.compile(r"^[0-9A-Z]{6}$")


def resolve_market_code(market: str) -> str:
    """Accept a contract code or a known market name; return the contract code."""
    clean = market.strip().upper()
    if clean in MARKET_CODES:
        return MARKET_CODES[clean]
    if _CODE_RE.match(clean) and any(ch.isdigit() for ch in clean):
        return clean
    raise KeyError(f"Unknown CFTC market '{market}'. Use a contract code or one of: {sorted(MARKET_CODES)}")


def _fetch(dataset: str, code: str, limit: int) -> pd.DataFrame:
    params = {
        "$where": f"cftc_contract_market_code = '{code}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(limit),
    }
    r = requests.get(BASE_URL.format(dataset=dataset), params=params, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def _clean(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_url"] = source
    frame["date"] = pd.to_datetime(frame["report_date_as_yyyy_mm_dd"], errors="coerce")
    numeric_cols = [col for col in frame.columns if "positions" in col or col.endswith("_all")]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_cot_market(market: str | None, limit: int = 1100) -> pd.DataFrame:
    """
    Weekly positioning for one market from the TFF or Disaggregated report
    (whichever covers it). Empty frame if the market is unknown or unreported.
    """
    if not market:
        return pd.DataFrame()
    try:
        code = resolve_market_code(market)
    except KeyError:
        return pd.DataFrame()

    datasets = [DISAGG, TFF] if code in DISAGG_CODES else [TFF, DISAGG]
    for dataset in datasets:
        frame = _fetch(dataset, code, limit)
        if not frame.empty:
            return _clean(frame, BASE_URL.format(dataset=dataset))
    return pd.DataFrame()


def fetch_open_interest(market: str, limit: int = 5000) -> pd.Series:
    """Weekly total futures open interest from the Legacy report, indexed by report date."""
    code = resolve_market_code(market)
    frame = _fetch(LEGACY, code, limit)
    if frame.empty:
        return pd.Series(dtype=float, name="open_interest")
    frame = _clean(frame, BASE_URL.format(dataset=LEGACY))
    s = frame.drop_duplicates("date", keep="last").set_index("date")["open_interest_all"]
    return s.dropna().rename("open_interest")
