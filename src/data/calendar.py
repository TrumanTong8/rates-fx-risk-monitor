# src/data/calendar.py
"""
Unified economic calendar from free sources:
  - ForexFactory weekly feeds: all currencies, with times, forecast and previous (this + next week)
  - FRED release dates: scheduled US data releases further out (CPI, payrolls, GDP, ...)
  - Federal Reserve FOMC calendar: meeting decision dates
Official US dates only fill in after the ForexFactory window, so events are not double-counted.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

FF_FEEDS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HEADERS = {"User-Agent": "Mozilla/5.0"}
NEW_YORK = ZoneInfo("America/New_York")

COLUMNS = ["datetime", "date", "time", "country", "title", "impact", "forecast", "previous", "source"]

# FRED release id -> (title, impact, release time in New York)
US_RELEASES = {
    50: ("Employment Situation (NFP)", "High", time(8, 30)),
    10: ("Consumer Price Index", "High", time(8, 30)),
    46: ("Producer Price Index", "High", time(8, 30)),
    54: ("Personal Income and Outlays (PCE)", "High", time(8, 30)),
    53: ("Gross Domestic Product", "High", time(8, 30)),
    9: ("Retail Sales", "High", time(8, 30)),
    192: ("JOLTS Job Openings", "High", time(10, 0)),
    11: ("Employment Cost Index", "High", time(8, 30)),
    180: ("Initial Jobless Claims", "Medium", time(8, 30)),
}

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _finish(frame: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Add local date/time display columns from a tz-aware 'datetime' column."""
    if frame.empty:
        return _empty()
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(tz)
    frame["date"] = frame["datetime"].dt.date
    frame["time"] = frame["datetime"].dt.strftime("%H:%M")
    for col in COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    return frame[COLUMNS]


def fetch_forexfactory_events(tz: str = "Asia/Singapore") -> pd.DataFrame:
    rows: list[dict] = []
    for feed in FF_FEEDS:
        try:
            r = requests.get(feed, headers=HEADERS, timeout=15)
            r.raise_for_status()
            rows.extend(r.json())
        except (requests.RequestException, ValueError):
            continue
    if not rows:
        return _empty()

    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["datetime"])
    frame["source"] = "ForexFactory"
    return _finish(frame, tz)


def fetch_us_release_events(api_key: str, start: date, tz: str = "Asia/Singapore") -> pd.DataFrame:
    if not api_key:
        return _empty()
    rows = []
    for release_id, (title, impact, release_time) in US_RELEASES.items():
        params = {
            "release_id": release_id,
            "api_key": api_key,
            "file_type": "json",
            "include_release_dates_with_no_data": "true",
            "realtime_start": start.isoformat(),
            "sort_order": "asc",
        }
        try:
            r = requests.get(FRED_RELEASE_DATES_URL, params=params, timeout=20)
            r.raise_for_status()
        except requests.RequestException:
            continue
        for item in r.json().get("release_dates", []):
            day = date.fromisoformat(item["date"])
            rows.append({
                "datetime": datetime.combine(day, release_time, tzinfo=NEW_YORK),
                "country": "USD",
                "title": title,
                "impact": impact,
                "source": "FRED release calendar",
            })
    return _finish(pd.DataFrame(rows), tz) if rows else _empty()


def fetch_fomc_events(tz: str = "Asia/Singapore") -> pd.DataFrame:
    try:
        r = requests.get(FOMC_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except requests.RequestException:
        return _empty()

    text = re.sub(r"<br\s*/?>", "\n", r.text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\s*\n\s*", "\n", text)

    # The page lists one "<year> FOMC Meetings" section per year; each meeting is
    # "<Month>\n<d1>-<d2>[*]" (or "Apr/May\n30-1" when it spans two months).
    rows = []
    sections = re.split(r"\n(\d{4}) FOMC Meetings\n", text)
    for year_str, body in zip(sections[1::2], sections[2::2]):
        year = int(year_str)
        pattern = re.compile(r"\n(" + "|".join(MONTHS) + r")(?:/(" + "|".join(MONTHS) + r"))?\n(\d{1,2})(?:-(\d{1,2}))?\*?(?=\n)")
        for m in pattern.finditer("\n" + body):
            month = MONTHS[m.group(2) or m.group(1)]
            day = int(m.group(4) or m.group(3))
            rows.append({
                "datetime": datetime(year, month, day, 14, 0, tzinfo=NEW_YORK),
                "country": "USD",
                "title": "FOMC rate decision",
                "impact": "High",
                "source": "Federal Reserve FOMC calendar",
            })
    if not rows:
        return _empty()
    return _finish(pd.DataFrame(rows).drop_duplicates("datetime"), tz)


def fetch_events(
    start: date | None = None,
    end: date | None = None,
    currencies: list[str] | None = None,
    impacts: tuple[str, ...] | None = ("High", "Medium"),
    tz: str = "Asia/Singapore",
    fred_api_key: str | None = None,
) -> pd.DataFrame:
    """
    Combined calendar, sorted by time. Columns:
    datetime (tz-aware, local), date, time, country (currency code), title, impact,
    forecast, previous, source.
    """
    start = start or date.today()
    ff = fetch_forexfactory_events(tz)
    official = pd.concat(
        [fetch_us_release_events(fred_api_key or "", start, tz), fetch_fomc_events(tz)],
        ignore_index=True,
    )
    if not ff.empty and not official.empty:
        official = official[official["date"] > ff["date"].max()]

    frames = [f for f in (ff, official) if not f.empty]
    if not frames:
        return _empty()
    events = pd.concat(frames, ignore_index=True)

    events = events[events["date"] >= start]
    if end is not None:
        events = events[events["date"] <= end]
    if currencies:
        events = events[events["country"].isin([c.upper() for c in currencies])]
    if impacts:
        events = events[events["impact"].isin(impacts)]
    return events.sort_values("datetime").reset_index(drop=True)
