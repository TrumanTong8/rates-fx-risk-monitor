# src/data/econ_calendar.py
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import requests


class CalendarAuthError(RuntimeError): ...
class CalendarRateLimitError(RuntimeError): ...
class CalendarRequestError(RuntimeError): ...


def fetch_te_calendar(
    api_key: str,
    start: date,
    end: date,
    tz: str = "Asia/Singapore",
    country: str = "united states",
    importance: int | None = 3,  # 1 low, 2 medium, 3 high
    fallback_to_snapshot: bool = True,
) -> pd.DataFrame:
    base = "https://api.tradingeconomics.com/calendar/country"
    country_path = requests.utils.quote(country.lower(), safe="")

    params = {"c": api_key, "f": "json"}
    if importance is not None:
        params["importance"] = importance

    def _request(url: str):
        try:
            r = requests.get(url, params=params, timeout=20)
        except requests.RequestException as ex:
            raise CalendarRequestError("Network error calling Trading Economics.") from ex

        if r.status_code in (401, 403):
            raise CalendarAuthError(f"Trading Economics denied access (HTTP {r.status_code}).")
        if r.status_code == 429:
            raise CalendarRateLimitError("Trading Economics rate limit hit (HTTP 429).")
        if not r.ok:
            raise CalendarRequestError(f"Trading Economics request failed (HTTP {r.status_code}): {r.text[:200]}")

        data = r.json()
        return data if isinstance(data, list) else []

    def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
        local_tz = ZoneInfo(tz)
        out = []

        for e in rows:
            dt = None
            s = (e.get("Date") or "").strip()
            if s:
                # handle "Z"
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                try:
                    dt = datetime.fromisoformat(s)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)  # TE date is UTC in docs
                    dt = dt.astimezone(local_tz)
                except Exception:
                    dt = None

            out.append(
                {
                    "time_dt": dt,
                    "time": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
                    "report": e.get("Event") or "",
                    "period": e.get("Reference") or "",
                    "actual": e.get("Actual"),
                    "forecast": e.get("Forecast"),
                    "previous": e.get("Previous"),
                    "country": e.get("Country") or "",
                    "importance": e.get("Importance"),
                    "category": e.get("Category") or "",
                    "currency": e.get("Currency") or "",
                    "unit": e.get("Unit") or "",
                    "source": e.get("Source") or "",
                }
            )

        df = pd.DataFrame(out)
        required = ["time", "report", "period", "actual", "forecast", "previous"]
        for c in required:
            if c not in df.columns:
                df[c] = None

        return df

    def _filter_to_range(df: pd.DataFrame) -> pd.DataFrame:
        if "time_dt" not in df.columns:
            return df

        # keep only rows with parsed datetime
        df = df[df["time_dt"].notna()].copy()

        # inclusive date filter (local timezone)
        start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=ZoneInfo(tz))
        end_dt = datetime.combine(end, datetime.max.time()).replace(tzinfo=ZoneInfo(tz))

        df = df[(df["time_dt"] >= start_dt) & (df["time_dt"] <= end_dt)]
        return df

    # 1) Try date-range endpoint first
    url_range = f"{base}/{country_path}/{start.isoformat()}/{end.isoformat()}"
    df = _rows_to_df(_request(url_range))
    df = _filter_to_range(df)

    # 2) If nothing matches the requested range, fallback to snapshot endpoint
    if df.empty and fallback_to_snapshot:
        url_snap = f"{base}/{country_path}"
        df = _rows_to_df(_request(url_snap))
        df = _filter_to_range(df)

    # Sort & output
    if "time_dt" in df.columns:
        df = df.sort_values("time_dt").drop(columns=["time_dt"])

    front = ["time", "report", "period", "actual", "forecast", "previous"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]