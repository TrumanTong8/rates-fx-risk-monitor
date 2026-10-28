"""
Smoke test for the free data layer: hits every source and ticker the dashboard relies on
and prints coverage. Run from the repo root:

    .venv/bin/python scripts/check_data_sources.py
"""
from __future__ import annotations

import sys
import tomllib
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import calendar, cftc, eia, fred, yahoo  # noqa: E402
from src.data.symbols import LEGACY_TO_FRED, LEGACY_TO_YAHOO  # noqa: E402

ENERGY_EIA_SERIES = [
    "PET.WCESTUS1.W", "NG.NW2_EPG0_SWO_R48_BCF.W", "PET.WGFSTUS1.W", "PET.WDISTUS1.W",
    "PET.WPRSTUS1.W", "PET.W_EPC0_SAX_YCUOK_MBBL.W", "PET.WCSSTUS1.W",
]
ENERGY_OI_MARKETS = ["WTI CRUDE", "NATURAL GAS", "RBOB GASOLINE", "ULSD", "PROPANE"]


def load_secrets() -> dict:
    path = ROOT / ".streamlit" / "secrets.toml"
    return tomllib.loads(path.read_text()) if path.exists() else {}


def describe(index: pd.Index) -> tuple[int, str, str]:
    if len(index) == 0:
        return 0, "", ""
    return len(index), str(pd.Timestamp(index.min()).date()), str(pd.Timestamp(index.max()).date())


def run_check(results: list[dict], source: str, item: str, fn) -> None:
    try:
        n, first, last = fn()
        status = "OK" if n else "EMPTY"
        results.append({"source": source, "item": item, "status": status, "rows": n, "first": first, "last": last})
    except Exception as exc:
        results.append({"source": source, "item": item, "status": "FAIL", "rows": 0, "first": "", "last": str(exc)[:70]})
    row = results[-1]
    print(f"{row['status']:5} {source:8} {item:32} {row['rows']:>6} {row['first']:>10} {row['last']:>10}", flush=True)


def main() -> int:
    secrets = load_secrets()
    fred_key = str(secrets.get("FRED_API_KEY", "")).strip() or None
    eia_key = str(secrets.get("EIA_API_KEY", "")).strip()
    results: list[dict] = []

    for symbol in sorted(set(LEGACY_TO_YAHOO.values())):
        run_check(results, "Yahoo", symbol,
                  lambda s=symbol: describe(pd.DatetimeIndex(yahoo.fetch_daily_prices(s, years=30)["date"])))

    for legacy, (ids, transform) in LEGACY_TO_FRED.items():
        label = f"{legacy} ({transform})"
        run_check(results, "FRED", label,
                  lambda i=ids, t=transform: describe(fred.fetch_fred_indicator(i, t, api_key=fred_key).index))

    for series_id in ENERGY_EIA_SERIES:
        run_check(results, "EIA", series_id, lambda s=series_id: describe(eia.fetch_eia_series(s, eia_key).index))

    cot_markets = {code: name for name, code in cftc.MARKET_CODES.items()}
    for code, name in cot_markets.items():
        run_check(results, "CFTC COT", f"{name} [{code}]",
                  lambda c=code: describe(pd.DatetimeIndex(cftc.fetch_cot_market(c).get("date", pd.Series(dtype="datetime64[ns]")))))
    for market in ENERGY_OI_MARKETS:
        run_check(results, "CFTC OI", market, lambda m=market: describe(cftc.fetch_open_interest(m).index))

    today = date.today()
    run_check(results, "Calendar", "ForexFactory (all ccy)",
              lambda: describe(pd.DatetimeIndex(calendar.fetch_forexfactory_events()["date"].astype("datetime64[ns]"))))
    run_check(results, "Calendar", "FRED US release dates",
              lambda: describe(pd.DatetimeIndex(calendar.fetch_us_release_events(fred_key or "", today)["date"].astype("datetime64[ns]"))))
    run_check(results, "Calendar", "FOMC meetings",
              lambda: describe(pd.DatetimeIndex(calendar.fetch_fomc_events()["date"].astype("datetime64[ns]"))))

    events = calendar.fetch_events(currencies=["USD"], fred_api_key=fred_key)
    print("\nNext USD medium/high-impact events (combined calendar):")
    print(events[["date", "time", "title", "impact", "forecast", "previous", "source"]].head(15).to_string(index=False))

    failed = [r for r in results if r["status"] != "OK"]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks OK")
    for r in failed:
        print(f"  {r['status']}: {r['source']} {r['item']} {r['last']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
