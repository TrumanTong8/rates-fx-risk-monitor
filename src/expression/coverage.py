from __future__ import annotations

import pandas as pd


def price_coverage(prices: pd.DataFrame, requested_years: int) -> dict[str, object]:
    if prices.empty:
        return {
            "first_date": "",
            "last_date": "",
            "observations": 0,
            "available_years": 0.0,
            "requested_years": requested_years,
            "coverage_ratio": 0.0,
            "status": "No data",
        }

    first = prices["date"].iloc[0]
    last = prices["date"].iloc[-1]
    available_years = max((last - first).days / 365.25, 0)
    ratio = min(available_years / max(float(requested_years), 1.0), 1.0)
    if available_years < 1:
        status = "Too short for backtest"
    elif ratio < 0.5:
        status = "Short history"
    elif len(prices) < 220:
        status = "Too few daily bars"
    else:
        status = "OK"
    return {
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "observations": int(len(prices)),
        "available_years": round(available_years, 1),
        "requested_years": requested_years,
        "coverage_ratio": round(ratio, 2),
        "status": status,
    }
