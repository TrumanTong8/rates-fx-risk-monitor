# src/data/yahoo.py
from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def fetch_daily_prices(
    symbol: str,
    years: int | None = 10,
    start: date | None = None,
    adjusted: bool = True,
) -> pd.DataFrame:
    """
    Daily OHLCV bars from Yahoo Finance.
    Columns: date (tz-naive), open, high, low, close, volume.
    adjusted=True folds dividends/splits into prices (total return for ETFs);
    it makes no difference for futures, FX or indices.
    """
    ticker = yf.Ticker(symbol)
    if start is not None:
        raw = ticker.history(start=start.isoformat(), interval="1d", auto_adjust=adjusted)
    elif years is None:
        raw = ticker.history(period="max", interval="1d", auto_adjust=adjusted)
    else:
        start_ts = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
        raw = ticker.history(start=start_ts.strftime("%Y-%m-%d"), interval="1d", auto_adjust=adjusted)

    if raw is None or raw.empty:
        raise ValueError(f"No Yahoo data returned for {symbol}")

    frame = raw.rename(columns=str.lower).reset_index()
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    for col in PRICE_COLUMNS[1:]:
        if col not in frame.columns:
            frame[col] = 0 if col == "volume" else pd.NA

    frame = frame[PRICE_COLUMNS].dropna(subset=["close"])
    return frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def fetch_close(symbol: str, years: int | None = 10, start: date | None = None) -> pd.Series:
    """Daily close as a date-indexed Series named after the symbol."""
    prices = fetch_daily_prices(symbol, years=years, start=start)
    return prices.set_index("date")["close"].astype(float).rename(symbol)
