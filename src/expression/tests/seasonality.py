from __future__ import annotations

import numpy as np
import pandas as pd

from src.expression.models import TestResult


def _remove_price_outliers(frame: pd.DataFrame, max_abs_daily_return: float = 0.50) -> pd.DataFrame:
    cleaned = frame.copy()
    for _ in range(3):
        daily_return = cleaned["close"].pct_change()
        outliers = daily_return.abs() > max_abs_daily_return
        if not outliers.any():
            break
        cleaned = cleaned.loc[~outliers].reset_index(drop=True)
    return cleaned


def monthly_seasonality(prices: pd.DataFrame, direction: str = "long") -> pd.DataFrame:
    """Create monthly box statistics for the 10Y, 20Y, and 30Y lookbacks."""
    if prices.empty:
        return pd.DataFrame(columns=[
            "lookback_years",
            "month_index",
            "month_name",
            "avg_return",
            "median_return",
            "win_rate",
            "q1",
            "q3",
            "lower",
            "upper",
            "sample",
        ])

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame = _remove_price_outliers(frame)
    frame["monthly_return"] = frame["close"].pct_change(21)
    if direction != "long":
        frame["monthly_return"] = -frame["monthly_return"]
    frame = frame.dropna(subset=["monthly_return"]).copy()

    rows = []
    for years in [10, 20, 30]:
        cutoff = frame["date"].iloc[-1] - pd.DateOffset(years=years)
        window = frame[frame["date"] >= cutoff].copy()
        if window.empty or len(window) < 30:
            continue

        window["month_index"] = window["date"].dt.month
        for month_idx in range(1, 13):
            month_values = window[window["month_index"] == month_idx]["monthly_return"].dropna()
            if month_values.empty:
                continue

            rows.append(
                {
                    "lookback_years": years,
                    "month_index": month_idx,
                    "month_name": pd.Timestamp(year=2000, month=month_idx, day=1).strftime("%b"),
                    "avg_return": float(month_values.mean()),
                    "median_return": float(month_values.median()),
                    "win_rate": float((month_values > 0).mean()),
                    "q1": float(month_values.quantile(0.25)),
                    "q3": float(month_values.quantile(0.75)),
                    "lower": float(month_values.min()),
                    "upper": float(month_values.max()),
                    "sample": int(month_values.count()),
                }
            )

    if not rows:
        return pd.DataFrame(columns=[
            "lookback_years",
            "month_index",
            "month_name",
            "avg_return",
            "median_return",
            "win_rate",
            "q1",
            "q3",
            "lower",
            "upper",
            "sample",
        ])

    return pd.DataFrame(rows).sort_values(["lookback_years", "month_index"]).reset_index(drop=True)


def score_seasonality(prices: pd.DataFrame, direction: str) -> TestResult:
    if len(prices) < 260:
        return TestResult(40, "low", "Less than one year of data; seasonality is weak evidence.")

    stats = monthly_seasonality(prices, direction)
    current_month = int(prices["date"].iloc[-1].month)
    row = stats[(stats["lookback_years"] == 10) & (stats["month_index"] == current_month)]
    if row.empty:
        return TestResult(50, "low", "No usable history for the current month.")

    avg = float(row["avg_return"].iloc[0])
    win_rate = float(row["win_rate"].iloc[0])
    score = 50 + np.clip(avg * 600, -30, 30) + np.clip((win_rate - 0.5) * 40, -15, 15)
    confidence = "high" if int(row["sample"].iloc[0]) >= 60 else "medium"
    reason = f"Current-month avg return {avg:.2%}, win rate {win_rate:.0%}."
    return TestResult(float(np.clip(score, 0, 100)), confidence, reason, str(prices["date"].iloc[-1].date()))


