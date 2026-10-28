from __future__ import annotations

import numpy as np
import pandas as pd


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 10:
        return 52
    deltas = np.diff(index.sort_values().values).astype("timedelta64[D]").astype(int)
    med = np.median(deltas[deltas > 0]) if np.any(deltas > 0) else 7
    if med <= 2:
        return 252
    if med <= 10:
        return 52
    if med <= 40:
        return 12
    return 1


def compute_zscores(series: pd.Series, lookback_periods: int) -> pd.Series:
    values = series.astype(float)
    mean = values.rolling(lookback_periods, min_periods=max(10, lookback_periods // 4)).mean()
    std = values.rolling(lookback_periods, min_periods=max(10, lookback_periods // 4)).std(ddof=0)
    return (values - mean) / std.replace(0, np.nan)


def align_weekly(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    left_weekly = left.sort_index().resample("W-FRI").last()
    right_weekly = right.sort_index().resample("W-FRI").last()
    idx = left_weekly.index.union(right_weekly.index).sort_values()

    def fill(weekly: pd.Series, name: str) -> pd.Series:
        # Fill gaps inside the series' own history, but never past its latest observation,
        # so a newer CFTC week doesn't carry stale EIA inventory forward (and vice versa).
        filled = weekly.reindex(idx).ffill()
        return filled.where(idx <= weekly.last_valid_index()).rename(name)

    return fill(left_weekly, "inventory"), fill(right_weekly, "open_interest")


def compute_dashboard_frame(inv: pd.Series, oi: pd.Series, price: pd.Series | None = None) -> pd.DataFrame:
    """Weekly (Friday) frame of inventory, open interest, inventory changes and weekly-close price."""
    inv_aligned, oi_aligned = align_weekly(inv, oi)
    df = pd.DataFrame({"inventory": inv_aligned, "open_interest": oi_aligned})
    df["inv_to_oi"] = df["inventory"] / df["open_interest"].replace(0, np.nan)
    df["inv_wow"] = df["inventory"].pct_change(1, fill_method=None) * 100
    df["inv_roll28d"] = df["inventory"].pct_change(4, fill_method=None) * 100

    if price is not None and len(price) > 0:
        df = df.join(price.sort_index().resample("W-FRI").last().rename("price"), how="left")
    return df
