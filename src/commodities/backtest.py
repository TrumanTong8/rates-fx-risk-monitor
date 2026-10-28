"""
Inventory-shock backtest: an unusually large weekly draw (z < -2) is a long signal,
an unusually large build (z > +2) a short signal. Signals come from the weekly EIA
inventory frame; trades enter on the first daily close after the EIA release and
forward returns are measured in trading days on the daily price series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.commodities.analytics import compute_zscores, infer_periods_per_year

Z_THRESHOLD = 2.0
Z_RESET_THRESHOLD = 0.5
HORIZONS_DAYS = [5, 10, 21, 42, 63]
LOOKBACK_YRS = [5, 10]
Z_COLS = ["inv_wow", "inv_roll28d"]
Z_LABELS = {"inv_wow": "WoW inventory change %", "inv_roll28d": "Rolling 28d inventory change %"}


def _find_signals(z: pd.Series, threshold: float, reset_threshold: float = Z_RESET_THRESHOLD, min_gap: int = 5) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    longs, shorts = [], []
    last_long = last_short = -999
    long_locked = False
    short_locked = False
    prev_value = np.nan
    for i, (dt, value) in enumerate(z.items()):
        if pd.isna(value):
            continue

        if long_locked and value > -reset_threshold:
            long_locked = False
        if short_locked and value < reset_threshold:
            short_locked = False

        crosses_long = value < -threshold and (pd.isna(prev_value) or prev_value >= -threshold)
        crosses_short = value > threshold and (pd.isna(prev_value) or prev_value <= threshold)

        if not long_locked and crosses_long and (i - last_long) >= min_gap:
            longs.append(dt)
            last_long = i
            long_locked = True
        elif not short_locked and crosses_short and (i - last_short) >= min_gap:
            shorts.append(dt)
            last_short = i
            short_locked = True
        prev_value = value
    return longs, shorts


def _entry_dates(signal_dates: list[pd.Timestamp], daily_price: pd.Series, release_lag_days: int) -> list[pd.Timestamp]:
    """First trading day on or after each signal's EIA release date."""
    index = daily_price.index
    entries = []
    for dt in signal_dates:
        pos = index.searchsorted(dt + pd.Timedelta(days=release_lag_days))
        if pos < len(index):
            entries.append(index[pos])
    return entries


def _calc_fwd_returns(dates: list[pd.Timestamp], price: pd.Series, horizons: list[int], direction: int) -> pd.DataFrame:
    rows = []
    for dt in dates:
        pos = price.index.get_loc(dt)
        p0 = price.iloc[pos]
        if pd.isna(p0) or p0 == 0:
            continue
        row = {"signal_date": dt}
        for horizon in horizons:
            fpos = pos + horizon
            row[f"+{horizon}d"] = (price.iloc[fpos] / p0 - 1.0) * 100.0 * direction if fpos < len(price) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("signal_date") if rows else pd.DataFrame()


def run_backtest(df: pd.DataFrame, daily_price: pd.Series, z_col: str, lookback_years: int, release_lag_days: int) -> dict | None:
    price = daily_price.astype(float).dropna().sort_index()
    if price.empty:
        return None
    ppy = infer_periods_per_year(df.index)
    z = compute_zscores(df[z_col].astype(float), lookback_periods=lookback_years * ppy)
    long_signals, short_signals = _find_signals(z, Z_THRESHOLD)
    long_dates = _entry_dates(long_signals, price, release_lag_days)
    short_dates = _entry_dates(short_signals, price, release_lag_days)
    df_long = _calc_fwd_returns(long_dates, price, HORIZONS_DAYS, 1)
    df_short = _calc_fwd_returns(short_dates, price, HORIZONS_DAYS, -1)

    rows = []
    for label, data in [("Long (z < -2)", df_long), ("Short (z > +2)", df_short)]:
        for horizon in [f"+{h}d" for h in HORIZONS_DAYS]:
            values = data[horizon].dropna() if not data.empty and horizon in data.columns else pd.Series(dtype=float)
            rows.append({
                "Direction": label,
                "Horizon": horizon,
                "N": len(values),
                "Avg %": values.mean() if len(values) else np.nan,
                "Median %": values.median() if len(values) else np.nan,
                "Hit Rate %": (values > 0).mean() * 100 if len(values) else np.nan,
                "Best %": values.max() if len(values) else np.nan,
                "Worst %": values.min() if len(values) else np.nan,
            })
    return {
        "z": z,
        "price": price,
        "long_dates": long_dates,
        "short_dates": short_dates,
        "df_long": df_long,
        "df_short": df_short,
        "summary": pd.DataFrame(rows),
    }


def build_backtest_results(df: pd.DataFrame, daily_price: pd.Series, release_lag_days: int) -> dict[tuple[str, int], dict]:
    results = {}
    for z_col in Z_COLS:
        if z_col not in df.columns or df[z_col].isna().all():
            continue
        for lb in LOOKBACK_YRS:
            result = run_backtest(df, daily_price, z_col, lb, release_lag_days)
            if result is not None:
                results[(z_col, lb)] = result
    return results


def best_setup_rows(results: dict[tuple[str, int], dict], min_trades: int = 10) -> tuple[pd.Series | None, pd.Series | None]:
    rows = []
    for (z_col, lb), result in results.items():
        summary = result["summary"].copy()
        if summary.empty:
            continue
        summary["z_col"] = z_col
        summary["Signal"] = Z_LABELS[z_col]
        summary["Lookback"] = f"{lb}Y"
        summary["Score"] = summary["Avg %"] * (summary["Hit Rate %"] / 100.0) * np.sqrt(summary["N"].clip(lower=0))
        rows.append(summary)

    if not rows:
        return None, None

    all_rows = pd.concat(rows, ignore_index=True)
    all_rows = all_rows[(all_rows["N"] >= min_trades) & (all_rows["Avg %"] > 0) & (all_rows["Hit Rate %"] >= 50)]
    if all_rows.empty:
        return None, None

    long_rows = all_rows[all_rows["Direction"].str.startswith("Long")].sort_values("Score", ascending=False)
    short_rows = all_rows[all_rows["Direction"].str.startswith("Short")].sort_values("Score", ascending=False)
    best_long = long_rows.iloc[0] if not long_rows.empty else None
    best_short = short_rows.iloc[0] if not short_rows.empty else None
    return best_long, best_short


def distribution_frame(result: dict) -> pd.DataFrame:
    rows = []
    for direction, data in [("Long", result["df_long"]), ("Short", result["df_short"])]:
        if data.empty:
            continue
        for horizon in [f"+{h}d" for h in HORIZONS_DAYS]:
            if horizon not in data.columns:
                continue
            for signal_date, value in data[horizon].dropna().items():
                rows.append({
                    "Direction": direction,
                    "Horizon": horizon,
                    "Forward return %": float(value),
                    "Signal date": signal_date,
                })
    return pd.DataFrame(rows)
