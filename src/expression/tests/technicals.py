from __future__ import annotations

import numpy as np
import pandas as pd

from src.expression.models import TestResult


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def realized_volatility(prices: pd.DataFrame, window_days: int) -> float:
    if len(prices) < max(window_days, 2):
        return float("nan")
    closes = pd.to_numeric(prices["close"], errors="coerce")
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if window_days == 1:
        if log_returns.empty:
            return float("nan")
        return float(abs(log_returns.iloc[-1]) * np.sqrt(252))
    trailing = log_returns.tail(window_days)
    if trailing.empty or trailing.shape[0] < 2:
        return float("nan")
    return float(trailing.std(ddof=1) * np.sqrt(252))


def realized_volatility_history(prices: pd.DataFrame, window_days: int) -> pd.Series:
    closes = pd.to_numeric(prices["close"], errors="coerce")
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if log_returns.empty:
        return pd.Series(dtype=float)
    if window_days == 1:
        return (log_returns.abs() * np.sqrt(252)).dropna()
    history = log_returns.rolling(window=window_days, min_periods=window_days).std(ddof=1) * np.sqrt(252)
    return history.dropna()


def percentile_rank_against_history(history: pd.Series, current_value: float | int | None) -> str:
    if current_value is None or pd.isna(current_value):
        return "N/A"
    if history.empty:
        return "N/A â€” insufficient history"
    history = history.dropna()
    if history.empty:
        return "N/A â€” insufficient history"
    below = float((history < current_value).sum())
    equal = float((history == current_value).sum())
    percentile = (below + 0.5 * equal) / float(len(history)) * 100.0
    return f"{percentile:.0f}%"


def _history_tail_by_days(history: pd.Series, trading_days: int) -> pd.Series:
    if history.empty:
        return history
    return history.dropna().tail(trading_days)


def rv_percentile_snapshot(prices: pd.DataFrame) -> dict[str, float | str]:
    rv_windows = {
        "1D": 1,
        "1M": 21,
        "3M": 63,
    }
    lookbacks = {
        "6M": 126,
        "1Y": 252,
        "3Y": 756,
        "10Y": 2520,
    }
    columns = [
        ("1D RV", "6M"),
        ("1D RV", "1Y"),
        ("1M RV", "6M"),
        ("1M RV", "1Y"),
        ("1M RV", "3Y"),
        ("1M RV", "10Y"),
        ("3M RV", "1Y"),
        ("3M RV", "3Y"),
        ("3M RV", "10Y"),
    ]
    values: dict[str, float | str] = {}
    histories = {label: realized_volatility_history(prices, days) for label, days in rv_windows.items()}
    current = {label: realized_volatility(prices, days) for label, days in rv_windows.items()}
    for rv_label, lookback_label in columns:
        base_label = rv_label.replace(" RV", "")
        history = _history_tail_by_days(histories[base_label], lookbacks[lookback_label])
        column_name = f"{rv_label} Percentile ({lookback_label})"
        values[column_name] = percentile_rank_against_history(history, current[base_label])
    return values


def _round_metric(value: float | int | None, metric_name: str, asset_class: str = "Other") -> float:
    if value is None or pd.isna(value):
        return float("nan")
    normalized = float(value)
    class_key = str(asset_class).lower()
    if metric_name in {"Current", "50 EMA", "100 EMA", "200 EMA"}:
        decimals = 4 if any(token in class_key for token in ["fx", "forex", "currency", "rate", "rates"]) else 3 if any(token in class_key for token in ["future", "futures", "commodity", "metal", "oil"]) else 2
    elif metric_name in {"1M Return", "3M Return"}:
        decimals = 4
    elif metric_name in {"1M RV", "3M RV", "6M RV"}:
        decimals = 4
    elif metric_name == "RSI (14D)":
        decimals = 1
    else:
        decimals = 3
    return round(normalized, decimals)


def technical_snapshot(prices: pd.DataFrame, asset_class: str = "Other", available_years: float | None = None) -> dict[str, float | str]:
    frame = technical_frame(prices)
    latest = frame.iloc[-1]
    rv_1m = realized_volatility(prices, 21)
    rv_3m = realized_volatility(prices, 63)
    rv_6m = realized_volatility(prices, 126)
    history_1m = realized_volatility_history(prices, 21)
    history_3m = realized_volatility_history(prices, 63)
    history_6m = realized_volatility_history(prices, 126)
    has_30y = bool(available_years is not None and float(available_years) >= 30.0)
    return {
        "Current": _round_metric(float(latest["close"]), "Current", asset_class),
        "1M Return": _round_metric(float(latest["return_1m"]) if pd.notna(latest["return_1m"]) else np.nan, "1M Return", asset_class),
        "3M Return": _round_metric(float(latest["return_3m"]) if pd.notna(latest["return_3m"]) else np.nan, "3M Return", asset_class),
        "50 EMA": _round_metric(float(latest["ema_50"]) if pd.notna(latest["ema_50"]) else np.nan, "50 EMA", asset_class),
        "100 EMA": _round_metric(float(latest["ema_100"]) if pd.notna(latest["ema_100"]) else np.nan, "100 EMA", asset_class),
        "200 EMA": _round_metric(float(latest["ema_200"]) if pd.notna(latest["ema_200"]) else np.nan, "200 EMA", asset_class),
        "RSI (14D)": _round_metric(float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else np.nan, "RSI (14D)", asset_class),
        "1M RV": _round_metric(rv_1m, "1M RV", asset_class),
        "1M RV Percentile (%) (30y)": percentile_rank_against_history(history_1m, rv_1m) if has_30y else "N/A - insufficient history",
        "3M RV": _round_metric(rv_3m, "3M RV", asset_class),
        "3M RV Percentile (%) (30y)": percentile_rank_against_history(history_3m, rv_3m) if has_30y else "N/A - insufficient history",
        "6M RV": _round_metric(rv_6m, "6M RV", asset_class),
        "6M RV Percentile (%) (30y)": percentile_rank_against_history(history_6m, rv_6m) if has_30y else "N/A - insufficient history",
    }


def technical_frame(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    for window in [20, 50, 100, 200]:
        frame[f"ma_{window}"] = frame["close"].rolling(window).mean()
        frame[f"ma_{window}_slope"] = frame[f"ma_{window}"].diff(20)
    for window in [50, 100, 200]:
        frame[f"ema_{window}"] = frame["close"].ewm(span=window, adjust=False).mean()
    frame["rsi_14"] = rsi(frame["close"])
    frame["return_1m"] = frame["close"].pct_change(21)
    frame["return_3m"] = frame["close"].pct_change(63)
    return frame


def score_technicals(prices: pd.DataFrame, direction: str) -> TestResult:
    if len(prices) < 220:
        return TestResult(35, "low", "Not enough data for 200-day moving average.")
    frame = technical_frame(prices)
    latest = frame.iloc[-1]
    close = float(latest["close"])
    ma_checks = [close > float(latest[f"ma_{w}"]) for w in [20, 50, 100, 200]]
    trend_score = sum(ma_checks) / 4
    momentum = float(latest["return_3m"]) if pd.notna(latest["return_3m"]) else 0
    if direction != "long":
        trend_score = 1 - trend_score
        momentum = -momentum
    rsi_value = float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else 50
    stretch_penalty = 12 if (direction == "long" and rsi_value > 75) or (direction != "long" and rsi_value < 25) else 0
    score = trend_score * 70 + np.clip(momentum * 120, -15, 15) + 15 - stretch_penalty
    reason = f"Price is above {sum(ma_checks)}/4 key MAs; 3M return {float(latest['return_3m']):.2%}; RSI {rsi_value:.1f}."
    return TestResult(float(np.clip(score, 0, 100)), "high", reason, str(latest["date"].date()))

