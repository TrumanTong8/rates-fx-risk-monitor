from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.expression.models import Expression, TestResult


REGIME_PREFERENCES = {
    "Goldilocks: growth up, inflation down": {},
    "Stagflation: growth down, inflation up": {},
    "Slowdown: growth down, inflation down": {},
    "Reflation: growth up, inflation up": {},
}

# Macro inputs come from FRED. "fred" is the series id (a tuple splices ids end-to-end),
# "transform" is a key of src.data.fred.TRANSFORMS, and "lag_days" shifts each observation
# from its reference-period date to roughly when it is published, so regimes never use
# data before release (FRED dates September CPI as Sep 1; it is released mid-October).
ECONOMIC_INDICATORS = [
    {"ticker": "GDP QoQ saar", "fred": "A191RL1Q225SBEA", "transform": "level", "lag_days": 120, "bucket": "growth", "direction": 1, "label": "US GDP QoQ"},
    {"ticker": "PAYEMS chg", "fred": "PAYEMS", "transform": "diff", "lag_days": 35, "bucket": "growth", "direction": 1, "label": "Nonfarm payrolls"},
    {"ticker": "UNRATE", "fred": "UNRATE", "transform": "level", "lag_days": 35, "bucket": "growth", "direction": -1, "label": "Unemployment rate"},
    {"ticker": "RSAFS MoM", "fred": "RSAFS", "transform": "mom_pct", "lag_days": 45, "bucket": "growth", "direction": 1, "label": "Retail sales MoM"},
    {"ticker": "ICSA", "fred": "ICSA", "transform": "thousands", "lag_days": 5, "bucket": "growth", "direction": -1, "label": "Initial jobless claims"},
    {"ticker": "JTSJOL", "fred": "JTSJOL", "transform": "level", "lag_days": 65, "bucket": "growth", "direction": 1, "label": "JOLTS openings"},
    {"ticker": "CPIAUCSL YoY", "fred": "CPIAUCSL", "transform": "yoy_pct", "lag_days": 43, "bucket": "inflation", "direction": 1, "label": "CPI YoY"},
    {"ticker": "CPILFESL YoY", "fred": "CPILFESL", "transform": "yoy_pct", "lag_days": 43, "bucket": "inflation", "direction": 1, "label": "Core CPI YoY"},
    {"ticker": "PCEPILFE YoY", "fred": "PCEPILFE", "transform": "yoy_pct", "lag_days": 60, "bucket": "inflation", "direction": 1, "label": "Core PCE YoY"},
    {"ticker": "PCEPILFE MoM", "fred": "PCEPILFE", "transform": "mom_pct", "lag_days": 60, "bucket": "inflation", "direction": 1, "label": "Core PCE MoM"},
]

MARKET_INDICATORS = [
    {"ticker": "T10YIE", "fred": "T10YIE", "transform": "level", "lag_days": 0, "bucket": "inflation", "direction": 1, "label": "10Y breakeven"},
    {"ticker": "T5YIFR", "fred": "T5YIFR", "transform": "level", "lag_days": 0, "bucket": "inflation", "direction": 1, "label": "5Y5Y inflation forward"},
    {"ticker": "Fed funds target", "fred": ("DFEDTAR", "DFEDTARU"), "transform": "level", "lag_days": 0, "bucket": "policy", "direction": 1, "label": "Fed funds target"},
    {"ticker": "SOFR", "fred": "SOFR", "transform": "level", "lag_days": 0, "bucket": "policy", "direction": 1, "label": "SOFR"},
    {"ticker": "DGS2", "fred": "DGS2", "transform": "level", "lag_days": 0, "bucket": "policy", "direction": 1, "label": "US 2Y yield"},
    {"ticker": "DGS10", "fred": "DGS10", "transform": "level", "lag_days": 0, "bucket": "policy", "direction": 1, "label": "US 10Y yield"},
]


def build_fred_regime_inputs(fetch) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Pull every indicator with fetch(fred_ids, transform) -> date-indexed Series and shape them
    like the old database tables: economic (ticker_name, date, actual, revised) and
    market (ticker_name, date, close). Also returns labels of indicators that failed to load.
    """
    def safe_fetch(row: dict) -> pd.Series | None:
        try:
            return fetch(row["fred"], row["transform"])
        except Exception:
            return None

    all_rows = ECONOMIC_INDICATORS + MARKET_INDICATORS
    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = dict(zip([row["ticker"] for row in all_rows], pool.map(safe_fetch, all_rows)))

    economic, market, failed = [], [], []
    for rows, frames, value_col in [(ECONOMIC_INDICATORS, economic, "actual"), (MARKET_INDICATORS, market, "close")]:
        for row in rows:
            series = fetched[row["ticker"]]
            if series is None:
                failed.append(row["label"])
                continue
            dates = pd.DatetimeIndex(series.index) + pd.Timedelta(days=row["lag_days"])
            frame = pd.DataFrame({"ticker_name": row["ticker"], "date": dates, value_col: series.values})
            frames.append(frame[frame["date"] <= pd.Timestamp.today()])

    economic_frame = pd.concat(economic, ignore_index=True) if economic else pd.DataFrame(columns=["ticker_name", "date", "actual"])
    economic_frame["revised"] = np.nan
    market_frame = pd.concat(market, ignore_index=True) if market else pd.DataFrame(columns=["ticker_name", "date", "close"])
    return economic_frame, market_frame, failed


def monthly_closes(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices[["date", "close"]].dropna().copy()
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
    return frame.groupby("month", as_index=False)["close"].last()


def build_market_regime_history(growth_proxy: pd.DataFrame, inflation_proxy: pd.DataFrame) -> pd.DataFrame:
    growth = monthly_closes(growth_proxy).rename(columns={"close": "growth_proxy"})
    inflation = monthly_closes(inflation_proxy).rename(columns={"close": "inflation_proxy"})
    frame = pd.merge(growth, inflation, on="month", how="inner").sort_values("month")
    frame["growth_6m"] = frame["growth_proxy"].pct_change(6)
    frame["inflation_6m"] = frame["inflation_proxy"].pct_change(6)
    frame["growth_threshold"] = frame["growth_6m"].rolling(60, min_periods=24).median()
    frame["inflation_threshold"] = frame["inflation_6m"].rolling(60, min_periods=24).median()
    frame["growth_up"] = frame["growth_6m"] >= frame["growth_threshold"]
    frame["inflation_up"] = frame["inflation_6m"] >= frame["inflation_threshold"]
    frame["regime"] = np.select(
        [
            frame["growth_up"] & ~frame["inflation_up"],
            ~frame["growth_up"] & frame["inflation_up"],
            ~frame["growth_up"] & ~frame["inflation_up"],
            frame["growth_up"] & frame["inflation_up"],
        ],
        [
            "Goldilocks: growth up, inflation down",
            "Stagflation: growth down, inflation up",
            "Slowdown: growth down, inflation down",
            "Reflation: growth up, inflation up",
        ],
        default="Unclassified",
    )
    return frame.dropna(subset=["growth_6m", "inflation_6m"]).reset_index(drop=True)


def _monthly_release_values(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["month", "ticker_name", "value"])
    data = frame.copy()
    data["value"] = pd.to_numeric(data["actual"], errors="coerce")
    data["value"] = data["value"].fillna(pd.to_numeric(data["revised"], errors="coerce"))
    data = data.dropna(subset=["value"])
    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp("M")
    return data.sort_values("date").groupby(["month", "ticker_name"], as_index=False)["value"].last()


def _monthly_market_values(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["month", "ticker_name", "value"])
    data = frame.rename(columns={"close": "value"}).dropna(subset=["value"]).copy()
    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp("M")
    return data.sort_values("date").groupby(["month", "ticker_name"], as_index=False)["value"].last()


def _indicator_signal(series: pd.Series) -> pd.Series:
    momentum = series - series.shift(6)
    baseline = momentum.rolling(60, min_periods=24).mean()
    volatility = momentum.rolling(60, min_periods=24).std()
    return (momentum - baseline) / volatility.replace(0, np.nan)


def _classified_regime(growth_score: pd.Series, inflation_score: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                (growth_score >= 0) & (inflation_score < 0),
                (growth_score < 0) & (inflation_score >= 0),
                (growth_score < 0) & (inflation_score < 0),
                (growth_score >= 0) & (inflation_score >= 0),
            ],
            [
                "Goldilocks: growth up, inflation down",
                "Stagflation: growth down, inflation up",
                "Slowdown: growth down, inflation down",
                "Reflation: growth up, inflation up",
            ],
            default="Unclassified",
        ),
        index=growth_score.index,
    )


def build_macro_regime_history(
    economic_data: pd.DataFrame,
    market_data: pd.DataFrame,
    max_stale_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicators = {row["ticker"]: row for row in ECONOMIC_INDICATORS + MARKET_INDICATORS}
    monthly = pd.concat([_monthly_release_values(economic_data), _monthly_market_values(market_data)], ignore_index=True)
    if monthly.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot = monthly.pivot(index="month", columns="ticker_name", values="value").sort_index()
    signal_rows = []
    adjusted_signals = {}
    latest_values = {}
    signal_source_months = {}
    for ticker, meta in indicators.items():
        if ticker not in pivot.columns:
            continue
        signal = _indicator_signal(pivot[ticker]) * meta["direction"]
        adjusted_signals[ticker] = signal.ffill(limit=max_stale_months)
        latest_values[ticker] = pivot[ticker].ffill(limit=max_stale_months)
        valid_signal_month = pd.Series(signal.index.where(signal.notna()), index=signal.index).ffill(limit=max_stale_months)
        signal_source_months[ticker] = valid_signal_month
        for month, value in adjusted_signals[ticker].dropna().items():
            source_month = valid_signal_month.at[month]
            latest = latest_values[ticker].at[month] if month in latest_values[ticker].index else np.nan
            signal_rows.append(
                {
                    "month": month,
                    "source_month": source_month,
                    "stale_months": 0 if pd.isna(source_month) else (month.to_period("M") - pd.Timestamp(source_month).to_period("M")).n,
                    "ticker": ticker,
                    "label": meta["label"],
                    "bucket": meta["bucket"],
                    "latest": latest,
                    "z_score": value,
                    "contribution": value,
                }
            )

    if not adjusted_signals:
        return pd.DataFrame(), pd.DataFrame()

    signals = pd.DataFrame(adjusted_signals).sort_index()
    growth_cols = [row["ticker"] for row in ECONOMIC_INDICATORS if row["bucket"] == "growth" and row["ticker"] in signals]
    inflation_cols = [
        row["ticker"]
        for row in ECONOMIC_INDICATORS + MARKET_INDICATORS
        if row["bucket"] == "inflation" and row["ticker"] in signals
    ]
    policy_cols = [row["ticker"] for row in MARKET_INDICATORS if row["bucket"] == "policy" and row["ticker"] in signals]
    if not growth_cols or not inflation_cols:
        return pd.DataFrame(), pd.DataFrame(signal_rows)

    regimes = pd.DataFrame(index=signals.index)
    regimes["growth_score"] = signals[growth_cols].mean(axis=1, skipna=True)
    regimes["inflation_score"] = signals[inflation_cols].mean(axis=1, skipna=True)
    regimes["policy_score"] = signals[policy_cols].mean(axis=1, skipna=True) if policy_cols else np.nan
    regimes["growth_indicators"] = signals[growth_cols].notna().sum(axis=1)
    regimes["inflation_indicators"] = signals[inflation_cols].notna().sum(axis=1)
    regimes["policy_indicators"] = signals[policy_cols].notna().sum(axis=1) if policy_cols else 0
    regimes["regime"] = _classified_regime(regimes["growth_score"], regimes["inflation_score"])
    regimes = regimes.dropna(subset=["growth_score", "inflation_score"]).reset_index()
    evidence = pd.DataFrame(signal_rows)
    return regimes, evidence


def regime_forward_returns(prices: pd.DataFrame, regimes: pd.DataFrame, direction: str, horizon_days: int) -> pd.DataFrame:
    horizon_months = max(round(horizon_days / 21), 1)
    asset = monthly_closes(prices)
    asset["forward_return"] = asset["close"].shift(-horizon_months) / asset["close"] - 1
    if direction == "short":
        asset["forward_return"] = -asset["forward_return"]
    merged = pd.merge(asset, regimes[["month", "regime"]], on="month", how="inner")
    return merged.dropna(subset=["forward_return", "regime"])


def summarize_regime_returns(prices: pd.DataFrame, regimes: pd.DataFrame, direction: str, horizon_days: int) -> pd.DataFrame:
    returns = regime_forward_returns(prices, regimes, direction, horizon_days)
    if returns.empty:
        return pd.DataFrame(columns=["regime", "sample", "avg_return", "median_return", "win_rate", "worst_return", "volatility"])
    grouped = returns.groupby("regime")["forward_return"]
    return pd.DataFrame(
        {
            "sample": grouped.count(),
            "avg_return": grouped.mean(),
            "median_return": grouped.median(),
            "win_rate": grouped.apply(lambda series: (series > 0).mean()),
            "worst_return": grouped.min(),
            "volatility": grouped.std(),
        }
    ).reset_index()


def score_regime(
    expression: Expression,
    prices: pd.DataFrame,
    regimes: pd.DataFrame | None,
    selected_regime: str,
    horizon_days: int,
) -> TestResult:
    if regimes is None or regimes.empty:
        return TestResult(50, "low", "No historical regime data available; neutral score.")

    summary = summarize_regime_returns(prices, regimes, expression.direction, horizon_days)
    row = summary[summary["regime"] == selected_regime]
    if row.empty:
        return TestResult(50, "low", f"No historical samples for {selected_regime}; neutral score.")

    sample = int(row["sample"].iloc[0])
    avg_return = float(row["avg_return"].iloc[0])
    median_return = float(row["median_return"].iloc[0])
    win_rate = float(row["win_rate"].iloc[0])
    worst_return = float(row["worst_return"].iloc[0])
    volatility = float(row["volatility"].iloc[0]) if pd.notna(row["volatility"].iloc[0]) else 0.0

    avg_contribution = np.clip(avg_return * 250, -25, 25)
    median_contribution = np.clip(median_return * 150, -15, 15)
    win_contribution = np.clip((win_rate - 0.5) * 70, -25, 25)
    drawdown_penalty = np.clip(abs(min(worst_return, 0)) * 80, 0, 20)
    sample_penalty = 15 if sample < 12 else 8 if sample < 24 else 0
    score = 50 + avg_contribution + median_contribution + win_contribution - drawdown_penalty - sample_penalty
    confidence = "high" if sample >= 48 else "medium" if sample >= 24 else "low"
    reason = (
        f"Historical {selected_regime} samples: {sample}. "
        f"Forward return over ~{horizon_days}d: avg {avg_return:.2%}, median {median_return:.2%}, "
        f"win rate {win_rate:.0%}, worst {worst_return:.2%}, forward-return volatility {volatility:.2%}. "
        f"Formula: 50 + avg {avg_contribution:.1f} + median {median_contribution:.1f} "
        f"+ win-rate {win_contribution:.1f} - drawdown {drawdown_penalty:.1f} - sample {sample_penalty:.1f}."
    )
    return TestResult(float(np.clip(score, 0, 100)), confidence, reason)

