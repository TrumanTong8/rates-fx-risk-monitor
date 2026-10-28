from __future__ import annotations

import numpy as np
import pandas as pd

from src.expression.models import TestResult


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def score_positioning(cot: pd.DataFrame, direction: str) -> TestResult:
    if cot.empty or len(cot) < 52:
        return TestResult(50, "low", "No reliable CFTC history mapped yet; neutral until a market mapping is confirmed.")

    long_col = _first_existing(cot, ["lev_money_positions_long", "m_money_positions_long_all", "noncomm_positions_long_all"])
    short_col = _first_existing(cot, ["lev_money_positions_short", "m_money_positions_short_all", "noncomm_positions_short_all"])
    oi_col = _first_existing(cot, ["open_interest_all", "open_interest"])
    if not long_col or not short_col:
        return TestResult(50, "low", "CFTC rows found, but long/short columns were not recognized.")

    series = (cot[long_col] - cot[short_col]).dropna()
    latest = float(series.iloc[-1])
    z = (latest - float(series.tail(156).mean())) / float(series.tail(156).std(ddof=0) or 1)
    percentile = (series <= latest).mean()
    crowded_long = percentile > 0.85 or z > 1.5
    crowded_short = percentile < 0.15 or z < -1.5
    if direction == "long":
        score = 35 if crowded_long else 78 if crowded_short else 58
    else:
        score = 35 if crowded_short else 78 if crowded_long else 58

    oi_note = ""
    if oi_col and pd.notna(cot[oi_col].iloc[-1]) and float(cot[oi_col].iloc[-1]) != 0:
        oi_note = f" Net position is {latest / float(cot[oi_col].iloc[-1]):.1%} of OI."
    reason = f"Net spec percentile {percentile:.0%}, z-score {z:.2f}.{oi_note}"
    return TestResult(float(np.clip(score, 0, 100)), "medium", reason, str(cot["date"].iloc[-1].date()))


