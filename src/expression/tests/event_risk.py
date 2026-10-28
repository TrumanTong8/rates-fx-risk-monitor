from __future__ import annotations

from datetime import date

import pandas as pd

from src.expression.models import TestResult


HIGH_IMPORTANCE = ["CPI", "Consumer Price", "PPI", "Producer Price", "Non-Farm", "NFP", "Employment Situation", "FOMC", "Federal Funds", "Powell", "PCE"]


def score_event_risk(events: pd.DataFrame, horizon_days: int) -> TestResult:
    if events.empty:
        return TestResult(50, "low", "No automatic event feed available; keep manual calendar checks.")
    upcoming = events.copy()
    if "days_until" not in upcoming.columns:
        today = date.today()
        upcoming["days_until"] = upcoming["date"].apply(lambda d: (d - today).days)
        upcoming = upcoming[(upcoming["days_until"] >= 0) & (upcoming["days_until"] <= horizon_days)]
    if upcoming.empty:
        return TestResult(65, "medium", "No medium/high-impact USD events inside the selected horizon.")

    titles = " | ".join(upcoming["title"].astype(str).tolist())
    key_count = sum(any(term.lower() in title.lower() for term in HIGH_IMPORTANCE) for title in upcoming["title"].astype(str))
    high_impact_count = int((upcoming["impact"].astype(str).str.lower() == "high").sum()) if "impact" in upcoming.columns else key_count
    near_key_count = sum(
        any(term.lower() in str(row["title"]).lower() for term in HIGH_IMPORTANCE)
        and int(row.get("days_until", horizon_days)) <= 21
        for _, row in upcoming.iterrows()
    )
    horizon_months = max(horizon_days / 30, 1)
    event_density = len(upcoming) / horizon_months
    high_impact_density = high_impact_count / horizon_months
    key_event_density = key_count / horizon_months
    event_penalty = min(event_density * 3.0, 18)
    high_impact_penalty = min(high_impact_density * 6.0, 24)
    key_event_penalty = min(key_event_density * 8.0, 28)
    near_key_penalty = min(near_key_count * 4.0, 16)
    score = 80 - event_penalty - high_impact_penalty - key_event_penalty - near_key_penalty
    reason = (
        f"{len(upcoming)} events inside horizon; {high_impact_count} high-impact; "
        f"{key_count} CPI/PPI/jobs/FOMC-type events; {near_key_count} key events inside 21 days. "
        f"Standardized density: {event_density:.1f} events / 30d, {key_event_density:.1f} key events / 30d. "
        f"Formula: 80 - event-density penalty {event_penalty:.1f} - high-impact-density penalty {high_impact_penalty:.1f} "
        f"- key-event penalty {key_event_penalty:.1f} - near-key penalty {near_key_penalty:.1f}."
    )
    return TestResult(float(max(20, score)), "medium", reason + f" Watch: {titles[:160]}.")

