from __future__ import annotations

from src.expression.models import TestResult


def weighted_score(results: dict[str, TestResult], weights: dict[str, float]) -> float:
    active = {key: max(float(value), 0.0) for key, value in weights.items() if key in results}
    total_weight = sum(active.values())
    if total_weight == 0:
        return 0.0
    return sum(results[key].score * weight for key, weight in active.items()) / total_weight


def verdict(score: float) -> str:
    if score >= 75:
        return "Strong"
    if score >= 60:
        return "Constructive"
    if score >= 45:
        return "Mixed"
    return "Weak"


