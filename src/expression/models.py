from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    name: str
    symbol: str
    asset_class: str
    direction: str
    view_link: str
    cftc_market: str | None = None


@dataclass(frozen=True)
class TestResult:
    score: float
    confidence: str
    reason: str
    as_of: str = ""

