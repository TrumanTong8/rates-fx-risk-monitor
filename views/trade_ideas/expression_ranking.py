from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.expression.charts.diagnostic_charts import (
    event_timeline,
    headline_spec_columns,
    net_percent_open_interest_chart,
    net_position_chart,
    participant_breakdown_chart,
    participant_net_percent_oi_chart,
    participant_weekly_net_change_chart,
    participant_columns,
    positioning_chart,
    regime_matrix,
    score_heatmap,
)
from src.data import cache, fred
from src.data.symbols import to_yahoo
from src.expression.charts.seasonality_chart import make_seasonality_chart
from src.expression.charts.technical_chart import make_technical_chart
from src.expression.coverage import price_coverage
from src.expression.models import Expression
from src.expression.scoring.ranker import verdict, weighted_score
from src.expression.tests.event_risk import score_event_risk
from src.expression.tests.positioning import score_positioning
from src.expression.tests.regime import (
    ECONOMIC_INDICATORS,
    MARKET_INDICATORS,
    REGIME_PREFERENCES,
    build_fred_regime_inputs,
    build_macro_regime_history,
    build_market_regime_history,
    score_regime,
    summarize_regime_returns,
)
from src.expression.tests.seasonality import monthly_seasonality, score_seasonality
from src.expression.tests.technicals import rv_percentile_snapshot, score_technicals, technical_frame, technical_snapshot


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config" / "expression"
SAVED_EXPRESSIONS_PATH = CONFIG_DIR / "saved_expressions.yaml"


@st.cache_data(ttl=60 * 60)
def load_yaml(name: str):
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cached_prices(symbol: str, years: int) -> pd.DataFrame:
    return cache.daily_prices(symbol, years)


def cached_events(country: str) -> pd.DataFrame:
    return cache.events(currencies=(country,), impacts=("High", "Medium"))


def cached_cot(market: str | None, cot_by_market: dict[str, pd.DataFrame | str]) -> pd.DataFrame:
    result = cot_by_market.get(market) if market else None
    if result is None:
        return pd.DataFrame()
    if isinstance(result, str):
        raise RuntimeError(result)
    return result


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def cached_macro_regimes(fred_api_key: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    def fetch(series_ids, transform):
        ids = list(series_ids) if isinstance(series_ids, tuple) else series_ids
        return fred.fetch_fred_indicator(ids, transform, api_key=fred_api_key or None)

    try:
        economic, market, failed = build_fred_regime_inputs(fetch)
        regimes, evidence = build_macro_regime_history(economic, market)
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"FRED regime model unavailable ({exc}); using market-proxy fallback."
    if regimes.empty:
        return regimes, evidence, "FRED returned insufficient macro history; using market-proxy fallback."
    note = f" Missing: {', '.join(failed)}." if failed else ""
    return regimes, evidence, f"FRED macro regime model.{note}"


def load_expressions(config: dict) -> list[Expression]:
    return [Expression(**row) for row in config["expressions"]]


def empty_expression_table() -> pd.DataFrame:
    columns = ["name", "symbol", "asset_class", "direction", "view_link", "cftc_market"]
    return pd.DataFrame(columns=columns)


def normalize_expression_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = empty_expression_table()
    for column in normalized.columns:
        normalized[column] = frame[column] if column in frame.columns else ""
    normalized["symbol"] = normalized["symbol"].astype(str).map(to_yahoo)
    normalized["direction"] = normalized["direction"].where(normalized["direction"].isin(["long", "short"]), "long")
    return normalized.drop_duplicates(["name", "symbol", "direction"], keep="last").reset_index(drop=True)


def load_saved_expression_frame(config: dict) -> pd.DataFrame:
    if SAVED_EXPRESSIONS_PATH.exists():
        try:
            with open(SAVED_EXPRESSIONS_PATH, "r", encoding="utf-8") as handle:
                saved = yaml.safe_load(handle) or {}
            saved_rows = saved.get("expressions", [])
            if saved_rows:
                return normalize_expression_frame(pd.DataFrame(saved_rows))
        except Exception:
            pass
    return normalize_expression_frame(pd.DataFrame([expression.__dict__ for expression in load_expressions(config)]))


def expression_frame_yaml(frame: pd.DataFrame) -> str:
    normalized = normalize_expression_frame(frame)
    rows = []
    for row in normalized.to_dict("records"):
        clean_row = {}
        for column in normalized.columns:
            value = row.get(column, "")
            clean_row[column] = "" if pd.isna(value) else str(value).strip()
        if clean_row["symbol"]:
            rows.append(clean_row)
    payload = {
        "expressions": rows,
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def save_expression_frame(frame: pd.DataFrame) -> None:
    SAVED_EXPRESSIONS_PATH.write_text(expression_frame_yaml(frame), encoding="utf-8")


def load_uploaded_expression_frame(content: bytes) -> pd.DataFrame:
    saved = yaml.safe_load(content.decode("utf-8")) or {}
    rows = saved.get("expressions", []) if isinstance(saved, dict) else []
    if not rows:
        raise ValueError("No 'expressions' list found in the uploaded file.")
    return normalize_expression_frame(pd.DataFrame(rows))


def add_example(example: dict) -> None:
    current = normalize_expression_frame(st.session_state.expression_universe)
    updated = pd.concat([current, pd.DataFrame([example])], ignore_index=True)
    st.session_state.expression_universe = normalize_expression_frame(updated)


def move_expression_row(frame: pd.DataFrame, row_index: int, target_index: int) -> pd.DataFrame:
    normalized = normalize_expression_frame(frame)
    if normalized.empty:
        return normalized
    row_index = max(0, min(row_index, len(normalized) - 1))
    target_index = max(0, min(target_index, len(normalized) - 1))
    rows = normalized.to_dict("records")
    row = rows.pop(row_index)
    rows.insert(target_index, row)
    return normalize_expression_frame(pd.DataFrame(rows))


def event_countries_for_expression(expression: Expression) -> list[str]:
    known = {"USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF", "CNH", "CNY"}
    asset_class = expression.asset_class.lower()
    symbol_root = expression.symbol.split()[0].upper().replace("/", "").replace("=X", "")
    countries: list[str] = []
    if "fx" in asset_class or expression.symbol.upper().endswith("=X"):
        if len(symbol_root) == 3:
            # Yahoo quotes "JPY=X" as USD/JPY
            countries = ["USD", symbol_root]
        elif len(symbol_root) >= 6:
            countries = [symbol_root[:3], symbol_root[3:6]]
        else:
            name_root = expression.name.upper().replace("/", "")
            countries = [code for code in known if code in name_root]
    else:
        countries = ["USD"]

    filtered = []
    for country in countries:
        if country in known and country not in filtered:
            filtered.append(country)
    return filtered or ["USD"]


def events_for_expression(expression: Expression) -> pd.DataFrame:
    frames = []
    for country in event_countries_for_expression(expression):
        frame = cached_events(country)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["title", "country", "date", "impact", "forecast", "previous", "source"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["title", "country", "date"]).sort_values(["date", "country", "title"])


def events_within_horizon(events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if events.empty:
        return events
    today = pd.Timestamp.today().date()
    frame = events.copy()
    frame["days_until"] = frame["date"].apply(lambda value: (value - today).days)
    return frame[(frame["days_until"] >= 0) & (frame["days_until"] <= horizon)].sort_values("date")


def format_regime_return_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.rename(
        columns={
            "regime": "Regime",
            "sample": "Sample",
            "avg_return": "Avg Forward Return",
            "median_return": "Median Forward Return",
            "win_rate": "Win Rate",
            "worst_return": "Worst Forward Return",
            "volatility": "Forward Return Volatility",
        }
    ).copy()
    percent_columns = ["Avg Forward Return", "Median Forward Return", "Win Rate", "Worst Forward Return", "Forward Return Volatility"]
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
    return display


def format_seasonality_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.rename(
        columns={
            "lookback_years": "Lookback Years",
            "month_index": "Month",
            "month_name": "Month Name",
            "avg_return": "Avg Return",
            "median_return": "Median Return",
            "win_rate": "Win Rate",
            "q1": "Q1 Return",
            "q3": "Q3 Return",
            "lower": "Worst Return",
            "upper": "Best Return",
            "sample": "Sample",
        }
    ).copy()
    percent_columns = ["Avg Return", "Median Return", "Win Rate", "Q1 Return", "Q3 Return", "Worst Return", "Best Return"]
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.2%}")
    for column in ["Lookback Years", "Month", "Sample"]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):,.0f}")
    return display


def regime_input_table(evidence: pd.DataFrame) -> pd.DataFrame:
    latest_tickers = set()
    last_signal_months = {}
    latest_source_months = {}
    latest_stale_months = {}
    if not evidence.empty:
        latest_month = evidence["month"].max()
        latest_rows = evidence[evidence["month"].eq(latest_month)]
        latest_tickers = set(latest_rows["ticker"])
        last_signal_months = evidence.groupby("ticker")["month"].max().to_dict()
        if "source_month" in latest_rows.columns:
            latest_source_months = latest_rows.set_index("ticker")["source_month"].to_dict()
        if "stale_months" in latest_rows.columns:
            latest_stale_months = latest_rows.set_index("ticker")["stale_months"].to_dict()
    rows = []
    for row in ECONOMIC_INDICATORS + MARKET_INDICATORS:
        last_signal = last_signal_months.get(row["ticker"])
        source_month = latest_source_months.get(row["ticker"])
        stale_months = latest_stale_months.get(row["ticker"])
        rows.append(
            {
                "Bucket": row["bucket"].title(),
                "Data Used": row["label"],
                "FRED Series": "+".join(row["fred"]) if isinstance(row["fred"], tuple) else row["fred"],
                "Interpretation": "Higher = stronger score" if row["direction"] == 1 else "Higher = weaker score, inverted",
                "Latest Signal Month": "" if pd.isna(last_signal) else pd.to_datetime(last_signal).strftime("%Y-%m-%d"),
                "Source Month Used": "" if pd.isna(source_month) else pd.to_datetime(source_month).strftime("%Y-%m-%d"),
                "Age": "" if pd.isna(stale_months) else "Fresh" if int(stale_months) == 0 else f"Carried {int(stale_months)}m",
                "Used Latest Month": "Yes" if row["ticker"] in latest_tickers else "No",
            }
        )
    return pd.DataFrame(rows)


def positioning_percentile_table(details: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, detail in details.items():
        if "error" in detail or "cot" not in detail:
            continue
        cot = detail["cot"]
        if cot.empty:
            continue
        long_col, short_col, participant = headline_spec_columns(cot)
        if not long_col or not short_col:
            continue
        frame = cot.copy()
        frame["long"] = pd.to_numeric(frame[long_col], errors="coerce")
        frame["short"] = pd.to_numeric(frame[short_col], errors="coerce")
        frame["net_position"] = frame["long"] - frame["short"]
        frame = frame.dropna(subset=["date", "long", "short", "net_position"]).sort_values("date")
        if frame.empty:
            continue

        latest_value = float(frame["net_position"].iloc[-1])
        windows = {
            "1Y": 52,
            "3Y": 156,
            "5Y": 260,
            "10Y": 520,
        }
        percentile_values = {}
        for label, periods in windows.items():
            if len(frame) < periods:
                percentile_values[label] = None
                continue
            history = frame["net_position"].tail(periods)
            percentile_values[label] = float((history <= latest_value).mean()) if not history.empty else None

        rows.append(
            {
                "Expression": name,
                "Participant": participant,
                "Current Net Position": latest_value,
                "1Y Net Contract Percentile": percentile_values["1Y"],
                "3Y Net Contract Percentile": percentile_values["3Y"],
                "5Y Net Contract Percentile": percentile_values["5Y"],
                "10Y Net Contract Percentile": percentile_values["10Y"],
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("Current Net Position", ascending=False).reset_index(drop=True)


def positioning_oi_percentile_table(details: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, detail in details.items():
        if "error" in detail or "cot" not in detail:
            continue
        cot = detail["cot"]
        if cot.empty:
            continue
        long_col, short_col, participant = headline_spec_columns(cot)
        oi_col = "open_interest_all" if "open_interest_all" in cot.columns else "open_interest" if "open_interest" in cot.columns else None
        if not long_col or not short_col or not oi_col:
            continue
        frame = cot.copy()
        frame["long"] = pd.to_numeric(frame[long_col], errors="coerce")
        frame["short"] = pd.to_numeric(frame[short_col], errors="coerce")
        frame["open_interest"] = pd.to_numeric(frame[oi_col], errors="coerce")
        frame["net_pct_oi"] = (frame["long"] - frame["short"]) / frame["open_interest"].replace(0, pd.NA)
        frame = frame.dropna(subset=["date", "net_pct_oi"]).sort_values("date")
        if frame.empty:
            continue

        latest_value = float(frame["net_pct_oi"].iloc[-1])
        windows = {
            "1Y": 52,
            "3Y": 156,
            "5Y": 260,
            "10Y": 520,
        }
        percentile_values = {}
        for label, periods in windows.items():
            if len(frame) < periods:
                percentile_values[label] = None
                continue
            history = frame["net_pct_oi"].tail(periods)
            percentile_values[label] = float((history <= latest_value).mean()) if not history.empty else None

        rows.append(
            {
                "Expression": name,
                "Participant": participant,
                "Current Net % OI": latest_value,
                "1Y Net % OI Percentile": percentile_values["1Y"],
                "3Y Net % OI Percentile": percentile_values["3Y"],
                "5Y Net % OI Percentile": percentile_values["5Y"],
                "10Y Net % OI Percentile": percentile_values["10Y"],
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("Current Net % OI", ascending=False).reset_index(drop=True)


def _format_number(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.0f}"


def _format_percent(value: float | None) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1%}"


def _format_float(value: float | None) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def participant_positioning_table(cot: pd.DataFrame) -> pd.DataFrame:
    if cot.empty:
        return pd.DataFrame()
    oi_col = "open_interest_all" if "open_interest_all" in cot.columns else "open_interest" if "open_interest" in cot.columns else None
    frame = cot.copy().sort_values("date")
    latest = frame.iloc[-1]
    rows = []
    for participant, long_col, short_col in participant_columns(cot):
        data = frame[["date", long_col, short_col] + ([oi_col] if oi_col else [])].copy()
        data["long"] = pd.to_numeric(data[long_col], errors="coerce")
        data["short"] = pd.to_numeric(data[short_col], errors="coerce")
        data["net"] = data["long"] - data["short"]
        if oi_col:
            data["open_interest"] = pd.to_numeric(data[oi_col], errors="coerce")
            data["net_pct_oi"] = data["net"] / data["open_interest"].replace(0, pd.NA)
        else:
            data["open_interest"] = pd.NA
            data["net_pct_oi"] = pd.NA
        data = data.dropna(subset=["net"])
        if data.empty:
            continue
        latest_row = data.iloc[-1]
        hist_3y = data["net"].tail(156)
        hist_std = hist_3y.std(ddof=0)
        z_3y = (latest_row["net"] - hist_3y.mean()) / hist_std if pd.notna(hist_std) and hist_std != 0 else pd.NA
        pct_3y = (hist_3y <= latest_row["net"]).mean() if len(hist_3y) >= 52 else pd.NA
        rows.append(
            {
                "Participant": participant,
                "Gross Long": _format_number(latest_row["long"]),
                "Gross Short": _format_number(latest_row["short"]),
                "Net": _format_number(latest_row["net"]),
                "Net % OI": _format_percent(latest_row["net_pct_oi"]),
                "3Y Z": _format_float(z_3y),
                "3Y Percentile": _format_percent(pct_3y),
                "1W Net Change": _format_number(latest_row["net"] - data["net"].iloc[-2] if len(data) > 1 else pd.NA),
                "4W Net Change": _format_number(latest_row["net"] - data["net"].iloc[-5] if len(data) > 4 else pd.NA),
                "13W Net Change": _format_number(latest_row["net"] - data["net"].iloc[-14] if len(data) > 13 else pd.NA),
            }
        )
    return pd.DataFrame(rows)


def speculative_flow_table(cot: pd.DataFrame) -> pd.DataFrame:
    if cot.empty:
        return pd.DataFrame()
    long_col, short_col, category = headline_spec_columns(cot)
    oi_col = "open_interest_all" if "open_interest_all" in cot.columns else "open_interest" if "open_interest" in cot.columns else None
    if not long_col or not short_col:
        return pd.DataFrame()
    frame = cot.copy().sort_values("date")
    frame["long"] = pd.to_numeric(frame[long_col], errors="coerce")
    frame["short"] = pd.to_numeric(frame[short_col], errors="coerce")
    frame["net"] = frame["long"] - frame["short"]
    frame["open_interest"] = pd.to_numeric(frame[oi_col], errors="coerce") if oi_col else pd.NA
    frame = frame.dropna(subset=["date", "net"])
    if frame.empty:
        return pd.DataFrame()
    latest = frame.iloc[-1]
    rows = []
    for label, periods in [("1W", 1), ("4W", 4), ("13W", 13)]:
        if len(frame) <= periods:
            continue
        prior = frame.iloc[-periods - 1]
        net_change = latest["net"] - prior["net"]
        long_change = latest["long"] - prior["long"]
        short_change = latest["short"] - prior["short"]
        rows.append(
            {
                "Participant": category,
                "Window": label,
                "Net Change": _format_number(net_change),
                "Net Change % OI": _format_percent(net_change / latest["open_interest"] if pd.notna(latest["open_interest"]) and latest["open_interest"] != 0 else pd.NA),
                "Gross Long Change": _format_number(long_change),
                "Gross Short Change": _format_number(short_change),
            }
        )
    return pd.DataFrame(rows)


def participant_flow_change_table(cot: pd.DataFrame) -> pd.DataFrame:
    if cot.empty:
        return pd.DataFrame()
    oi_col = "open_interest_all" if "open_interest_all" in cot.columns else "open_interest" if "open_interest" in cot.columns else None
    frame = cot.copy().sort_values("date")
    if len(frame) < 2:
        return pd.DataFrame()
    rows = []
    for participant, long_col, short_col in participant_columns(cot):
        data = frame[["date", long_col, short_col] + ([oi_col] if oi_col else [])].copy()
        data["long"] = pd.to_numeric(data[long_col], errors="coerce")
        data["short"] = pd.to_numeric(data[short_col], errors="coerce")
        data["net"] = data["long"] - data["short"]
        if oi_col:
            data["open_interest"] = pd.to_numeric(data[oi_col], errors="coerce")
            data["net_pct_oi"] = data["net"] / data["open_interest"].replace(0, pd.NA)
        else:
            data["open_interest"] = pd.NA
            data["net_pct_oi"] = pd.NA
        data = data.dropna(subset=["long", "short", "net"]).sort_values("date")
        if len(data) < 2:
            continue
        latest = data.iloc[-1]
        prior = data.iloc[-2]
        hist_pct = data["net_pct_oi"].dropna().tail(156)
        percentile_3y = (hist_pct <= latest["net_pct_oi"]).mean() if len(hist_pct) >= 52 and pd.notna(latest["net_pct_oi"]) else pd.NA
        rows.append(
            {
                "Participant": participant,
                "Gross Long Change": _format_number(latest["long"] - prior["long"]),
                "Gross Short Change": _format_number(latest["short"] - prior["short"]),
                "Net Change": _format_number(latest["net"] - prior["net"]),
                "Current Net": _format_number(latest["net"]),
                "Current Net % OI": _format_percent(latest["net_pct_oi"]),
                "3Y Net % OI Percentile": _format_percent(percentile_3y),
            }
        )
    return pd.DataFrame(rows)


def squeeze_risk_table(details: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, detail in details.items():
        if "error" in detail or "cot" not in detail:
            continue
        cot = detail["cot"]
        expression = detail["expression"]
        if cot.empty:
            continue
        oi_col = "open_interest_all" if "open_interest_all" in cot.columns else "open_interest" if "open_interest" in cot.columns else None
        if not oi_col:
            continue

        frame = cot.copy().sort_values("date")
        frame["open_interest"] = pd.to_numeric(frame[oi_col], errors="coerce")
        if frame.empty or len(frame) < 2:
            continue

        latest = frame.iloc[-1]
        prior = frame.iloc[-2]
        open_interest = float(latest["open_interest"]) if pd.notna(latest["open_interest"]) else pd.NA
        oi_change = latest["open_interest"] - prior["open_interest"] if pd.notna(latest["open_interest"]) and pd.notna(prior["open_interest"]) else pd.NA

        participant_rows = {}
        for participant, long_col, short_col in participant_columns(cot):
            data = frame[["date", long_col, short_col, oi_col]].copy()
            data["long"] = pd.to_numeric(data[long_col], errors="coerce")
            data["short"] = pd.to_numeric(data[short_col], errors="coerce")
            data["open_interest"] = pd.to_numeric(data[oi_col], errors="coerce")
            data["net"] = data["long"] - data["short"]
            data["net_pct_oi"] = data["net"] / data["open_interest"].replace(0, pd.NA)
            data = data.dropna(subset=["date", "long", "short", "net", "net_pct_oi"]).sort_values("date")
            if len(data) < 2:
                continue
            latest_row = data.iloc[-1]
            prior_row = data.iloc[-2]
            history = data["net_pct_oi"].tail(156)
            percentile_3y = float((history <= latest_row["net_pct_oi"]).mean()) if len(history) >= 52 else pd.NA
            participant_rows[participant] = {
                "net": float(latest_row["net"]),
                "net_pct_oi": float(latest_row["net_pct_oi"]),
                "short_change": float(latest_row["short"] - prior_row["short"]),
                "long_change": float(latest_row["long"] - prior_row["long"]),
                "net_change": float(latest_row["net"] - prior_row["net"]),
                "percentile_3y": percentile_3y,
            }

        if not participant_rows:
            continue

        long_col, short_col, headline = headline_spec_columns(cot)
        headline_metrics = participant_rows.get(headline)
        if not headline_metrics:
            continue

        asset_manager_short_change = participant_rows.get("Asset Manager", {}).get("short_change", pd.NA)
        leveraged_short_change = participant_rows.get("Leveraged Money", {}).get("short_change", pd.NA)
        managed_short_change = participant_rows.get("Managed Money", {}).get("short_change", pd.NA)
        primary_short_change = leveraged_short_change if pd.notna(leveraged_short_change) else managed_short_change

        net_pct = headline_metrics["net_pct_oi"]
        percentile = headline_metrics["percentile_3y"]
        net_change = headline_metrics["net_change"]
        flow_threshold = 0.01 * open_interest if pd.notna(open_interest) else 0
        material_short_flow = pd.notna(primary_short_change) and primary_short_change > flow_threshold
        material_long_flow = pd.notna(headline_metrics["long_change"]) and headline_metrics["long_change"] > flow_threshold
        crowded_short = pd.notna(percentile) and percentile <= 0.20 and net_pct < 0
        crowded_long = pd.notna(percentile) and percentile >= 0.80 and net_pct > 0

        if crowded_short and material_short_flow:
            driver = "Both: crowded short + fresh short flow"
        elif crowded_long and material_long_flow:
            driver = "Both: crowded long + fresh long flow"
        elif material_short_flow or material_long_flow:
            driver = "Flow-driven"
        elif crowded_short or crowded_long:
            driver = "Level-driven"
        else:
            driver = "Not stretched"

        if crowded_short:
            risk_type = "Short squeeze risk"
            expression_read = "Supportive for long expressions; painful for shorts if shorts cover."
        elif crowded_long:
            risk_type = "Long unwind risk"
            expression_read = "Risk for long expressions; supportive for shorts if longs unwind."
        elif net_change < 0:
            risk_type = "Shorts building"
            expression_read = "Watch for squeeze risk if the short reaches an extreme."
        elif net_change > 0:
            risk_type = "Longs building"
            expression_read = "Watch for unwind risk if the long reaches an extreme."
        else:
            risk_type = "Neutral"
            expression_read = "No clear CFTC squeeze or unwind signal."

        rows.append(
            {
                "Expression": name,
                "Asset Class": expression.asset_class,
                "CFTC Contract": expression.cftc_market or "",
                "Headline Participant": headline,
                "Asset Manager Weekly Short Change": asset_manager_short_change,
                "Leveraged / Managed Money Weekly Short Change": primary_short_change,
                "Open Interest Change": oi_change,
                "Current Net % OI": net_pct,
                "3Y Net % OI Percentile": percentile,
                "Move Type": driver,
                "Risk Read": risk_type,
                "Expression Read": expression_read,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["Risk Read", "3Y Net % OI Percentile"], ascending=[True, True]).reset_index(drop=True)


def style_squeeze_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    if frame.empty:
        return frame.style

    numeric = frame.copy()
    numeric_columns = [
        "Asset Manager Weekly Short Change",
        "Leveraged / Managed Money Weekly Short Change",
        "Open Interest Change",
        "Current Net % OI",
        "3Y Net % OI Percentile",
    ]
    for column in numeric_columns:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    def risk_style(row):
        styles = []
        for column, value in row.items():
            if column == "Risk Read":
                if value == "Short squeeze risk":
                    styles.append("background-color: rgba(34,197,94,0.22); color: #166534; font-weight: 700;")
                elif value == "Long unwind risk":
                    styles.append("background-color: rgba(239,68,68,0.20); color: #991b1b; font-weight: 700;")
                else:
                    styles.append("")
            elif column == "Move Type" and isinstance(value, str) and value.startswith("Both"):
                styles.append("background-color: rgba(245,158,11,0.22); color: #92400e; font-weight: 700;")
            else:
                styles.append("")
        return styles

    formatters = {
        "Asset Manager Weekly Short Change": lambda value: "" if pd.isna(value) else f"{float(value):,.0f}",
        "Leveraged / Managed Money Weekly Short Change": lambda value: "" if pd.isna(value) else f"{float(value):,.0f}",
        "Open Interest Change": lambda value: "" if pd.isna(value) else f"{float(value):,.0f}",
        "Current Net % OI": lambda value: "" if pd.isna(value) else f"{float(value):.1%}",
        "3Y Net % OI Percentile": lambda value: "N/A - insufficient history" if pd.isna(value) else f"{float(value):.0%}",
    }
    return numeric.style.apply(risk_style, axis=1).format(formatters)


def positioning_source_table(details: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, detail in details.items():
        if "error" in detail or "cot" not in detail:
            continue
        cot = detail["cot"]
        expression = detail["expression"]
        if cot.empty:
            rows.append(
                {
                    "Expression": name,
                    "CFTC Mapping": expression.cftc_market or "",
                    "Rows": 0,
                    "First Date": "",
                    "Last Date": "",
                    "Source": "No CFTC rows returned",
                    "Matched Contracts": "",
                    "Long Column": "",
                    "Short Column": "",
                }
            )
            continue
        long_col = next((col for col in ["lev_money_positions_long", "m_money_positions_long_all", "noncomm_positions_long_all"] if col in cot.columns), None)
        short_col = next((col for col in ["lev_money_positions_short", "m_money_positions_short_all", "noncomm_positions_short_all"] if col in cot.columns), None)
        source = str(cot["source_url"].iloc[-1]) if "source_url" in cot.columns else ""
        source_name = "TFF financial futures" if "gpe5-46if" in source else "Disaggregated commodities" if "kh3c-gbw2" in source else source
        contracts = ", ".join(sorted(cot["contract_market_name"].dropna().astype(str).unique())[:6])
        rows.append(
            {
                "Expression": name,
                "CFTC Mapping": expression.cftc_market or "",
                "Rows": len(cot),
                "First Date": cot["date"].min().date(),
                "Last Date": cot["date"].max().date(),
                "Source": source_name,
                "Matched Contracts": contracts,
                "Long Column": long_col or "",
                "Short Column": short_col or "",
            }
        )
    return pd.DataFrame(rows)


def style_positioning_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    if frame.empty:
        return frame.style

    percent_columns = [
        column
        for column in frame.columns
        if "Percentile" in column or column in ["Current Net % OI"]
    ]
    numeric = frame.copy()
    for column in percent_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    if "Current Net Position" in numeric.columns:
        numeric["Current Net Position"] = pd.to_numeric(numeric["Current Net Position"], errors="coerce")

    def color_net_position(row):
        styles = []
        for column, value in row.items():
            if column in ["Current Net Position", "Current Net % OI"]:
                if pd.isna(value):
                    styles.append("background-color: #f3f4f6; color: #111827;")
                elif value > 0:
                    styles.append("background-color: rgba(34,197,94,0.18); color: #166534;")
                elif value < 0:
                    styles.append("background-color: rgba(239,68,68,0.16); color: #991b1b;")
                else:
                    styles.append("background-color: #f3f4f6; color: #111827;")
            elif column in percent_columns:
                if pd.isna(value):
                    styles.append("color: #6b7280; background-color: #f9fafb;")
                else:
                    styles.append("")
            else:
                styles.append("")
        return styles

    styled = numeric.style.apply(color_net_position, axis=1)
    formatters = {
        column: (lambda value: "N/A - insufficient history" if pd.isna(value) else f"{float(value):.0%}")
        for column in percent_columns
    }
    if "Current Net Position" in numeric.columns:
        formatters["Current Net Position"] = lambda value: "" if pd.isna(value) else f"{float(value):,.0f}"
    if "Current Net % OI" in numeric.columns:
        formatters["Current Net % OI"] = lambda value: "" if pd.isna(value) else f"{float(value):.1%}"
    styled = styled.format(formatters)
    for column in percent_columns:
        if "Percentile" in column:
            styled = styled.background_gradient(cmap="Blues", subset=[column], vmin=0, vmax=1)
    return styled


def style_rv_percentile_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    if frame.empty:
        return frame.style

    numeric = frame.copy()
    for column in numeric.columns:
        numeric[column] = (
            numeric[column]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace({"N/A": pd.NA, "N/A - insufficient history": pd.NA, "N/A â€” insufficient history": pd.NA})
        )
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce") / 100

    styled = numeric.style.format(lambda value: "N/A - insufficient history" if pd.isna(value) else f"{float(value):.0%}")
    styled = styled.background_gradient(cmap="YlOrRd", axis=None, vmin=0, vmax=1)
    styled = styled.set_properties(**{"text-align": "right"})
    return styled


def data_source_audit_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Component": "Prices",
                "Data Source": "Yahoo Finance (yfinance)",
                "Tickers / Feed": "Daily bars for futures (=F), FX (=X), indices (^) and ETFs",
                "Used For": "Technicals, RV, seasonality, regime forward returns",
                "Refresh / Lag": "Delayed daily bars; cached for 1 hour",
                "Main Caveat": "Futures are front-month continuous series with roll jumps; ETF prices are dividend-adjusted.",
            },
            {
                "Component": "CFTC positioning",
                "Data Source": "CFTC Public Reporting Socrata",
                "Tickers / Feed": "TFF financial futures + Disaggregated commodities",
                "Used For": "Positioning, crowding, flow, squeeze / unwind risk",
                "Refresh / Lag": "Weekly COT report, usually Friday for Tuesday positions",
                "Main Caveat": "Futures positioning is a proxy for ETFs, indices, FX pairs, and rates views.",
            },
            {
                "Component": "Events",
                "Data Source": "ForexFactory, FRED release calendar, Federal Reserve",
                "Tickers / Feed": "FX calendar JSON (this + next week), FRED US release dates, FOMC calendar",
                "Used For": "Event-risk timeline and event-risk score",
                "Refresh / Lag": "Calendar dependent",
                "Main Caveat": "Forecast fields can be sparse; FX uses both currencies in the pair.",
            },
            {
                "Component": "Macro regime",
                "Data Source": "FRED (St. Louis Fed)",
                "Tickers / Feed": "GDP, payrolls, CPI, PCE, claims, JOLTS, yields, breakevens, SOFR",
                "Used For": "Growth / inflation / policy regime classification",
                "Refresh / Lag": "Release dependent; recent data can be carried forward",
                "Main Caveat": "FRED holds revised values, not first prints; dates are shifted by a typical release lag.",
            },
            {
                "Component": "Derived analytics",
                "Data Source": "Calculated inside dashboard",
                "Tickers / Feed": "Close-to-close returns, CFTC long/short/OI fields",
                "Used For": "RV, percentiles, z-scores, seasonality, net % OI",
                "Refresh / Lag": "Same as underlying source",
                "Main Caveat": "Bad input data will distort derived signals unless filtered.",
            },
        ]
    )


def known_data_caveats() -> None:
    st.markdown(
        """
        - **Prices are Yahoo Finance daily bars.** Futures (`GC=F`, `ZN=F`, ...) are front-month continuous series, so long histories include roll jumps.
        - **Gold and silver use front COMEX futures** (`GC=F`, `SI=F`) because Yahoo has no spot history; **CFTC `GOLD` / `SILVER`** are used for positioning.
        - **Macro data is revised, not first-print.** FRED stores the latest revised values, so the regime history is slightly cleaner than what was known in real time.
        - **FX seasonality filters extreme bad prints** before calculating monthly return distributions.
        - **CFTC is futures-based.** `NASDAQ MINI` is a Nasdaq futures proxy, not QQQ itself.
        - **Rates are contract-specific.** `UST 2Y NOTE`, `UST 10Y NOTE`, and `UST BOND` are not automatically combined into one Treasury curve signal.
        - **FX CFTC convention:** currency futures are exposure to the named non-USD currency versus USD; `JAPANESE YEN` maps naturally to a JPY/USD view.
        """
    )


st.title("Expression Ranking")
st.caption("Compare ways to express one macro view across seasonality, technicals, positioning, event risk, and regime fit.")

config = load_yaml("instruments.yaml")
examples_config = load_yaml("expression_examples.yaml")
default_weights = load_yaml("scoring_weights.yaml")
default_view = config["default_view"]
expressions = load_expressions(config)

if "expression_universe" not in st.session_state:
    st.session_state.expression_universe = load_saved_expression_frame(config)
if "expression_editor_version" not in st.session_state:
    st.session_state.expression_editor_version = 0

with st.sidebar:
    st.header("View")
    view_name = st.text_input("View name", default_view["name"])
    thesis = st.text_area("Thesis", default_view["thesis"], height=110)
    horizon_days = st.number_input("Horizon days", min_value=5, max_value=365, value=int(default_view["horizon_days"]), step=5)
    years = st.slider("Price history years", 1, 30, 30)
    regime = st.selectbox("Current / forecast regime", list(REGIME_PREFERENCES.keys()))

    with st.expander("Scoring weights", expanded=False):
        st.caption("Equal-weighted by default. Adjust only if you deliberately want one evidence type to matter more.")
        weights = {}
        for key in default_weights:
            weights[key] = st.slider(key.replace("_", " ").title(), 0, 50, int(default_weights[key]))

st.subheader(view_name)
st.write(thesis)

st.markdown("### Expression Universe")
st.caption("Edit Yahoo Finance symbols, directions, and CFTC mappings here. Add rows for more FX pairs, futures, ETFs, stocks, commodities, or rates proxies.")
st.markdown(
    """
    **FX convention:** use currency pairs in **base/quote** form. For example, **JPYUSD** means **long JPY / short USD**, while **USDJPY** means **long USD / short JPY**. CFTC currency futures are usually reported as exposure to the **named non-USD currency versus USD**, so **Japanese Yen** positioning maps naturally to a **JPY/USD** view. If you express the trade as **USD/JPY**, the CFTC positioning signal needs to be interpreted in the opposite direction.
    """
)

with st.popover("? How are scores calculated?"):
    st.markdown(
        """
        **Important:** the total score is a navigation aid, not a trading signal.

        **Seasonality**: current calendar month historical 21-trading-day returns. Positive average return and higher win rate help the selected trade direction.

        **Technicals**: price versus 20/50/100/200-day moving averages, 3-month momentum, and RSI stretch penalty.

        **Positioning**: CFTC net speculative positioning where mapped. The app calculates percentile and a 3-year z-score. Crowded longs hurt long trades; crowded shorts can support long trades.

        **Event risk**: medium/high-impact upcoming events inside the horizon. More major events currently reduce the score because they add uncertainty.

        **Regime**: monthly growth and inflation regimes are built from FRED macro data. Each macro indicator is converted into a 6-month momentum z-score versus its own rolling 5-year history, then the expression is scored from its own historical forward returns during similar regimes.

        **Total**: `sum(component score * weight) / sum(weights)`.

        The weights are subjective. For now, treat the score map as a compact summary and inspect each evidence tab before trusting the total.
        """
    )

with st.expander("Add examples", expanded=False):
    st.caption("Click any example to add it to the editable universe with symbol, asset class, direction, and CFTC market pre-filled.")
    for group, examples in examples_config["groups"].items():
        st.write(f"**{group}**")
        cols = st.columns(4)
        for index, example in enumerate(examples):
            if cols[index % 4].button(example["name"], key=f"example-{group}-{example['symbol']}-{example['direction']}"):
                add_example(example)
                st.rerun()

editable = st.data_editor(
    st.session_state.expression_universe,
    num_rows="dynamic",
    use_container_width=True,
    key=f"expr_expression_editor_{st.session_state.expression_editor_version}",
    column_config={
        "direction": st.column_config.SelectboxColumn("direction", options=["long", "short"], required=True),
    },
)
st.session_state.expression_universe = normalize_expression_frame(editable)

order_options = [
    f"{index + 1}. {row.get('name') or row.get('symbol') or 'Unnamed'}"
    for index, row in enumerate(st.session_state.expression_universe.to_dict("records"))
]
if order_options:
    order_col, top_col, up_col, down_col, bottom_col = st.columns([3, 1, 1, 1, 1])
    with order_col:
        selected_order_row = st.selectbox("Reorder expression", order_options, help="Move the selected expression in the universe. Press Save options to keep the order after closing the app.")
    selected_order_index = order_options.index(selected_order_row)
    move_actions = [
        (top_col, "Top", 0, selected_order_index == 0),
        (up_col, "Up", selected_order_index - 1, selected_order_index == 0),
        (down_col, "Down", selected_order_index + 1, selected_order_index == len(order_options) - 1),
        (bottom_col, "Bottom", len(order_options) - 1, selected_order_index == len(order_options) - 1),
    ]
    for column, label, target_index, disabled in move_actions:
        with column:
            if st.button(label, disabled=disabled, use_container_width=True):
                st.session_state.expression_universe = move_expression_row(
                    st.session_state.expression_universe,
                    selected_order_index,
                    target_index,
                )
                st.session_state.expression_editor_version += 1
                st.rerun()

save_col, download_col, saved_status_col = st.columns([1, 1, 3])
with save_col:
    if st.button("Save options", type="primary", use_container_width=True):
        save_expression_frame(st.session_state.expression_universe)
        st.success("Saved")
with download_col:
    st.download_button(
        "Download list",
        data=expression_frame_yaml(st.session_state.expression_universe).encode("utf-8"),
        file_name="saved_expressions.yaml",
        mime="application/x-yaml",
        use_container_width=True,
    )
with saved_status_col:
    st.caption(
        f"**Save options** writes `{SAVED_EXPRESSIONS_PATH.name}` so the list reloads next time. "
        "On the hosted app that file resets whenever the app restarts, so use **Download list** "
        "to keep a copy and load it back below."
    )

uploaded_universe = st.file_uploader("Load a saved list (.yaml)", type=["yaml", "yml"], key="expr_universe_upload")
if uploaded_universe is not None and st.session_state.get("expr_universe_upload_id") != uploaded_universe.file_id:
    st.session_state["expr_universe_upload_id"] = uploaded_universe.file_id
    try:
        st.session_state.expression_universe = load_uploaded_expression_frame(uploaded_universe.getvalue())
        st.session_state.expression_editor_version += 1
        st.rerun()
    except Exception as exc:
        st.error(f"Could not load that file: {exc}")
expressions = [
    Expression(
        name=str(row.get("name", "")).strip() or str(row.get("symbol", "")).strip(),
        symbol=str(row.get("symbol", "")).strip(),
        asset_class=str(row.get("asset_class", "Other")).strip() or "Other",
        direction=str(row.get("direction", "long")).strip() or "long",
        view_link=str(row.get("view_link", "")).strip(),
        cftc_market=str(row.get("cftc_market", "")).strip() or None,
    )
    for row in editable.to_dict("records")
    if str(row.get("symbol", "")).strip()
]

with st.spinner("Loading prices, positioning and macro data..."):
    macro_regime_history, macro_regime_evidence, regime_source = cached_macro_regimes(cache.get_secret("FRED_API_KEY"))
    cot_by_market = cache.cot_markets(tuple(sorted({e.cftc_market for e in expressions if e.cftc_market})))
try:
    if macro_regime_history.empty:
        regime_history = build_market_regime_history(cached_prices("SPY", years), cached_prices("DBC", years))
    else:
        regime_history = macro_regime_history
except Exception:
    regime_history = pd.DataFrame()

rows = []
coverage_rows = []
technical_comparison_rows = []
rv_percentile_rows = []
details: dict[str, dict] = {}
for expression in expressions:
    try:
        prices = cached_prices(expression.symbol, years)
        coverage = price_coverage(prices, years)
        cot = cached_cot(expression.cftc_market, cot_by_market)
        expression_events = events_for_expression(expression)
        visible_expression_events = events_within_horizon(expression_events, int(horizon_days))
        technical_snapshot_row = technical_snapshot(prices, expression.asset_class, coverage["available_years"])
        rv_percentile_row = rv_percentile_snapshot(prices)
        results = {
            "seasonality": score_seasonality(prices, expression.direction),
            "technicals": score_technicals(prices, expression.direction),
            "positioning": score_positioning(cot, expression.direction),
            "event_risk": score_event_risk(visible_expression_events, horizon_days),
            "regime": score_regime(expression, prices, regime_history, regime, int(horizon_days)),
        }
        total = weighted_score(results, weights)
        rows.append(
            {
                "Expression": expression.name,
                "Symbol": expression.symbol,
                "Asset Class": expression.asset_class,
                "Total": round(total, 1),
                "Verdict": verdict(total),
                "Seasonality": round(results["seasonality"].score, 1),
                "Technicals": round(results["technicals"].score, 1),
                "Positioning": round(results["positioning"].score, 1),
                "Event Risk": round(results["event_risk"].score, 1),
                "Regime": round(results["regime"].score, 1),
                "History": coverage["status"],
            }
        )
        coverage_rows.append(
            {
                "Expression": expression.name,
                "Symbol": expression.symbol,
                "First Date": coverage["first_date"],
                "Last Date": coverage["last_date"],
                "Daily Bars": coverage["observations"],
                "Available Years": coverage["available_years"],
                "Requested Years": coverage["requested_years"],
                "Coverage": f"{coverage['coverage_ratio']:.0%}",
                "Status": coverage["status"],
            }
        )
        technical_comparison_rows.append({"Expression": expression.name, **technical_snapshot_row})
        rv_percentile_rows.append({"Expression": expression.name, **rv_percentile_row})
        details[expression.name] = {
            "expression": expression,
            "prices": prices,
            "cot": cot,
            "results": results,
            "coverage": coverage,
            "technical_snapshot": technical_snapshot_row,
            "rv_percentile_snapshot": rv_percentile_row,
            "events": expression_events,
            "visible_events": visible_expression_events,
            "event_countries": event_countries_for_expression(expression),
        }
    except Exception as exc:
        rows.append(
            {
                "Expression": expression.name,
                "Symbol": expression.symbol,
                "Asset Class": expression.asset_class,
                "Total": 0,
                "Verdict": "Data error",
                "Seasonality": None,
                "Technicals": None,
                "Positioning": None,
                "Event Risk": None,
                "Regime": None,
                "History": "Data error",
            }
        )
        coverage_rows.append(
            {
                "Expression": expression.name,
                "Symbol": expression.symbol,
                "First Date": "",
                "Last Date": "",
                "Daily Bars": 0,
                "Available Years": 0,
                "Requested Years": years,
                "Coverage": "0%",
                "Status": f"Data error: {exc}",
            }
        )
        details[expression.name] = {"error": str(exc), "expression": expression}

ranking = pd.DataFrame(rows).sort_values("Total", ascending=False)

st.markdown("### Score Map")
st.caption("Compact summary only. The weights are adjustable and subjective, so use the detailed tabs below for the actual evidence.")
st.plotly_chart(score_heatmap(ranking), use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

with st.expander("Ranking table", expanded=False):
    st.dataframe(ranking, use_container_width=True, hide_index=True)

with st.expander("Data coverage check", expanded=True):
    st.caption("Backtests are only as good as the available history. New ETFs, some volatility symbols, and source-limited series may have less history than requested.")
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)

available_expressions = ranking["Expression"].tolist() if not ranking.empty else []
if not available_expressions:
    available_expressions = [expression.name for expression in expressions]
current_selection = st.session_state.get("expr_inspect_expression")
if current_selection not in available_expressions:
    current_selection = available_expressions[0] if available_expressions else None
st.session_state["expr_inspect_expression"] = current_selection
selected = st.selectbox("Inspect expression", available_expressions, index=available_expressions.index(current_selection) if current_selection in available_expressions else 0, key="expr_inspect_expression")
detail = details[selected]

if "error" in detail:
    st.error(detail["error"])
else:
    expression = detail["expression"]
    prices = detail["prices"]
    results = detail["results"]
    latest = technical_frame(prices).iloc[-1]

    st.markdown(f"### {expression.name}")
    st.write(expression.view_link)

    coverage = detail["coverage"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Last close", f"{latest['close']:.2f}")
    col2.metric("RSI 14", f"{latest['rsi_14']:.1f}")
    col3.metric("3M return", f"{latest['return_3m']:.2%}")
    col4.metric("As of", str(latest["date"].date()))
    col5.metric("History", f"{coverage['available_years']}y")

    tabs = st.tabs(["Test explanations", "Seasonality", "Technicals", "Positioning", "Events", "Regime", "Methodology & Sources"])
    with tabs[0]:
        for name, result in results.items():
            st.write(f"**{name.replace('_', ' ').title()}**: {result.score:.1f} ({result.confidence})")
            st.caption(result.reason)
    with tabs[1]:
        stats = monthly_seasonality(prices, expression.direction)
        with st.expander("? Seasonality methodology", expanded=False):
            st.markdown(
                """
                Seasonality uses the selected expression's price history from Yahoo Finance. The dashboard calculates 21-trading-day returns by calendar month, filters extreme bad price jumps, and flips returns for short expressions so positive values mean the selected trade made money.

                The box shows the historical distribution for each month. The number above each box is the **hit rate**, or the share of observations with a positive direction-adjusted return.
                """
            )
        st.markdown(
            """
            This box-and-whisker chart shows the distribution of historical direction-adjusted returns for the current calendar month across different lookback windows. For short expressions, returns are flipped so positive values mean the selected expression made money. The box shows the middle range of outcomes, the whiskers show wider historical outcomes, and points/outliers show unusually strong or weak months.

            The number shown above each box is the **hit rate**: the percentage of historical observations where the direction-adjusted return was positive for the selected expression.
            """
        )
        st.plotly_chart(make_seasonality_chart(stats), use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        st.dataframe(format_seasonality_table(stats), use_container_width=True, hide_index=True)
    with tabs[2]:
        with st.expander("? Technicals and RV terms", expanded=False):
            st.markdown(
                """
                **1D RV** is an annualized absolute daily move, so it is best treated as a shock indicator. **1M RV** and **3M RV** are annualized standard deviations of daily returns over roughly 21 and 63 trading days.

                **RV percentile** ranks the current RV reading against its own history over the stated lookback. A high percentile means volatility is high versus that lookback, not automatically good or bad. **RSI** measures recent price stretch; moving averages and EMA levels summarize trend.
                """
            )
        st.plotly_chart(make_technical_chart(technical_frame(prices)), use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        comparison_columns = [
            "Current",
            "1M Return",
            "3M Return",
            "50 EMA",
            "100 EMA",
            "200 EMA",
            "RSI (14D)",
            "1M RV",
            "1M RV Percentile (%) (30y)",
            "3M RV",
            "3M RV Percentile (%) (30y)",
            "6M RV",
            "6M RV Percentile (%) (30y)",
        ]
        comparison_df = pd.DataFrame(technical_comparison_rows).set_index("Expression")
        comparison_df = comparison_df.reindex(columns=comparison_columns)
        percent_columns = ["1M Return", "3M Return"]
        for column in percent_columns:
            if column in comparison_df.columns:
                comparison_df[column] = comparison_df[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
                )
        rv_percent_columns = ["1M RV", "3M RV", "6M RV"]
        for column in rv_percent_columns:
            if column in comparison_df.columns:
                comparison_df[column] = comparison_df[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
                )
        st.caption("Comparison across the current expression universe. Returns and RV values are displayed as percentages; RV percentile cells show the current valueâ€™s placement against the same metricâ€™s 30-year history for that expression.")
        st.dataframe(comparison_df, use_container_width=True, hide_index=False)

        st.markdown("#### RV Percentile Matrix")
        st.markdown(
            """
            RV percentile compares the current realized-volatility reading against that same expression's own historical realized-volatility observations. **1D RV** is best read as a short-term shock indicator. **1M RV** and **3M RV** are better for volatility-regime context. The color scale highlights hotter volatility percentiles, from low/quiet to high/stressed; it is **not** a good/bad signal by itself.
            """
        )
        rv_percentile_columns = [
            "1D RV Percentile (6M)",
            "1D RV Percentile (1Y)",
            "1M RV Percentile (6M)",
            "1M RV Percentile (1Y)",
            "1M RV Percentile (3Y)",
            "1M RV Percentile (10Y)",
            "3M RV Percentile (1Y)",
            "3M RV Percentile (3Y)",
            "3M RV Percentile (10Y)",
        ]
        rv_percentile_df = pd.DataFrame(rv_percentile_rows).set_index("Expression")
        rv_percentile_df = rv_percentile_df.reindex(columns=rv_percentile_columns)
        st.dataframe(style_rv_percentile_table(rv_percentile_df), use_container_width=True, hide_index=False)
    with tabs[3]:
        with st.expander("? CFTC positioning terms", expanded=False):
            st.markdown(
                """
                **Gross long** and **gross short** are the reported long and short books before netting. **Net position** is gross long minus gross short. **Net % OI** divides net position by total open interest to compare positioning through time.

                Financial futures use **Leveraged Money** as the headline speculative group. Commodity futures use **Managed Money**. Participant breakdown charts show other groups such as Asset Manager, Dealer, Producer/Merchant, Swap Dealer, Other Reportables, and Non-Reportables where available.
                """
            )
        gross_chart = positioning_chart(detail["cot"])
        net_chart = net_position_chart(detail["cot"])
        net_oi_chart = net_percent_open_interest_chart(detail["cot"])
        participant_chart = participant_breakdown_chart(detail["cot"])
        if gross_chart or net_chart or net_oi_chart or participant_chart:
            st.markdown("#### Positioning")
            st.markdown(
                "Shows what the key speculative participants currently hold. "
                "For financial futures this means **Leveraged Money** (hedge funds / aggressive money managers). "
                "For commodity futures this means **Managed Money** (CTAs, CPOs, and fund managers)."
            )
            left, right = st.columns(2)
            with left:
                st.caption(
                    "Gross speculative long vs gross speculative short "
                    "(hedge funds / aggressive money managers for financial futures; managed money for commodities). "
                    "This shows the long and short books before netting."
                )
                if gross_chart:
                    st.plotly_chart(gross_chart, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
                else:
                    st.info("Gross long/short data not available for this CFTC mapping.")
            with right:
                st.markdown(
                    "**Net speculative position** = **gross speculative long** - **gross speculative short**. "
                    "**Positive** means the selected speculative group is **net long**; "
                    "**negative** means it is **net short**."
                )
                if net_chart:
                    st.plotly_chart(net_chart, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
                else:
                    st.info("Net position data not available for this CFTC mapping.")
            st.markdown(
                "**Speculative Net Contract Percentile** compares the current **raw net contract** position with "
                "that same market's own historical raw net contract positions over **1Y, 3Y, 5Y, and 10Y** windows. "
                "It **does not adjust for open interest**. A high percentile means the raw net position is high or more positive versus history. "
                "If the raw net position is still **negative**, a high percentile means participants are still **net short**, "
                "but **less short than usual**."
            )
            position_table = positioning_percentile_table(details)
            if not position_table.empty:
                st.dataframe(style_positioning_table(position_table), use_container_width=True, hide_index=True)
            else:
                st.info("No net-contract percentile data available for the current universe yet.")

            st.markdown("#### Crowding")
            st.markdown(
                "Shows whether current speculative positioning is stretched relative to market size or its own history. "
                "This helps avoid treating a large raw contract count as crowded when open interest has also grown."
            )
            if net_oi_chart:
                st.markdown(
                    "**Current Net % OI** = (**speculative gross long** - **speculative gross short**) / **total open interest**. "
                    "The **sign** shows direction: **positive = net long**, **negative = net short**. "
                    "The **magnitude** shows how large the net position is relative to market size; for example, **-26%** means "
                    "speculative net shorts equal about **26% of total open interest**."
                )
                st.plotly_chart(net_oi_chart, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            else:
                st.info("Net position as % of open interest is not available for this CFTC mapping.")

            st.markdown(
                "**Speculative Net % OI Percentile** compares today's **Current Net % OI** with the same market's own "
                "historical **Net % OI** over **1Y, 3Y, 5Y, and 10Y** windows. It is **not a cross-asset comparison** "
                "and it is **not raw contracts**. A **high percentile** means today's value is high or more positive versus its own history. "
                "If **Current Net % OI is negative** and the percentile is high, participants are still **net short**, "
                "but **less short than they have usually been**."
            )
            oi_percentile_table = positioning_oi_percentile_table(details)
            if not oi_percentile_table.empty:
                st.dataframe(style_positioning_table(oi_percentile_table), use_container_width=True, hide_index=True)
            else:
                st.info("No net-%OI percentile data available for the current universe yet.")

            st.markdown("#### Squeeze / Unwind Risk")
            st.markdown(
                """
                This panel compares **fresh weekly flow** with the current **positioning level** across the whole expression universe. It is not just for bonds: it uses the mapped CFTC contract for each rates, FX, equity-index, commodity, or volatility expression.

                **Flow-driven** means recent long/short additions are material versus open interest. **Level-driven** means current net positioning is already stretched versus its own 3-year history. **Short squeeze risk** can support long expressions if shorts are forced to cover; **long unwind risk** can hurt long expressions if crowded longs liquidate.
                """
            )
            squeeze_table = squeeze_risk_table(details)
            if squeeze_table.empty:
                st.info("No squeeze-risk table available for the current CFTC mappings.")
            else:
                st.dataframe(style_squeeze_table(squeeze_table), use_container_width=True, hide_index=True)

            st.markdown("#### Participant Flow")
            st.markdown(
                """
                Shows **how each CFTC participant group is changing exposure**, not just where the headline speculative group sits today. The line chart uses **net position as % of open interest** so participant behavior is comparable through time. The weekly bar chart and table show whether the latest move came from **gross longs**, **gross shorts**, or both.
                """
            )
            participant_net_oi_chart = participant_net_percent_oi_chart(detail["cot"])
            participant_weekly_chart = participant_weekly_net_change_chart(detail["cot"])
            if participant_net_oi_chart:
                st.plotly_chart(participant_net_oi_chart, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            else:
                st.info("Participant net-%OI chart is not available for this CFTC mapping.")
            if participant_weekly_chart:
                st.plotly_chart(participant_weekly_chart, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            else:
                st.info("Participant weekly-flow chart is not available for this CFTC mapping.")
            participant_flow = participant_flow_change_table(detail["cot"])
            if participant_flow.empty:
                st.info("Participant gross long/short flow table is not available for this CFTC mapping.")
            else:
                st.dataframe(participant_flow, use_container_width=True, hide_index=True)

            st.markdown("#### Flow")
            st.markdown(
                "Shows whether speculative participants are adding or reducing exposure recently. "
                "Flow is different from crowding: a market can be crowded but no longer adding, or uncrowded but accumulating quickly."
            )
            flow_table = speculative_flow_table(detail["cot"])
            if flow_table.empty:
                st.info("No speculative flow table available for this CFTC mapping.")
            else:
                st.dataframe(flow_table, use_container_width=True, hide_index=True)

            st.markdown("#### Participant Breakdown")
            st.markdown(
                "Compares CFTC participant groups, not just the headline speculative group. "
                "Financial futures include Dealer, Asset Manager, Leveraged Money, Other Reportables, and Non-Reportables. "
                "Commodity futures include Producer/Merchant, Swap Dealer, Managed Money, Other Reportables, and Non-Reportables."
            )
            if participant_chart:
                st.caption(
                    "Latest participant gross long and gross short. Shorts are plotted below zero so the balance of each group is easier to read."
                )
                st.plotly_chart(participant_chart, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            else:
                st.info("Participant breakdown chart is not available for this CFTC mapping.")
            participant_table = participant_positioning_table(detail["cot"])
            if participant_table.empty:
                st.info("Participant breakdown table is not available for this CFTC mapping.")
            else:
                st.dataframe(participant_table, use_container_width=True, hide_index=True)
        else:
            st.info("No mapped CFTC chart available for this expression yet.")
        with st.expander("CFTC source audit", expanded=True):
            st.caption(
                "Positioning is pulled from CFTC Public Reporting Socrata datasets. "
                "Financial futures use Traders in Financial Futures leveraged-money columns. "
                "Commodity futures use Disaggregated money-manager columns. "
                "The dashboard does not sum commercial and non-commercial positions; it uses speculative long minus speculative short for scoring. "
                "For FX, CFTC futures positioning is in the named currency versus USD, such as Japanese Yen futures meaning long/short JPY versus USD. "
                "A negative Japanese Yen net position means speculative accounts are net short JPY futures, which is effectively long USD/JPY. "
                "For rates, each expression uses its mapped CFTC contract such as UST 2Y NOTE, UST 10Y NOTE, or UST BOND. "
                "This is not an aggregate Treasury curve signal unless multiple contracts are explicitly combined."
            )
            source_table = positioning_source_table(details)
            if source_table.empty:
                st.info("No CFTC source information available.")
            else:
                st.dataframe(source_table, use_container_width=True, hide_index=True)
        st.caption(results["positioning"].reason)
    with tabs[4]:
        with st.expander("? Event sources and FX rules", expanded=False):
            st.markdown(
                """
                Events come from ForexFactory calendar feeds (this week and next) plus scheduled US releases from the FRED release calendar and FOMC dates from the Federal Reserve. For FX expressions, the dashboard pulls events for both currencies in the pair. For example, `EURUSD=X` uses **EUR** and **USD** events.

                Event risk is based on medium/high-impact events inside the selected horizon. Forecast and previous fields depend on what the source feed provides.
                """
            )
        selected_events = detail.get("visible_events", pd.DataFrame())
        selected_event_countries = detail.get("event_countries", ["USD"])
        st.caption(f"Event countries for this expression: {', '.join(selected_event_countries)}")
        timeline = event_timeline(selected_events)
        if timeline:
            st.plotly_chart(timeline, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            st.dataframe(selected_events, use_container_width=True, hide_index=True)
        elif detail.get("events", pd.DataFrame()).empty:
            st.info("No events returned from the automatic calendar feed for this expression's countries.")
        else:
            st.info(f"No events found inside the next {int(horizon_days)} days.")
    with tabs[5]:
        st.caption(f"Regime source: {regime_source}")
        with st.expander("? Regime terms", expanded=False):
            st.markdown(
                """
                The macro regime model converts each available growth, inflation, and policy input into a 6-month momentum z-score versus its own rolling 5-year history. Growth and inflation signs determine the regime bucket: Goldilocks, Reflation, Stagflation, or Slowdown.

                The **policy z-score** is shown as context. It does not directly define the growth/inflation quadrant. Historical regime returns are calculated from the selected expression's own forward returns during matching historical regimes.
                """
            )
        if not regime_history.empty and {"growth_score", "inflation_score"}.issubset(regime_history.columns):
            latest_regime = regime_history.sort_values("month").iloc[-1]
            st.markdown("#### Current model regime")
            st.markdown(f"**{latest_regime['regime']}**")
            st.caption(f"Latest regime month: {pd.to_datetime(latest_regime['month']).date()}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Growth z", f"{latest_regime['growth_score']:.2f}")
            c2.metric("Inflation z", f"{latest_regime['inflation_score']:.2f}")
            c3.metric("Policy context z", "n/a" if pd.isna(latest_regime.get("policy_score")) else f"{latest_regime['policy_score']:.2f}")
            c4.metric("Growth inputs", int(latest_regime.get("growth_indicators", 0)))
            c5.metric("Inflation inputs", int(latest_regime.get("inflation_indicators", 0)))
            if latest_regime.get("growth_indicators", 0) < 3 or latest_regime.get("inflation_indicators", 0) < 3:
                st.warning(
                    "The latest regime month has limited inputs. This usually happens because monthly macro releases "
                    "arrive with a lag. The model can carry recent inputs forward for up to two months, but older stale data is excluded."
                )
        else:
            st.info("No FRED macro regime score available for the current run.")
        st.markdown("#### Growth / inflation / policy z-scores")
        st.caption(
            "Growth and inflation z-scores classify the regime. Policy context z is shown as supporting information only; "
            "it does not currently determine the regime label or directly change the regime score."
        )
        with st.expander("Regime inputs used", expanded=True):
            st.caption(
                "These are the exact FRED series the regime model tries to use. "
                "'Used Latest Month' means the series contributed to the latest monthly growth, inflation, or policy z-score. "
                "Recent monthly releases can be carried forward for up to two months while waiting for the next release."
            )
            inputs_view = regime_input_table(macro_regime_evidence)
            st.dataframe(inputs_view, use_container_width=True, hide_index=True)
        with st.expander("Macro regime evidence"):
            if macro_regime_evidence.empty:
                st.info("No FRED macro evidence available for this run.")
            else:
                latest_month = macro_regime_evidence["month"].max()
                evidence_view = macro_regime_evidence[macro_regime_evidence["month"].eq(latest_month)].copy()
                evidence_view["month"] = evidence_view["month"].dt.strftime("%Y-%m-%d")
                if "source_month" in evidence_view.columns:
                    evidence_view["source_month"] = pd.to_datetime(evidence_view["source_month"]).dt.strftime("%Y-%m-%d")
                evidence_view["latest"] = evidence_view["latest"].round(3)
                evidence_view["z_score"] = evidence_view["z_score"].round(2)
                evidence_view["contribution"] = evidence_view["contribution"].round(2)
                evidence_columns = ["month", "bucket", "label", "ticker", "latest", "z_score", "contribution"]
                if "source_month" in evidence_view.columns:
                    evidence_columns.insert(1, "source_month")
                if "stale_months" in evidence_view.columns:
                    evidence_columns.insert(2, "stale_months")
                st.dataframe(
                    evidence_view[evidence_columns].sort_values(["bucket", "label"]),
                    use_container_width=True,
                    hide_index=True,
                )
        with st.expander("Regime methodology"):
            st.markdown(
                """
                Macro inputs come from FRED:

                - Growth score: GDP, payrolls, unemployment, retail sales, jobless claims, and JOLTS where available (ISM PMI has no free source)
                - Inflation score: CPI, core CPI, core PCE, breakevens, and inflation forwards where available
                - Policy context: Fed Funds, SOFR, 2Y yield, and 10Y yield
                - Each indicator is converted into a 6-month momentum z-score versus its own rolling 5-year history
                - Each observation is shifted from its reference period to a typical release date (e.g. CPI ~43 days, GDP ~120 days), so a regime month only uses data that had been published
                - Recent monthly data can be carried forward for up to two months, so July data can inform August while August releases are still pending
                - Negative-direction indicators such as unemployment and jobless claims are inverted before aggregation
                - Monthly regimes are assigned from growth and inflation scores

                Policy context z is shown for interpretation only. It does not currently decide whether the regime is Goldilocks, Reflation, Slowdown, or Stagflation, and it does not directly change the regime score.

                If FRED is unavailable, the app falls back to the older market-proxy method using `SPY` and `DBC`.

                The selected expression is scored from its own forward returns in matching historical regimes, so the same regime model can be applied across ETFs, rates, FX, futures, commodities, and volatility proxies when price history is available.

                In the historical return table, volatility means the standard deviation of that expression's forward returns after months with the same regime. It is not comparing the asset against another ETF; it compares the dispersion of its own historical outcomes within each regime bucket. Higher volatility means the average return was less stable across past samples.
                """
            )
        with st.expander("Historical sample counts"):
            st.caption(
                "This shows how many historical months were classified into each regime. "
                "It is a sample-size check for backtest reliability, not the current regime signal."
            )
            st.plotly_chart(regime_matrix(regime, regime_history), use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        with st.expander("Historical regime return table"):
            st.caption(results["regime"].reason)
            if regime_history.empty:
                st.info("No regime history available.")
            else:
                regime_return_table = summarize_regime_returns(prices, regime_history, expression.direction, int(horizon_days))
                st.dataframe(
                    format_regime_return_table(regime_return_table),
                    use_container_width=True,
                    hide_index=True,
                )
    with tabs[6]:
        st.markdown("#### Data Source Audit")
        st.caption("Reference map for where each dashboard component comes from and what can go wrong.")
        st.dataframe(data_source_audit_table(), use_container_width=True, hide_index=True)

        st.markdown("#### Known Data Caveats")
        known_data_caveats()

        st.markdown("#### Symbol and Asset-Class Conventions")
        st.markdown(
            """
            - **Symbols** are Yahoo Finance tickers: `=F` futures, `=X` FX, `^` indices, plain ETFs/stocks. Old Bloomberg-style tickers in saved lists are translated automatically.
            - **FX symbols** are interpreted in base/quote form. `EURUSD=X` means EUR versus USD; `JPYUSD=X` means JPY versus USD.
            - **Price proxy and positioning proxy can differ.** For example, gold uses the `GC=F` future for price analytics and CFTC `GOLD` for futures positioning.
            - **ETF expressions can use futures positioning proxies.** For example, Nasdaq exposure may use `NDX Index` for price data and `NASDAQ MINI` for CFTC positioning.
            """
        )

        st.markdown("#### Core Term Glossary")
        glossary = pd.DataFrame(
            [
                {"Term": "Hit rate", "Meaning": "Share of historical observations where direction-adjusted return was positive."},
                {"Term": "1D RV", "Meaning": "Annualized absolute daily log return; useful as a shock indicator."},
                {"Term": "1M / 3M RV", "Meaning": "Annualized standard deviation of daily returns over roughly 21 / 63 trading days."},
                {"Term": "Percentile", "Meaning": "Empirical rank of the current value versus historical observations. It does not assume a normal distribution."},
                {"Term": "Z-score", "Meaning": "Distance from historical average measured in standard-deviation units: (current - average) / standard deviation."},
                {"Term": "Open interest", "Meaning": "Total outstanding futures contracts in the market."},
                {"Term": "Net position", "Meaning": "Gross long contracts minus gross short contracts."},
                {"Term": "Net % OI", "Meaning": "Net position divided by open interest; useful for comparing crowding through time."},
                {"Term": "Flow-driven", "Meaning": "Recent weekly long/short changes are material relative to open interest."},
                {"Term": "Level-driven", "Meaning": "Current net positioning is stretched versus its own history."},
            ]
        )
        st.dataframe(glossary, use_container_width=True, hide_index=True)

        st.markdown("#### Component Methodology")
        with st.expander("Seasonality"):
            st.markdown(
                """
                Seasonality uses 21-trading-day returns grouped by calendar month across 10Y, 20Y, and 30Y windows where enough data exists. Returns are direction-adjusted: short expressions are multiplied by -1 so positive values represent profitable short-expression outcomes. Extreme bad price jumps are filtered before the monthly distributions are built.
                """
            )
        with st.expander("Technicals and RV"):
            st.markdown(
                """
                Technicals are derived from close prices: moving averages, EMA levels, RSI, 1M/3M returns, and realized volatility. RV percentiles compare current RV against the same expression's own historical RV observations over the stated lookback. Hotter RV colors mean higher volatility percentile, not automatically a better or worse trade.
                """
            )
        with st.expander("CFTC positioning"):
            st.markdown(
                """
                CFTC data comes from public COT datasets. Financial futures use Traders in Financial Futures participant buckets such as Leveraged Money and Asset Manager. Commodities use Disaggregated buckets such as Managed Money, Producer/Merchant, and Swap Dealer. The dashboard does not add commercial and non-commercial positions together for the headline score; it uses the mapped speculative participant long/short fields.
                """
            )
        with st.expander("Events"):
            st.markdown(
                """
                Event risk uses medium/high-impact events inside the selected horizon. Non-FX expressions default to USD events. FX expressions use both currencies in the pair, so `EURUSD=X` includes EUR and USD events. Beyond ForexFactory's two-week window, USD events come from the FRED release calendar and the FOMC calendar.
                """
            )
        with st.expander("Regime"):
            st.markdown(
                """
                The regime model uses growth and inflation z-scores to classify months into Goldilocks, Reflation, Stagflation, or Slowdown. Inputs are pulled from FRED, lagged to typical release dates, and converted into 6-month momentum z-scores versus each series' own rolling 5-year history. Policy z-score is shown for context but does not currently define the regime quadrant.
                """
            )





