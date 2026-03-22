# src/workflows/daily_pack.py
from __future__ import annotations

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px

from src.data.econ_calendar import fetch_te_calendar
from src.data.market_data import build_rates_fx_frame  # <-- real market data


def _fmt_bp(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.1f} bp"


def _fmt_px(x: float | None, dp: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{dp}f}"

def last_two_valid(s: pd.Series):
    s = s.dropna()
    if len(s) < 2:
        return None, None
    return s.iloc[-1], s.iloc[-2]


def build_daily_pack(
    asof: date,
    lookback_days: int,
    pack_type: str,
    # Market data (FRED)
    fred_api_key: str | None = None,
    # Economic calendar (Trading Economics)
    te_api_key: str | None = None,
    te_country: str = "united states",
    te_importance: int | None = 3,
) -> dict:
    """
    Returns:
      - figures: list[plotly Figure]
      - summary_md: str
      - data: DataFrame (rates/fx)
      - events: DataFrame or None
    """

    # ----------------------------
    # 1) Pull real Rates/FX data
    # ----------------------------
    if not fred_api_key or not fred_api_key.strip():
        summary_md = f"""## Daily Pack — {asof} ({pack_type})

**Rates & FX**
- Missing `FRED_API_KEY` → cannot load UST yields (DGS2, DGS10).

**Next steps**
- Add `FRED_API_KEY` to `.streamlit/secrets.toml` and rerun.
"""
        return {"figures": [], "summary_md": summary_md, "data": pd.DataFrame(), "events": None}

    # lookback_days here is treated as a "target rows" window.
    # Fetch a wider calendar window so we definitely get enough business days.
    fetch_start = asof - timedelta(days=max(int(lookback_days * 2), 60))
    fetch_end = asof

    df = build_rates_fx_frame(
        fred_api_key=fred_api_key.strip(),
        start=fetch_start,
        end=fetch_end,
    )

    # Ensure datetime index and take last N rows
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df.tail(lookback_days)

    # If insufficient data, handle gracefully
    if df.shape[0] < 2:
        summary_md = f"""## Daily Pack — {asof} ({pack_type})

**Rates & FX**
- Not enough data returned to compute daily changes.

"""
        figs = []
        return {"figures": figs, "summary_md": summary_md, "data": df, "events": None}

    # ----------------------------
    # 2) Build charts (real data)
    # ----------------------------
    df_plot = df.copy()
    df_plot.index.name = "date"
    df_plot = df_plot.reset_index()

    figs = [
        px.line(df_plot, x="date", y=["UST_2Y", "UST_10Y"], title="UST 2Y & 10Y (FRED, daily)"),
        px.line(df_plot, x="date", y="2s10s", title="2s10s (computed)"),
        px.line(df_plot, x="date", y="EURUSD", title="EURUSD (Frankfurter/ECB, daily)"),
    ]

    # ----------------------------
    # 3) Summary block (real moves)
    # ----------------------------
    y2_last, y2_prev = last_two_valid(df["UST_2Y"])
    y10_last, y10_prev = last_two_valid(df["UST_10Y"])
    s_last, s_prev = last_two_valid(df["2s10s"])
    fx_last, fx_prev = last_two_valid(df["EURUSD"])

    d2y_bp = (y2_last - y2_prev) * 100 if y2_last is not None else None
    d10y_bp = (y10_last - y10_prev) * 100 if y10_last is not None else None
    d2s10s_bp = (s_last - s_prev) * 100 if s_last is not None else None
    deurusd = (fx_last - fx_prev) if fx_last is not None else None

    summary_md = f"""## Daily Pack — {asof} ({pack_type})

**Moves (vs prior close / prior observation)**
- 2Y: {_fmt_bp(d2y_bp)}
- 10Y: {_fmt_bp(d10y_bp)}
- 2s10s: {_fmt_bp(d2s10s_bp)}
- EURUSD: {_fmt_px(deurusd, dp=4)}

**Read**
- Curve move looks: _(level/slope driven?)_
- USD/FX driver: _(rates-driven / risk-driven / mixed)_
- Watch next: _(events / levels)_
- Invalidate if: _(what would change your view)_
"""

    # ----------------------------
    # 4) Calendar section (Trading Economics)
    # ----------------------------
    events = None
    if te_api_key and te_api_key.strip():
        try:
            events = fetch_te_calendar(
                api_key=te_api_key.strip(),
                start=asof,
                end=asof + timedelta(days=2),
                tz="Asia/Singapore",
                country=te_country,
                importance=te_importance,
            )

            top = events.head(8) if events is not None else None
            summary_md += "\n**Next 48h key events**\n"
            if top is not None and len(top) > 0:
                for _, r in top.iterrows():
                    summary_md += (
                        f"- {r['time']} — {r['report']} ({r['period']}) | "
                        f"F:{r.get('forecast')} P:{r.get('previous')}\n"
                    )
            else:
                summary_md += "- (No events in this window)\n"
        except Exception:
            summary_md += "\n**Next 48h key events**\n- (Calendar unavailable — check TE access / rate limits)\n"
    else:
        summary_md += "\n**Next 48h key events**\n- (No TE API key configured)\n"

    return {"figures": figs, "summary_md": summary_md, "data": df, "events": events}