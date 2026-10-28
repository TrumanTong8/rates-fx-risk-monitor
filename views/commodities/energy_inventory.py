# views/commodities/energy_inventory.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.commodities.analytics import compute_dashboard_frame
from src.commodities.backtest import (
    LOOKBACK_YRS,
    Z_COLS,
    Z_LABELS,
    Z_RESET_THRESHOLD,
    Z_THRESHOLD,
    best_setup_rows,
    build_backtest_results,
)
from src.commodities.charts import (
    plot_horizon_bars,
    plot_inventory_vs_oi,
    plot_line,
    plot_return_distribution,
    plot_seasonal_inventory,
    plot_signal_timeline,
)
from src.commodities.config import ENERGY, EnergyConfig
from src.data import cache

st.title("Energy Inventory & Backtest")
st.caption("EIA weekly inventories against CFTC open interest and front-month futures, with an inventory-shock signal backtest.")


def load_energy_data(cfg: EnergyConfig) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    inv = cache.eia_series(cfg.eia_id)
    oi = cache.open_interest(cfg.cftc_market)
    price = cache.close_series(cfg.price_symbol, None) if cfg.price_symbol else None
    return inv, oi, price


def kpi_values(df: pd.DataFrame) -> dict[str, float | str]:
    base = df.dropna(subset=["inventory", "open_interest"])
    if base.empty:
        return {}
    last = base.iloc[-1]
    prev = base.iloc[-2] if len(base) >= 2 else None
    return {
        "As of": base.index[-1].date().isoformat(),
        "Inventory": float(last["inventory"]),
        "Inventory delta": float(last["inventory"] - prev["inventory"]) if prev is not None else np.nan,
        "Open interest": float(last["open_interest"]),
        "Inventory / OI": float(last["inv_to_oi"]),
        "Price": float(last["price"]) if "price" in base.columns and pd.notna(last.get("price", np.nan)) else np.nan,
    }


def render_data_sources_expander(cfg: EnergyConfig) -> None:
    with st.expander("Data sources & definitions", expanded=False):
        st.markdown("#### Data sources")
        st.markdown(
            f"""
            | Field | Source | Series / market | What it represents |
            |---|---|---|---|
            | Inventory | EIA | `{cfg.eia_id}` | {cfg.inventory_desc} |
            | Open interest | CFTC Legacy Futures-Only | `{cfg.cftc_market}` | {cfg.open_interest_desc} |
            | Price | Yahoo Finance | `{cfg.price_symbol or "Not available"}` | Front-month continuous future used for charts and the backtest. |
            """
        )
        st.markdown("#### Term explanations")
        st.markdown(
            """
            - **Inventory**: Physical stock/storage level for the selected commodity.
            - **Open interest (OI)**: Number of outstanding futures contracts. It is not trading volume.
            - **Inventory delta**: Latest inventory minus the previous inventory observation. Positive means a build; negative means a draw.
            - **Inventory / OI**: Inventory divided by open interest. This is a rough physical-stock-to-paper-positioning ratio, not a valuation measure.
            - **WoW inventory change %**: Week-over-week percentage change in inventory.
            - **Rolling 28d inventory change %**: Approximate four-week percentage change in inventory.
            """
        )
        st.info(cfg.caveat)


def render_setup_card(column, title: str, row: pd.Series | None) -> None:
    if row is None:
        column.metric(title, "No robust setup")
        column.caption("No row passed N, hit-rate, and positive-return filters.")
        return

    column.metric(title, row["Horizon"], delta=f"{row['Avg %']:+.2f}% avg")
    column.caption(
        f"{row['Signal']} | {row['Lookback']} | "
        f"Hit rate {row['Hit Rate %']:.1f}% | N={int(row['N'])}"
    )


def render_backtest_section(df: pd.DataFrame, daily_price: pd.Series, cfg: EnergyConfig) -> None:
    st.markdown("### Inventory Shock Signal Backtest")
    st.caption("Tests whether extreme inventory draws/builds historically preceded forward price moves.")

    logic_cols = st.columns(3)
    logic_cols[0].metric("Signal threshold", f"±{Z_THRESHOLD:.0f}σ")
    logic_cols[0].caption(f"Signals trigger on the first threshold breach, then lock until z-score normalizes inside ±{Z_RESET_THRESHOLD:.1f}.")
    logic_cols[1].metric("Long signal", "Draw shock")
    logic_cols[1].caption("z < -2: unusually large inventory draw.")
    logic_cols[2].metric("Short signal", "Build shock")
    logic_cols[2].caption("z > +2: unusually large inventory build.")
    st.caption(
        f"Trades enter at the first daily close on or after the EIA release ({cfg.release_lag_days} days after the "
        "week-ending date), and horizons are trading days on the daily front-month price."
    )

    results = build_backtest_results(df, daily_price, cfg.release_lag_days)
    if not results:
        st.info("No backtest results are available for this selection.")
        return

    best_long, best_short = best_setup_rows(results)
    best_cols = st.columns(2)
    render_setup_card(best_cols[0], "Best Long Setup", best_long)
    render_setup_card(best_cols[1], "Best Short Setup", best_short)

    st.markdown("#### Horizon Performance")
    control_cols = st.columns([1, 1, 2])
    z_col = control_cols[0].selectbox(
        "Signal type",
        Z_COLS,
        format_func=lambda value: Z_LABELS[value],
        key=f"{cfg.name}_bt_signal",
    )
    lb = control_cols[1].selectbox(
        "Lookback",
        LOOKBACK_YRS,
        format_func=lambda value: f"{value}Y",
        key=f"{cfg.name}_bt_lookback",
    )
    result = results.get((z_col, lb))
    if result is None:
        st.warning("No results for this signal/lookback combination.")
        return

    summary = result["summary"]
    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(
        plot_horizon_bars(summary, "Hit Rate %", "Hit Rate by Horizon", "Hit Rate %"),
        width="stretch",
    )
    chart_cols[1].plotly_chart(
        plot_horizon_bars(summary, "Avg %", "Average Return by Horizon", "Average forward return %"),
        width="stretch",
    )

    st.markdown("#### Signal Timeline")
    st.plotly_chart(
        plot_signal_timeline(result, f"{cfg.name}: {Z_LABELS[z_col]} ({lb}Y lookback)"),
        width="stretch",
    )

    st.markdown("#### Return Distribution")
    st.plotly_chart(
        plot_return_distribution(result, f"{cfg.name}: forward return distribution"),
        width="stretch",
    )

    with st.expander("Detailed backtest tables", expanded=False):
        for table_z_col in Z_COLS:
            st.markdown(f"##### {Z_LABELS[table_z_col]}")
            for table_lb in LOOKBACK_YRS:
                table_result = results.get((table_z_col, table_lb))
                if table_result is None:
                    st.caption(f"{table_lb}Y lookback: no results.")
                    continue
                st.markdown(f"**{table_lb}Y lookback**")
                st.caption(
                    f"Long signals: {len(table_result['long_dates'])} | "
                    f"Short signals: {len(table_result['short_dates'])}"
                )
                st.dataframe(table_result["summary"].round(2), width="stretch")


commodity_options = list(ENERGY)
if st.session_state.get("inventory_commodity") not in commodity_options:
    st.session_state["inventory_commodity"] = commodity_options[0]
st.session_state.setdefault("inventory_start", pd.Timestamp("2000-01-01").date())
st.session_state.setdefault("inventory_end", pd.Timestamp.today().date())

with st.sidebar:
    st.markdown("### Inventory Dashboard")
    name = st.selectbox("Commodity", commodity_options, key="inventory_commodity")
    start = st.date_input("Start", key="inventory_start")
    end = st.date_input("End", key="inventory_end")

cfg = ENERGY[name]
try:
    with st.spinner("Loading commodity data..."):
        inv, oi, price = load_energy_data(cfg)
except Exception as exc:
    st.error(f"{type(exc).__name__}: {exc}")
    st.stop()

df = compute_dashboard_frame(inv, oi, price=price)
df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].copy()
if df.empty:
    st.warning("No data in the selected date range.")
    st.stop()

kpis = kpi_values(df)
cols = st.columns(6)
for col, label in zip(cols, ["As of", "Inventory", "Inventory delta", "Open interest", "Inventory / OI", "Price"]):
    value = kpis.get(label, np.nan)
    if isinstance(value, str):
        text = value
    elif pd.isna(value):
        text = "—"
    else:
        text = f"{value:.4g}" if label == "Inventory / OI" else f"{value:,.2f}"
    col.metric(label, text)

render_data_sources_expander(cfg)

st.plotly_chart(
    plot_seasonal_inventory(df["inventory"], f"{cfg.name}: seasonal inventory range", cfg.inventory_units),
    width="stretch",
)
st.plotly_chart(plot_inventory_vs_oi(df, cfg.name, cfg.inventory_units), width="stretch")
st.plotly_chart(plot_line(df, ["inv_wow", "inv_roll28d"], "Inventory changes", "% change"), width="stretch")
st.plotly_chart(plot_line(df, ["inv_to_oi"], "Inventory / open interest", "ratio"), width="stretch")
st.dataframe(df.tail(20).round(4), width="stretch")

st.divider()
if price is None:
    st.info(cfg.caveat)
else:
    daily_price = price[(price.index >= pd.Timestamp(start)) & (price.index <= pd.Timestamp(end))]
    render_backtest_section(df, daily_price, cfg)
