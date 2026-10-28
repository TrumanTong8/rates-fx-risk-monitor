import streamlit as st

st.set_page_config(page_title="Trading Dashboard", layout="wide")

pages = {
    "Rates & FX": [
        st.Page("views/rates_fx/overview.py", title="Overview", default=True),
        st.Page("views/rates_fx/curve_factors.py", title="Curve Factors"),
        st.Page("views/rates_fx/fx_drivers.py", title="FX Drivers"),
        st.Page("views/rates_fx/carry_roll.py", title="Carry & Roll"),
    ],
    "Trade Ideas": [
        st.Page("views/trade_ideas/expression_ranking.py", title="Expression Ranking"),
    ],
    "Commodities": [
        st.Page("views/commodities/energy_inventory.py", title="Energy Inventory & Backtest"),
        st.Page("views/commodities/energy_seasonals.py", title="Energy Seasonals"),
        st.Page("views/commodities/spr.py", title="Strategic Petroleum Reserve"),
    ],
}

st.navigation(pages).run()
