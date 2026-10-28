# Trading Dashboard (Streamlit)

One Streamlit app for Sales & Trading workflows: rates and FX monitoring, ranking trade expressions of a macro view, and energy inventory analysis with a signal backtest.

Everything runs on **free data sources**: Yahoo Finance, FRED, EIA, the CFTC, ForexFactory and the Federal Reserve.

> Note: Most "true real-time" market feeds are paid. Prices here are delayed daily bars, and macro data is FRED's latest revised series.

---

## Using the website

Access the [link here](https://rates-fx-risk-monitor-truman.streamlit.app/) to try it out!

---

## Pages

### Rates & FX
- **Overview**
  - **Economic Calendar**: filter by currency and importance. ForexFactory covers this week and next for all currencies (with forecast/previous). Beyond that, scheduled US releases come from the FRED release calendar and FOMC dates from the Fed.
  - **Daily Pack**: one click builds UST 2Y/10Y, 2s10s and EURUSD charts, a copy/paste discussion summary with the latest moves, and the next 48h of key events. Downloadable as `.md`.
- **Curve Factors**, **FX Drivers**, **Carry & Roll**: coming soon.

### Trade Ideas
- **Expression Ranking**: compare ways to express one macro view (e.g. "Rates will not hike"). Each expression is scored on:
  - seasonality
  - technicals
  - CFTC positioning
  - event risk
  - fit with the growth/inflation regime, built from FRED macro data

  The universe is editable and uses Yahoo symbols (`GC=F`, `EURUSD=X`, `^NDX`, `SPY`). Save it locally, or download it and load it back on the hosted app.

### Commodities (energy)
- **Energy Inventory & Backtest**: for WTI crude, natural gas, gasoline, distillate and propane, shows:
  - EIA weekly inventory, with its 5Y/10Y seasonal range
  - CFTC open interest
  - the front-month futures price
  - an inventory-shock backtest: z-score draws/builds, with entry on the first close after the EIA release
- **Energy Seasonals**: current year against 5Y/10Y ranges for seven EIA inventory series.
- **Strategic Petroleum Reserve**: SPR history with a date-range picker.

---

## Data sources

| Data | Source | Notes |
|---|---|---|
| Prices (futures, FX, indices, ETFs) | Yahoo Finance via `yfinance` | Futures are front-month continuous series; ETFs are dividend-adjusted |
| UST yields, macro indicators, release dates | FRED | API key needed for the release calendar and the Daily Pack |
| EURUSD (Daily Pack) | Frankfurter / ECB reference rates | |
| Energy inventories | EIA Open Data v2 | Free API key |
| Positioning, open interest | CFTC Public Reporting (Socrata) | Queried by stable contract code |
| Economic calendar | ForexFactory, FRED, Federal Reserve | ForexFactory rate-limits; the app falls back to its last good download |

---

## Setup (Local)

### 1) Create venv (Python 3.12 recommended)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Add API keys (Streamlit secrets)
Create this file at repo root:

`.streamlit/secrets.toml`
```toml
FRED_API_KEY = "YOUR_FRED_KEY"
EIA_API_KEY = "YOUR_EIA_KEY"
```
Notes:
* `FRED_API_KEY`: free key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).
* `EIA_API_KEY`: free key from [EIA](https://www.eia.gov/opendata/register.php).
* On Streamlit Community Cloud, add the same keys under the app's **Settings → Secrets**.

### 3) Run the app
```bash
streamlit run app.py
```

### 4) Check the data sources (optional)
Hits every source and ticker the app relies on and prints coverage:
```bash
python scripts/check_data_sources.py
```

---

## Project layout

```
app.py                  # navigation: Rates & FX, Trade Ideas, Commodities
views/                  # one file per page
src/data/               # free data layer: yahoo, fred, eia, cftc, calendar, cache
src/expression/         # expression ranking: scoring tests, charts, ranker
src/commodities/        # energy config, analytics, backtest, charts
src/workflows/          # Daily Pack
config/expression/      # default universe, examples, scoring weights, saved list
scripts/                # data-source smoke test
```

---

## Known limitations
- **Gold and silver** use front-month futures (`GC=F`, `SI=F`), because Yahoo has no spot history.
- **ISM PMI** is not included in the regime model, because it has no free source.
- **Propane** has no free futures price, so it has no price chart or backtest.
- **FRED stores revised values**, not first prints. Regime inputs are shifted to typical release dates to avoid look-ahead, but the history is still slightly cleaner than real time.
- **Front-month futures include roll jumps**. These are notably large for RBOB at its seasonal specification changes.
