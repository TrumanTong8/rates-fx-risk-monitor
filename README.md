# Rates–FX Risk Monitor (Streamlit)

A lightweight Rates/FX monitoring dashboard built in **Streamlit** for Sales & Trading workflows.

It includes:
- **Economic Calendar** (Trading Economics API)
- **Daily Pack generator**: auto-builds 3–5 charts + a copy/paste discussion summary + key upcoming events
- **Rates & FX charts** using **free data sources**
  - UST yields (2Y, 10Y) from **FRED** (daily)
  - EURUSD from **Frankfurter / ECB reference rates** (daily)

> Note: Most “true real-time” market feeds are paid. This project is intentionally built around free, reliable sources suitable for an MVP.

---

## Features

### 1) Economic Calendar
- Pulls upcoming (and/or historical) macro releases with fields:
  - `time, report, period, actual, forecast, previous`
- Data source: **Trading Economics API**
- Supports country + importance filtering

### 2) Daily Pack (one-click workflow)
Generates a meeting-ready pack:
- 3 core charts:
  - UST 2Y & 10Y
  - 2s10s curve spread
  - EURUSD
- Short discussion template (copy/paste)
- “Next 48h key events” pulled from Trading Economics
- Downloadable summary as `.md`

### 3) Rates & FX (Free Sources)
- **FRED**: DGS2, DGS10 (daily yields)
- **Frankfurter**: EURUSD (ECB ref rates, daily)

---

## Setup (Local)

### 1) Create venv (Python 3.12 recommended)
``` bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```


### 2) Add API keys (Streamlit secrets)
Create this file at repo root:

.streamlit/secrets.toml
````
FRED_API_KEY = "YOUR_FRED_KEY"
TE_API_KEY = "guest:guest"
````
Notes:
* FRED_API_KEY: free key from FRED.
* TE_API_KEY: guest:guest works as a limited/demo key; use a real key for fuller coverage.

### 3) Run the app
```` 
streeamlit run app.py
````

