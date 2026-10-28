# src/data/symbols.py
"""
Translation from the old Bloomberg-style database tickers to free sources.
Used to migrate saved configs; new configs should use Yahoo symbols directly.
"""
from __future__ import annotations

# Old database ticker -> Yahoo Finance symbol
LEGACY_TO_YAHOO = {
    "SPY US Equity": "SPY",
    "TLT US Equity": "TLT",
    "SHY US Equity": "SHY",
    "DBC US Equity": "DBC",
    "GLD US Equity": "GLD",
    "NDX Index": "^NDX",
    "SPX Index": "^GSPC",
    "RTY1 Index": "RTY=F",
    "VIX Index": "^VIX",
    "VVIX Index": "^VVIX",
    "TU1 Comdty": "ZT=F",
    "FV1 Comdty": "ZF=F",
    "TY1 Comdty": "ZN=F",
    "US1 Comdty": "ZB=F",
    "XAU Curncy": "GC=F",   # Yahoo has no spot gold history; front COMEX future instead
    "XAG Curncy": "SI=F",   # likewise for silver
    "GC1 Comdty": "GC=F",
    "SI1 Comdty": "SI=F",
    "HG1 Comdty": "HG=F",
    "CL1 Comdty": "CL=F",
    "CO1 Comdty": "BZ=F",
    "NG1 Comdty": "NG=F",
    "XB1 Comdty": "RB=F",
    "HO1 Comdty": "HO=F",
    "EURUSD Curncy": "EURUSD=X",
    "AUDUSD Curncy": "AUDUSD=X",
    "JPYUSD Curncy": "JPYUSD=X",
    "DXY Curncy": "DX-Y.NYB",
}


def to_yahoo(symbol: str) -> str:
    """Map an old database ticker to Yahoo; pass anything else through unchanged."""
    return LEGACY_TO_YAHOO.get(symbol.strip(), symbol.strip())
