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

# Old database macro ticker -> (FRED id or ids to splice, transform in src.data.fred.TRANSFORMS)
LEGACY_TO_FRED = {
    "GDP CQOQ Index": ("A191RL1Q225SBEA", "level"),          # real GDP, % QoQ annualised
    "NFP TCH Index": ("PAYEMS", "diff"),                     # payrolls monthly change, thousands
    "USURTOT Index": ("UNRATE", "level"),
    "RSTAMOM Index": ("RSAFS", "mom_pct"),
    "INJCJC Index": ("ICSA", "thousands"),
    "JOLTTOTL Index": ("JTSJOL", "level"),
    "CPI YOY Index": ("CPIAUCSL", "yoy_pct"),
    "CPI XYOY Index": ("CPILFESL", "yoy_pct"),
    "PCE CYOY Index": ("PCEPILFE", "yoy_pct"),
    "PCE CMOM Index": ("PCEPILFE", "mom_pct"),
    "USGGBE10 Index": ("T10YIE", "level"),
    "USGG5Y5Y Index": ("T5YIFR", "level"),
    "FDTR Index": (["DFEDTAR", "DFEDTARU"], "level"),        # target, then upper bound from Dec 2008
    "SOFRRATE Index": ("SOFR", "level"),
    "USGG2YR Index": ("DGS2", "level"),
    "USGG5YR Index": ("DGS5", "level"),
    "USGG10YR Index": ("DGS10", "level"),
}


def to_yahoo(symbol: str) -> str:
    """Map an old database ticker to Yahoo; pass anything else through unchanged."""
    return LEGACY_TO_YAHOO.get(symbol.strip(), symbol.strip())
