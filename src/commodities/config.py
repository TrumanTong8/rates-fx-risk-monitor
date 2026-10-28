from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnergyConfig:
    name: str
    eia_id: str                          # EIA v1 series id for weekly inventory
    inventory_units: str
    cftc_market: str                     # key of src.data.cftc.MARKET_CODES (open interest)
    price_symbol: Optional[str]          # Yahoo front-month future; None when no free price exists
    release_lag_days: int                # days from EIA week-ending date to publication
    inventory_desc: str
    open_interest_desc: str
    caveat: str


# EIA weekly petroleum stocks (week ending Friday) are published the following Wednesday;
# natural gas storage the following Thursday. Backtest entries wait for the release.
ENERGY: dict[str, EnergyConfig] = {
    cfg.name: cfg
    for cfg in [
        EnergyConfig(
            "WTI Crude (US stocks)", "PET.WCESTUS1.W", "Thousand barrels", "WTI CRUDE", "CL=F", 5,
            "EIA weekly US commercial crude oil stocks, excluding SPR.",
            "CFTC Legacy Futures-Only open interest for physical WTI crude (NYMEX, code 067651).",
            "Inventory is a US physical stock measure, while open interest is futures market participation.",
        ),
        EnergyConfig(
            "US Nat Gas (storage)", "NG.NW2_EPG0_SWO_R48_BCF.W", "Billion cubic feet (BCF)", "NATURAL GAS", "NG=F", 6,
            "EIA weekly Lower 48 working natural gas storage.",
            "CFTC Legacy Futures-Only open interest for Henry Hub natural gas (NYMEX, code 023651).",
            "Storage is physical gas inventory; open interest is futures contracts outstanding.",
        ),
        EnergyConfig(
            "US Gasoline (Finished stocks)", "PET.WGFSTUS1.W", "Thousand barrels", "RBOB GASOLINE", "RB=F", 5,
            "EIA weekly US finished motor gasoline stocks.",
            "CFTC Legacy Futures-Only open interest for RBOB gasoline (NYMEX, code 111659).",
            "RBOB front-month prices jump at the spring/autumn specification rolls, which can distort forward returns.",
        ),
        EnergyConfig(
            "US Distillate (stocks)", "PET.WDISTUS1.W", "Thousand barrels", "ULSD", "HO=F", 5,
            "EIA weekly US distillate fuel oil stocks.",
            "CFTC Legacy Futures-Only open interest for NY Harbor ULSD (NYMEX, code 022651).",
            "Distillate stocks include products such as diesel/heating oil; open interest is futures participation.",
        ),
        EnergyConfig(
            "US Propane/Propylene (stocks)", "PET.WPRSTUS1.W", "Thousand barrels", "PROPANE", None, 5,
            "EIA weekly US propane/propylene stocks.",
            "CFTC Legacy Futures-Only open interest for Mont Belvieu propane (NYMEX, code 06665O).",
            "There is no free propane futures price, so this commodity has no price chart or backtest.",
        ),
    ]
}

ENERGY_SEASONAL_SERIES = {
    "Crude oil inventories": ("PET.WCESTUS1.W", "Thousand barrels"),
    "Cushing crude inventories": ("PET.W_EPC0_SAX_YCUOK_MBBL.W", "Thousand barrels"),
    "Natural gas storage": ("NG.NW2_EPG0_SWO_R48_BCF.W", "Billion cubic feet"),
    "Motor gasoline inventories": ("PET.WGFSTUS1.W", "Thousand barrels"),
    "Distillate fuel inventories": ("PET.WDISTUS1.W", "Thousand barrels"),
    "Propane/propylene inventories": ("PET.WPRSTUS1.W", "Thousand barrels"),
    "Strategic Petroleum Reserve": ("PET.WCSSTUS1.W", "Thousand barrels"),
}

SPR_SERIES_ID = "PET.WCSSTUS1.W"
