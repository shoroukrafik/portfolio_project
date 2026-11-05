"""
Ticker lists per universe — used for labeling stocks in concentration charts.
Sourced from dashboard_legacy/data/meta.json (DJI, SP100) and data/stock_list_*.txt (NDX).
The order must match the index of weights in the saved curve records.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY_META = PROJECT_ROOT / "dashboard_legacy" / "data" / "meta.json"


def _read_stock_list(filename: str) -> list:
    p = PROJECT_ROOT / 'data' / filename
    if not p.exists():
        return []
    try:
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    except OSError:
        return []


def get_tickers(universe: str) -> list:
    """Returns ordered ticker list matching the weights array index."""
    if universe == 'DJI':
        if LEGACY_META.exists():
            try:
                m = json.load(open(LEGACY_META))
                t = m.get('tickers_dji', [])
                if t: return t
            except (json.JSONDecodeError, OSError):
                pass
        return _read_stock_list('stock_list_ext.txt')

    if universe == 'SP100':
        if LEGACY_META.exists():
            try:
                m = json.load(open(LEGACY_META))
                t = m.get('tickers_sp100', [])
                if t: return t
            except (json.JSONDecodeError, OSError):
                pass
        return _read_stock_list('stock_list_sp100.txt')

    if universe == 'NDX':
        return _read_stock_list('stock_list_ndx.txt')

    if universe == 'NDX_2017':
        return _read_stock_list('stock_list_ndx_2017.txt')

    return []
