"""
Stock concentration metrics from per-day portfolio weights.

Sources weights from dashboard_legacy/data/curves/ where available
(DJI_v6 and SP100_v6 production variants have day-by-day weights saved).

Computes:
  - HHI (Herfindahl-Hirschman Index) — sum of squared weights
  - Effective number of stocks = 1 / HHI
  - Top-K weight concentration (K = 1, 3, 5)
  - Per-stock average weight over period
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import plotly.graph_objects as go

from .theme import (
    PRIMARY, SUCCESS, WARNING, DANGER, NEUTRAL,
    TEXT, TEXT_MUTED, TEXT_DIM, BORDER, SURFACE, PRIMARY_DIM,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY_DIR = PROJECT_ROOT / "dashboard_legacy" / "data" / "curves"


# Map variant folder → legacy curve slug (where weights are available)
VARIANT_TO_LEGACY_SLUG = {
    'universal_v6_dji':            'DJI_v6',
    'universal_v6_sp100':          'SP100_v6',
    'universal_v6_ndx':            'NDX_v6',
    'universal_v6_ndx_2017':       'NDX_2017',
    'universal_v6_ndx_2017_mom':   'NDX_2017_mom',
    'universal_v6_sp100_2015_mom': 'SP100_2015_mom',
    # MAC variants — only some have legacy curves
    'mac_dji':            'MARS_v2_MAC_v1cplusws_lineup_DJI_30',
    'mac_dji_v6':         'MARS_v2_MAC_v6_lineup_DJI_30',
    'mac_sp100':          'MARS_v2_MAC_v1cplusws_lineup_SP100',
    'mac_sp100_v6':       'MARS_v2_MAC_v6_lineup_SP100',
}


def load_weights_from_legacy(folder: str, period: str) -> Optional[List[Dict]]:
    """
    Returns list of dicts: [{date, weights, cash_weight, regime_name}, ...]
    Or None if not available.
    """
    slug = VARIANT_TO_LEGACY_SLUG.get(folder)
    if not slug:
        return None

    fpath = LEGACY_DIR / f"{slug}__{period}.json"
    if not fpath.exists():
        return None

    try:
        data = json.load(open(fpath))
        records = data.get('records', [])
        if not records:
            return None
        return [
            {
                'date':         r['date'],
                'weights':      np.array(r.get('weights', [])),
                'cash_weight':  r.get('cash_weight', 0),
                'regime_name':  r.get('regime_name', ''),
            }
            for r in records
        ]
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def compute_hhi(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index (sum of squared weights)."""
    return float(np.sum(weights ** 2))


def compute_effective_n(weights: np.ndarray) -> float:
    """Effective number of holdings = 1 / HHI."""
    hhi = compute_hhi(weights)
    return 1.0 / hhi if hhi > 0 else 0


def compute_top_k(weights: np.ndarray, k: int = 5) -> float:
    """Top-K weight concentration."""
    if len(weights) == 0: return 0
    sorted_w = np.sort(weights)[::-1]
    return float(np.sum(sorted_w[:k]))


def concentration_timeseries(records: List[Dict]) -> Dict[str, List]:
    """Compute HHI / top-K / effective N over time."""
    out = {
        'date':           [r['date'] for r in records],
        'hhi':            [compute_hhi(r['weights']) for r in records],
        'effective_n':    [compute_effective_n(r['weights']) for r in records],
        'top1':           [compute_top_k(r['weights'], 1) * 100 for r in records],
        'top5':           [compute_top_k(r['weights'], 5) * 100 for r in records],
        'cash':           [r['cash_weight'] * 100 for r in records],
        'regime':         [r['regime_name'] for r in records],
    }
    return out


def concentration_chart(
    records: List[Dict],
    variant_label: str,
    period_label: str,
    n_stocks_in_universe: int,
    height: int = 380,
) -> go.Figure:
    """
    4-panel time-series: top-5 weight, effective N, HHI, cash %.
    Shows day-by-day concentration evolution.
    """
    ts = concentration_timeseries(records)

    # Theoretical equal-weight baseline
    ew_top5 = (5 / n_stocks_in_universe) * 100   # if uniform, top-5 = 5/N
    ew_eff_n = float(n_stocks_in_universe)
    ew_hhi = 1.0 / n_stocks_in_universe

    fig = go.Figure()

    # Top-5 weight %
    fig.add_trace(go.Scatter(
        x=ts['date'], y=ts['top5'], mode='lines',
        line=dict(color=PRIMARY, width=2),
        name='Top-5 weight %',
        hovertemplate='%{x}<br>Top-5: %{y:.1f}%<extra></extra>',
    ))
    # EW reference line
    fig.add_hline(y=ew_top5, line_dash='dot', line_color=NEUTRAL,
                  annotation_text=f"EW top-5 = {ew_top5:.1f}%", annotation_position='top right',
                  annotation_font=dict(color=TEXT_DIM, size=10))

    fig.update_layout(
        title=dict(
            text=f"<b>{variant_label} — {period_label}: Top-5 weight concentration over time</b>",
            font=dict(family='Inter', size=13, color=TEXT),
            x=0.0, xanchor='left', y=0.95,
        ),
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(family='SF Mono', size=10, color=TEXT_MUTED),
                   gridcolor='rgba(255,255,255,0.04)', showgrid=True),
        yaxis=dict(title=dict(text='Top-5 weight %', font=dict(color=TEXT_MUTED, size=11)),
                   tickfont=dict(family='SF Mono', size=10, color=TEXT_MUTED),
                   gridcolor='rgba(255,255,255,0.04)', range=[0, 100], ticksuffix='%'),
        showlegend=False,
    )
    return fig


def effective_n_chart(
    records: List[Dict],
    variant_label: str,
    period_label: str,
    n_stocks_in_universe: int,
    height: int = 340,
) -> go.Figure:
    """Effective N stocks over time — diversification measure."""
    ts = concentration_timeseries(records)
    ew_eff_n = float(n_stocks_in_universe)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts['date'], y=ts['effective_n'], mode='lines',
        line=dict(color=SUCCESS, width=2),
        name='Effective N',
        hovertemplate='%{x}<br>Effective N: %{y:.1f}<extra></extra>',
    ))
    fig.add_hline(y=ew_eff_n, line_dash='dot', line_color=NEUTRAL,
                  annotation_text=f"EW = {ew_eff_n:.0f}", annotation_position='top right',
                  annotation_font=dict(color=TEXT_DIM, size=10))

    fig.update_layout(
        title=dict(
            text=f"<b>Effective number of stocks held (1/HHI)</b>",
            font=dict(family='Inter', size=13, color=TEXT),
            x=0.0, xanchor='left', y=0.95,
        ),
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(family='SF Mono', size=10, color=TEXT_MUTED),
                   gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(title=dict(text='Effective N stocks', font=dict(color=TEXT_MUTED, size=11)),
                   tickfont=dict(family='SF Mono', size=10, color=TEXT_MUTED),
                   gridcolor='rgba(255,255,255,0.04)',
                   range=[0, n_stocks_in_universe * 1.1]),
        showlegend=False,
    )
    return fig


def average_allocation_chart(
    records: List[Dict],
    variant_label: str,
    period_label: str,
    tickers: List[str],
    height: int = 380,
    top_n: int = 15,
) -> go.Figure:
    """
    Bar chart of TOP-N stocks by average allocation over the period.
    Shows which stocks the agent concentrates on.
    """
    if not records or not tickers:
        return go.Figure()

    # Stack weights matrix: (n_days × n_stocks)
    weights_matrix = np.stack([r['weights'] for r in records])
    avg_weights = weights_matrix.mean(axis=0) * 100  # convert to %

    # Sort by average weight
    n_stocks = min(len(tickers), len(avg_weights))
    ticker_weights = list(zip(tickers[:n_stocks], avg_weights[:n_stocks]))
    ticker_weights.sort(key=lambda x: x[1], reverse=True)

    # Take top N
    top = ticker_weights[:top_n]
    labels = [t for t, w in top]
    values = [w for t, w in top]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation='h',
        marker=dict(color=PRIMARY, line=dict(width=0)),
        text=[f"{v:.2f}%" for v in values],
        textposition='outside',
        textfont=dict(family='SF Mono', size=11, color=TEXT),
        hovertemplate='%{y}: %{x:.3f}%<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Top-{top_n} stocks by average allocation</b>",
            font=dict(family='Inter', size=13, color=TEXT),
            x=0.0, xanchor='left', y=0.95,
        ),
        height=height,
        margin=dict(l=20, r=80, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(family='SF Mono', size=10, color=TEXT_MUTED),
                   gridcolor='rgba(255,255,255,0.04)', ticksuffix='%'),
        yaxis=dict(autorange='reversed',
                   tickfont=dict(family='Inter', size=11, color=TEXT),
                   showgrid=False),
        showlegend=False,
    )
    return fig
