"""
Plotly chart helpers — bar charts per period vs benchmarks.

Used by per-universe pages to render the headline 4-chart grid
(val / bear / 2023 / bull) consistently.
"""

from typing import Dict, List, Optional

import plotly.graph_objects as go

from .theme import (
    PRIMARY, SUCCESS, WARNING, DANGER, NEUTRAL,
    TEXT, TEXT_MUTED, BORDER, SURFACE,
    PRIMARY_DIM,
)


PERIOD_COLORS = {
    'val':       '#8b7cff',   # primary violet
    'test_bear': '#f87171',   # softer red
    'test_2023': '#fbbf24',   # warm amber
    'test_bull': '#34d399',   # softer green
}


def _color_for_return(value: float, benchmark: float) -> str:
    """Color a bar by whether it beats the benchmark."""
    if value > benchmark:
        return SUCCESS
    if value < benchmark:
        return DANGER
    return NEUTRAL


def period_bar_chart(
    period_label: str,
    benchmarks: Dict[str, float],     # {name: cum_return_pct}
    variants:   List[Dict],           # [{label, value, is_production, is_ablation}, ...]
    primary_benchmark: Optional[str] = None,  # the name used for green/red coloring
    height: int = 320,
) -> go.Figure:
    """
    Build one horizontal bar chart for one period.
    Benchmarks shown as gray bars at top; variants colored by category below.
    """
    # Order: benchmarks first (gray), then production (violet), then ablations (faded red/blue)
    rows = []
    for name, value in benchmarks.items():
        rows.append({'label': name, 'value': value, 'kind': 'benchmark'})
    for v in variants:
        rows.append({
            'label': v['label'],
            'value': v['value'],
            'kind':  v.get('kind', 'variant'),
        })

    # Determine primary benchmark for color reference
    if primary_benchmark is None and benchmarks:
        primary_benchmark = list(benchmarks.keys())[0]
    primary_bm_value = benchmarks.get(primary_benchmark, 0)

    # Colors by kind
    def color_of(row):
        if row['kind'] == 'benchmark':
            return NEUTRAL
        if row['kind'] == 'production':
            return PRIMARY
        if row['kind'] == 'production_mac':
            return '#5db8f7'   # cooler blue
        if row['kind'] == 'ablation':
            return WARNING
        if row['kind'] == 'experimental':
            return '#a78bfa'   # lighter violet
        # variant: green if beats primary bench, red if loses
        return _color_for_return(row['value'], primary_bm_value)

    fig = go.Figure(go.Bar(
        x=[r['value'] for r in rows],
        y=[r['label'] for r in rows],
        orientation='h',
        marker=dict(
            color=[color_of(r) for r in rows],
            line=dict(width=0),
        ),
        text=[f"{r['value']:+.2f}%" for r in rows],
        textposition='outside',
        textfont=dict(family='SF Mono, monospace', size=12, color=TEXT),
        hovertemplate='%{y}: %{x:+.2f}%<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{period_label}</b>",
            font=dict(family='Inter, system-ui, sans-serif', size=14, color=TEXT),
            x=0.0, xanchor='left', y=0.95,
        ),
        height=height,
        margin=dict(l=20, r=80, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            zeroline=True, zerolinecolor=BORDER, zerolinewidth=2,
            tickfont=dict(family='SF Mono, monospace', size=11, color=TEXT_MUTED),
            gridcolor='rgba(255,255,255,0.04)',
            showgrid=True,
            ticksuffix='%',
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(family='Inter, system-ui, sans-serif', size=12, color=TEXT),
            showgrid=False,
        ),
        showlegend=False,
        font=dict(family='Inter, system-ui, sans-serif'),
    )

    return fig


def four_period_grid(
    title: str,
    benchmarks_by_period: Dict[str, Dict[str, float]],
    variants_by_period: Dict[str, List[Dict]],
    primary_benchmark: Optional[str] = None,
):
    """
    Return 4 figures: one per period. Pages will st.plotly_chart each.
    """
    out = {}
    period_titles = {
        'val':       'Validation 2021',
        'test_bear': 'Bear 2022',
        'test_2023': 'Recovery 2023',
        'test_bull': 'Bull 2024',
    }
    for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
        out[period] = period_bar_chart(
            period_label=period_titles[period],
            benchmarks=benchmarks_by_period.get(period, {}),
            variants=variants_by_period.get(period, []),
            primary_benchmark=primary_benchmark,
        )
    return out


def risk_metric_chart(
    metric_label: str,            # e.g. "Val Sharpe" or "Bear MDD"
    variants: List[Dict],          # [{label, value, kind}, ...]
    higher_is_better: bool = True,
    height: int = 240,
    suffix: str = '',              # '%' for MDD, '' for Sharpe
) -> go.Figure:
    """
    One horizontal bar chart for ONE metric on ONE period.
    Used for Sharpe / MDD comparison views.
    """
    def color_of(row):
        if row.get('kind') == 'production':         return PRIMARY
        if row.get('kind') == 'production_mac':     return '#5db8f7'
        if row.get('kind') == 'ablation':           return WARNING
        if row.get('kind') == 'experimental':       return '#a78bfa'
        return NEUTRAL

    fig = go.Figure(go.Bar(
        x=[v['value'] for v in variants],
        y=[v['label'] for v in variants],
        orientation='h',
        marker=dict(color=[color_of(v) for v in variants], line=dict(width=0)),
        text=[f"{v['value']:+.2f}{suffix}" for v in variants],
        textposition='outside',
        textfont=dict(family='SF Mono, monospace', size=11, color=TEXT),
        hovertemplate='%{y}: %{x:+.3f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{metric_label}</b>",
            font=dict(family='Inter, system-ui, sans-serif', size=13, color=TEXT),
            x=0.0, xanchor='left', y=0.95,
        ),
        height=height,
        margin=dict(l=20, r=80, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            zeroline=True, zerolinecolor=BORDER, zerolinewidth=2,
            tickfont=dict(family='SF Mono, monospace', size=10, color=TEXT_MUTED),
            gridcolor='rgba(255,255,255,0.04)',
            showgrid=True, ticksuffix=suffix,
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(family='Inter, system-ui, sans-serif', size=11, color=TEXT),
            showgrid=False,
        ),
        showlegend=False,
    )
    return fig


def period_focus_panel(
    period_label: str,
    return_bars: List[Dict],        # variants + benchmarks for return
    sharpe_bars: List[Dict],
    mdd_bars: List[Dict],
    return_suffix: str = '%',
    height: int = 360,
):
    """
    Three full-width bar charts side-by-side for ONE period:
      Return / Sharpe / MDD
    Used for the "Bull 2024 — focused analysis" section.
    """
    return_fig = period_bar_chart(
        period_label=f"{period_label} — Cumulative return",
        benchmarks={},  # benchmarks already in return_bars list with kind='benchmark'
        variants=return_bars,
        primary_benchmark=None,
        height=height,
    )

    sharpe_fig = risk_metric_chart(
        metric_label=f"{period_label} — Sharpe ratio",
        variants=sharpe_bars,
        higher_is_better=True,
        height=height,
        suffix='',
    )

    mdd_fig = risk_metric_chart(
        metric_label=f"{period_label} — Max drawdown",
        variants=mdd_bars,
        higher_is_better=False,
        height=height,
        suffix='%',
    )

    return return_fig, sharpe_fig, mdd_fig


def risk_metrics_grid(
    sharpe_by_period: Dict[str, List[Dict]],   # {period: [{label, value, kind}]}
    mdd_by_period:    Dict[str, List[Dict]],
):
    """
    Return 8 figures: 4 Sharpe charts + 4 MDD charts, one per period.
    """
    period_titles = {
        'val':       'Validation 2021',
        'test_bear': 'Bear 2022',
        'test_2023': 'Recovery 2023',
        'test_bull': 'Bull 2024',
    }
    sharpe_figs = {}
    mdd_figs    = {}
    for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
        sharpe_figs[period] = risk_metric_chart(
            metric_label=f"{period_titles[period]} — Sharpe",
            variants=sharpe_by_period.get(period, []),
            higher_is_better=True,
            suffix='',
        )
        mdd_figs[period] = risk_metric_chart(
            metric_label=f"{period_titles[period]} — Max Drawdown",
            variants=mdd_by_period.get(period, []),
            higher_is_better=False,   # MDD less-negative = better
            suffix='%',
        )
    return sharpe_figs, mdd_figs
