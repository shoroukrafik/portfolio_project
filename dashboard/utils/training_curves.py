"""
Training curves loader — reads val_sharpe_history.json from each variant
and produces a multi-line training trajectory chart for one universe.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import plotly.graph_objects as go

from .theme import (
    PRIMARY, SUCCESS, WARNING, DANGER, NEUTRAL,
    TEXT, TEXT_MUTED, BORDER, SURFACE, PRIMARY_DIM,
)


# Distinct colors for variants on the training curve
VARIANT_COLORS = [
    '#8b7cff',  # violet (production)
    '#5db8f7',  # blue (MAC)
    '#fbbf24',  # amber (mom)
    '#a78bfa',  # light violet (2015+mom)
    '#34d399',  # green (2015 no-mom)
    '#f87171',  # red (noregime)
    '#facc15',  # yellow (h2soft)
    '#fb7185',  # rose
]


def load_val_history(result_dir: str) -> Optional[list]:
    """Returns list of (step, val_sharpe) tuples, or None."""
    p = os.path.join(result_dir, 'val_sharpe_history.json')
    if not os.path.exists(p):
        return None
    try:
        data = json.load(open(p))
        return data.get('history') or []
    except (json.JSONDecodeError, OSError):
        return None


def training_curves_chart(
    variants: List[dict],            # [{label, result_dir, color_idx, picks?}, ...]
    title: str = "Validation Sharpe over training",
    height: int = 480,
) -> go.Figure:
    """
    Multi-line chart of val_sharpe vs training step for each variant.
    """
    fig = go.Figure()

    for idx, v in enumerate(variants):
        hist = load_val_history(v['result_dir'])
        if not hist:
            continue
        steps = [h[0] for h in hist]
        vals  = [h[1] for h in hist]
        color = VARIANT_COLORS[idx % len(VARIANT_COLORS)]

        fig.add_trace(go.Scatter(
            x=steps, y=vals,
            mode='lines',
            line=dict(color=color, width=2),
            name=v['label'],
            hovertemplate=f"<b>{v['label']}</b><br>step %{{x:,}}<br>val_Sh %{{y:.3f}}<extra></extra>",
        ))

        # Markers for both selection rules — different shapes so they're distinguishable
        picks = v.get('picks', {})

        # HEADLINE pick = star (max wins + sum-Sharpe tiebreaker, step >= 20%)
        if picks.get('headline'):
            h = picks['headline']
            y_val = next((vv for ss, vv in hist if ss == h['step']), h.get('val_sharpe'))
            fig.add_trace(go.Scatter(
                x=[h['step']], y=[y_val],
                mode='markers',
                marker=dict(color=color, size=14, symbol='star',
                            line=dict(width=1.5, color='white')),
                name=f"{v['label']} HEADLINE",
                showlegend=False,
                hovertemplate=(f"<b>{v['label']} — HEADLINE pick</b><br>"
                               f"step %{{x:,}}<br>val_Sh %{{y:.3f}}<br>"
                               f"(max wins + sum-Sharpe, step ≥ 20%)<extra></extra>"),
            ))

        # DEFENSIVE pick = diamond (max val-Sharpe, step >= 20%)
        if picks.get('defensive'):
            d = picks['defensive']
            y_val = next((vv for ss, vv in hist if ss == d['step']), d.get('val_sharpe'))
            # Skip if defensive pick == headline pick (same point, would overlap)
            same_as_headline = picks.get('headline') and picks['headline']['step'] == d['step']
            if not same_as_headline:
                fig.add_trace(go.Scatter(
                    x=[d['step']], y=[y_val],
                    mode='markers',
                    marker=dict(color=color, size=12, symbol='diamond-open',
                                line=dict(width=2.5, color=color)),
                    name=f"{v['label']} DEFENSIVE",
                    showlegend=False,
                    hovertemplate=(f"<b>{v['label']} — DEFENSIVE pick</b><br>"
                                   f"step %{{x:,}}<br>val_Sh %{{y:.3f}}<br>"
                                   f"(max val-Sharpe, step ≥ 20%)<extra></extra>"),
                ))

    # Add dummy traces for the marker-shape legend (so the symbol key is visible)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='star', size=12, color='#aaaaaa', line=dict(width=1.5, color='white')),
        name='★ HEADLINE pick (max wins + sum-Sh)', showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='diamond-open', size=12, color='#aaaaaa', line=dict(width=2.5, color='#aaaaaa')),
        name='◇ DEFENSIVE pick (max val-Sharpe)', showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family='Inter, system-ui, sans-serif', size=14, color=TEXT),
            x=0.0, xanchor='left', y=0.97,
        ),
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(text='Training step', font=dict(color=TEXT_MUTED, size=11)),
            tickfont=dict(family='SF Mono, monospace', size=10, color=TEXT_MUTED),
            gridcolor='rgba(255,255,255,0.04)',
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text='Validation Sharpe', font=dict(color=TEXT_MUTED, size=11)),
            tickfont=dict(family='SF Mono, monospace', size=10, color=TEXT_MUTED),
            gridcolor='rgba(255,255,255,0.04)',
            showgrid=True,
            zeroline=True, zerolinecolor=BORDER, zerolinewidth=1,
        ),
        legend=dict(
            orientation='h', y=-0.18,
            font=dict(color=TEXT_MUTED, size=10),
            bgcolor='rgba(0,0,0,0)',
        ),
        font=dict(family='Inter, system-ui, sans-serif'),
    )
    return fig
