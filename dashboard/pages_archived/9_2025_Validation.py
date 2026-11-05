"""
2025 Forward Simulator page.

Out-of-sample validation: how do the three canonical agents (locked
on 2022 to 2024 data) perform on 2025 data they have never seen?

Reads from results/_canonical/simulator_2025.json (main results) and
results/_canonical/simulator_2025_dji_floor_sweep.json (DJI floor
diagnostic). Both files are produced by models/simulator_2025.py.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import apply_theme, badge
from utils.theme import (
    TEXT, TEXT_MUTED, TEXT_DIM, SURFACE, SURFACE_ALT, BORDER, BG,
    PRIMARY, PRIMARY_DIM, SUCCESS, WARNING, DANGER, NEUTRAL,
)


# ============================== Config ==============================
st.set_page_config(
    page_title="2025 Forward Simulator · DRL Portfolio Thesis",
    page_icon="◆",
    layout="wide",
)
apply_theme()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR  = PROJECT_ROOT / 'results' / '_canonical'
SIM_PATH     = RESULTS_DIR / 'simulator_2025.json'
FLOOR_PATH   = RESULTS_DIR / 'simulator_2025_dji_floor_sweep.json'


# ============================== Sidebar ==============================
with st.sidebar:
    st.markdown(
        f"<div style='font-size:0.7rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
        f"text-transform:uppercase; margin-top:1rem;'>Forward window</div>"
        f"<div style='font-size:1.0rem; color:{TEXT}; font-weight:600; margin-top:0.2rem;'>"
        f"2025-01-01 to 2025-12-31</div>"
        f"<div style='font-size:0.78rem; color:{TEXT_MUTED}; margin-top:0.3rem; line-height:1.5;'>"
        f"Same canonical checkpoints reported in Chapter 4.<br/>No retraining. No floor adjustment.</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.78rem; color:{TEXT_MUTED}; line-height:1.5;'>"
        f"<b style='color:{TEXT};'>Reproducibility</b><br/>"
        f"<code style='font-size:0.72rem;'>python models/simulator_2025.py</code><br/>"
        f"Outputs are cached; reruns take seconds.</div>",
        unsafe_allow_html=True
    )


# ============================== Header ==============================
st.title("2025 Forward Simulator")
st.markdown(
    f"<div style='color:{TEXT_MUTED}; font-size:0.95rem; margin-bottom:1.5rem; line-height:1.6;'>"
    "Honest out-of-sample test. The three canonical agents from Chapter 4 "
    "were locked using only 2022 to 2024 data; this page runs them forward on "
    "2025 market data they have never seen. The HMM is refit on the original "
    "2010 to 2020 window with a fixed seed. No retraining, no per-period tuning."
    "</div>",
    unsafe_allow_html=True
)


# ============================== Load data ==============================
if not SIM_PATH.exists():
    st.error(
        f"Simulator results not found at `{SIM_PATH}`. "
        f"Run `python models/simulator_2025.py` first."
    )
    st.stop()

with open(SIM_PATH) as f:
    SIM = json.load(f)

FLOOR_SWEEP = None
if FLOOR_PATH.exists():
    with open(FLOOR_PATH) as f:
        FLOOR_SWEEP = json.load(f)


# ============================== Helper renderers ==============================
def color_for(value, ref):
    if value > ref + 0.5:   return SUCCESS
    if value < ref - 0.5:   return DANGER
    return NEUTRAL


def fmt_pct(v: float, signed: bool = True) -> str:
    return f"{v:+.2f}%" if signed else f"{v:.2f}%"


def metric_row(agent_m: dict, bh_m: dict, ew_m: dict):
    """Three-by-three metric grid: cum ret / Sharpe / MDD across agent, BH, EW."""
    cols = st.columns(3)
    metrics = [
        ('Cumulative return', 'cumulative_return', '%'),
        ('Sharpe ratio',      'sharpe_ratio',      ''),
        ('Maximum drawdown',  'max_drawdown',      '%'),
    ]
    for col, (label, key, unit) in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div style='font-size:0.72rem; color:{TEXT_DIM}; "
                f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>{label}</div>",
                unsafe_allow_html=True
            )
            a_v, b_v, e_v = agent_m[key], bh_m[key], ew_m[key]
            for tag, v, ref in [('Agent', a_v, b_v),
                                ('BH',    b_v, b_v),
                                ('EW',    e_v, b_v)]:
                col_v = SUCCESS if tag == 'Agent' and v > b_v else (
                        DANGER if tag == 'Agent' and v < b_v else TEXT)
                if tag != 'Agent':
                    col_v = TEXT_MUTED
                if key == 'sharpe_ratio':
                    val_str = f"{v:+.3f}"
                else:
                    val_str = f"{v:+.2f}{unit}"
                weight = 600 if tag == 'Agent' else 400
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-family:SF Mono, Menlo, monospace; padding:0.25rem 0;'>"
                    f"<span style='color:{TEXT_DIM}; font-size:0.78rem;'>{tag}</span>"
                    f"<span style='color:{col_v}; font-weight:{weight};'>{val_str}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )


def equity_curve(name: str, agent: dict, bh: dict, ew: dict):
    """Rebased equity curves: agent solid violet, BH dashed grey, EW dotted muted."""
    fig = go.Figure()

    # Agent series
    if agent.get('dates') and agent.get('values'):
        v = np.asarray(agent['values'], dtype=float)
        v_norm = (v / v[0]) * 100.0
        fig.add_trace(go.Scatter(
            x=agent['dates'], y=v_norm,
            name='Canonical agent',
            line=dict(color=PRIMARY, width=2.4),
            mode='lines',
            hovertemplate='%{x|%b %d}<br>Agent: %{y:.2f}<extra></extra>',
        ))

    # BH series (we only have raw values; use sequence index for x if dates absent)
    if bh.get('values'):
        v = np.asarray(bh['values'], dtype=float)
        v_norm = (v / v[0]) * 100.0
        # Build a synthetic date axis if agent dates exist; else just index
        x = list(range(len(v_norm)))
        if agent.get('dates') and len(agent['dates']) > 0:
            # Stretch agent dates to BH length proportionally (informational only)
            import pandas as pd
            try:
                full = pd.date_range(start=agent['dates'][0], periods=len(v_norm), freq='B')
                x = [d.strftime('%Y-%m-%d') for d in full]
            except Exception:
                pass
        fig.add_trace(go.Scatter(
            x=x, y=v_norm,
            name='Broad-market BH',
            line=dict(color=TEXT_MUTED, width=1.6, dash='dash'),
            mode='lines',
            hovertemplate='%{x}<br>BH: %{y:.2f}<extra></extra>',
        ))

    # EW series
    if ew.get('values'):
        v = np.asarray(ew['values'], dtype=float)
        v_norm = (v / v[0]) * 100.0
        x = list(range(len(v_norm)))
        if agent.get('dates') and len(agent['dates']) > 0:
            import pandas as pd
            try:
                full = pd.date_range(start=agent['dates'][0], periods=len(v_norm), freq='B')
                x = [d.strftime('%Y-%m-%d') for d in full]
            except Exception:
                pass
        fig.add_trace(go.Scatter(
            x=x, y=v_norm,
            name='Equal-weight',
            line=dict(color=WARNING, width=1.4, dash='dot'),
            mode='lines',
            hovertemplate='%{x}<br>EW: %{y:.2f}<extra></extra>',
        ))

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(family='Inter, system-ui, sans-serif', color=TEXT, size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
            bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_MUTED, size=11),
        ),
        xaxis=dict(gridcolor=BORDER, zeroline=False, showline=False, color=TEXT_DIM),
        yaxis=dict(gridcolor=BORDER, zeroline=False, showline=False, color=TEXT_DIM,
                   title=dict(text='Rebased to 100', font=dict(color=TEXT_DIM, size=10))),
        hovermode='x unified',
    )
    return fig


# ============================== Universe sections ==============================
UNIVERSE_HEADLINES = {
    'DJI': {
        'name': 'DJI 30',
        'subtitle': '30-stock blue-chip universe · canonical: mom_2m step 800k + 70% bear floor',
        'icon_color': SUCCESS,
        'narrative': (
            "Agent beats DJI buy-and-hold on Sharpe (+1.156 vs +0.855) and on drawdown "
            "(-15.82% vs -16.12%). Cumulative return edge over BH narrowed to ≈1.4 pp "
            "(vs ≈17 pp in 2024) because 2025 contained roughly 32 HMM-labelled bear "
            "days that triggered the 70% cash floor. The risk-adjusted advantage held; "
            "the cumulative-return drag is regime-distribution dependent and consistent "
            "with the limitation already stated in Section 4.8.2."
        ),
    },
    'SP100': {
        'name': 'S&P 100',
        'subtitle': '90-stock S&P 100 universe · canonical: mom_2m step 1.15M + 30% bear floor',
        'icon_color': PRIMARY,
        'narrative': (
            "Agent's risk-adjusted edge over SP500 BH held (Sharpe +1.042 vs +0.945, "
            "MDD -14.40% vs -18.90%). Cumulative return trails BH (+13.14% vs +17.51%), "
            "reflecting the cap-weighted SP500 dominance pattern that the thesis "
            "documented for 2024. Same cross-universe shape, smaller drawdown."
        ),
    },
    'NDX': {
        'name': 'NASDAQ-100 2017',
        'subtitle': '90-stock NDX 2017 universe · canonical: mom step 100k, no floor',
        'icon_color': WARNING,
        'narrative': (
            "Defensive trade as predicted by the encoder-capacity hypothesis (Section 5.3.1). "
            "Cumulative return trails BH and EW, but the agent has the smallest "
            "drawdown of the three NDX strategies (-19.45% vs BH -22.93%). "
            "Bull-capture ceiling on the highest-dispersion universe holds in 2025."
        ),
    },
}


for code in ['DJI', 'SP100', 'NDX']:
    if code not in SIM:
        continue
    data = SIM[code]
    headline = UNIVERSE_HEADLINES[code]

    st.markdown("---")
    st.markdown(
        f"<div style='display:flex; align-items:baseline; gap:0.8rem; margin-bottom:0.3rem;'>"
        f"<div style='width:6px; height:24px; background:{headline['icon_color']}; border-radius:2px;'></div>"
        f"<h2 style='margin:0; padding:0;'>{headline['name']}</h2>"
        f"<span style='color:{TEXT_DIM}; font-size:0.78rem; font-family:SF Mono, Menlo, monospace;'>"
        f"{data['eval_window']}</span>"
        f"</div>"
        f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; margin-bottom:1.2rem;'>{headline['subtitle']}</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns([5, 4], gap='large')

    with left:
        st.plotly_chart(
            equity_curve(code, data['agent'], data['BH'], data['EW']),
            use_container_width=True,
            key=f"chart_{code}",
        )

    with right:
        metric_row(data['agent'], data['BH'], data['EW'])

    st.markdown(
        f"<div style='background:{SURFACE}; border-left:3px solid {headline['icon_color']}; "
        f"padding:0.9rem 1.1rem; border-radius:0 8px 8px 0; color:{TEXT_MUTED}; "
        f"font-size:0.88rem; line-height:1.6; margin-top:1rem; margin-bottom:0.5rem;'>"
        f"{headline['narrative']}"
        f"</div>",
        unsafe_allow_html=True
    )


# ============================== DJI Floor Sweep Diagnostic ==============================
if FLOOR_SWEEP:
    st.markdown("---")
    st.markdown("## Diagnostic — DJI cash-floor sweep on 2025")
    st.markdown(
        f"<div style='color:{TEXT_MUTED}; font-size:0.9rem; margin-bottom:1.2rem; line-height:1.6;'>"
        "Out-of-sample test only. Selecting the floor based on 2025 data is not legitimate "
        "academically (it would be data snooping); these numbers diagnose why the canonical "
        "70% floor was over-defensive in 2025, not what the canonical floor should have been. "
        "The story: as 2025 had ≈32 HMM-bear-labelled days (mostly short corrections), the "
        "70% floor cost about 7 pp of cumulative return relative to no floor at all."
        f"</div>",
        unsafe_allow_html=True
    )

    floors = sorted([float(k) for k in FLOOR_SWEEP['floors'].keys()])
    cum_rets = [FLOOR_SWEEP['floors'][str(f)]['cumulative_return'] for f in floors]
    sharpes  = [FLOOR_SWEEP['floors'][str(f)]['sharpe_ratio']      for f in floors]
    mdds     = [FLOOR_SWEEP['floors'][str(f)]['max_drawdown']      for f in floors]

    cs1, cs2 = st.columns([3, 2], gap='large')

    with cs1:
        fig = go.Figure()
        x_lab = [f"{int(f*100)}%" for f in floors]
        fig.add_trace(go.Bar(
            x=x_lab, y=cum_rets,
            name='Cum. ret (%)',
            marker_color=[PRIMARY if abs(f - 0.7) < 1e-6 else PRIMARY_DIM for f in floors],
            text=[f"{v:+.1f}" for v in cum_rets],
            textposition='outside',
            textfont=dict(color=TEXT, size=11),
            hovertemplate='Floor %{x}<br>Cum.Ret: %{y:.2f}%<extra></extra>',
        ))
        # BH and EW reference lines
        fig.add_hline(y=FLOOR_SWEEP['BH']['cumulative_return'],
                      line=dict(color=NEUTRAL, dash='dash', width=1),
                      annotation_text=f"DJI BH {FLOOR_SWEEP['BH']['cumulative_return']:+.1f}%",
                      annotation_position='top right',
                      annotation_font=dict(color=NEUTRAL, size=10))
        fig.add_hline(y=FLOOR_SWEEP['EW']['cumulative_return'],
                      line=dict(color=WARNING, dash='dot', width=1),
                      annotation_text=f"EW DJI {FLOOR_SWEEP['EW']['cumulative_return']:+.1f}%",
                      annotation_position='bottom right',
                      annotation_font=dict(color=WARNING, size=10))
        fig.update_layout(
            template='plotly_dark', plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(family='Inter, system-ui, sans-serif', color=TEXT, size=12),
            margin=dict(l=10, r=10, t=30, b=10), height=320, showlegend=False,
            title=dict(text='Cumulative return by cash floor (DJI 2025)',
                       font=dict(color=TEXT, size=14), x=0, xanchor='left'),
            xaxis=dict(title=dict(text='Cash floor on bear-regime days',
                                  font=dict(color=TEXT_DIM, size=11)),
                       gridcolor=BORDER, color=TEXT_DIM),
            yaxis=dict(gridcolor=BORDER, color=TEXT_DIM, zeroline=False,
                       title=dict(text='Cumulative return (%)',
                                  font=dict(color=TEXT_DIM, size=10))),
        )
        st.plotly_chart(fig, use_container_width=True)

    with cs2:
        st.markdown(
            f"<div style='font-size:0.72rem; color:{TEXT_DIM}; text-transform:uppercase; "
            f"letter-spacing:0.08em; margin-bottom:0.6rem;'>Floor sweep table</div>",
            unsafe_allow_html=True
        )
        header = (f"<div style='display:grid; grid-template-columns:0.8fr 1fr 1fr 1fr; "
                  f"gap:0.5rem; padding:0.4rem 0; border-bottom:1px solid {BORDER}; "
                  f"color:{TEXT_DIM}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em;'>"
                  f"<span>Floor</span><span style='text-align:right;'>Cum.Ret</span>"
                  f"<span style='text-align:right;'>Sharpe</span><span style='text-align:right;'>MDD</span>"
                  f"</div>")
        st.markdown(header, unsafe_allow_html=True)
        for f in floors:
            f_str = f"{int(f*100)}%"
            cr = FLOOR_SWEEP['floors'][str(f)]['cumulative_return']
            sh = FLOOR_SWEEP['floors'][str(f)]['sharpe_ratio']
            md = FLOOR_SWEEP['floors'][str(f)]['max_drawdown']
            is_canon = abs(f - 0.7) < 1e-6
            tag = f' <span style="color:{PRIMARY}; font-size:0.65rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:600;">canonical</span>' if is_canon else ''
            row = (f"<div style='display:grid; grid-template-columns:0.8fr 1fr 1fr 1fr; "
                   f"gap:0.5rem; padding:0.45rem 0; border-bottom:1px solid {BORDER}; "
                   f"font-family:SF Mono, Menlo, monospace; font-size:0.86rem;'>"
                   f"<span style='color:{TEXT};'>{f_str}{tag}</span>"
                   f"<span style='text-align:right; color:{TEXT};'>{cr:+.2f}%</span>"
                   f"<span style='text-align:right; color:{TEXT};'>{sh:+.3f}</span>"
                   f"<span style='text-align:right; color:{TEXT};'>{md:+.2f}%</span>"
                   f"</div>")
            st.markdown(row, unsafe_allow_html=True)

    st.markdown(
        f"<div style='background:{SURFACE}; border-left:3px solid {WARNING}; "
        f"padding:0.9rem 1.1rem; border-radius:0 8px 8px 0; color:{TEXT_MUTED}; "
        f"font-size:0.88rem; line-height:1.6; margin-top:1.2rem;'>"
        f"<b style='color:{TEXT};'>Reading the diagnostic.</b> The canonical 70% floor was "
        f"chosen by the universe-blind selection rule of Section 3.5.2 (max wins on 2022 to "
        f"2024). On 2025 specifically, the optimal-in-hindsight floor would have been 30% "
        f"(highest Sharpe) or 0% (highest cumulative return). This is the regime-distribution "
        f"dependence that Section 4.8.2 names as the overlay's 'upper bound' limitation: the "
        f"floor's benefit depends on whether HMM-bear-labelled days are sustained downtrends "
        f"(as in the 2022 bear test, where the 70% floor saved 8 pp) or short corrections "
        f"(as in 2025, where it cost 7 pp). The canonical model is not retroactively changed."
        f"</div>",
        unsafe_allow_html=True
    )


# ============================== Footer note ==============================
st.markdown("---")
st.markdown(
    f"<div style='color:{TEXT_DIM}; font-size:0.78rem; line-height:1.6; margin-top:1rem;'>"
    f"<b style='color:{TEXT_MUTED};'>Methodology notes.</b><br/>"
    f"&bull; HMMs refit on 2010 to 2020 features (DJI HMM for the DJI agent; SP500 HMM for "
    f"S&amp;P 100 and NDX, matching Section 3.2.6). Fixed seed = 42 for reproducibility.<br/>"
    f"&bull; All technical indicators and normalisation use the same v6 feature pipeline. "
    f"Per-stock normalisation parameters loaded from <code>data/normalization_params_*.json</code>.<br/>"
    f"&bull; Same 10 bps transaction-cost assumption as the thesis. Same cash-floor logic. "
    f"No retraining, no checkpoint reselection, no floor adjustment.<br/>"
    f"&bull; Agent has 228 trading days vs benchmarks' 249 because the env reserves a "
    f"warm-up window for technical-indicator initialisation, matching the convention used "
    f"throughout Chapter 4 (Section 4.7.4)."
    f"</div>",
    unsafe_allow_html=True
)
