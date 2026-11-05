"""
Live trading demo — the canonical regime-aware DRL agent on 2025 data.

Type an investment amount, pick a universe, then scrub or play through
2025 day by day to experience what the agent would have done with the
money in real out-of-sample market conditions.

Floor mechanism toggle (DJI and S&P 100 only):
  - HARD (canonical): the canonical 70% / 30% bear-regime floor reported
    in Chapter 4.
  - SOFT (refinement): the soft posterior-weighted floor proposed in
    Section 4.8.4, which scales by HMM bear confidence.

Data sources:
  results/_canonical/simulator_2025_full.json       (HARD canonical)
  results/_canonical/simulator_2025_full_soft.json  (SOFT refinement)
Produced by models/simulator_2025_full.py and
models/simulator_2025_full_soft.py.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import apply_theme
from utils.theme import (
    TEXT, TEXT_MUTED, TEXT_DIM, SURFACE, SURFACE_ALT, BORDER, BG,
    PRIMARY, PRIMARY_DIM, SUCCESS, WARNING, DANGER, NEUTRAL,
)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
DATA_PATH_HARD = PROJECT_ROOT / 'results' / '_canonical' / 'simulator_2025_full.json'
DATA_PATH_SOFT = PROJECT_ROOT / 'results' / '_canonical' / 'simulator_2025_full_soft.json'


# ============================== Page setup ==============================
st.set_page_config(
    page_title="Live Trading Demo · Canonical Agent on 2025 Data",
    page_icon="◆",
    layout="wide",
)
apply_theme()

if not DATA_PATH_HARD.exists():
    st.error(
        f"Canonical simulator data missing at `{DATA_PATH_HARD}`. "
        f"Run `python models/simulator_2025_full.py` first."
    )
    st.stop()

with open(DATA_PATH_HARD) as f:
    DATA_HARD = json.load(f)
DATA_SOFT = None
if DATA_PATH_SOFT.exists():
    with open(DATA_PATH_SOFT) as f:
        DATA_SOFT = json.load(f)


# ============================== Header ==============================
st.markdown(
    f"<div style='display:flex; align-items:baseline; gap:0.8rem; margin-bottom:0.2rem;'>"
    f"<h1 style='margin:0; padding:0;'>Live Trading Demo</h1>"
    f"<span style='color:{TEXT_DIM}; font-size:0.78rem; letter-spacing:0.06em; "
    f"text-transform:uppercase;'>Canonical agent on 2025 data</span>"
    f"</div>"
    f"<div style='color:{TEXT_MUTED}; font-size:0.93rem; margin-bottom:1.5rem; line-height:1.55;'>"
    f"Choose an investment amount and a universe, then scrub or play through 2025 to "
    f"experience the regime-aware DRL agent making its day-by-day decisions on real "
    f"out-of-sample market data."
    f"</div>",
    unsafe_allow_html=True
)


# ============================== Sidebar controls ==============================
UNIVERSE_TOOLTIPS = {
    'DJI':   '30 blue-chip U.S. stocks. The smallest and most concentrated universe.',
    'SP100': '90 large-cap U.S. stocks (S&P 100 minus continuity filter exclusions).',
    'NDX':   '90 NASDAQ-100 stocks (2017-cutoff variant). Most mega-cap concentrated.',
}

with st.sidebar:
    st.markdown(f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
                f"text-transform:uppercase; margin-bottom:0.5rem; margin-top:1rem;'>Your investment</div>",
                unsafe_allow_html=True)
    amount = st.number_input(
        "Amount (USD)", min_value=100, max_value=10_000_000,
        value=100_000, step=1_000,
        label_visibility='collapsed',
        format='%d',
        help='How much money to simulate investing. The portfolio scales linearly; '
             'all percentages and ratios remain identical regardless of amount.',
    )

    st.markdown(f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
                f"text-transform:uppercase; margin-top:1.4rem; margin-bottom:0.5rem;'>Universe</div>",
                unsafe_allow_html=True)
    universe = st.radio(
        "Universe",
        options=['DJI', 'SP100', 'NDX'],
        format_func=lambda c: {
            'DJI':   '🟢 DJI 30',
            'SP100': '🟣 S&P 100',
            'NDX':   '🟡 NASDAQ-100 2017',
        }[c],
        label_visibility='collapsed',
        help='Which equity universe the agent invests in. Each has a different canonical '
             'cash-floor setting per the thesis.',
    )
    st.caption(UNIVERSE_TOOLTIPS[universe])

    # ===== Floor mechanism toggle (only for DJI and SP100) =====
    use_soft = False
    if universe in ('DJI', 'SP100') and DATA_SOFT is not None:
        st.markdown(f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
                    f"text-transform:uppercase; margin-top:1.4rem; margin-bottom:0.5rem;'>Floor mechanism</div>",
                    unsafe_allow_html=True)
        floor_choice = st.radio(
            "Floor mechanism",
            options=['HARD (canonical)', 'SOFT (refinement)'],
            label_visibility='collapsed',
            help='HARD = canonical binary cash floor reported in Chapter 4. '
                 'SOFT = posterior-weighted refinement from Section 4.8.4 that scales '
                 'the floor by HMM bear confidence.',
        )
        use_soft = floor_choice.startswith('SOFT')

    st.markdown(f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
                f"text-transform:uppercase; margin-top:1.4rem; margin-bottom:0.5rem;'>Compare against</div>",
                unsafe_allow_html=True)
    show_bh = st.checkbox('Broad-market BH', value=True)
    show_ew = st.checkbox('Equal-weight', value=True)

    st.markdown('---')
    st.markdown(f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
                f"text-transform:uppercase; margin-bottom:0.5rem;'>Playback</div>",
                unsafe_allow_html=True)
    speed = st.slider('Speed (days/sec)', min_value=1, max_value=30, value=8,
                      help='How many trading days to advance per second when playing.')

    is_playing = st.session_state.get('is_playing', False)
    play_label = '⏸  Pause' if is_playing else '▶  Play'
    bcol1, bcol2 = st.columns(2)
    if bcol1.button(play_label, use_container_width=True):
        st.session_state['is_playing'] = not is_playing
        st.rerun()
    if bcol2.button('⏮  Reset', use_container_width=True):
        st.session_state['day_idx'] = 0
        st.session_state['is_playing'] = False
        st.rerun()


# ============================== Pick data source ==============================
DATA = DATA_SOFT if (use_soft and DATA_SOFT is not None) else DATA_HARD
ud         = DATA[universe]
agent      = ud['agent']
bh         = ud['BH']
ew         = ud['EW']
universe_name = ud['universe_name']
floor      = ud['floor']

dates      = agent['dates']
values     = np.array(agent['values'], dtype=float)
returns    = np.array(agent['returns'], dtype=float)
cash_pct   = np.array(agent['cash_pct'], dtype=float)
regimes    = agent['regime_names']
holdings   = agent['top_holdings']
n_days     = len(dates)


def rebase(series_values, target_dates):
    v = np.asarray(series_values, dtype=float)
    if len(v) == 0:
        return np.full(len(target_dates), float(amount))
    xp = np.linspace(0, 1, len(v))
    fp = v / v[0] * float(amount)
    xq = np.linspace(0, 1, len(target_dates))
    return np.interp(xq, xp, fp)


agent_dollars = values / values[0] * float(amount)
bh_dollars    = rebase(bh['values'], dates) if show_bh else None
ew_dollars    = rebase(ew['values'], dates) if show_ew else None


# ============================== Day slider + auto-advance ==============================
if 'day_idx' not in st.session_state:
    st.session_state['day_idx'] = 0
st.session_state['day_idx'] = min(st.session_state['day_idx'], n_days - 1)

day_idx = st.slider(
    f"Trading day  ({dates[0]} → {dates[-1]})",
    min_value=0, max_value=n_days - 1,
    value=st.session_state['day_idx'],
    key='day_slider',
)
st.session_state['day_idx'] = day_idx


# ============================== Current-state computation ==============================
cur_date = dates[day_idx]
cur_val  = float(agent_dollars[day_idx])
cur_ret  = float(returns[day_idx]) * 100.0
cur_cash = float(cash_pct[day_idx])
cur_regime = regimes[day_idx]
cur_holdings = holdings[day_idx]

day_change_dollars = cur_val - (agent_dollars[day_idx - 1] if day_idx > 0 else float(amount))
cum_pct = (cur_val / float(amount) - 1.0) * 100.0
cum_dollars = cur_val - float(amount)


# ============================== Top row: value + regime ==============================
hero_col, regime_col = st.columns([3, 2], gap='large')

with hero_col:
    delta_color = SUCCESS if day_change_dollars >= 0 else DANGER
    delta_arrow = '▲' if day_change_dollars >= 0 else '▼'
    st.markdown(
        f"<div style='background:linear-gradient(135deg, {SURFACE} 0%, {SURFACE_ALT} 100%); "
        f"border:1px solid {BORDER}; border-radius:14px; padding:1.6rem 1.8rem;'>"
        f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
        f"text-transform:uppercase; margin-bottom:0.5rem;'>Portfolio value</div>"
        f"<div style='font-size:2.6rem; font-weight:700; color:{TEXT}; "
        f"font-family:SF Mono, Menlo, monospace; line-height:1.0; margin-bottom:0.4rem;'>"
        f"${cur_val:,.2f}</div>"
        f"<div style='font-size:0.95rem; color:{delta_color}; font-weight:500; "
        f"font-family:SF Mono, Menlo, monospace;'>"
        f"{delta_arrow}  ${abs(day_change_dollars):,.2f}  today  ({cur_ret:+.2f}%)"
        f"</div>"
        f"<div style='font-size:0.82rem; color:{TEXT_MUTED}; margin-top:0.7rem; "
        f"font-family:SF Mono, Menlo, monospace;'>"
        f"Cumulative since Jan 2: <span style='color:{SUCCESS if cum_pct >= 0 else DANGER}; font-weight:600;'>"
        f"${cum_dollars:+,.0f}  ({cum_pct:+.2f}%)</span></div>"
        f"</div>",
        unsafe_allow_html=True
    )

with regime_col:
    regime_colors = {
        'bull':     (SUCCESS, '🟢', 'Take risk — light penalty'),
        'sideways': (WARNING, '🟡', 'Moderate caution'),
        'bear':     (DANGER, '🔴', ''),
    }
    rc, ricon, rdesc = regime_colors.get(cur_regime, (NEUTRAL, '○', ''))
    # Describe floor state under the active mechanism
    if cur_regime == 'bear':
        if floor == 0:
            rdesc = 'Heavy penalty active (no floor)'
        elif use_soft:
            rdesc = f'Soft floor active (up to {int(floor*100)}%, scaled by HMM confidence)'
        else:
            rdesc = f'{int(floor*100)}% hard cash floor active'
    cash_bar_width = min(100, int(cur_cash))
    st.markdown(
        f"<div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; "
        f"padding:1.6rem 1.8rem; height:100%;'>"
        f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
        f"text-transform:uppercase; margin-bottom:0.5rem;'>Today's regime</div>"
        f"<div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:0.4rem;'>"
        f"<span style='font-size:1.6rem;'>{ricon}</span>"
        f"<span style='font-size:1.4rem; font-weight:700; color:{rc}; text-transform:uppercase; "
        f"letter-spacing:0.04em;'>{cur_regime}</span></div>"
        f"<div style='font-size:0.82rem; color:{TEXT_MUTED}; margin-bottom:1rem;'>{rdesc}</div>"
        f"<div style='font-size:0.72rem; color:{TEXT_DIM}; text-transform:uppercase; "
        f"letter-spacing:0.08em; margin-bottom:0.3rem;'>Cash position</div>"
        f"<div style='font-size:1.4rem; color:{TEXT}; font-weight:600; "
        f"font-family:SF Mono, Menlo, monospace; margin-bottom:0.5rem;'>"
        f"{cur_cash:.1f}%  <span style='color:{TEXT_MUTED}; font-size:0.85rem; font-weight:400;'>"
        f"(${cur_cash/100*cur_val:,.0f})</span></div>"
        f"<div style='height:6px; background:{BORDER}; border-radius:3px; overflow:hidden;'>"
        f"<div style='height:100%; width:{cash_bar_width}%; "
        f"background:{rc}; transition:width 0.4s ease;'></div></div>"
        f"</div>",
        unsafe_allow_html=True
    )


# ============================== Equity curve ==============================
st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

fig = go.Figure()

past_x  = dates[:day_idx + 1]
past_y  = agent_dollars[:day_idx + 1]
fig.add_trace(go.Scatter(
    x=past_x, y=past_y, name='Agent',
    line=dict(color=PRIMARY, width=2.4),
    mode='lines',
    hovertemplate='%{x}<br>Agent: $%{y:,.2f}<extra></extra>',
))
if day_idx < n_days - 1:
    fut_x = dates[day_idx:]
    fut_y = agent_dollars[day_idx:]
    fig.add_trace(go.Scatter(
        x=fut_x, y=fut_y, name='Agent (future)',
        line=dict(color=BORDER, width=1.4, dash='dot'),
        mode='lines', showlegend=False, hoverinfo='skip',
    ))

if show_bh and bh_dollars is not None:
    fig.add_trace(go.Scatter(
        x=dates[:day_idx + 1], y=bh_dollars[:day_idx + 1],
        name=bh.get('name', 'BH'),
        line=dict(color=TEXT_MUTED, width=1.6, dash='dash'),
        mode='lines',
        hovertemplate='%{x}<br>BH: $%{y:,.2f}<extra></extra>',
    ))
if show_ew and ew_dollars is not None:
    fig.add_trace(go.Scatter(
        x=dates[:day_idx + 1], y=ew_dollars[:day_idx + 1],
        name=ew.get('name', 'EW'),
        line=dict(color=WARNING, width=1.4, dash='dot'),
        mode='lines',
        hovertemplate='%{x}<br>EW: $%{y:,.2f}<extra></extra>',
    ))

fig.add_trace(go.Scatter(
    x=[cur_date], y=[cur_val],
    name='Today',
    mode='markers',
    marker=dict(color=PRIMARY, size=11, line=dict(color=BG, width=2)),
    hovertemplate=f'{cur_date}<br>$%{{y:,.2f}}<extra></extra>',
    showlegend=False,
))

fig.update_layout(
    template='plotly_dark', plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family='Inter, system-ui, sans-serif', color=TEXT, size=12),
    margin=dict(l=10, r=10, t=10, b=10), height=340,
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_MUTED, size=11)),
    xaxis=dict(gridcolor=BORDER, zeroline=False, color=TEXT_DIM),
    yaxis=dict(gridcolor=BORDER, zeroline=False, color=TEXT_DIM,
               title=dict(text='Portfolio value (USD)', font=dict(color=TEXT_DIM, size=10)),
               tickformat='$,.0f'),
    hovermode='x unified',
)
st.plotly_chart(fig, use_container_width=True,
                key=f"eq_{universe}_{'soft' if use_soft else 'hard'}_{day_idx}")


# ============================== Bottom row: holdings + strategy ==============================
hcol, ncol = st.columns([3, 2], gap='large')

with hcol:
    st.markdown(
        f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
        f"text-transform:uppercase; margin-bottom:0.6rem;'>Top holdings · {cur_date}</div>",
        unsafe_allow_html=True
    )
    invested_val = cur_val * (1.0 - cur_cash / 100.0)
    cash_dollars = cur_cash / 100.0 * cur_val
    cash_bar_pct_holdings = min(100, int(cur_cash))
    # Cash always appears as the first row so it is visible alongside the stocks.
    cash_row = (
        f"<div style='display:grid; grid-template-columns:60px 1fr 80px 100px; "
        f"gap:0.7rem; align-items:center; padding:0.55rem 0.5rem; "
        f"border-bottom:1px solid {BORDER}; "
        f"background: rgba(52, 211, 153, 0.05); "
        f"border-left: 3px solid {SUCCESS}; margin-bottom: 0.2rem; border-radius: 0 4px 4px 0;'>"
        f"<span style='font-family:SF Mono, Menlo, monospace; color:{SUCCESS}; "
        f"font-weight:700; font-size:0.9rem; letter-spacing:0.04em;'>CASH</span>"
        f"<div style='height:8px; background:{BORDER}; border-radius:4px; overflow:hidden;'>"
        f"<div style='height:100%; width:{cash_bar_pct_holdings}%; background:{SUCCESS}; "
        f"transition:width 0.4s ease;'></div></div>"
        f"<span style='font-family:SF Mono, Menlo, monospace; color:{TEXT}; "
        f"font-weight:600; font-size:0.85rem; text-align:right;'>{cur_cash:.2f}%</span>"
        f"<span style='font-family:SF Mono, Menlo, monospace; color:{TEXT}; "
        f"font-weight:600; font-size:0.85rem; text-align:right;'>${cash_dollars:,.0f}</span>"
        f"</div>"
    )

    if cur_holdings:
        stock_rows = []
        for h in cur_holdings[:8]:
            tk = h['ticker']
            w  = h['weight']
            wp = w * 100.0
            dollars = w * cur_val
            bar_pct = min(100, int(wp * 3))
            stock_rows.append(
                f"<div style='display:grid; grid-template-columns:60px 1fr 80px 100px; "
                f"gap:0.7rem; align-items:center; padding:0.55rem 0; border-bottom:1px solid {BORDER};'>"
                f"<span style='font-family:SF Mono, Menlo, monospace; color:{TEXT}; "
                f"font-weight:600; font-size:0.9rem;'>{tk}</span>"
                f"<div style='height:8px; background:{BORDER}; border-radius:4px; overflow:hidden;'>"
                f"<div style='height:100%; width:{bar_pct}%; background:{PRIMARY};'></div></div>"
                f"<span style='font-family:SF Mono, Menlo, monospace; color:{TEXT_MUTED}; "
                f"font-size:0.85rem; text-align:right;'>{wp:.2f}%</span>"
                f"<span style='font-family:SF Mono, Menlo, monospace; color:{TEXT}; "
                f"font-size:0.85rem; text-align:right;'>${dollars:,.0f}</span>"
                f"</div>"
            )
        st.markdown(cash_row + ''.join(stock_rows), unsafe_allow_html=True)
    else:
        # Initial day: only the cash row exists (agent starts 100% cash).
        st.markdown(cash_row, unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; padding:0.8rem 0;'>"
            f"Agent has not opened any stock positions yet (initial day). "
            f"Advance the slider to see allocation."
            f"</div>",
            unsafe_allow_html=True
        )

with ncol:
    st.markdown(
        f"<div style='font-size:0.72rem; color:{TEXT_DIM}; letter-spacing:0.1em; "
        f"text-transform:uppercase; margin-bottom:0.6rem;'>Strategy</div>",
        unsafe_allow_html=True
    )
    if floor == 0:
        floor_desc = "<b>No cash floor</b>"
    elif use_soft:
        floor_desc = f"<b>{int(floor*100)}% soft posterior-weighted bear floor</b>"
    else:
        floor_desc = f"<b>{int(floor*100)}% hard bear-regime cash floor</b>"

    st.markdown(
        f"<div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; "
        f"padding:1rem 1.1rem; font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.7;'>"
        f"<div style='color:{TEXT}; font-weight:600; margin-bottom:0.4rem;'>{universe_name}</div>"
        f"Canonical PPO agent with attention extractor and HMM regime conditioning, "
        f"using {floor_desc}.<br/>"
        f"Out-of-sample evaluation on 2025 data the agent never saw. No retraining, "
        f"no per-period tuning."
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div style='margin-top:1rem; font-size:0.72rem; color:{TEXT_DIM}; "
        f"letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;'>"
        f"Year-to-date (today)</div>",
        unsafe_allow_html=True
    )
    high = float(np.max(agent_dollars[:day_idx + 1]))
    low  = float(np.min(agent_dollars[:day_idx + 1]))
    drawdown = (cur_val - high) / high * 100.0 if high > 0 else 0.0
    st.markdown(
        f"<div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; "
        f"padding:0.9rem 1.1rem; font-family:SF Mono, Menlo, monospace; font-size:0.86rem;'>"
        f"<div style='display:flex; justify-content:space-between; padding:0.25rem 0;'>"
        f"<span style='color:{TEXT_DIM};'>YTD High</span>"
        f"<span style='color:{TEXT};'>${high:,.0f}</span></div>"
        f"<div style='display:flex; justify-content:space-between; padding:0.25rem 0;'>"
        f"<span style='color:{TEXT_DIM};'>YTD Low</span>"
        f"<span style='color:{TEXT};'>${low:,.0f}</span></div>"
        f"<div style='display:flex; justify-content:space-between; padding:0.25rem 0;'>"
        f"<span style='color:{TEXT_DIM};'>Drawdown</span>"
        f"<span style='color:{DANGER if drawdown < -0.5 else TEXT};'>{drawdown:+.2f}%</span></div>"
        f"</div>",
        unsafe_allow_html=True
    )


# ============================== Footer ==============================
mech_label = ("SOFT posterior-weighted refinement (Section 4.8.4)"
              if use_soft else "HARD canonical floor (Chapter 4)")
st.markdown(
    f"<div style='color:{TEXT_DIM}; font-size:0.78rem; line-height:1.6; margin-top:2rem; "
    f"padding-top:1rem; border-top:1px solid {BORDER};'>"
    f"<b style='color:{TEXT_MUTED};'>Methodology.</b> "
    f"Currently running: <b style='color:{TEXT_MUTED};'>{mech_label}</b>. "
    f"The agent's decisions are deterministic, using the same canonical PPO checkpoint "
    f"reported in Chapter 4. HMM regime labels for 2025 come from a model refit on 2010 "
    f"to 2020 data (DJI HMM for the DJI agent; SP500 HMM for S&amp;P 100 and NDX, "
    f"matching Section 3.2.6). Transaction cost is 10 bps per turnover. The simulator "
    f"does not retrain or re-pick the canonical floor based on 2025 data; that would be "
    f"data snooping."
    f"</div>",
    unsafe_allow_html=True
)


# ============================== Auto-play ==============================
if st.session_state.get('is_playing', False) and day_idx < n_days - 1:
    advance = max(1, int(speed / 4))
    next_idx = min(n_days - 1, day_idx + advance)
    time.sleep(max(0.05, 1.0 / max(1, speed)))
    st.session_state['day_idx'] = next_idx
    st.rerun()
elif day_idx >= n_days - 1:
    st.session_state['is_playing'] = False
