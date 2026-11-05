"""
Regime-Aware DRL Portfolio Management — Defense demo dashboard.

Home / landing page.
Runs:  streamlit run dashboard/app.py

The dashboard is intentionally minimal for the thesis defense: one
interactive page (Investor Simulator) where the audience can experience
the canonical agent operating on 2025 out-of-sample data day by day.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import apply_theme
from utils.theme import (
    PRIMARY, TEXT, TEXT_MUTED, TEXT_DIM, SUCCESS, BORDER, SURFACE, SURFACE_ALT,
)


# ----------------------------- Page config -----------------------------
st.set_page_config(
    page_title="Regime-Aware DRL Portfolio Thesis · Defense Demo",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ----------------------------- Sidebar header --------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0 0 1.5rem 0; border-bottom: 1px solid {BORDER}; margin-bottom: 1rem;">
        <div style="font-size: 0.7rem; color: {TEXT_DIM}; letter-spacing: 0.1em; text-transform: uppercase;">
            Bachelor's Thesis
        </div>
        <div style="font-size: 1.05rem; font-weight: 600; color: {TEXT}; margin-top: 0.3rem;">
            Regime-Aware<br/>DRL Portfolio Management
        </div>
        <div style="font-size: 0.75rem; color: {TEXT_MUTED}; margin-top: 0.4rem;">
            PPO · HMM · Attention · Cash-floor overlay
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:0.78rem; color:{TEXT_MUTED}; line-height:1.55;'>"
        f"Open the <b style='color:{PRIMARY};'>Investor Simulator</b> page "
        f"from the navigation above to start the live demo."
        f"</div>",
        unsafe_allow_html=True
    )


# ----------------------------- Hero ------------------------------------
st.markdown(f"""
<div style="background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE_ALT} 100%);
            border: 1px solid {BORDER}; border-radius: 14px;
            padding: 2.2rem 2.4rem; margin-bottom: 2rem;">
    <div style="font-size: 0.75rem; color: {PRIMARY}; letter-spacing: 0.1em;
                text-transform: uppercase; font-weight: 600;">
        Defense Demo
    </div>
    <h1 style="margin: 0.6rem 0 0.8rem 0; font-size: 1.9rem !important; line-height: 1.2;">
        Regime-Aware Deep Reinforcement Learning<br/>
        for Portfolio Management
    </h1>
    <div style="color: {TEXT_MUTED}; font-size: 1.0rem; line-height: 1.65; max-width: 900px;">
        A single-agent PPO recipe with multi-head self-attention over per-stock features,
        Hidden Markov Model regime conditioning, and a bear-regime cash-floor overlay,
        validated across DJI&nbsp;30, S&amp;P&nbsp;100, and NASDAQ-100 universes under a
        uniform selection rule.
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------- The demo card ---------------------------
st.markdown(f"""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem;">
    <div style="background: {SURFACE}; border: 1px solid {BORDER}; border-left: 4px solid {PRIMARY};
                border-radius: 10px; padding: 1.4rem 1.6rem;">
        <div style="font-size: 0.72rem; color: {TEXT_DIM}; letter-spacing: 0.1em;
                    text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">
            Live demo · Investor Simulator
        </div>
        <div style="color: {TEXT}; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.6rem;">
            Experience the agent on 2025 data
        </div>
        <div style="color: {TEXT_MUTED}; font-size: 0.88rem; line-height: 1.6;">
            Type an investment amount, pick a universe, and scrub or play through 2025
            day by day. Watch the portfolio value, the HMM regime, the cash position, and
            the top holdings update in real time. Toggle between the canonical hard floor
            and the soft posterior-weighted refinement to see the floor mechanism in action.
        </div>
        <div style="margin-top: 1rem; color: {PRIMARY}; font-size: 0.85rem; font-weight: 500;">
            → Open the <b>Investor Simulator</b> page in the sidebar
        </div>
    </div>
    <div style="background: {SURFACE}; border: 1px solid {BORDER};
                border-radius: 10px; padding: 1.4rem 1.6rem;">
        <div style="font-size: 0.72rem; color: {TEXT_DIM}; letter-spacing: 0.1em;
                    text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">
            Thesis headline
        </div>
        <div style="color: {TEXT}; font-size: 0.92rem; line-height: 1.7;">
            <b style="color: {SUCCESS};">DJI 30:</b> 8 of 8 benchmark wins<br/>
            <b style="color: {PRIMARY};">S&amp;P 100:</b> 5 of 8 wins<br/>
            <b style="color: {TEXT_MUTED};">NDX 2017:</b> defensive trade, smallest drawdown<br/>
            <span style="color: {TEXT_DIM}; font-size: 0.82rem;">
                Same architecture, same hyperparameters, same selection rule applied across
                all three universes.
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------- What the demo shows ---------------------
st.markdown(f"""
<div style="background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
            padding: 1.3rem 1.6rem;">
    <div style="font-size: 0.72rem; color: {TEXT_DIM}; letter-spacing: 0.1em;
                text-transform: uppercase; margin-bottom: 0.7rem; font-weight: 600;">
        What you will see in the demo
    </div>
    <ul style="color: {TEXT_MUTED}; font-size: 0.88rem; line-height: 1.8;
               padding-left: 1.2rem; margin: 0;">
        <li><b style="color: {TEXT};">Portfolio value</b> in dollars, updating day by day.</li>
        <li><b style="color: {TEXT};">Today's regime</b> (bull / sideways / bear) from the HMM
            with a coloured pill.</li>
        <li><b style="color: {TEXT};">Cash position</b> as a percentage and dollar amount, with
            the cash bar growing on bear-labelled days.</li>
        <li><b style="color: {TEXT};">Top holdings</b> with weights and dollar allocations,
            sorted by position size.</li>
        <li><b style="color: {TEXT};">Equity curve</b> with broad-market and equal-weight
            comparison lines overlaid.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ----------------------------- Footer ----------------------------------
st.markdown(
    f"<div style='color:{TEXT_DIM}; font-size:0.76rem; line-height:1.6; margin-top:2rem; "
    f"padding-top:1rem; border-top:1px solid {BORDER};'>"
    f"The dashboard is intentionally minimal for defense. Development pages used during "
    f"the writing of the thesis are archived under "
    f"<code style='font-size:0.72rem;'>dashboard/pages_archived/</code> "
    f"and can be re-activated by moving them back into "
    f"<code style='font-size:0.72rem;'>dashboard/pages/</code>."
    f"</div>",
    unsafe_allow_html=True
)
