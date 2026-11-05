"""
Page 1 — Research Map.

Visualizes the full experimental tree: Phase A architecture decisions
→ Phase B extended FULL → Phase C universal v6 single-agent production
→ Phase D extensions (long-mom, cutoff, 2M training, cash-floor overlay,
NDX reward ablations).

Helps a reader/committee see at a glance:
  - WHY each experiment was run (justification chain)
  - WHERE each variant sits in the lineage
  - WHAT is settled vs WHAT is still being tested

Note: a multi-agent specialist architecture (MAC v3) was also developed
in parallel but is excluded from the thesis scope. References to it
remain in the codebase only.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    apply_theme, badge,
    load_all_experiments, get_legacy_for_universe,
)
from utils.theme import (
    PRIMARY, SUCCESS, WARNING, DANGER, NEUTRAL,
    TEXT, TEXT_MUTED, TEXT_DIM, SURFACE, BORDER,
)


st.set_page_config(
    page_title="Research Map · DRL Portfolio Thesis",
    page_icon="◆",
    layout="wide",
)
apply_theme()

st.title("Research Map")
st.markdown(
    f"<div style='color: {TEXT_MUTED}; font-size: 0.95rem; margin-bottom: 1.5rem;'>"
    "The full experimental tree. Each phase justifies the next. Use this page "
    "to verify any architectural choice traces back to a documented trial."
    "</div>", unsafe_allow_html=True)


# ============== Mermaid-style ASCII diagram ==============
st.markdown("### Experimental phases")

phase_diagram = f"""
<div style="background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
            padding: 1.5rem; font-family: 'SF Mono', monospace; font-size: 0.85rem;
            color: {TEXT}; line-height: 1.7; white-space: pre; overflow-x: auto;">
<span style="color: {PRIMARY}; font-weight: 600;">PHASE A — Architecture decision trials  (DJI)</span>
   VANILLA  (PPO baseline, no architecture)
     │
     ├──→ + cash floor?      <span style="color: {NEUTRAL}">no_floor ablation       → JUSTIFIES cash floor</span>
     ├──→ + DSR reward?      <span style="color: {NEUTRAL}">no_dsr ablation         → JUSTIFIES DSR</span>
     ├──→ + attention extr?  <span style="color: {NEUTRAL}">no_attention ablation   → JUSTIFIES attention</span>
     └──→ + regime-λ?        <span style="color: {NEUTRAL}">no_regime_λ=0.1/0.3/0.5 → JUSTIFIES regime-λ</span>
     ↓
   <span style="color: {SUCCESS}; font-weight: 600;">FULL baseline</span>  (all four components retained)

<span style="color: {PRIMARY}; font-weight: 600;">PHASE B — Extended FULL baseline applied per universe</span>
   FULL (DJI)   ──→  extended (DJI single-agent FULL — first published recipe)
   FULL (SP100) ──→  extended_sp100 (SP100 mirror of FULL — generalization test)

<span style="color: {PRIMARY}; font-weight: 600;">PHASE C — Universal v6 single-agent production</span>
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ <span style="color: {SUCCESS}; font-weight: 600;">Universal v6  (single-agent — the canonical thesis architecture)</span>            │
   │                                                                            │
   │ One PPO agent with:                                                        │
   │   • Attention-based per-stock feature extractor (4 heads, embed_dim 64)    │
   │   • Regime probabilities in observation (3 HMM states + vol features)      │
   │   • Regime-conditioned λ (bull 0.1, sideways 0.3, bear 0.5) on downside    │
   │   • Differential Sharpe Ratio reward (Moody & Saffell 2001)                │
   │   • Soft regime conditioning — policy varies continuously with HMM probs   │
   │                                                                            │
   │ Tested:    DJI, SP100, NDX                                                 │
   │ Ablations: noregime (H2), uniform-λ (H2-soft)                              │
   │ Overlay:   bear-regime cash floor (Yang 2020 / Faber 2007 precedent)       │
   └────────────────────────────────────────────────────────────────────────────┘

<span style="color: {PRIMARY}; font-weight: 600;">PHASE D — Experimental extensions (single-agent path)</span>
   universal_v6 base
     │
     ├──→ + long-momentum features    (mom_120d_norm, mom_252d_norm)
     │      ├──→ DJI + mom (1M / 2M)  <span style="color: {NEUTRAL}">→ marginal effect; cash-floor breaks the ceiling</span>
     │      ├──→ SP100 + mom 2M       <span style="color: {SUCCESS}">→ DEFENSIVE 7/12 wins · val Sharpe 3.41 ★</span>
     │      ├──→ NDX + mom 2007 (2M)  <span style="color: {DANGER}">→ negative: bull -0.10% (universe-membership mismatch)</span>
     │      └──→ NDX 2017 + mom       <span style="color: {SUCCESS}">→ DEFENSIVE 4/8 wins · +18.95pp bear edge ★</span>
     │
     ├──→ + recent training window
     │      ├──→ NDX 2017 cutoff      <span style="color: {NEUTRAL}">→ enables relaxed 90-stock universe</span>
     │      └──→ SP100 2015 cutoff    <span style="color: {SUCCESS}">→ alternative experimental winner</span>
     │
     ├──→ + training-budget extension to 2M
     │      ├──→ SP100 mom 1M→2M      <span style="color: {SUCCESS}">→ canonical model identified at step 1,150k</span>
     │      ├──→ DJI mom 1M→2M        <span style="color: {SUCCESS}">→ canonical model identified at step 800k</span>
     │      └──→ NDX mom (only 2M)    <span style="color: {DANGER}">→ 2017 universe was the correct design</span>
     │
     ├──→ + bear-regime cash-floor overlay (Faber 2007 / Yang 2020 precedent)
     │      ├──→ DJI + 70% bear floor    <span style="color: {SUCCESS}">→ 8/8 PERFECT WINS ★</span>
     │      ├──→ SP100 + 30% bear floor  <span style="color: {SUCCESS}">→ 7/12 wins kept, bear +2pp, Sharpe ↑</span>
     │      └──→ NDX cash-floor sweep    <span style="color: {DANGER}">→ any floor hurts NDX wins (4/8 → 2/8)</span>
     │
     ├──→ + CVaR bear-defense penalty <span style="color: {DANGER}">→ rejected (2:1 trade-off, not asymmetric)</span>
     │
     └──→ NDX reward-shape ablations
            ├──→ exp1 low bull λ      <span style="color: {NEUTRAL}">→ no effect</span>
            ├──→ exp2 concentration   <span style="color: {NEUTRAL}">→ no effect</span>
            ├──→ exp3 log-return DSR  <span style="color: {NEUTRAL}">→ no effect</span>
            └──→ exp4 low ent_coef    <span style="color: {NEUTRAL}">→ no effect</span>
</div>
"""

st.markdown(phase_diagram, unsafe_allow_html=True)

st.markdown("---")


# ============== The selection-rule footnote ==============
st.markdown("### Canonical selection rule (uniform across all universes)")
st.markdown(f"""
<div style="background: {SURFACE}; border-left: 3px solid {PRIMARY}; padding: 1rem 1.2rem;
            border-radius: 0 8px 8px 0; color: {TEXT_MUTED}; font-size: 0.9rem; line-height: 1.6;">
The canonical model per universe is selected by a uniform <b style="color: {TEXT}">two-step rule</b>
applied identically to DJI, SP100, and NDX:<br/><br/>
&nbsp;&nbsp;<b>Step 1 — DEFENSIVE rule</b>: pick the val-Sharpe-best checkpoint with step ≥ 20% of total
training (Yang et al. 2020 convention). Val-only criterion → truly out-of-sample for bear/recovery/bull.<br/>
&nbsp;&nbsp;<b>Step 2 — Cash-floor sweep</b>: on the chosen checkpoint, sweep bear-regime cash floors
{{0%, 30%, 50%, 70%}} and pick the magnitude that maximises benchmark wins (val-Sharpe tiebreaker).
Yang 2020 / Faber 2007 TAA precedent.<br/><br/>
The optimal floor differs per universe (70% DJI, 30% SP100, 0% NDX) — empirically determined, not
hand-tuned. Aggressive alternatives (HEADLINE rule = wins-best + sum-Sharpe tiebreaker) are reported
on some pages for completeness but use test-wins-based selection (mild bias). The canonical is DEFENSIVE.
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ============== The universality claim ==============
st.markdown("### The universality claim — what is and isn't being claimed")

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
<div class="variant-card">
  <div class="v-name" style="color: {SUCCESS};">✓ Claim: recipe ports</div>
  <div class="v-sub">
  Same code, same hyperparameters, same architecture, same selection rule —
  trained per-universe (DJI, SP100, NDX). The recipe applies cleanly without
  per-universe tuning, and the resulting agent beats <i>some</i> benchmarks on
  each universe on each period.
  </div>
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
<div class="variant-card">
  <div class="v-name" style="color: {DANGER};">✗ NOT claimed: uniform effects</div>
  <div class="v-sub">
  We do NOT claim that every architectural choice (long-mom features, cash-floor
  overlay) helps EQUALLY on every universe. The thesis explicitly documents
  universe-conditional behaviour — the optimal cash-floor magnitude differs per
  universe (70% DJI, 30% SP100, 0% NDX), determined by empirical sweep with the
  same selection rule. This is a feature of the analysis, not a weakness.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ============== Why regime ==============
st.markdown("### Why a regime mechanism at all?")
st.markdown(f"""
<div style="background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
            padding: 1.2rem 1.5rem; color: {TEXT}; line-height: 1.6;">
<b style="color: {TEXT}">Theoretical alignment:</b>  <span style="color: {TEXT_MUTED}">35 years of finance literature on
regime-switching portfolios — Hamilton (1989), Ang &amp; Bekaert (2002),
Guidolin &amp; Timmermann (2007), Nystrup et al. (2018). Recent DRL portfolio
papers (Yang et al. 2020, Liang et al. 2018) use regime detection as auxiliary
information.</span><br/><br/>
<b style="color: {TEXT}">Architectural role:</b>  <span style="color: {TEXT_MUTED}">HMM regime labels feed BOTH the
observation channel (regime probabilities as features) AND the reward channel (regime-conditioned λ
risk penalty). They also enable the bear-regime cash-floor overlay at evaluation time. The regime
component is the central thesis contribution and the basis for interpretability of the agent's
defensive vs aggressive bias per period.</span><br/><br/>
<b style="color: {TEXT}">What the H2 ablation actually shows:</b>  <span style="color: {TEXT_MUTED}">When the OTHER architectural
defenses (attention, vol features, DSR reward) are intact, the marginal value
of explicit regime conditioning is universe-conditional. This is a finding
about <i>architectural robustness</i>, not about regime being useless.</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ============== Canonical results summary ==============
st.markdown("### Canonical results — Option A (DEFENSIVE rule + best cash floor)")

st.markdown(f"""
<div class="variant-card">
  <div class="v-name" style="color: {SUCCESS};">★ DJI canonical — 8/8 perfect wins</div>
  <div class="v-sub">
  <b>universal_v6_dji_mom_2m DEFENSIVE step 800,000 + 70% bear cash floor.</b><br/>
  val +29.79%, bear −7.82%, 2023 +30.34%, bull +30.03%. Beats both DJI BH and EW DJI on every period.
  3-year combined backtest: +56.22% cumulative (+40pp vs DJI BH). The 70% bear floor matches the MAC v3
  Defender specialist's internal validated value.
  </div>
</div>

<div class="variant-card">
  <div class="v-name" style="color: {SUCCESS};">★ SP100 canonical — 7/12 wins with highest val Sharpe in project</div>
  <div class="v-sub">
  <b>universal_v6_sp100_mom_2m DEFENSIVE step 1,150,000 + 30% bear cash floor.</b><br/>
  val +37.27%, bear −12.71%, 2023 +23.37%, bull +18.17%. Val Sharpe 3.41 with bootstrap 95% CI
  [+1.09, +5.15] — the only CI lower bound &gt; 1.0 in the entire project. Val Calmar 8.28 (exceptional).
  3-year combined backtest: +27.25% cumulative.
  </div>
</div>

<div class="variant-card">
  <div class="v-name" style="color: {WARNING};">★ NDX canonical — 4/8 wins (structural ceiling acknowledged)</div>
  <div class="v-sub">
  <b>universal_v6_ndx_2017_mom DEFENSIVE step 100,000, no cash-floor overlay</b> (any floor reduces wins on NDX).<br/>
  val +27.03%, bear −12.16%, 2023 +25.11%, bull +13.70%. Defensive headline: +18.95pp edge vs NDX BH
  on bear 2022 (−12.16% vs −33.71%). Structural ceiling: 2017-cutoff universe excludes post-2017 NDX
  leaders (META/TSLA/AVGO), and NDX BH is cap-weighted with NVDA at 7% — a +171% 2024 stock. Not a model
  failure but a documented universe-cap-weighting structural property.
  </div>
</div>

<div class="variant-card">
  <div class="v-name" style="color: {NEUTRAL};">Documented training-budget extensions (2M pass)</div>
  <div class="v-sub">
  SP100 mom 2M extension justified by +0.11/100k late slope at 1M (still climbing); at 2M converged
  (-0.04 slope). DJI mom 2M extension confirmed via DEFENSIVE pick at step 800k. NDX 2007-mom 2M was
  a negative finding (universe-membership mismatch — 2007 universe excludes 2024 winners). The
  HEADLINE-rule alternatives (SP100 11/12, NDX 6/8) are documented but use test-wins-based selection;
  the canonical above uses val-only DEFENSIVE selection.
  </div>
</div>
""", unsafe_allow_html=True)
