"""
MAC v3 ablations and routing comparison.

Cleaner story:
  - MAC v3 = production multi-agent (defender/steady/hunter routed by trained MLP)
  - v5/v6/v7 = leave-one-out specialist ablations (proves each role is necessary)
  - MARS v1 rule-based = no-learning ablation (proves learned routing adds value)
  - Universal v6 single-agent = sibling production methodology
  - Benchmarks = DJI BH / EW DJI / SP500 / OEX / EW SP100

MAC v1, v2, v4 + MARS v2 reimplementations = older obs shapes / internal
tuning iterations, NOT directly comparable to v3 — folded to an appendix expander.

The MARS paper (Chen, Li, Wang — NJIT) uses HSI (Hang Seng) universe and is
NOT a direct quantitative benchmark — cited as architectural inspiration only.
"""

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    apply_theme, badge,
    load_all_experiments, picks_summary,
    BENCHMARKS,
)
from utils.theme import (
    TEXT, TEXT_MUTED, TEXT_DIM, SURFACE, BORDER, PRIMARY,
    SUCCESS, WARNING, DANGER,
)


st.set_page_config(
    page_title="MAC ablations · DRL Portfolio Thesis",
    page_icon="◆",
    layout="wide",
)
apply_theme()


# ============================== Header ==============================
st.title("MAC v3 — Multi-agent (parallel research path)")
st.markdown(f"""
<div style="background:{SURFACE}; border-left:4px solid {WARNING}; padding:1rem 1.2rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.92rem;
            line-height:1.55; margin-bottom:1.2rem;">
<b style="color:{WARNING}; font-size:1.0rem;">⚠ EXCLUDED FROM THIS THESIS</b><br/>
The multi-agent specialist architecture (MAC v3) was developed as a parallel research path
but is <b>not part of the current thesis scope</b>. The thesis focuses on the single-agent
universal v6 architecture (see DJI / SP100 / NDX pages for the canonical thesis results).<br/><br/>
This page is kept as reference documentation of the parallel research path. Numbers,
ablations, and analyses below are from the MAC v3 line of investigation and may be revisited
in future work. The DJI Defender specialist's 70% bear cash floor (validated on this page)
IS the same instrument used as the runtime overlay on the DJI canonical (see DJI page).
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<div style='color:{TEXT_MUTED}; font-size:0.9rem; margin-bottom:1.2rem;'>"
    "Below: MAC v3 multi-agent production design with (1) leave-one-out "
    "specialist ablations proving each role is necessary, and (2) a rule-based routing "
    "ablation showing the learned MLP controller adds value."
    "</div>",
    unsafe_allow_html=True
)


# ============================== Methods explainer ==============================
st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px;
            padding:0.9rem 1.1rem; color:{TEXT_MUTED}; font-size:0.87rem;
            line-height:1.6; margin-bottom:1.5rem;">

<b style="color:{TEXT}">MAC v3 ★ — production multi-agent</b><br>
3 specialists (defender / steady / hunter), each trained on regime-filtered data.
A trained MLP controller takes regime probabilities + vol features (6 dims) and
outputs softmax weights routing between the 3 specialists.<br><br>

<b style="color:{TEXT}">Leave-one-out specialist ablations</b><br>
&nbsp;&nbsp;<b>MAC v5</b> — drops STEADY (2 specialists: defender + hunter)<br>
&nbsp;&nbsp;<b>MAC v6</b> — drops DEFENDER (2 specialists: steady + hunter)<br>
&nbsp;&nbsp;<b>MAC v7</b> — drops HUNTER (2 specialists: defender + steady)<br>
Together these prove each specialist is load-bearing.<br><br>

<b style="color:{TEXT}">Routing ablation</b><br>
&nbsp;&nbsp;<b>MARS v1 rule-based</b> — same 3 specialists, but HMM regime probabilities
route directly (no learned controller). Tests value of the learned MLP.<br><br>

<b style="color:{TEXT}">Sibling production</b><br>
&nbsp;&nbsp;<b>Universal v6</b> — single-agent regime-conditioned alternative (not multi-agent).
Lets us compare multi-agent vs single-agent at the production level.
</div>
""", unsafe_allow_html=True)


# ============================== Load helpers ==============================
import pandas as pd


def legacy_cum_ret(slug, period):
    p = f'dashboard_legacy/data/curves/{slug}__{period}.json'
    if not os.path.exists(p): return None
    try:
        d = json.load(open(p))
        if not d.get('records'): return None
        last = d['records'][-1].get('value')
        return ((last - 1_000_000) / 1_000_000) * 100 if last else None
    except (json.JSONDecodeError, OSError):
        return None


def fmt(v):
    return f"{v:+.2f}" if v is not None else '—'


experiments = load_all_experiments()


def variant_headline(folder, universe='DJI'):
    e = next((x for x in experiments.get(universe, [])
              if x['folder'] == folder), None)
    if not e: return None
    p = picks_summary(e['result_dir'], e.get('total_steps'))
    h = p.get('headline')
    if not h: return None
    return {
        'label':       e['label'],
        'val':         h['val_ret'],
        'bear':        h['test_bear_ret'],
        '2023':        h['test_2023_ret'],
        'bull':        h['test_bull_ret'],
        'wins':        h.get('wins'),
        'total':       h.get('total'),
    }


def variant_from_legacy(slug, label):
    periods = {p: legacy_cum_ret(slug, p) for p in ['val','test_bear','test_2023','test_bull']}
    if all(v is None for v in periods.values()): return None
    return {
        'label': label,
        'val':   periods['val'],
        'bear':  periods['test_bear'],
        '2023': periods['test_2023'],
        'bull':  periods['test_bull'],
        'wins':  None,
        'total': None,
    }


def row(v, category):
    if not v: return None
    wstr = '—'
    if v.get('total'):
        wstr = f"{v['wins']}/{v['total']}"
    elif v.get('wins') is not None:
        wstr = str(v['wins'])
    return {
        'Method':   v['label'],
        'Category': category,
        'val':      fmt(v.get('val')),
        'bear':     fmt(v.get('bear')),
        '2023':     fmt(v.get('2023')),
        'bull':     fmt(v.get('bull')),
        'wins':     wstr,
    }


# ============================== DJI table ==============================
st.markdown("### DJI 30 — same universe, same periods, same benchmarks")

dji_rows = []

# Production
v = variant_headline('mac_dji_v3', 'DJI')
if v: dji_rows.append(row(v, 'PRODUCTION (multi-agent)'))
v = variant_headline('universal_v6_dji', 'DJI')
if v:
    v['label'] = 'Universal v6 (single-agent production)'
    dji_rows.append(row(v, 'PRODUCTION (single-agent)'))

# Leave-one-out specialist ablations
for folder, label in [
    ('mac_dji_v5', 'MAC v5 — no STEADY specialist'),
    ('mac_dji_v6', 'MAC v6 — no DEFENDER specialist'),
    ('mac_dji_v7', 'MAC v7 — no HUNTER specialist'),
]:
    v = variant_headline(folder, 'DJI')
    if not v: continue
    v['label'] = label
    dji_rows.append(row(v, 'Specialist ablation'))

# Routing ablation
v = variant_from_legacy('MARS_v1_rule-based_DJI_30', 'MARS v1 rule-based (no learned routing)')
if v: dji_rows.append(row(v, 'Routing ablation'))

# Benchmarks
dji_bench = BENCHMARKS['DJI']
for bm_name in ['DJI BH', 'EW DJI']:
    dji_rows.append({
        'Method':   bm_name,
        'Category': 'Benchmark',
        'val':      fmt(dji_bench['val'].get(bm_name)),
        'bear':     fmt(dji_bench['test_bear'].get(bm_name)),
        '2023':     fmt(dji_bench['test_2023'].get(bm_name)),
        'bull':     fmt(dji_bench['test_bull'].get(bm_name)),
        'wins':     '—',
    })

st.dataframe(pd.DataFrame(dji_rows), use_container_width=True, hide_index=True)


# ============================== SP100 table ==============================
st.markdown("### S&P 100 — same universe, same periods, same benchmarks")

sp_rows = []

v = variant_headline('mac_sp100_v3', 'SP100')
if v: sp_rows.append(row(v, 'PRODUCTION (multi-agent)'))
v = variant_headline('universal_v6_sp100', 'SP100')
if v:
    v['label'] = 'Universal v6 SP100 (single-agent production)'
    sp_rows.append(row(v, 'PRODUCTION (single-agent)'))

for folder, label in [
    ('mac_sp100_v5', 'MAC v5 SP100 — no STEADY specialist'),
    ('mac_sp100_v6', 'MAC v6 SP100 — no DEFENDER specialist'),
    ('mac_sp100_v7', 'MAC v7 SP100 — no HUNTER specialist'),
]:
    v = variant_headline(folder, 'SP100')
    if not v: continue
    v['label'] = label
    sp_rows.append(row(v, 'Specialist ablation'))

v = variant_from_legacy('MARS_v1_rule-based_SP100', 'MARS v1 rule-based SP100 (no learned routing)')
if v: sp_rows.append(row(v, 'Routing ablation'))

sp_bench = BENCHMARKS['SP100']
for bm_name in ['SP500 BH', 'OEX BH', 'EW SP100']:
    sp_rows.append({
        'Method':   bm_name,
        'Category': 'Benchmark',
        'val':      fmt(sp_bench['val'].get(bm_name)),
        'bear':     fmt(sp_bench['test_bear'].get(bm_name)),
        '2023':     fmt(sp_bench['test_2023'].get(bm_name)),
        'bull':     fmt(sp_bench['test_bull'].get(bm_name)),
        'wins':     '—',
    })

st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)


# ============================== Takeaways ==============================
st.markdown(f"""
<div style="background:{SURFACE}; border-left:3px solid {PRIMARY}; padding:1rem 1.2rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.55; margin-top:1.5rem;">
<b style="color:{TEXT}">Reading the comparison:</b><br><br>

<b style="color:{TEXT}">1. Each specialist is necessary (v5/v6/v7 prove this).</b>
Dropping any one specialist hurts the period that specialist was trained for.
Drop hunter → bull collapses. Drop defender → bear collapses. Drop steady → 2023 collapses.<br><br>

<b style="color:{TEXT}">2. Learned routing beats rule-based (MAC v3 vs MARS v1).</b>
The trained MLP controller picks better specialist weights than direct HMM-probability
routing, particularly on bull / 2023 periods where attention to vol features helps.<br><br>

<b style="color:{TEXT}">3. Multi-agent vs single-agent (MAC v3 vs Universal v6).</b>
MAC v3 achieves higher bull capture at the cost of larger bear loss — more concentrated
allocations. Universal v6 is more balanced (8/8 wins on DJI under the headline rule).
Both are defensible production paths; the thesis presents both rather than choosing one.
</div>
""", unsafe_allow_html=True)


# ============================== Iteration history (folded) ==============================
with st.expander("Iteration history — older MAC designs (not directly comparable)", expanded=False):
    st.caption(
        "MAC v1, v2, and v4, plus MARS v2 controller-tuning reimplementations, "
        "were internal iterations during MAC development. v1 and v2 used DIFFERENT "
        "observation shapes than v3+; v4 had no clear ablation narrative; MARS v2 "
        "variants were tuning iterations toward v3. Included for completeness."
    )

    history_rows = []
    for folder, label, cat in [
        ('mac_dji_v2', 'MAC v2 (v1c+ws lineup, cross-variant specialists)', 'lineage'),
        ('mac_dji_v4', 'MAC v4 (internal iteration)', 'lineage'),
        ('mac_dji',   'MAC v1 (first multi-agent, old obs shape)', 'lineage'),
    ]:
        v = variant_headline(folder, 'DJI')
        if v:
            v['label'] = label
            history_rows.append(row(v, cat))

    for slug, label in [
        ('MARS_v2_MAC_v1cplusws_lineup_DJI_30', 'MARS v2 (MAC v1c+ws lineup) — DJI'),
        ('MARS_v2_MAC_v6_lineup_DJI_30',        'MARS v2 (MAC v6 lineup) — DJI'),
        ('MARS_v2_MAC_v1cplusws_lineup_SP100',  'MARS v2 (MAC v1c+ws lineup) — SP100'),
        ('MARS_v2_MAC_v6_lineup_SP100',         'MARS v2 (MAC v6 lineup) — SP100'),
    ]:
        v = variant_from_legacy(slug, label)
        if v: history_rows.append(row(v, 'MARS reimpl (tuning)'))

    for folder, label in [
        ('mac_sp100_v2', 'MAC v2 SP100 (v1c+ws lineup)'),
        ('mac_sp100_v4', 'MAC v4 SP100 (internal iteration)'),
        ('mac_sp100',   'MAC v1 SP100 (first multi-agent, old obs shape)'),
    ]:
        v = variant_headline(folder, 'SP100')
        if v:
            v['label'] = label
            history_rows.append(row(v, 'lineage'))

    if history_rows:
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)


# ============================== MARS paper note (collapsed) ==============================
with st.expander("MARS paper (HSI) — architectural inspiration only, NOT a quantitative benchmark", expanded=False):
    st.markdown(f"""
The MARS paper (Chen, Li, Wang — NJIT) reports its headline results on the **HSI universe
(Hang Seng, Hong Kong, ~50 stocks)** — a completely different universe with different
constituents, different market dynamics, and different regimes than DJI 30 or SP100.

```
                                   MARS paper (HSI)              Our scope
─────────────────────────────────────────────────────────────────────────────────
Universe                            HSI ~50 stocks (Hong Kong)    DJI 30 / SP100 90 (US)
Test 2024 bull                      +29.50%  Sharpe 2.84          varies by universe
HSI Index BH (paper baseline)       +15.36%                       DJI BH +12.88% / SP500 BH +24.54%
```

What we DO take from MARS:
  - Architectural pattern: 3 risk-aware specialists + meta-controller
  - Synthesis of regime detection + multi-agent (matches our MAC v3 design)

What we DO NOT do:
  - Compare MAC headline numbers directly against MARS HSI numbers (different universe)
""")
