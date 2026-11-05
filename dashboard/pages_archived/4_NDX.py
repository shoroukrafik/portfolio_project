"""
NASDAQ 100 universe page.

Same template as DJI/SP100, plus NDX-specific experimental section:
  - 2017 cutoff variant (relaxed late-IPO admission)
  - long-momentum extension (Pareto improvement)
  - CVaR bear-defense (negative result)
  - Reward-shaping ablations (exp1-4)
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    apply_theme, badge,
    load_all_experiments, get_legacy_for_universe,
    BENCHMARKS, BENCHMARK_SHARPE, BENCHMARK_MDD, PERIOD_LABELS,
    picks_summary,
    four_period_grid, risk_metrics_grid, period_focus_panel,
    training_curves_chart,
    load_weights_from_legacy, concentration_chart,
    effective_n_chart, average_allocation_chart,
    get_tickers,
)
from utils.theme import TEXT, TEXT_MUTED, TEXT_DIM, SURFACE, BORDER, PRIMARY, SUCCESS, WARNING


st.set_page_config(
    page_title="NASDAQ 100 · DRL Portfolio Thesis",
    page_icon="◆",
    layout="wide",
)
apply_theme()


# ============================== Sidebar rule switch ==============================
with st.sidebar:
    st.markdown(f"<div style='font-size:0.75rem; color:#9ba0b3; letter-spacing:0.1em; "
                f"text-transform:uppercase; margin-top: 1rem;'>Selection rule</div>",
                unsafe_allow_html=True)
    rule_choice = st.radio(
        "Selection rule",
        options=['Defensive (val-Sharpe) — CANONICAL', 'Headline (wins-best) — aggressive'],
        index=0,
        label_visibility='collapsed',
        help="DEFENSIVE = val-Sharpe-best (canonical). HEADLINE = wins-best (aggressive alternative).",
    )
    USE_HEADLINE = rule_choice.startswith('Headline')


# ============================== Header ==============================
st.title("NASDAQ 100")
st.markdown(
    f"<div style='color:{TEXT_MUTED}; font-size:0.95rem; margin-bottom:1.2rem;'>"
    "NASDAQ-100 universe · 90 stocks (2017-cutoff used for canonical) · "
    "Train 2017-2020 · Val 2021 · Tests 2022 / 2023 / 2024."
    "</div>",
    unsafe_allow_html=True
)

st.markdown(f"""
<div style="background:{SURFACE}; border-left:3px solid {WARNING}; padding:0.8rem 1.1rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.85rem;
            line-height:1.5; margin-bottom:1.0rem;">
<b style="color:{TEXT}">Canonical model (Option A — consistent DEFENSIVE+floor rule):</b>
<code>universal_v6_ndx_2017_mom</code> DEFENSIVE pick (step 100,000, val_Sh 1.882) + NO cash floor.
Achieves <b>4/8 wins</b>, val Sharpe 1.73. Cash floor sweep showed any positive floor REDUCES wins
on NDX (val NDX BH tie flips to loss; bull falls below EW NDX).<br><br>
<b style="color:{TEXT}">Aggressive alternative (selection bias):</b>
Joint (checkpoint × floor) search finds <b>step 475k + 70% bear floor → 6/8 wins</b>
with val Sharpe 2.43. Uses wins-based checkpoint selection (NOT val-best). Documented for
completeness as the universe's "upper bound under aggressive selection."<br><br>
<b style="color:{TEXT}">Note on training steps:</b> NDX 2017-cutoff variants use 500k
total PPO steps. The "passes through training data" is ≈500 for 500k×4yr (vs ≈286 for production 1M×14yr).
Val-Sharpe peaks before step 100k then declines — more training would only overfit.
</div>
""", unsafe_allow_html=True)


# ============================== Pull data ==============================
experiments = load_all_experiments()
ndx = experiments.get('NDX', [])

def find_by_folder(folder_name):
    return next((e for e in ndx if e['folder'] == folder_name), None)


prod          = find_by_folder('universal_v6_ndx')
ndx_2017      = find_by_folder('universal_v6_ndx_2017')
ndx_2017_mom  = find_by_folder('universal_v6_ndx_2017_mom')
ndx_2017_cvar = find_by_folder('universal_v6_ndx_2017_mom_bear')
ndx_mom_2m    = find_by_folder('universal_v6_ndx_mom_2m')


def selected_pick(exp):
    if exp is None: return None
    picks = picks_summary(exp['result_dir'], exp.get('total_steps'))
    return picks.get('headline' if USE_HEADLINE else 'defensive')


def headline_period_values(exp):
    if exp is None:
        return {}
    pick = selected_pick(exp)
    if pick:
        return {
            'val':       pick['val_ret'],
            'test_bear': pick['test_bear_ret'],
            'test_2023': pick['test_2023_ret'],
            'test_bull': pick['test_bull_ret'],
        }
    out = {}
    for p in ['val', 'test_bear', 'test_2023', 'test_bull']:
        v = exp['periods'].get(p, {}).get('cumulative_return')
        if v is not None:
            out[p] = v
    return out


def headline_risk_metrics(exp):
    if exp is None: return {}
    pick = selected_pick(exp)
    out = {}
    if pick:
        for p in ['val','test_bear','test_2023','test_bull']:
            out[p] = {'sharpe': pick.get(f'{p}_sharpe'), 'mdd': pick.get(f'{p}_mdd')}
    else:
        for p in ['val','test_bear','test_2023','test_bull']:
            pd = exp.get('periods', {}).get(p, {})
            out[p] = {'sharpe': pd.get('sharpe_ratio'), 'mdd': pd.get('max_drawdown')}
    return out


# ============================== Headline 4-chart grid ==============================
benchmarks_by_period = BENCHMARKS['NDX']

variants_by_period = {'val': [], 'test_bear': [], 'test_2023': [], 'test_bull': []}

for exp, label, kind in [
    (prod,         'Universal v6',                 'production'),
    (ndx_2017,     'v6 + 2017 cutoff',             'experimental'),
    (ndx_2017_mom, 'v6 + 2017 + long-mom ★',       'experimental'),
    (ndx_mom_2m,   'v6 + long-mom (2007, 2M)',     'experimental'),
]:
    if exp is None:
        continue
    values = headline_period_values(exp)
    for period, value in values.items():
        variants_by_period[period].append({
            'label': label,
            'value': value,
            'kind':  kind,
        })

st.markdown("### Headline results vs benchmarks")
st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px;
            padding:0.7rem 1rem; margin-bottom:0.8rem; color:{TEXT_MUTED}; font-size:0.85rem;">
<b style="color:{TEXT}">Selection rules applied:</b><br>
&nbsp;&nbsp;<span style="color:{PRIMARY}">●</span> <b>HEADLINE</b> = max benchmark wins, tied → max sum-Sharpe, step ≥ 20% of training
&nbsp;&nbsp; (used in charts)<br>
&nbsp;&nbsp;<span style="color:{TEXT_MUTED}">●</span> <b>DEFENSIVE</b> = max val-Sharpe with step ≥ 20% (shown in detailed table for comparison)
</div>
""", unsafe_allow_html=True)
st.caption(
    "NDX BH = NASDAQ-100 buy-and-hold; EW NDX = equal-weight 68 (production) / 90 (2017-cutoff) stocks. "
    "Note: all variants cluster around val ~27% because val is the SELECTION criterion — "
    "tests reveal where they actually differ."
)

figs = four_period_grid(
    title="NDX",
    benchmarks_by_period=benchmarks_by_period,
    variants_by_period=variants_by_period,
    primary_benchmark='NDX BH',
)

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(figs['val'],       use_container_width=True)
    st.plotly_chart(figs['test_2023'], use_container_width=True)
with col_b:
    st.plotly_chart(figs['test_bear'], use_container_width=True)
    st.plotly_chart(figs['test_bull'], use_container_width=True)


# ============================== Caption ==============================
st.markdown(f"""
<div style="background:{SURFACE}; border-left:3px solid {PRIMARY}; padding:1rem 1.2rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.92rem;
            line-height:1.55; margin:1rem 0;">
<b style="color:{TEXT}">Reading the NDX story:</b><br>
The canonical model — <b>universal_v6_ndx_2017_mom</b> DEFENSIVE checkpoint (step 100,000) — achieves
<b>4/8 wins</b> with val Sharpe 1.73. The headline NDX finding is the <b style="color:{SUCCESS}">+18.95
percentage-point defensive edge against NDX BH on bear 2022</b> (−12.16% vs −33.71%) — the largest
defensive outperformance in the project. The 4/8 limit reflects a STRUCTURAL universe-membership
ceiling, not a model failure: NDX BH is cap-weighted with NVDA at ~7% (+171% in 2024), and a long-only
diversified strategy with bounded single-stock weights cannot match a cap-weighted benchmark
dominated by a +171% stock. Against the FAIR diversified comparator (EW NDX), the canonical model
wins on bear AND bull. Cash-floor sweep confirmed any positive floor REDUCES NDX wins (the val 2021
NDX BH tie flips to a loss at any floor &gt; 0), so the canonical uses no overlay. The CVaR
bear-defense experiment was REJECTED — see expander below.
</div>
""", unsafe_allow_html=True)


# ============================== Bull 2024 focus panel ==============================
st.markdown("### Bull 2024 — focused analysis (return + Sharpe + MDD)")
st.caption("NDX bull 2024 was the AI rally year. Compare variants vs benchmarks on all three risk-adjusted metrics.")

bull_return_bars = [{'label': bm, 'value': v, 'kind': 'benchmark'} for bm, v in BENCHMARKS['NDX']['test_bull'].items()]
bull_sharpe_bars = [{'label': bm, 'value': v, 'kind': 'benchmark'} for bm, v in BENCHMARK_SHARPE['NDX']['test_bull'].items()]
bull_mdd_bars    = [{'label': bm, 'value': v, 'kind': 'benchmark'} for bm, v in BENCHMARK_MDD['NDX']['test_bull'].items()]

for exp, label, kind in [
    (prod,         'Universal v6',           'production'),
    (ndx_2017,     'v6 + 2017 cutoff',       'experimental'),
    (ndx_2017_mom, 'v6 + 2017 + long-mom ★', 'experimental'),
    (ndx_2017_cvar,'v6 + 2017 + CVaR',       'ablation'),
]:
    if exp is None: continue
    vals = headline_period_values(exp)
    rm = headline_risk_metrics(exp)
    if vals.get('test_bull') is not None:
        bull_return_bars.append({'label': label, 'value': vals['test_bull'], 'kind': kind})
    if rm.get('test_bull', {}).get('sharpe') is not None:
        bull_sharpe_bars.append({'label': label, 'value': rm['test_bull']['sharpe'], 'kind': kind})
    if rm.get('test_bull', {}).get('mdd') is not None:
        bull_mdd_bars.append({'label': label, 'value': rm['test_bull']['mdd'], 'kind': kind})

bull_ret_fig, bull_sh_fig, bull_mdd_fig = period_focus_panel(
    period_label='Bull 2024', return_bars=bull_return_bars,
    sharpe_bars=bull_sharpe_bars, mdd_bars=bull_mdd_bars,
)
c1, c2, c3 = st.columns(3)
with c1: st.plotly_chart(bull_ret_fig, use_container_width=True)
with c2: st.plotly_chart(bull_sh_fig,  use_container_width=True)
with c3: st.plotly_chart(bull_mdd_fig, use_container_width=True)


# ============================== Why NDX 2023 is the extreme outlier ==============================
st.markdown("### Why NDX 2023 is the extreme outlier (the mega-cap concentration gap)")
st.markdown(f"""
<div style="background:{SURFACE}; border-left:3px solid {WARNING}; padding:1rem 1.2rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1.5rem;">
<b style="color:{TEXT}">All NDX variants miss ^NDX 2023 (+54.90%) by 20-30pp. This is structural.</b><br><br>
^NDX 2023 was the "AI rally" year driven by EXTREME mega-cap concentration. The top 6
constituents contributed roughly +63pp of the +54.9% total:<br><br>
<code style="color:{TEXT}">
&nbsp; NVDA &nbsp;+239% × 15% NDX cap = +36pp toward ^NDX<br>
&nbsp; META &nbsp;+194% ×  5% NDX cap = +10pp<br>
&nbsp; AAPL &nbsp; +49% × 12% NDX cap =  +6pp<br>
&nbsp; AMZN &nbsp; +81% ×  5% NDX cap =  +4pp<br>
&nbsp; GOOG &nbsp; +58% ×  6% NDX cap =  +3.5pp<br>
&nbsp; TSLA &nbsp;+101% ×  4% NDX cap =  +4pp
</code><br><br>
^NDX achieves this through <b>market-cap weighting</b> — NVDA at 15% portfolio weight,
not 1.5%. Our attention-based extractor diversifies allocations across all 68-90 stocks,
typically capping any single position around 2-3%. NVDA at 2% × +239% contributes only
~5pp instead of ~36pp. Multiply this across 6 top stocks and the agent recovers ~25-30pp
of the +54.9% — exactly what we observe (+24 to +31% across variants).<br><br>
<b style="color:{TEXT}">The gap is not a model failure — it's the inherent tension between
diversified DRL allocation and extreme-concentration benchmarks.</b> Closing the gap
would require: market-cap-weighting (defeats DRL's purpose), forward-looking signals
(out of scope), or heavy concentration bonus (we tested exp2 — no effect).
The Equal-Weight NDX benchmark (+37% in 2023) is closer to what diversified strategies
CAN achieve, and shows our agent's gap shrinks from -24pp (vs ^NDX) to +0 to -8pp (vs EW NDX).
</div>
""", unsafe_allow_html=True)


# ============================== Training gradient view ==============================
st.markdown("### Training trajectory — validation Sharpe over training steps")
st.caption(
    "Each line is one variant. ★ STAR = HEADLINE pick (max wins + sum-Sharpe, step ≥ 20%). "
    "◇ DIAMOND = DEFENSIVE pick (max val-Sharpe, step ≥ 20%). "
    "NDX 2017 variants use 500k steps (4-year window — different x-axis range). "
    "Rapid early decline visible on 2017 variants = small-data overfitting."
)

curve_variants = []
for exp, label in [
    (prod,           'Universal v6 prod'),
    (ndx_2017,       'v6 + 2017 cutoff'),
    (ndx_2017_mom,   'v6 + 2017 + long-mom ★'),
    (ndx_2017_cvar,  'v6 + 2017 + CVaR'),
]:
    if exp is None: continue
    p = picks_summary(exp['result_dir'], exp.get('total_steps'))
    curve_variants.append({
        'label': label,
        'result_dir': exp['result_dir'],
        'picks': p,
    })

if curve_variants:
    fig = training_curves_chart(curve_variants, title="NDX universe — training trajectories")
    st.plotly_chart(fig, use_container_width=True)


# ============================== Risk-adjusted metrics ==============================
st.markdown("### Risk-adjusted view — Sharpe and MDD per period")
st.caption(
    "Returns alone don't tell the full story. Sharpe ratio (higher = better risk-adjusted "
    "return) and max drawdown (less negative = better tail risk) per variant."
)

sharpe_by_period = {p: [] for p in ['val','test_bear','test_2023','test_bull']}
mdd_by_period    = {p: [] for p in ['val','test_bear','test_2023','test_bull']}

# Benchmarks first
for p in ['val','test_bear','test_2023','test_bull']:
    for bm_name, sh in BENCHMARK_SHARPE['NDX'].get(p, {}).items():
        sharpe_by_period[p].append({'label': bm_name, 'value': sh, 'kind': 'benchmark'})
    for bm_name, mdd in BENCHMARK_MDD['NDX'].get(p, {}).items():
        mdd_by_period[p].append({'label': bm_name, 'value': mdd, 'kind': 'benchmark'})

for exp, label, kind in [
    (prod,         'Universal v6',           'production'),
    (ndx_2017,     'v6 + 2017 cutoff',       'experimental'),
    (ndx_2017_mom, 'v6 + 2017 + long-mom ★', 'experimental'),
    (ndx_2017_cvar,'v6 + 2017 + CVaR',       'ablation'),
]:
    if exp is None: continue
    rm = headline_risk_metrics(exp)
    for p in ['val','test_bear','test_2023','test_bull']:
        if rm.get(p, {}).get('sharpe') is not None:
            sharpe_by_period[p].append({'label': label, 'value': rm[p]['sharpe'], 'kind': kind})
        if rm.get(p, {}).get('mdd') is not None:
            mdd_by_period[p].append({'label': label, 'value': rm[p]['mdd'], 'kind': kind})

sharpe_figs, mdd_figs = risk_metrics_grid(sharpe_by_period, mdd_by_period)

st.markdown("**Sharpe ratio per period** (higher is better)")
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(sharpe_figs['val'],       use_container_width=True)
    st.plotly_chart(sharpe_figs['test_2023'], use_container_width=True)
with col_b:
    st.plotly_chart(sharpe_figs['test_bear'], use_container_width=True)
    st.plotly_chart(sharpe_figs['test_bull'], use_container_width=True)

st.markdown("**Max drawdown per period** (less negative is better)")
col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(mdd_figs['val'],       use_container_width=True)
    st.plotly_chart(mdd_figs['test_2023'], use_container_width=True)
with col_d:
    st.plotly_chart(mdd_figs['test_bear'], use_container_width=True)
    st.plotly_chart(mdd_figs['test_bull'], use_container_width=True)


# ============================== Stock concentration view ==============================
st.markdown("### Stock concentration over time")
st.caption(
    "Per-day portfolio concentration: Top-5 weight, effective number of stocks (1/HHI), "
    "and top-15 stocks by average allocation. Weights computed by re-running the headline "
    "production model through each period. Variants with no weights data show a placeholder."
)

# Variant selector — which model to inspect
ndx_variant_choice = st.selectbox(
    "Variant for concentration view",
    options=[
        ('universal_v6_ndx',           'Universal v6 (production)',   'NDX'),
        ('universal_v6_ndx_2017_mom',  'v6 + 2017 + long-mom ★',      'NDX_2017'),
    ],
    format_func=lambda t: t[1],
    key='ndx_conc_variant',
)
ndx_conc_folder, ndx_conc_label, ndx_conc_ticker_key = ndx_variant_choice

ndx_conc_period = st.selectbox(
    "Period for concentration view",
    options=['val', 'test_bear', 'test_2023', 'test_bull'],
    format_func=lambda p: PERIOD_LABELS.get(p, p),
    index=3,
    key='ndx_conc_period',
)

ndx_tickers = get_tickers(ndx_conc_ticker_key)
ndx_conc_records = load_weights_from_legacy(ndx_conc_folder, ndx_conc_period)

if ndx_conc_records and ndx_tickers:
    period_label = PERIOD_LABELS.get(ndx_conc_period, ndx_conc_period)
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            concentration_chart(ndx_conc_records, ndx_conc_label, period_label,
                                 n_stocks_in_universe=len(ndx_tickers)),
            use_container_width=True,
        )
        st.plotly_chart(
            effective_n_chart(ndx_conc_records, ndx_conc_label, period_label,
                               n_stocks_in_universe=len(ndx_tickers)),
            use_container_width=True,
        )
    with col_r:
        st.plotly_chart(
            average_allocation_chart(ndx_conc_records, ndx_conc_label, period_label,
                                      tickers=ndx_tickers, top_n=15),
            use_container_width=True,
        )
else:
    st.info(
        f"Concentration data not yet computed for {ndx_conc_label}. "
        f"Run: python3 models/compute_weights_for_dashboard.py --variant {ndx_conc_folder.replace('universal_v6_', '')}"
    )


# ============================== Experimental sandbox ==============================
st.markdown("### NDX experimental sandbox (extra trials)")
st.caption("NDX received additional experimental trials beyond DJI/SP100 to explore "
           "the dispersion-bull-capture limit. Documented for completeness.")

with st.expander("CVaR bear-defense experiment (negative result)", expanded=False):
    if ndx_2017_cvar:
        v = headline_period_values(ndx_2017_cvar)
        st.markdown(f"""
**Hypothesis:** CVaR penalty (tail-risk-aware reward shaping) would asymmetrically
improve bear defense without hurting bull capture.

**Result:** REJECTED. CVaR trades bull for bear at ~2:1 ratio (lose 2pp bull per
1pp bear gain). Not asymmetric — because the 60-day CVaR window catches NVDA's
high realised volatility during bull rallies, penalizing concentration on it.

**Numbers (headline pick):**
- val {v.get('val','-')}  bear {v.get('test_bear','-')}  2023 {v.get('test_2023','-')}  bull {v.get('test_bull','-')}

**Best NDX experimental pick remains:** v6 + 2017 + long-momentum (no CVaR).
""")

with st.expander("Long-mom on full 2007-2020 window (negative result)", expanded=False):
    if ndx_mom_2m:
        v = headline_period_values(ndx_mom_2m)
        st.markdown(f"""
**Hypothesis:** Long-momentum features that worked on SP100 (2M HEADLINE 11/12 wins)
would also help NDX when trained on the same full 2007-2020 window. Tests whether
the dispersion-gradient finding generalizes to the highest-dispersion universe.

**Result:** REJECTED. Val and recovery (2023) improve modestly, but the model
loses bull capture almost entirely (≈0%) and degrades bear defense. The HEADLINE
pick lands at step 1.975M (very late in training).

**Numbers (HEADLINE pick):**
- val {v.get('val', 0):+.2f}%   bear {v.get('test_bear', 0):+.2f}%   2023 {v.get('test_2023', 0):+.2f}%   bull {v.get('test_bull', 0):+.2f}%

**Interpretation:** Long-mom features calibrated on 2007-2020 NDX tickers do not
transfer to 2024 NDX (NVDA-driven mega-cap rally). The 2007-2020 distribution's
"high momentum + high vol" stocks teach the model to AVOID exactly the kind of
asset that drove 2024's bull. On SP100 the same recipe succeeded because SP100's
composition is more stable across decades.

**Implication for thesis:** Universe-membership stability is a precondition for
long-mom features to help — dispersion alone is insufficient. The 2017-cutoff NDX
variant (4-year window, recent composition) remains the correct NDX recipe.
""")
    else:
        st.caption("NDX 2007+long-mom 2M variant data not loaded.")

with st.expander("Reward-shaping ablations exp1-4 (no effect)", expanded=False):
    st.caption(
        "Four reward-shaping ablations tested on NDX production (2007-cutoff) "
        "showed no meaningful effect — all cluster around production v6 baseline. "
        "Confirms reward-shape isn't the bottleneck; training data + universe are."
    )
    import pandas as pd
    rows = []
    for folder_name, label in [
        ('universal_v6_ndx_exp1', 'exp1 — lower bull λ (0.1 → 0.01)'),
        ('universal_v6_ndx_exp2', 'exp2 — top-5 concentration bonus'),
        ('universal_v6_ndx_exp3', 'exp3 — log-return reward (vs DSR)'),
        ('universal_v6_ndx_exp4', 'exp4 — low entropy coefficient'),
    ]:
        e = find_by_folder(folder_name)
        if not e:
            continue
        v = headline_period_values(e)
        rows.append({
            'Ablation': label,
            'val':  f"{v.get('val', 0):+.2f}",
            'bear': f"{v.get('test_bear', 0):+.2f}",
            '2023': f"{v.get('test_2023', 0):+.2f}",
            'bull': f"{v.get('test_bull', 0):+.2f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================== Detailed metrics ==============================
with st.expander("Detailed metrics table — every variant under both selection rules", expanded=False):
    import pandas as pd

    st.caption(
        "Headline = wins-best + sum-Sharpe tiebreaker, step ≥ 20% of training. "
        "Defensive = val-Sharpe-best, step ≥ 20% of training. "
        "Reward-shaping ablations (exp1-4) are kept under the 'exp' expander above."
    )

    rows = []
    for exp, name in [
        (prod,           'Universal v6 (production, 2007-cutoff)'),
        (ndx_2017,       'v6 + 2017 cutoff (relaxed universe)'),
        (ndx_2017_mom,   'v6 + 2017 + long-momentum ★'),
        (ndx_2017_cvar,  'v6 + 2017 + long-mom + CVaR (rejected)'),
    ]:
        if not exp:
            continue
        picks = picks_summary(exp['result_dir'], exp.get('total_steps'))
        h = picks.get('headline')
        d = picks.get('defensive')
        if h:
            rows.append({
                'Variant': name, 'Rule': 'Headline',
                'Step':    f"{h['step']:,}",
                'val':     f"{h['val_ret']:+.2f}",
                'bear':    f"{h['test_bear_ret']:+.2f}",
                '2023':    f"{h['test_2023_ret']:+.2f}",
                'bull':    f"{h['test_bull_ret']:+.2f}",
                'vSh':     f"{h['val_sharpe']:.2f}",
                'wins':    (f"{h.get('wins')}/{h.get('total')}" if h.get('total') else (f"{h.get('wins')}" if h.get('wins') is not None else '-')),
            })
        if d and (not h or d['step'] != h['step']):
            rows.append({
                'Variant': '', 'Rule': 'Defensive',
                'Step':    f"{d['step']:,}",
                'val':     f"{d['val_ret']:+.2f}",
                'bear':    f"{d['test_bear_ret']:+.2f}",
                '2023':    f"{d['test_2023_ret']:+.2f}",
                'bull':    f"{d['test_bull_ret']:+.2f}",
                'vSh':     f"{d['val_sharpe']:.2f}", 'wins': '-',
            })
        if not h and not d:
            p = exp.get('periods', {})
            def n(k):
                v = p.get(k, {}).get('cumulative_return')
                return f"{v:+.2f}" if v is not None else '-'
            rows.append({
                'Variant': name, 'Rule': 'Saved best',
                'Step':    f"{exp.get('best_step', '-'):,}" if exp.get('best_step') else '-',
                'val':     n('val'),
                'bear':    n('test_bear'),
                '2023':    n('test_2023'),
                'bull':    n('test_bull'),
                'vSh':     f"{exp.get('best_val_sharpe', 0):.2f}" if exp.get('best_val_sharpe') else '-',
                'wins':    'n/a',
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================== Note on NDX scope ==============================
st.markdown(f"""
<div style="background:{SURFACE}; border-left:3px solid {WARNING}; padding:0.8rem 1.1rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.85rem;
            line-height:1.5; margin-top:1.5rem;">
<b style="color:{TEXT}">Note on NDX scope:</b> NDX joined the project after universal v6 was
established, so it does not have the full lineage of Phase-A architecture ablations — those were
settled on DJI and ported to NDX via the universal recipe. NDX-specific experimental trials
(2017-cutoff, long-momentum, CVaR bear-defense, reward-shape exp1-4) explored extensions of the
universal v6 recipe. The 2017-cutoff with long-momentum at 500k steps (DEFENSIVE step 100k) is
the canonical NDX choice.
</div>
""", unsafe_allow_html=True)
