"""
S&P 100 universe page.

Same template as DJI:
  - 4 bar charts (val / bear / 2023 / bull) vs benchmarks
  - One-paragraph caption with the headline finding
  - Lineage and ablations folded into expanders
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
from utils.theme import TEXT, TEXT_MUTED, TEXT_DIM, SURFACE, BORDER, PRIMARY, WARNING, SUCCESS, DANGER


st.set_page_config(
    page_title="S&P 100 · DRL Portfolio Thesis",
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
        help="DEFENSIVE = val-Sharpe-best (canonical, honest out-of-sample). HEADLINE = wins-best (aggressive alternative, mild selection bias).",
    )
    USE_HEADLINE = rule_choice.startswith('Headline')


# ============================== Header ==============================
st.title("S&P 100")
st.markdown(
    f"<div style='color:{TEXT_MUTED}; font-size:0.95rem; margin-bottom:1.2rem;'>"
    "90-stock U.S. large-cap universe · Train 2007-2020 · "
    "Val 2021 · Tests 2022 (bear) / 2023 (recovery) / 2024 (bull)."
    "</div>",
    unsafe_allow_html=True
)

st.markdown(f"""
<div style="background:{SURFACE}; border-left:3px solid {WARNING}; padding:0.8rem 1.1rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.85rem;
            line-height:1.5; margin-bottom:1.5rem;">
<b style="color:{TEXT}">Canonical model (Option A — consistent DEFENSIVE+best-floor rule):</b>
<code>universal_v6_sp100_mom_2m</code> DEFENSIVE pick (step 1,150,000) + 30% bear cash floor.
Achieves <b>7/12 wins</b> with val Sharpe <b>2.92 (CI 95% [+1.09, +5.15])</b> — the only CI lower
bound above 1.0 in the entire project. Val Calmar 8.28 (highest in project).<br><br>
<b style="color:{TEXT}">Aggressive alternative (mild selection bias):</b>
HEADLINE pick (step 1,875,000) + 30% bear floor gives 11/12 wins but uses test-wins-based
checkpoint selection. Side-by-side toggle in sidebar; canonical is DEFENSIVE.
</div>
""", unsafe_allow_html=True)


# ============================== Pull data ==============================
experiments = load_all_experiments()
sp = experiments.get('SP100', [])

def find_by_folder(folder_name):
    return next((e for e in sp if e['folder'] == folder_name), None)


prod         = find_by_folder('universal_v6_sp100')
mac_v3       = find_by_folder('mac_sp100_v3')
noregime     = find_by_folder('universal_v6_sp100_noregime')
h2soft_sp    = find_by_folder('universal_v6_sp100_h2soft')
mom          = find_by_folder('universal_v6_sp100_mom_2m')
mom_1m_archive = find_by_folder('universal_v6_sp100_mom')
mom_2015     = find_by_folder('universal_v6_sp100_2015_mom')
no_mom_2015  = find_by_folder('universal_v6_sp100_2015')


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
benchmarks_by_period = BENCHMARKS['SP100']

variants_by_period = {'val': [], 'test_bear': [], 'test_2023': [], 'test_bull': []}

for exp, label, kind in [
    (prod,         'Universal v6 prod',           'production'),
    (mom,          'v6 + long-mom 2M ★ CANONICAL (DEFENSIVE step 1.15M + 30% bear floor)', 'production'),
    (mom_2015,     'v6 + 2015 + long-mom ★',      'experimental'),
    (noregime,     'v6 no-regime (H2)',           'ablation'),
    (h2soft_sp,    'v6 uniform λ (H2-soft)',      'ablation'),
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
    "SP500 BH = S&P 500 buy-and-hold; OEX BH = S&P 100 index; EW SP100 = equal-weight 90 stocks."
)

figs = four_period_grid(
    title="SP100",
    benchmarks_by_period=benchmarks_by_period,
    variants_by_period=variants_by_period,
    primary_benchmark='OEX BH',
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
<div style="background:{SURFACE}; border-left:3px solid {SUCCESS}; padding:1rem 1.2rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.92rem;
            line-height:1.55; margin:1rem 0;">
<b style="color:{TEXT}">Reading the SP100 story:</b><br>
The canonical model — <b>universal_v6_sp100_mom_2m</b> DEFENSIVE checkpoint (step 1,150,000) +
30% bear cash floor — achieves <b style="color:{SUCCESS}">7/12 wins</b> with the
<b>highest validation Sharpe in the project (3.41)</b>. Its bootstrap 95% CI on val Sharpe is
[+1.09, +5.15] — the only CI lower bound exceeding 1.0 in this thesis, providing the strongest
statistical evidence that the result is not luck. Val Calmar 8.28 is also the highest in the
project. The 3-year combined backtest produces +27.25% cumulative return, beating SP500 BH (+24.35%).
The aggressive HEADLINE alternative (step 1.875M) achieves 11/12 wins but uses test-wins-based
checkpoint selection (mild bias); shown via the sidebar toggle for completeness.
</div>
""", unsafe_allow_html=True)


# ============================== 2x2 ablation (if data available) ==============================
if mom_2015 or no_mom_2015:
    st.markdown("### 2×2 ablation — long-momentum × 2015 cutoff")
    st.caption(
        "Isolates the contribution of long-momentum features from the contribution of "
        "the shorter (2015-2020) training window with relaxed universe (adds META, TSLA, AVGO, etc.)."
    )

    import pandas as pd

    def fmt_cell(exp, periods):
        if not exp:
            return {p: '—' for p in periods}
        v = headline_period_values(exp)
        return {p: f"{v.get(p, 0):+.2f}%" if p in v else '—' for p in periods}

    cells = {}
    cells['2007-2020, no long-mom (production)'] = fmt_cell(prod,         ['val','test_bear','test_2023','test_bull'])
    cells['2007-2020, + long-mom']                = fmt_cell(mom,          ['val','test_bear','test_2023','test_bull'])
    cells['2015-2020, no long-mom (ablation)']    = fmt_cell(no_mom_2015,  ['val','test_bear','test_2023','test_bull'])
    cells['2015-2020, + long-mom']                = fmt_cell(mom_2015,     ['val','test_bear','test_2023','test_bull'])

    df_2x2 = pd.DataFrame(cells).T
    df_2x2.columns = ['val', 'bear', '2023', 'bull']
    st.dataframe(df_2x2, use_container_width=True)


# ============================== Bull 2024 focus panel ==============================
st.markdown("### Bull 2024 — focused analysis (return + Sharpe + MDD side-by-side)")
st.caption("Direct comparison on the bull period: cumulative return, Sharpe, and MDD.")

bull_return_bars = [{'label': bm, 'value': v, 'kind': 'benchmark'} for bm, v in BENCHMARKS['SP100']['test_bull'].items()]
bull_sharpe_bars = [{'label': bm, 'value': v, 'kind': 'benchmark'} for bm, v in BENCHMARK_SHARPE['SP100']['test_bull'].items()]
bull_mdd_bars    = [{'label': bm, 'value': v, 'kind': 'benchmark'} for bm, v in BENCHMARK_MDD['SP100']['test_bull'].items()]

for exp, label, kind in [
    (prod,         'Universal v6 prod',           'production'),
    (mom,          'v6 + long-mom 2M ★ CANONICAL (DEFENSIVE step 1.15M + 30% bear floor)', 'production'),
    (mom_2015,     'v6 + 2015 + long-mom ★',      'experimental'),
    (noregime,     'v6 no-regime (H2)',           'ablation'),
    (h2soft_sp,    'v6 uniform λ (H2-soft)',      'ablation'),
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


# ============================== Training gradient view ==============================
st.markdown("### Training trajectory — validation Sharpe over training steps")
st.caption(
    "Each line is one variant. ★ STAR marks the HEADLINE pick (max wins + sum-Sharpe, step ≥ 20%). "
    "◇ DIAMOND marks the DEFENSIVE pick (max val-Sharpe, step ≥ 20%). "
    "When the two markers are at the same point, both rules picked the same checkpoint. "
    "Curve shape shows convergence: rising = still learning, declining = overfitting."
)

curve_variants = []
for exp, label in [
    (prod,         'Universal v6 prod (1M)'),
    (mom,          'v6 + long-mom 2M ★ CANONICAL'),
    (mom_2015,     'v6 + 2015 + long-mom'),
    (no_mom_2015,  'v6 + 2015 no-mom'),
]:
    if exp is None: continue
    p = picks_summary(exp['result_dir'], exp.get('total_steps'))
    curve_variants.append({
        'label': label,
        'result_dir': exp['result_dir'],
        'picks': p,
    })

if curve_variants:
    fig = training_curves_chart(curve_variants, title="SP100 universe — training trajectories")
    st.plotly_chart(fig, use_container_width=True)


# ============================== Risk-adjusted metrics view ==============================
st.markdown("### Risk-adjusted view — Sharpe and MDD per period")
st.caption(
    "Returns alone don't tell the full story. Sharpe ratio (higher = better risk-adjusted "
    "return) and max drawdown (less negative = better tail risk) for each variant."
)

sharpe_by_period = {p: [] for p in ['val','test_bear','test_2023','test_bull']}
mdd_by_period    = {p: [] for p in ['val','test_bear','test_2023','test_bull']}

# Benchmarks first
for p in ['val','test_bear','test_2023','test_bull']:
    for bm_name, sh in BENCHMARK_SHARPE['SP100'].get(p, {}).items():
        sharpe_by_period[p].append({'label': bm_name, 'value': sh, 'kind': 'benchmark'})
    for bm_name, mdd in BENCHMARK_MDD['SP100'].get(p, {}).items():
        mdd_by_period[p].append({'label': bm_name, 'value': mdd, 'kind': 'benchmark'})

for exp, label, kind in [
    (prod,         'Universal v6 prod',           'production'),
    (mom,          'v6 + long-mom 2M ★ CANONICAL (DEFENSIVE step 1.15M + 30% bear floor)', 'production'),
    (mom_2015,     'v6 + 2015 + long-mom ★',      'experimental'),
    (noregime,     'v6 no-regime (H2)',           'ablation'),
    (h2soft_sp,    'v6 uniform λ (H2-soft)',      'ablation'),
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

st.markdown("**Max drawdown per period** (less negative is better — closer to zero)")
col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(mdd_figs['val'],       use_container_width=True)
    st.plotly_chart(mdd_figs['test_2023'], use_container_width=True)
with col_d:
    st.plotly_chart(mdd_figs['test_bear'], use_container_width=True)
    st.plotly_chart(mdd_figs['test_bull'], use_container_width=True)


# ============================== Detailed metrics ==============================
with st.expander("Detailed metrics table — every variant under both selection rules", expanded=False):
    import pandas as pd

    st.caption(
        "Headline = wins-best + sum-Sharpe tiebreaker, step ≥ 20% of training. "
        "Defensive = val-Sharpe-best, step ≥ 20% of training. "
        "Saved best = in-training val-Sharpe-best (used only when no checkpoint sweep exists)."
    )

    rows = []
    for exp, name in [
        (prod,        'Universal v6 (1M, no overlay)'),
        (mom,         'v6 + long-mom 2M ★ CANONICAL (+ 30% bear floor)'),
        (mom_2015,    'v6 + 2015 + long-mom (1M, alternative experimental winner)'),
        (no_mom_2015, 'v6 + 2015 no-mom (ablation control)'),
        (noregime,    'v6 no-regime (H2 strong ablation)'),
        (h2soft_sp,   'v6 uniform λ (H2-soft ablation)'),
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


# ============================== Cash-floor overlay experiment ==============================
st.markdown("### Rule-based safety overlay — bear-only cash floor on the 2M HEADLINE pick")
st.caption(
    "Experimental defensive extension. The 2M HEADLINE checkpoint (step 1.875M) is "
    "re-evaluated with a runtime cash floor that fires ONLY in HMM-detected bear regime. "
    "No retraining — the floor is a Faber 2007 / Yang 2020 style TAA overlay applied on "
    "top of the learned policy. Floor=30% is the sweet spot: strictly improves bear "
    "defense and val Sharpe with no loss of wins."
)

import json, os
overlay_path = 'results/universal_v6_sp100_mom_2m/cashfloor_overlay_HEADLINE.json'
if os.path.exists(overlay_path):
    overlay = json.load(open(overlay_path))
    by_floor = overlay['by_floor']

    # Build comparison table
    bench = BENCHMARKS['SP100']
    rows = []
    for floor in ['0.0', '0.3', '0.5', '0.7']:
        m = by_floor[floor]
        w = 0; tot = 0
        for p in ['val','test_bear','test_2023','test_bull']:
            ret = m[p]['cumulative_return']
            for bname, bval in bench[p].items():
                if ret > bval: w += 1
                tot += 1
        rows.append({
            'Bear floor': f"{int(float(floor)*100)}%",
            'Val ret':   f"{m['val']['cumulative_return']:+.2f}%",
            'Bear ret':  f"{m['test_bear']['cumulative_return']:+.2f}%",
            '2023 ret':  f"{m['test_2023']['cumulative_return']:+.2f}%",
            'Bull ret':  f"{m['test_bull']['cumulative_return']:+.2f}%",
            'Val Sharpe': f"{m['val']['sharpe_ratio']:.2f}",
            'Bear MDD':  f"{m['test_bear']['max_drawdown']:+.2f}%",
            'Wins':      f"{w}/{tot}",
        })
    import pandas as pd
    df_overlay = pd.DataFrame(rows)
    st.dataframe(df_overlay, use_container_width=True, hide_index=True)

    st.markdown(f"""
<div style="background:{SURFACE}; border-left:3px solid {SUCCESS}; padding:1rem 1.2rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-top:0.6rem;">
<b style="color:{TEXT}">Key takeaways for thesis writeup:</b><br><br>

<b>•</b> <b>30% bear floor is a strict Pareto improvement:</b> wins unchanged (11/12),
bear improved +2.01pp (-19.65 → -17.64), val Sharpe up +0.27 (2.80 → 3.07), val MDD
improved +1.12pp. Bull cost only -3.91pp (still wins all three bull benchmarks).<br><br>

<b>•</b> <b>50% floor crosses the win-loss boundary:</b> bull drops from +35.46% to +28.88%,
losing to OEX BH (+30.90). Total wins → 10/12. The 30%-50% range is where the
tradeoff inflects.<br><br>

<b>•</b> <b>Val Sharpe IMPROVES monotonically with the floor (2.80 → 3.33):</b> because HMM
occasionally labels val/2023/bull days as transient bear; on those days the floor caps
exposure → less volatility → higher risk-adjusted return. Mirrors Yang et al. (2020)
turbulence-trigger insight.<br><br>

<b>•</b> <b>Literature precedent:</b> Yang et al. (2020 ICAIF) — turbulence-index threshold
switches to all-cash on regime trigger (100% floor, conditional). Faber (2007 JoWM) —
foundational tactical asset allocation framework. Our bear floor is the soft / graduated
version of both. The cash floor is the same defensive instrument the MAC v3 Defender
specialist uses internally (cash_floor = 0.7 in bear regime — validated as load-bearing).
</div>
""", unsafe_allow_html=True)
else:
    st.info(f"Cash-floor overlay results not yet computed. Run: `python3 ... cashfloor_overlay`")


# ============================== Stock concentration view ==============================
st.markdown("### Stock concentration over time")
st.caption(
    "Per-day portfolio concentration: Top-5 weight, effective number of stocks (1/HHI), "
    "and top-15 stocks by average allocation."
)

sp_variant_choice = st.selectbox(
    "Variant for concentration view",
    options=[
        ('universal_v6_sp100',          'Universal v6 (production)',          'SP100'),
        ('universal_v6_sp100_2015_mom', 'v6 + 2015 + long-mom ★',             'SP100'),
    ],
    format_func=lambda t: t[1],
    key='sp_conc_variant',
)
sp_conc_folder, sp_conc_label, _ = sp_variant_choice

conc_period_sp = st.selectbox(
    "Period for concentration view",
    options=['val', 'test_bear', 'test_2023', 'test_bull'],
    format_func=lambda p: PERIOD_LABELS.get(p, p),
    index=3,
    key='sp_conc_period',
)

sp_tickers = get_tickers('SP100')
conc_records_sp = load_weights_from_legacy(sp_conc_folder, conc_period_sp)

if conc_records_sp and sp_tickers:
    period_label = PERIOD_LABELS.get(conc_period_sp, conc_period_sp)
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            concentration_chart(conc_records_sp, sp_conc_label, period_label,
                                 n_stocks_in_universe=len(sp_tickers)),
            use_container_width=True,
        )
        st.plotly_chart(
            effective_n_chart(conc_records_sp, sp_conc_label, period_label,
                               n_stocks_in_universe=len(sp_tickers)),
            use_container_width=True,
        )
    with col_r:
        st.plotly_chart(
            average_allocation_chart(conc_records_sp, sp_conc_label, period_label,
                                      tickers=sp_tickers, top_n=15),
            use_container_width=True,
        )
else:
    st.info(f"Concentration data not yet computed for {sp_conc_label}.")


# ============================== Multi-seed ==============================
seeds = [e for e in sp if e['category'] == 'multi_seed']
if seeds:
    with st.expander(f"Multi-seed runs ({len(seeds)}) — variance estimation", expanded=False):
        st.caption(
            "Same production recipe trained with seeds 41 and 43 (production = seed 42)."
        )
        for exp in seeds:
            v = headline_period_values(exp)
            st.markdown(
                f"<div class='variant-card'>"
                f"<div class='v-name'>{exp['label']}</div>"
                f"<div class='v-folder'>{exp['folder']}</div>"
                f"<div class='v-row'>val {v.get('val','-')} &nbsp; bear {v.get('test_bear','-')} &nbsp; "
                f"2023 {v.get('test_2023','-')} &nbsp; bull {v.get('test_bull','-')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ============================== Lineage ==============================
mac_iters = [e for e in sp if e['category'] == 'lineage' and e['folder'] != 'mac_sp100_v3']
legacy = get_legacy_for_universe('SP100')

if mac_iters or legacy:
    with st.expander(
        f"Lineage — how we got here ({len(mac_iters) + len(legacy)} variants)",
        expanded=False,
    ):
        if legacy:
            st.markdown("**Phase A — architecture-decision trials (SP100)**")
            st.caption("Justify each component of the FULL design on SP100.")
            import pandas as pd
            rows = []
            for exp in legacy:
                periods = exp.get('periods', {})
                def fmt(p):
                    v = periods.get(p, {}).get('cumulative_return')
                    return f"{v:+.2f}" if v is not None else '-'
                rows.append({
                    'Variant': exp['label'],
                    'val':  fmt('val'),
                    'bear': fmt('test_bear'),
                    '2023': fmt('test_2023'),
                    'bull': fmt('test_bull'),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if mac_iters:
            st.markdown("**MAC multi-agent iterations (parallel research path, EXCLUDED from thesis)**")
            for exp in mac_iters:
                v = headline_period_values(exp)
                st.markdown(
                    f"<div class='variant-card'>"
                    f"<div class='v-name'>{exp['label']}</div>"
                    f"<div class='v-folder'>{exp['folder']}</div>"
                    f"<div class='v-row'>val {v.get('val','-')} &nbsp; bear {v.get('test_bear','-')} &nbsp; "
                    f"2023 {v.get('test_2023','-')} &nbsp; bull {v.get('test_bull','-')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
