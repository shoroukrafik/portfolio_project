"""
Loads pre-v6 design ablations from dashboard_legacy/data/curves/.

These are the Phase-A trials (VANILLA, FULL baseline, no_floor, no_dsr,
no_attention, no_regime_λ=0.1/0.3/0.5) that JUSTIFIED the architecture
choices baked into universal v6 — cash floor, DSR reward, attention
extractor, regime-conditioned λ.

We summarize them by extracting the final cumulative return from each
day-by-day curve file. No sweep is available — these were one-shot
evaluations of the in-training val-best checkpoint.
"""

import json
import os
from pathlib import Path
from typing import Optional

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY_CURVES_DIR = PROJECT_ROOT / "dashboard_legacy" / "data" / "curves"


# label (in legacy) → (universe, friendly_label, subtitle)
LEGACY_REGISTRY = {
    # DJI Phase-A trials
    'FULL (baseline)':
        ('DJI', 'FULL (Phase-A baseline)',
         'The DJI single-agent FULL design with all features. Was the original baseline before the universal-v6 refactor.'),
    'VANILLA':
        ('DJI', 'VANILLA (no architecture)',
         'PPO with no cash floor, no regime gating, no DSR, no attention. Justifies adding each component.'),
    'no_floor (eval)':
        ('DJI', 'FULL — cash floor removed',
         'Ablation: removes the regime-conditioned cash floor. Shows cash floor is load-bearing for bear defense.'),
    'no_dsr':
        ('DJI', 'FULL — DSR reward removed',
         'Ablation: replaces Differential Sharpe Reward with raw return. Justifies DSR.'),
    'no_attention':
        ('DJI', 'FULL — attention extractor removed',
         'Ablation: replaces attention extractor with flat MLP. Justifies attention.'),
    'no_regime_λ=0.1':
        ('DJI', 'FULL — uniform λ=0.1',
         'Ablation: removes regime-conditioned risk penalty (all periods λ=0.1). Justifies regime λ.'),
    'no_regime_λ=0.3':
        ('DJI', 'FULL — uniform λ=0.3',
         'Ablation: removes regime-conditioned risk penalty (all periods λ=0.3).'),
    'no_regime_λ=0.5':
        ('DJI', 'FULL — uniform λ=0.5',
         'Ablation: removes regime-conditioned risk penalty (all periods λ=0.5).'),

    # SP100 Phase-A trials
    'VANILLA SP100':
        ('SP100', 'VANILLA SP100 (no architecture)',
         'PPO on SP100 with no cash floor / regime gating / DSR / attention. SP100 mirror of VANILLA DJI.'),
    'SP100 no_floor':
        ('SP100', 'FULL SP100 — cash floor removed',
         'Cash-floor ablation on SP100. Tests whether finding generalizes from DJI.'),
    'SP100 no_dsr':
        ('SP100', 'FULL SP100 — DSR removed',
         'DSR ablation on SP100.'),
    'SP100 no_attention':
        ('SP100', 'FULL SP100 — attention removed',
         'Attention ablation on SP100.'),
    'SP100 no_regime_λ=0.1':
        ('SP100', 'FULL SP100 — uniform λ=0.1',
         'Uniform-λ ablation on SP100.'),
    'SP100 no_regime_λ=0.3':
        ('SP100', 'FULL SP100 — uniform λ=0.3',
         'Uniform-λ ablation on SP100.'),
    'SP100 no_regime_λ=0.5':
        ('SP100', 'FULL SP100 — uniform λ=0.5',
         'Uniform-λ ablation on SP100.'),
}


def _label_to_slug(label: str) -> str:
    """Match the slug used by precompute_data.py in dashboard_legacy."""
    # e.g. 'FULL (baseline)' → 'FULL_baseline'
    return (label.replace(' ', '_')
                  .replace('(', '').replace(')', '')
                  .replace('=', '_')
                  .replace('λ', 'lambda'))


@st.cache_data(ttl=300, show_spinner=False)
def load_legacy_ablations() -> dict:
    """
    Returns {universe: [ {label, subtitle, periods}, ... ]}
    periods has keys val/test_bear/test_2023/test_bull, each {'cumulative_return': float}.
    Returns empty dict if legacy data missing.
    """
    out = {'DJI': [], 'SP100': []}
    if not LEGACY_CURVES_DIR.exists():
        return out

    period_keys = ['val', 'test_bear', 'test_2023', 'test_bull']

    for legacy_label, (universe, friendly_label, subtitle) in LEGACY_REGISTRY.items():
        slug = _label_to_slug(legacy_label)
        periods = {}
        for period in period_keys:
            curve_file = LEGACY_CURVES_DIR / f"{slug}__{period}.json"
            if not curve_file.exists():
                continue
            try:
                curve = json.loads(curve_file.read_text())
                records = curve.get('records', [])
                if not records:
                    continue
                last = records[-1].get('value')
                first_val = 1_000_000
                cum_ret = ((last - first_val) / first_val) * 100 if last else None
                periods[period] = {'cumulative_return': cum_ret}
            except (json.JSONDecodeError, OSError, ValueError):
                continue
        if not periods:
            continue
        out.setdefault(universe, []).append({
            'folder':   f'(legacy) {slug}',
            'label':    friendly_label,
            'subtitle': subtitle,
            'category': 'lineage',
            'universe': universe,
            'periods':  periods,
            'source':   'legacy_curves',
        })

    return out


def get_legacy_for_universe(universe: str) -> list:
    return load_legacy_ablations().get(universe, [])
