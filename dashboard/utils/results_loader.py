"""
Auto-discovery of experiment results.

Scans `results/` folder for any `final_results.json` file and builds a
unified registry. New experiments dropped into `results/` automatically
appear in the dashboard — no code changes needed.

Each variant has a CATEGORY:
  - production   : the canonical universal-v6 result for that universe
  - experimental : extensions tested in this thesis (long-mom, 2017/2015 cutoff, CVaR)
  - ablation     : one component removed (no_regime, reward-shaping ablations)
  - multi_seed   : same model, different seed (variance estimation)
  - lineage      : pre-universal-v6 baselines (the "history" of how we got here)

Each variant has a LABEL (human-readable, meaningful) and the original
FOLDER NAME (so you can still grep results/).
"""

import json
import os
from pathlib import Path
from typing import Optional

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# VARIANT REGISTRY
# Format: folder_name → (universe, label, subtitle, category, sort_order, total_steps)
#
# universe   : "DJI" / "SP100" / "NDX" (for tab grouping)
# label      : short, displayed everywhere
# subtitle   : longer one-line description of WHAT this variant tests
# category   : production / experimental / ablation / multi_seed / lineage
# sort_order : lower = earlier in list
# total_steps: required for the min-step rule to apply correctly
# ---------------------------------------------------------------------------

CATEGORIES = {
    'production':   {'label': 'Production',   'color_class': 'badge-accent'},
    'experimental': {'label': 'Experimental', 'color_class': 'badge-warning'},
    'ablation':     {'label': 'Ablation',     'color_class': 'badge-neutral'},
    'multi_seed':   {'label': 'Multi-seed',   'color_class': 'badge-neutral'},
    'lineage':      {'label': 'Lineage',      'color_class': 'badge-neutral'},
}


VARIANT_REGISTRY = {
    # ============================ DJI ===============================
    'universal_v6_dji': {
        'universe': 'DJI',
        'label': 'Universal v6 (production)',
        'subtitle': 'Canonical recipe: PPO + attention + HMM regime gating + 60d momentum. Train 2007-2020.',
        'category': 'production', 'sort_order': 10, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_mom': {
        'universe': 'DJI',
        'label': 'v6 + long-momentum',
        'subtitle': 'Production v6 plus 6-month and 12-month momentum features. Tests Jegadeesh-Titman in DRL.',
        'category': 'experimental', 'sort_order': 20, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_mom_2m': {
        'universe': 'DJI',
        'label': 'v6 + long-momentum (2M extension)',
        'subtitle': 'DJI mom re-trained to 2M steps to rule out undertraining as the explanation for the 1M result. Confirmatory negative: wins unchanged (6/8 → 6/8), bear collapses to -46%, bull soars to +84% — proving DJI dispersion ceiling is structural, not budget-driven.',
        'category': 'experimental', 'sort_order': 21, 'total_steps': 2_000_000,
    },
    'universal_v6_dji_noregime': {
        'universe': 'DJI',
        'label': 'v6 — regime removed (H2 strong ablation)',
        'subtitle': 'Removes ALL regime mechanisms: regime probs + vol features + uniform λ. The full no-regime test.',
        'category': 'ablation', 'sort_order': 30, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_h2soft': {
        'universe': 'DJI',
        'label': 'v6 — uniform λ (H2-soft ablation)',
        'subtitle': 'Keeps regime probs + vol features in observation. Replaces regime-conditioned λ with uniform 0.3. Isolates λ-conditioning contribution.',
        'category': 'ablation', 'sort_order': 31, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_2015': {
        'universe': 'DJI',
        'label': 'v6 + 2015 cutoff (NO long-mom) — ablation',
        'subtitle': 'Shorter training window (2015-2020) on same 30-stock universe. Control for the DJI 2×2 long-mom × window ablation.',
        'category': 'experimental', 'sort_order': 26, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_2015_mom': {
        'universe': 'DJI',
        'label': 'v6 + 2015 cutoff + long-momentum',
        'subtitle': 'Shorter window (2015-2020) + long-mom features. Tests the dispersion-gradient prediction on DJI (low dispersion → expected modest effect).',
        'category': 'experimental', 'sort_order': 25, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_lr3e4': {
        'universe': 'DJI',
        'label': 'LR-test B — lr=3e-4 fixed (mismatched pair)',
        'subtitle': 'Tests SB3 default LR alone with our n_steps=1024. Mismatched pair (fast LR on small rollout) — predicted to be unstable. Result confirmed: val curve volatile, std 0.808.',
        'category': 'ablation', 'sort_order': 35, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_lrsched': {
        'universe': 'DJI',
        'label': 'LR-test C — linear schedule 3e-4 → 5e-5',
        'subtitle': 'Tests standard PPO best practice — high LR for fast initial progress, decay to slow LR for stable convergence. Same n_steps=1024.',
        'category': 'ablation', 'sort_order': 36, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_sb3default': {
        'universe': 'DJI',
        'label': 'LR-test E — SB3 default matched pair (lr=3e-4 + n_steps=2048)',
        'subtitle': 'Full SB3 PPO default — the cleanest matched-pair test of literature standard vs our conservative pair.',
        'category': 'ablation', 'sort_order': 37, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_s41': {
        'universe': 'DJI',
        'label': 'v6 production (seed 41)',
        'subtitle': 'Same model, different random seed. Used for multi-seed variance bands.',
        'category': 'multi_seed', 'sort_order': 50, 'total_steps': 1_000_000,
    },
    'universal_v6_dji_s43': {
        'universe': 'DJI',
        'label': 'v6 production (seed 43)',
        'subtitle': 'Same model, different random seed. Used for multi-seed variance bands.',
        'category': 'multi_seed', 'sort_order': 51, 'total_steps': 1_000_000,
    },
    'extended': {
        'universe': 'DJI',
        'label': 'FULL extended baseline (pre-v6)',
        'subtitle': 'Original DJI single-agent design before universal-v6 refactor. Same architecture, different training pipeline.',
        'category': 'lineage', 'sort_order': 80, 'total_steps': 1_000_000,
    },

    # ============================ SP100 =============================
    'universal_v6_sp100': {
        'universe': 'SP100',
        'label': 'Universal v6 (production)',
        'subtitle': 'Same recipe as DJI. 90-stock universe with full 2007-2024 history. Tests universality.',
        'category': 'production', 'sort_order': 10, 'total_steps': 1_000_000,
    },
    'universal_v6_sp100_mom_2m': {
        'universe': 'SP100',
        'label': 'v6 + long-momentum ★',
        'subtitle': '+ 6-month and 12-month momentum features, trained 2M steps. HEADLINE pick wins 11/12 benchmarks — current best SP100 variant.',
        'category': 'experimental', 'sort_order': 20, 'total_steps': 2_000_000,
    },
    'universal_v6_sp100_mom': {
        'universe': 'SP100',
        'label': 'v6 + long-momentum (1M archive)',
        'subtitle': 'Original 1M-step run. Superseded by the 2M version (still climbing at 1M with +0.11 late slope). Kept for 1M-vs-2M training-budget comparison.',
        'category': 'lineage', 'sort_order': 70, 'total_steps': 1_000_000,
    },
    'universal_v6_sp100_2015_mom': {
        'universe': 'SP100',
        'label': 'v6 + 2015 cutoff + long-momentum',
        'subtitle': 'Shorter window (2015-2020) + relaxed cutoff brings in META/TSLA/AVGO/PYPL. Mirrors NDX 2017 recipe.',
        'category': 'experimental', 'sort_order': 25, 'total_steps': 500_000,
    },
    'universal_v6_sp100_2015': {
        'universe': 'SP100',
        'label': 'v6 + 2015 cutoff (NO long-mom) — ablation',
        'subtitle': 'Pairs with 2015+mom variant to isolate long-momentum contribution from window+universe effect.',
        'category': 'experimental', 'sort_order': 26, 'total_steps': 500_000,
    },
    'universal_v6_sp100_noregime': {
        'universe': 'SP100',
        'label': 'v6 — regime removed (H2 strong ablation)',
        'subtitle': 'Removes ALL regime mechanisms: regime probs + vol features + uniform λ. Full no-regime test on SP100.',
        'category': 'ablation', 'sort_order': 30, 'total_steps': 1_000_000,
    },
    'universal_v6_sp100_h2soft': {
        'universe': 'SP100',
        'label': 'v6 — uniform λ (H2-soft ablation)',
        'subtitle': 'Keeps regime probs + vol features. Replaces regime-conditioned λ with uniform 0.3. Tests literature-standard fixed-λ choice.',
        'category': 'ablation', 'sort_order': 31, 'total_steps': 1_000_000,
    },
    'universal_v6_sp100_s41': {
        'universe': 'SP100',
        'label': 'v6 production (seed 41)',
        'subtitle': 'Same model, different seed. Used for multi-seed variance bands.',
        'category': 'multi_seed', 'sort_order': 50, 'total_steps': 1_000_000,
    },
    'universal_v6_sp100_s43': {
        'universe': 'SP100',
        'label': 'v6 production (seed 43)',
        'subtitle': 'Same model, different seed. Used for multi-seed variance bands.',
        'category': 'multi_seed', 'sort_order': 51, 'total_steps': 1_000_000,
    },
    'extended_sp100': {
        'universe': 'SP100',
        'label': 'FULL extended baseline (pre-v6)',
        'subtitle': 'Original SP100 single-agent design before universal-v6 refactor. The first SP100 result.',
        'category': 'lineage', 'sort_order': 80, 'total_steps': 1_000_000,
    },

    # ============================ NDX ===============================
    'universal_v6_ndx': {
        'universe': 'NDX',
        'label': 'Universal v6 (production)',
        'subtitle': 'Same recipe applied to NASDAQ-100. 2007-2024 full-history filter (68 surviving stocks).',
        'category': 'production', 'sort_order': 10, 'total_steps': 1_000_000,
    },
    'universal_v6_ndx_2017': {
        'universe': 'NDX',
        'label': 'v6 + 2017 cutoff',
        'subtitle': 'Relaxed first-trade cutoff to 2017-01-15. Adds META/TSLA/AVGO. Train window 2017-2020 only.',
        'category': 'experimental', 'sort_order': 20, 'total_steps': 500_000,
    },
    'universal_v6_ndx_2017_mom': {
        'universe': 'NDX',
        'label': 'v6 + 2017 + long-momentum ★',
        'subtitle': 'Pareto improvement: bull, bear, AND 2023 all improve. The headline NDX experimental result.',
        'category': 'experimental', 'sort_order': 25, 'total_steps': 500_000,
    },
    'universal_v6_ndx_mom_2m': {
        'universe': 'NDX',
        'label': 'v6 + long-momentum (2007-2020, 2M)',
        'subtitle': 'Long-mom features on the FULL 2007-2020 window (68-stock survivor universe). 2M steps. Negative finding: bull capture collapses to ~0% despite val improvement — suggests universe-membership stability matters more than dispersion alone.',
        'category': 'experimental', 'sort_order': 22, 'total_steps': 2_000_000,
    },
    'universal_v6_ndx_2017_mom_bear': {
        'universe': 'NDX',
        'label': 'v6 + 2017 + long-mom + CVaR',
        'subtitle': 'CVaR penalty added on top — tested for asymmetric bear defense. Result: 2:1 bull-for-bear trade-off (not asymmetric).',
        'category': 'experimental', 'sort_order': 30, 'total_steps': 500_000,
    },
    'universal_v6_ndx_exp1': {
        'universe': 'NDX',
        'label': 'NDX exp1 — lower bull λ',
        'subtitle': 'Reward-shape ablation: bull risk penalty 0.1 → 0.01. Tests if low bull aversion unlocks more bull capture.',
        'category': 'ablation', 'sort_order': 40, 'total_steps': 1_000_000,
    },
    'universal_v6_ndx_exp2': {
        'universe': 'NDX',
        'label': 'NDX exp2 — concentration bonus',
        'subtitle': 'Adds top-5 concentration bonus to reward. Tests if explicit concentration helps NDX bull capture.',
        'category': 'ablation', 'sort_order': 41, 'total_steps': 1_000_000,
    },
    'universal_v6_ndx_exp3': {
        'universe': 'NDX',
        'label': 'NDX exp3 — log-return reward',
        'subtitle': 'DSR reward replaced with raw log-return. Tests if differential Sharpe was the bottleneck.',
        'category': 'ablation', 'sort_order': 42, 'total_steps': 1_000_000,
    },
    'universal_v6_ndx_exp4': {
        'universe': 'NDX',
        'label': 'NDX exp4 — low ent_coef',
        'subtitle': 'Entropy coefficient 0.01 → 0.001. Tests if reducing exploration helps converge to concentrated policy.',
        'category': 'ablation', 'sort_order': 43, 'total_steps': 1_000_000,
    },

    # ============================ MAC iterations (DJI) =====================
    'mac_dji': {
        'universe': 'DJI',
        'label': 'MAC v1 (first multi-agent)',
        'subtitle': 'First specialist controller. 3 specialists (defender/steady/hunter) routed by HMM regime. Pre-production iteration.',
        'category': 'lineage', 'sort_order': 60, 'total_steps': 1_000_000,
    },
    'mac_dji_v2': {
        'universe': 'DJI',
        'label': 'MAC v2 (routing refinement)',
        'subtitle': 'Adds soft-routing between specialists. Iteration toward MAC v3 production multi-agent.',
        'category': 'lineage', 'sort_order': 61, 'total_steps': 1_000_000,
    },
    'mac_dji_v3': {
        'universe': 'DJI',
        'label': 'MAC v3 (multi-agent production) ★',
        'subtitle': 'The multi-agent production method. 3 specialists (defender/steady/hunter) trained on regime-filtered data, hard-switched by HMM. Sibling production to universal v6.',
        'category': 'production', 'sort_order': 11, 'total_steps': 1_000_000,
    },
    'mac_dji_v4': {
        'universe': 'DJI',
        'label': 'MAC v4 (entropy scaled)',
        'subtitle': 'Reward scaling iteration. Tests whether tuning entropy improves specialist diversity.',
        'category': 'lineage', 'sort_order': 63, 'total_steps': 1_000_000,
    },
    'mac_dji_v5': {
        'universe': 'DJI',
        'label': 'MAC v5 (cash floor tuned)',
        'subtitle': 'Per-specialist cash floor variation. Final MAC iteration before single-agent universal v6 pivot.',
        'category': 'lineage', 'sort_order': 64, 'total_steps': 1_000_000,
    },
    'mac_dji_v6': {
        'universe': 'DJI',
        'label': 'MAC v6 (universal-v6 hyperparams)',
        'subtitle': 'MAC controller using universal-v6 hyperparameters. Lets us compare MAC vs universal v6 with everything else equal.',
        'category': 'lineage', 'sort_order': 65, 'total_steps': 1_000_000,
    },
    'mac_dji_v7': {
        'universe': 'DJI',
        'label': 'MAC v7 (final)',
        'subtitle': 'Final MAC iteration. After this, the thesis pivoted to single-agent universal v6 as the cleaner story.',
        'category': 'lineage', 'sort_order': 66, 'total_steps': 1_000_000,
    },

    # ============================ MAC iterations (SP100) ===================
    'mac_sp100': {
        'universe': 'SP100',
        'label': 'MAC v1 SP100 (first multi-agent)',
        'subtitle': 'First specialist controller on SP100. Tests whether MAC generalizes beyond DJI.',
        'category': 'lineage', 'sort_order': 60, 'total_steps': 1_000_000,
    },
    'mac_sp100_v2': {
        'universe': 'SP100',
        'label': 'MAC v2 SP100',
        'subtitle': 'SP100 mirror of MAC v2.',
        'category': 'lineage', 'sort_order': 61, 'total_steps': 1_000_000,
    },
    'mac_sp100_v3': {
        'universe': 'SP100',
        'label': 'MAC v3 SP100 (multi-agent production) ★',
        'subtitle': 'The SP100 multi-agent production result — sibling to universal v6 single-agent. Tests whether MAC generalizes from DJI to broader universe.',
        'category': 'production', 'sort_order': 11, 'total_steps': 1_000_000,
    },
    'mac_sp100_v4': {
        'universe': 'SP100', 'label': 'MAC v4 SP100',
        'subtitle': 'SP100 mirror of MAC v4 iteration.',
        'category': 'lineage', 'sort_order': 63, 'total_steps': 1_000_000,
    },
    'mac_sp100_v5': {
        'universe': 'SP100', 'label': 'MAC v5 SP100',
        'subtitle': 'SP100 mirror of MAC v5 iteration.',
        'category': 'lineage', 'sort_order': 64, 'total_steps': 1_000_000,
    },
    'mac_sp100_v6': {
        'universe': 'SP100', 'label': 'MAC v6 SP100',
        'subtitle': 'SP100 with universal-v6 hyperparams. Lets us compare MAC vs universal v6 on SP100 directly.',
        'category': 'lineage', 'sort_order': 65, 'total_steps': 1_000_000,
    },
    'mac_sp100_v7': {
        'universe': 'SP100', 'label': 'MAC v7 SP100',
        'subtitle': 'Final MAC iteration on SP100.',
        'category': 'lineage', 'sort_order': 66, 'total_steps': 1_000_000,
    },
}


# ---------------------------------------------------------------------------
# Period-format normalization
# Old "extended" baselines use agent_val/agent_test_bear keys.
# New universal_v6 results use periods.val/test_bear etc.
# Normalize both into {'val', 'test_bear', 'test_2023', 'test_bull'}.
# ---------------------------------------------------------------------------

def _normalize_periods(raw: dict) -> dict:
    """Return dict {period_name: metrics_dict} from either format."""
    if 'periods' in raw:
        return raw['periods'] or {}
    # Old format: keys like agent_val, agent_test_bear, agent_test_2023, agent_test_bull
    out = {}
    for old_key, new_key in [
        ('agent_val',       'val'),
        ('agent_test_bear', 'test_bear'),
        ('agent_test_2023', 'test_2023'),
        ('agent_test_bull', 'test_bull'),
    ]:
        if old_key in raw:
            out[new_key] = raw[old_key]
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10, show_spinner=False)
def load_all_experiments() -> dict:
    """
    Scan results/ for final_results.json files. Cached for 10s so the
    dashboard auto-refreshes when new experiments complete.
    """
    registry: dict = {'DJI': [], 'SP100': [], 'NDX': [], 'Other': []}

    if not RESULTS_DIR.exists():
        return registry

    for folder in sorted(RESULTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith('_'):
            continue

        results_file = folder / 'final_results.json'
        sweep_file   = folder / 'checkpoint_sweep.json'

        # Source priority: final_results.json > checkpoint_sweep.json
        # (MAC variants have sweep but no final_results — extract periods from sweep best)
        data = None
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        elif sweep_file.exists() and folder.name in VARIANT_REGISTRY:
            # Only auto-promote sweep-only folders that are in the registry
            # (avoids surfacing every junk folder)
            try:
                sweep = json.loads(sweep_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # Synthesize a final_results-like dict using val-Sharpe-best from sweep
            if sweep:
                best = max(sweep, key=lambda r: r.get('val_sharpe', float('-inf')))
                synth_periods = {}
                for p in ['val', 'test_bear', 'test_2023', 'test_bull']:
                    synth_periods[p] = {
                        'cumulative_return': best.get(f'{p}_ret') if p == 'val' else best.get(f'test_{p[5:]}_ret') if p.startswith('test_') else best.get(f'{p}_ret'),
                        'sharpe_ratio':      best.get(f'{p}_sharpe') if p == 'val' else best.get(f'test_{p[5:]}_sharpe') if p.startswith('test_') else best.get(f'{p}_sharpe'),
                        'max_drawdown':      best.get(f'{p}_mdd') if p == 'val' else best.get(f'test_{p[5:]}_mdd') if p.startswith('test_') else best.get(f'{p}_mdd'),
                    }
                # Manual fix for naming: sweep uses 'val_ret' / 'test_bear_ret' / 'test_2023_ret' / 'test_bull_ret'
                synth_periods = {
                    'val':       {'cumulative_return': best.get('val_ret'),       'sharpe_ratio': best.get('val_sharpe'),       'max_drawdown': best.get('val_mdd')},
                    'test_bear': {'cumulative_return': best.get('test_bear_ret'), 'sharpe_ratio': best.get('test_bear_sharpe'), 'max_drawdown': best.get('test_bear_mdd')},
                    'test_2023': {'cumulative_return': best.get('test_2023_ret'), 'sharpe_ratio': best.get('test_2023_sharpe'), 'max_drawdown': best.get('test_2023_mdd')},
                    'test_bull': {'cumulative_return': best.get('test_bull_ret'), 'sharpe_ratio': best.get('test_bull_sharpe'), 'max_drawdown': best.get('test_bull_mdd')},
                }
                data = {
                    'method':          folder.name,
                    'best_step':       best.get('step'),
                    'best_val_sharpe': best.get('val_sharpe'),
                    'periods':         synth_periods,
                    '_synthesized_from_sweep': True,
                }

        if data is None:
            continue

        # Look up registry entry
        meta = VARIANT_REGISTRY.get(folder.name)
        if meta:
            universe   = meta['universe']
            label      = meta['label']
            subtitle   = meta['subtitle']
            category   = meta['category']
            sort_order = meta['sort_order']
            total_steps = meta.get('total_steps')
        else:
            # Unregistered folder — assign to Other with low priority
            name_lower = folder.name.lower()
            if 'ndx' in name_lower:
                universe = 'NDX'
            elif 'sp100' in name_lower or 'sp_100' in name_lower:
                universe = 'SP100'
            elif 'dji' in name_lower or 'dow' in name_lower:
                universe = 'DJI'
            else:
                universe = 'Other'
            label    = folder.name.replace('_', ' ')
            subtitle = 'Unregistered variant — add to VARIANT_REGISTRY for proper labeling.'
            category = 'lineage'
            sort_order = 999
            total_steps = None

        entry = {
            'folder':         folder.name,
            'label':          label,
            'subtitle':       subtitle,
            'category':       category,
            'sort_order':     sort_order,
            'total_steps':    total_steps,
            'universe':       universe,
            'result_dir':     str(folder),
            'method':         data.get('method', folder.name),
            'universe_desc':  data.get('universe', universe),
            'best_step':      data.get('best_step'),
            'best_val_sharpe': data.get('best_val_sharpe'),
            'periods':        _normalize_periods(data),
            'raw':            data,
        }
        registry.setdefault(universe, []).append(entry)

    for universe in registry:
        registry[universe].sort(key=lambda e: (e['sort_order'], e['folder']))

    return registry


def get_universe_experiments(universe: str) -> list:
    return load_all_experiments().get(universe, [])


def count_total_experiments() -> int:
    return sum(len(v) for v in load_all_experiments().values())
