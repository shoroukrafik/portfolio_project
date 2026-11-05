"""
Build per-universe multi-seed comparison tables (MAC v3 vs v6 universal).

Saves:
  results/multi_seed_dji.md
  results/multi_seed_sp100.md
  results/multi_seed_dji.png
  results/multi_seed_sp100.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


SEEDS = [42, 41, 43]

with open('results/multi_seed_summary.json') as f:
    summary = json.load(f)


def collect(system, universe, period, metric):
    """Return list of metric values across seeds."""
    return [summary[f'{system}|{universe}'][str(s)][period][metric] for s in SEEDS]


def winner(mac_mean, v6_mean, higher_is_better=True):
    if higher_is_better:
        return 'MAC' if mac_mean > v6_mean else 'v6 ' if v6_mean > mac_mean else 'TIE'
    return 'MAC' if mac_mean < v6_mean else 'v6 ' if v6_mean < mac_mean else 'TIE'


def build_table_for_universe(u):
    """Produce markdown + PNG for one universe."""
    rows = []
    for p in ['val','test_bear','test_2023','test_bull']:
        mac_ret  = collect('MAC_v3', u, p, 'cum_ret')
        mac_shp  = collect('MAC_v3', u, p, 'sharpe')
        mac_mdd  = collect('MAC_v3', u, p, 'mdd')
        v6_ret   = collect('universal_v6', u, p, 'cum_ret')
        v6_shp   = collect('universal_v6', u, p, 'sharpe')
        v6_mdd   = collect('universal_v6', u, p, 'mdd')

        # bear: less negative is better (so MDD/return: max wins)
        # all others: max is better
        is_bear = (p == 'test_bear')
        ret_winner = winner(np.mean(mac_ret), np.mean(v6_ret), higher_is_better=True)
        shp_winner = winner(np.mean(mac_shp), np.mean(v6_shp), higher_is_better=not is_bear if False else True)
        rows.append({
            'period': p,
            'mac_ret_mean': np.mean(mac_ret), 'mac_ret_sd': np.std(mac_ret, ddof=1),
            'v6_ret_mean':  np.mean(v6_ret),  'v6_ret_sd':  np.std(v6_ret, ddof=1),
            'mac_shp_mean': np.mean(mac_shp), 'mac_shp_sd': np.std(mac_shp, ddof=1),
            'v6_shp_mean':  np.mean(v6_shp),  'v6_shp_sd':  np.std(v6_shp, ddof=1),
            'mac_mdd_mean': np.mean(mac_mdd), 'mac_mdd_sd': np.std(mac_mdd, ddof=1),
            'v6_mdd_mean':  np.mean(v6_mdd),  'v6_mdd_sd':  np.std(v6_mdd, ddof=1),
            'mac_ret_per_seed': mac_ret,      'v6_ret_per_seed': v6_ret,
            'mac_shp_per_seed': mac_shp,      'v6_shp_per_seed': v6_shp,
            'ret_winner': ret_winner,         'shp_winner': shp_winner,
        })

    # Markdown
    md = [f'# Multi-seed comparison — {u.upper()}  (3 seeds: 42 / 41 / 43)\n',
          'Each cell: mean ± SD across 3 seeds. **bold** indicates per-period winner.\n',
          '## Returns (cumulative %)\n',
          '| Period | MAC v3 (mean ± SD) | per-seed | v6 universal (mean ± SD) | per-seed | Δ MAC−v6 | Winner |',
          '|---|---|---|---|---|---|---|']
    for r in rows:
        delta = r['mac_ret_mean'] - r['v6_ret_mean']
        is_bear = (r['period'] == 'test_bear')
        # For bear, MAC wins if delta is positive (less negative loss)
        w = r['ret_winner']
        mac_per = '/'.join(f"{v:+.2f}" for v in r['mac_ret_per_seed'])
        v6_per  = '/'.join(f"{v:+.2f}" for v in r['v6_ret_per_seed'])
        mac_cell = f"**{r['mac_ret_mean']:+.2f} ± {r['mac_ret_sd']:.2f}**" if w == 'MAC' else f"{r['mac_ret_mean']:+.2f} ± {r['mac_ret_sd']:.2f}"
        v6_cell  = f"**{r['v6_ret_mean']:+.2f} ± {r['v6_ret_sd']:.2f}**"   if w == 'v6 ' else f"{r['v6_ret_mean']:+.2f} ± {r['v6_ret_sd']:.2f}"
        md.append(f"| {r['period']} | {mac_cell} | {mac_per} | {v6_cell} | {v6_per} | {delta:+.2f} pp | {w.strip()} |")
    md.append('\n## Sharpe ratios\n')
    md.append('| Period | MAC v3 (mean ± SD) | v6 universal (mean ± SD) | Δ MAC−v6 | Winner |')
    md.append('|---|---|---|---|---|')
    for r in rows:
        delta = r['mac_shp_mean'] - r['v6_shp_mean']
        w = r['shp_winner']
        mac_cell = f"**{r['mac_shp_mean']:+.3f} ± {r['mac_shp_sd']:.3f}**" if w == 'MAC' else f"{r['mac_shp_mean']:+.3f} ± {r['mac_shp_sd']:.3f}"
        v6_cell  = f"**{r['v6_shp_mean']:+.3f} ± {r['v6_shp_sd']:.3f}**"   if w == 'v6 ' else f"{r['v6_shp_mean']:+.3f} ± {r['v6_shp_sd']:.3f}"
        md.append(f"| {r['period']} | {mac_cell} | {v6_cell} | {delta:+.3f} | {w.strip()} |")
    md.append('\n## Max drawdown (%)\n')
    md.append('| Period | MAC v3 (mean ± SD) | v6 universal (mean ± SD) | Δ MAC−v6 |')
    md.append('|---|---|---|---|')
    for r in rows:
        delta = r['mac_mdd_mean'] - r['v6_mdd_mean']
        md.append(f"| {r['period']} | {r['mac_mdd_mean']:+.2f} ± {r['mac_mdd_sd']:.2f} | "
                  f"{r['v6_mdd_mean']:+.2f} ± {r['v6_mdd_sd']:.2f} | {delta:+.2f} pp |")

    # Summary line
    mac_wins = sum(1 for r in rows if r['ret_winner'] == 'MAC')
    v6_wins  = sum(1 for r in rows if r['ret_winner'] == 'v6 ')
    md.append(f'\n**Period wins (returns):** MAC v3 = {mac_wins}/4, v6 universal = {v6_wins}/4')
    md.append(f'\n**Generated 2026-05-23 from `results/multi_seed_summary.json`**')

    md_path = f'results/multi_seed_{u}.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"  ✅ → {md_path}")

    # PNG — side-by-side bars per period for returns + Sharpe
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    periods = ['val','test_bear','test_2023','test_bull']
    x = np.arange(len(periods))
    w = 0.35

    # Returns
    mac_means_r = [r['mac_ret_mean'] for r in rows]
    mac_sds_r   = [r['mac_ret_sd']   for r in rows]
    v6_means_r  = [r['v6_ret_mean']  for r in rows]
    v6_sds_r    = [r['v6_ret_sd']    for r in rows]
    axes[0].bar(x - w/2, mac_means_r, w, yerr=mac_sds_r, label='MAC v3', color='#1f77b4', capsize=5, alpha=0.85)
    axes[0].bar(x + w/2, v6_means_r,  w, yerr=v6_sds_r,  label='v6 universal', color='#d62728', capsize=5, alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(periods)
    axes[0].set_ylabel('Cumulative return (%)')
    axes[0].set_title(f'{u.upper()} — Returns across 3 seeds (mean ± SD)')
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()
    # Annotate per-seed for v6 (since it varies more)
    for i, r in enumerate(rows):
        seeds_str = '/'.join(f"{v:+.1f}" for v in r['v6_ret_per_seed'])
        axes[0].text(x[i] + w/2, r['v6_ret_mean'] - r['v6_ret_sd'] - 2, seeds_str,
                     fontsize=7, ha='center', color='darkred')

    # Sharpe
    mac_means_s = [r['mac_shp_mean'] for r in rows]
    mac_sds_s   = [r['mac_shp_sd']   for r in rows]
    v6_means_s  = [r['v6_shp_mean']  for r in rows]
    v6_sds_s    = [r['v6_shp_sd']    for r in rows]
    axes[1].bar(x - w/2, mac_means_s, w, yerr=mac_sds_s, label='MAC v3', color='#1f77b4', capsize=5, alpha=0.85)
    axes[1].bar(x + w/2, v6_means_s,  w, yerr=v6_sds_s,  label='v6 universal', color='#d62728', capsize=5, alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(periods)
    axes[1].set_ylabel('Sharpe ratio')
    axes[1].set_title(f'{u.upper()} — Sharpe across 3 seeds (mean ± SD)')
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()

    plt.suptitle(f'{u.upper()} multi-seed comparison: MAC v3 vs v6 universal (n=3 seeds: 42, 41, 43)',
                  fontsize=12, weight='bold')
    plt.tight_layout()
    png_path = f'results/multi_seed_{u}.png'
    plt.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✅ → {png_path}")
    return rows


print("\n=== Building per-universe multi-seed tables ===\n")
for u in ['dji', 'sp100']:
    build_table_for_universe(u)
print("\n=== Done. ===\n")
