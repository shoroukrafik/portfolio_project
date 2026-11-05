"""
Build the full ablation table from all saved sweep / final-results JSONs.

Output:
  results/ablation_summary.json   — machine-readable per-ablation x per-period rows
  results/ablation_summary.md     — markdown table for thesis
  results/ablation_summary.png    — visualization for dashboard
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Ablation registry ---------------------------------------------------------
# Each entry: (label, suffix or special_loader, "what was changed", justifies)
MAC_ABLATIONS = [
    ('v3 (PRODUCTION)',     '_v3',         'Production lineup, soft warm-start',                'baseline'),
    ('Hard-switch',         'hardswitch',  'No learned controller, HMM argmax routing',         'controller'),
    ('v4 (one-hot ws)',     '_v4',         'One-hot warm-start instead of soft',                'controller'),
    ('A: no warm-start',    '_v3_nowarm',  'PPO from scratch, no supervised warm-start',        'warm-start'),
    ('B: shuffled regime',  '_v3_shuf',    'Random Dirichlet replaces real HMM probs in MAC',   'regime signal'),
    ('C: no vol features',  '_v3_novol',   'MAC obs = 3D regime probs only (drop vol)',         'vol features'),
    ('F: ent_coef 0.05',    '_v3_ent05',   'PPO entropy coef 0.05 (default 0.01)',              'hyperparameter'),
    ('v5 (drop balanced)',  '_v5',         '2-way router, no steady specialist',                'specialist count'),
    ('v6 (drop defender)',  '_v6',         '2-way router, no defender specialist',              'specialist count'),
    ('v7 (drop hunter)',    '_v7',         '2-way router, no hunter specialist',                'specialist count'),
    ('v2 (MARS legacy)',    '_v2',         'Per-role envs (legacy MARS arch)',                  'architecture'),
]

# Best-checkpoint selection per (universe, ablation). Use val-sharpe-best.
def best_by_val_sharpe(sweep_rows):
    return max(sweep_rows, key=lambda r: r['val_sharpe'])


def load_mac_ablation(universe, suffix):
    path = f'results/mac_{universe}{suffix}/checkpoint_sweep.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f)
    return best_by_val_sharpe(rows)


def load_hardswitch(universe):
    path = f'results/mac_{universe}_v3/hmm_hard_switch.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    # Flatten to same format as sweep_rows entries
    return {
        'val_ret': d['val']['metrics']['cumulative_return'],
        'val_sharpe': d['val']['metrics']['sharpe_ratio'],
        'test_bear_ret': d['test_bear']['metrics']['cumulative_return'],
        'test_bear_sharpe': d['test_bear']['metrics']['sharpe_ratio'],
        'test_2023_ret': d['test_2023']['metrics']['cumulative_return'],
        'test_2023_sharpe': d['test_2023']['metrics']['sharpe_ratio'],
        'test_bull_ret': d['test_bull']['metrics']['cumulative_return'],
        'test_bull_sharpe': d['test_bull']['metrics']['sharpe_ratio'],
        'step': None,
    }


# --- Build table rows ---------------------------------------------------------
rows = []
for label, suffix, change, justifies in MAC_ABLATIONS:
    for u in ['dji', 'sp100']:
        if suffix == 'hardswitch':
            r = load_hardswitch(u)
        else:
            r = load_mac_ablation(u, suffix)
        if r is None:
            continue
        rows.append({
            'ablation': label,
            'universe': u.upper(),
            'change':   change,
            'justifies': justifies,
            'step':     r.get('step'),
            'val_ret':  r['val_ret'],
            'val_shp':  r['val_sharpe'],
            'bear_ret': r['test_bear_ret'],
            'bear_shp': r['test_bear_sharpe'],
            't23_ret':  r['test_2023_ret'],
            't23_shp':  r['test_2023_sharpe'],
            'bull_ret': r['test_bull_ret'],
            'bull_shp': r['test_bull_sharpe'],
        })

# H2 separately — universal v6 single agent
H2_ROWS = []
for u in ['dji', 'sp100']:
    for kind in ['ORIG', 'H2']:
        path = f'results/universal_v6_{u}{"_noregime" if kind == "H2" else ""}/final_results.json'
        with open(path) as f:
            d = json.load(f)
        per = d['periods']
        H2_ROWS.append({
            'system': f'universal v6 {kind}',
            'universe': u.upper(),
            'val_ret':  per['val']['cumulative_return'],
            'val_shp':  per['val']['sharpe_ratio'],
            'bear_ret': per['test_bear']['cumulative_return'],
            'bear_shp': per['test_bear']['sharpe_ratio'],
            't23_ret':  per['test_2023']['cumulative_return'],
            't23_shp':  per['test_2023']['sharpe_ratio'],
            'bull_ret': per['test_bull']['cumulative_return'],
            'bull_shp': per['test_bull']['sharpe_ratio'],
        })

# --- Save JSON ---------------------------------------------------------------
out = {'mac_ablations': rows, 'h2_universal_no_regime': H2_ROWS}
with open('results/ablation_summary.json', 'w') as f:
    json.dump(out, f, indent=2, default=float)
print("✅ saved results/ablation_summary.json")

# --- Markdown table ----------------------------------------------------------
df = pd.DataFrame(rows)
lines = ['# Ablation summary (all single-seed, val-Sharpe-best checkpoint)\n']
lines.append('## MAC ablations (cumulative return % | Sharpe)\n')
lines.append('| Ablation | What changed | Justifies | Universe | val | bear | 2023 | bull |')
lines.append('|---|---|---|---|---|---|---|---|')
for r in rows:
    lines.append(
        f"| {r['ablation']} | {r['change']} | {r['justifies']} | {r['universe']} | "
        f"{r['val_ret']:+.2f} ({r['val_shp']:+.3f}) | "
        f"{r['bear_ret']:+.2f} ({r['bear_shp']:+.3f}) | "
        f"{r['t23_ret']:+.2f} ({r['t23_shp']:+.3f}) | "
        f"{r['bull_ret']:+.2f} ({r['bull_shp']:+.3f}) |"
    )
lines.append('\n## H2: universal v6 single agent — regime ON vs OFF\n')
lines.append('| System | Universe | val | bear | 2023 | bull |')
lines.append('|---|---|---|---|---|---|')
for r in H2_ROWS:
    lines.append(
        f"| {r['system']} | {r['universe']} | "
        f"{r['val_ret']:+.2f} ({r['val_shp']:+.3f}) | "
        f"{r['bear_ret']:+.2f} ({r['bear_shp']:+.3f}) | "
        f"{r['t23_ret']:+.2f} ({r['t23_shp']:+.3f}) | "
        f"{r['bull_ret']:+.2f} ({r['bull_shp']:+.3f}) |"
    )
with open('results/ablation_summary.md', 'w') as f:
    f.write('\n'.join(lines))
print("✅ saved results/ablation_summary.md")


# --- PNG chart ---------------------------------------------------------------
# Build: 2 universes (rows) x 4 periods (cols) heatmap-style chart
# Showing delta vs v3 production
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
period_keys = [('val_ret', 'VAL'), ('bear_ret', 'BEAR'), ('t23_ret', '2023'), ('bull_ret', 'BULL')]

for ax_idx, u in enumerate(['DJI', 'SP100']):
    ax = axes[ax_idx]
    u_rows = [r for r in rows if r['universe'] == u]
    v3 = next(r for r in u_rows if r['ablation'] == 'v3 (PRODUCTION)')

    labels = [r['ablation'] for r in u_rows]
    n_abl = len(u_rows)
    n_per = len(period_keys)

    # Make heatmap of deltas vs v3
    data = np.zeros((n_abl, n_per))
    for i, r in enumerate(u_rows):
        for j, (k, _) in enumerate(period_keys):
            data[i, j] = r[k] - v3[k]

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-12, vmax=12)
    ax.set_xticks(range(n_per))
    ax.set_xticklabels([lbl for _, lbl in period_keys], fontsize=11)
    ax.set_yticks(range(n_abl))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(f'{u} — Δ return vs MAC v3 production (pp)', fontsize=13, weight='bold')

    # Annotate each cell with the actual delta
    for i in range(n_abl):
        for j in range(n_per):
            txt = f"{data[i,j]:+.2f}"
            color = 'black' if abs(data[i,j]) < 7 else 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label='Δ return (pp)')

plt.suptitle('Ablation study: every MAC variant vs v3 production',
             fontsize=14, weight='bold', y=1.005)
plt.tight_layout()
plt.savefig('results/ablation_summary.png', dpi=130, bbox_inches='tight')
print("✅ saved results/ablation_summary.png")
plt.close()


# --- A second PNG just for H2 -----------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
periods = ['val', 'bear', '2023', 'bull']
x = np.arange(len(periods))
w = 0.2

dji_orig = [r for r in H2_ROWS if r['system'] == 'universal v6 ORIG' and r['universe'] == 'DJI'][0]
dji_h2   = [r for r in H2_ROWS if r['system'] == 'universal v6 H2'   and r['universe'] == 'DJI'][0]
sp_orig  = [r for r in H2_ROWS if r['system'] == 'universal v6 ORIG' and r['universe'] == 'SP100'][0]
sp_h2    = [r for r in H2_ROWS if r['system'] == 'universal v6 H2'   and r['universe'] == 'SP100'][0]

def vals(r): return [r['val_ret'], r['bear_ret'], r['t23_ret'], r['bull_ret']]

ax.bar(x - 1.5*w, vals(dji_orig), w, label='DJI orig (regime ON)', color='#1f77b4')
ax.bar(x - 0.5*w, vals(dji_h2),   w, label='DJI H2 (regime OFF)',  color='#aec7e8', hatch='//')
ax.bar(x + 0.5*w, vals(sp_orig),  w, label='SP100 orig (regime ON)', color='#d62728')
ax.bar(x + 1.5*w, vals(sp_h2),    w, label='SP100 H2 (regime OFF)',  color='#ff9896', hatch='//')

ax.set_xticks(x)
ax.set_xticklabels([p.upper() for p in periods], fontsize=11)
ax.set_ylabel('Cumulative return (%)')
ax.set_title('H2 ablation: Universal v6 single agent — regime architecture ON vs OFF',
             fontsize=12, weight='bold')
ax.axhline(0, color='gray', lw=0.5)
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/ablation_h2.png', dpi=130, bbox_inches='tight')
print("✅ saved results/ablation_h2.png")
plt.close()
