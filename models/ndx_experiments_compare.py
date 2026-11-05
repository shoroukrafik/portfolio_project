"""
Comparative learning-curve chart across all NDX experiments.

Plots val Sharpe over training steps for each of the 6 NDX variants:
  1. Production NDX v6              (2007-2020 train, 68 stocks, 1M steps)
  2. NDX exp1 — lower bull penalty   (500k)
  3. NDX exp2 — concentration bonus  (500k)
  4. NDX exp3 — pure log-return      (500k)
  5. NDX exp4 — lower ent_coef       (500k)
  6. NDX 2017 — 4-year recent train  (90 stocks, 500k)

Plus annotates each variant's val-best checkpoint and the final test_bull
result at that checkpoint.

Output:
  results/ndx_experiments_learning_curves.png
  results/ndx_experiments_summary.md
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


VARIANTS = [
    # (display_name, results_subdir, color, linestyle, what_changed)
    ('production v6',       'universal_v6_ndx',       '#1f77b4', '-',  'baseline (14yr train, 68 stocks, 1M)'),
    ('exp1 lower bull λ',   'universal_v6_ndx_exp1',  '#ff7f0e', '--', 'bull penalty 0.1 → 0.01'),
    ('exp2 concentration',  'universal_v6_ndx_exp2',  '#2ca02c', '--', 'concentration bonus top-5, λ=0.05'),
    ('exp3 log-return',     'universal_v6_ndx_exp3',  '#d62728', '--', 'DSR → raw return reward'),
    ('exp4 low ent_coef',   'universal_v6_ndx_exp4',  '#9467bd', '--', 'ent_coef 0.01 → 0.001'),
    ('2017-variant',        'universal_v6_ndx_2017',  '#8c564b', '-',  '4yr recent train, 90 stocks',),
]


def load_history(subdir):
    path = f'results/{subdir}/val_sharpe_history.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    # history is list of [step, val_sharpe]
    hist = d['history']
    steps = [h[0] for h in hist]
    vals  = [h[1] for h in hist]
    return {
        'steps': steps, 'vals': vals,
        'best_step': d['best_step'],
        'best_val_sharpe': d['best_val_sharpe'],
    }


def load_final(subdir):
    path = f'results/{subdir}/final_results.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # === LEFT PANEL — val Sharpe learning curves ===
    ax = axes[0]
    summary_rows = []
    for label, subdir, color, ls, what in VARIANTS:
        h = load_history(subdir)
        if h is None:
            print(f"  ⚠  missing {subdir}")
            continue
        ax.plot(h['steps'], h['vals'], color=color, linestyle=ls, lw=1.6,
                label=label, alpha=0.9)
        # Mark val-best checkpoint
        bs, bv = h['best_step'], h['best_val_sharpe']
        ax.scatter([bs], [bv], color=color, s=80, zorder=5, edgecolor='black', lw=1)
        ax.annotate(f"{label}\nstep {bs/1000:.0f}k → {bv:.2f}",
                    xy=(bs, bv), xytext=(8, 5),
                    textcoords='offset points', fontsize=7, color=color,
                    alpha=0.0)  # hide annotation; legend has it

        # Pull final-checkpoint test numbers
        fr = load_final(subdir)
        if fr:
            per = fr['periods']
            summary_rows.append({
                'variant': label, 'change': what,
                'best_step': bs, 'best_val_sharpe': bv,
                'val_ret':   per['val']['cumulative_return'],
                'bear_ret':  per['test_bear']['cumulative_return'],
                't23_ret':   per['test_2023']['cumulative_return'],
                'bull_ret':  per['test_bull']['cumulative_return'],
            })

    ax.set_xlabel('PPO training step')
    ax.set_ylabel('Validation Sharpe ratio')
    ax.set_title('NDX experiments — validation Sharpe across training', fontsize=12, weight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=1)
    ax.set_ylim(bottom=0.8)

    # === RIGHT PANEL — final-checkpoint test results (bar chart) ===
    ax = axes[1]
    labels = [r['variant'] for r in summary_rows]
    n = len(labels)
    x = np.arange(n)
    w = 0.2
    val_rets  = [r['val_ret']   for r in summary_rows]
    bear_rets = [r['bear_ret']  for r in summary_rows]
    t23_rets  = [r['t23_ret']   for r in summary_rows]
    bull_rets = [r['bull_ret']  for r in summary_rows]
    ax.bar(x - 1.5*w, val_rets,  w, label='val',       color='#9c27b0', alpha=0.85)
    ax.bar(x - 0.5*w, bear_rets, w, label='bear',      color='#d62728', alpha=0.85)
    ax.bar(x + 0.5*w, t23_rets,  w, label='2023',      color='#ff7f0e', alpha=0.85)
    ax.bar(x + 1.5*w, bull_rets, w, label='bull',      color='#2ca02c', alpha=0.85)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(28.13, color='green', lw=1.0, linestyle=':', label='^NDX bull 2024 (+28.13%)', alpha=0.7)
    ax.axhline(-33.71, color='red',  lw=1.0, linestyle=':', label='^NDX bear 2022 (-33.71%)', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([l.replace(' ','\n') for l in labels], fontsize=8, rotation=0)
    ax.set_ylabel('Cumulative return (%)')
    ax.set_title('Val-Sharpe-BEST checkpoint test returns per variant',
                  fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

    plt.suptitle('NDX experimental variants — comparative learning trajectory',
                  fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/ndx_experiments_learning_curves.png', dpi=130, bbox_inches='tight')
    plt.close()

    # Markdown summary
    md = ['# NDX experiments — comparative summary\n',
          'All variants compared at val-Sharpe-best checkpoint.',
          '^NDX 2024 cap-weighted benchmark: +28.13%   ^NDX 2022 benchmark: -33.71%\n',
          '| Variant | What changed | Best step | Val Sharpe | val | bear | 2023 | bull |',
          '|---|---|---|---|---|---|---|---|']
    for r in summary_rows:
        md.append(f"| {r['variant']} | {r['change']} | {r['best_step']:,} | "
                  f"{r['best_val_sharpe']:.3f} | {r['val_ret']:+.2f} | "
                  f"{r['bear_ret']:+.2f} | {r['t23_ret']:+.2f} | {r['bull_ret']:+.2f} |")
    with open('results/ndx_experiments_summary.md','w') as f:
        f.write('\n'.join(md))

    print("✅ saved results/ndx_experiments_learning_curves.png")
    print("✅ saved results/ndx_experiments_summary.md")

    # Console table
    print("\n" + '='*108)
    print("  NDX EXPERIMENTS — at val-Sharpe-best checkpoint")
    print('='*108)
    print(f"  {'Variant':<22}{'Change':<40}{'BestStep':>10}{'ValShp':>9}{'val':>8}{'bear':>8}{'2023':>8}{'bull':>8}")
    print('  ' + '-'*106)
    for r in summary_rows:
        print(f"  {r['variant']:<22}{r['change']:<40}{r['best_step']:>9,} {r['best_val_sharpe']:>8.3f}"
              f"{r['val_ret']:>+8.2f}{r['bear_ret']:>+8.2f}{r['t23_ret']:>+8.2f}{r['bull_ret']:>+8.2f}")


if __name__ == '__main__':
    main()
