"""
Robustness check on model-selection criterion.

Our training scripts save the checkpoint with the SINGLE-eval peak val Sharpe.
This is standard practice but is sensitive to lucky single-eval spikes —
the saved model may not be representative of nearby-checkpoint behavior.

This script answers, retrospectively, two questions:
  1. How "spiky" was the saved peak? (Compare single-eval peak vs the
     5-eval rolling mean AT THAT STEP.)
  2. If we had used a more robust criterion — pick the step with the
     highest 5-eval rolling-mean val Sharpe — would we have selected a
     different checkpoint?

A small gap (< ~0.1 Sharpe) between single-peak and rolling-mean-at-peak
means the saved model was on a stable plateau, NOT a transient spike, so
the saved metrics are trustworthy. A large gap is a robustness warning.

No retraining required — operates on existing val_sharpe_history arrays.

Outputs:
  results/ablation_ext_summary/robustness_check.png        — per-system charts
  results/ablation_ext_summary/robustness_check.md         — markdown table
  results/ablation_ext_summary/robustness_check.json       — machine-readable
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WINDOW = 5     # # of consecutive evals to average (each eval = 25k steps)


SYSTEMS = [
    # (label, primary path, optional separate history path)
    ('FULL',           'results/extended/final_results.json',
                       'results/extended/val_sharpe_history.json'),
    ('VANILLA',        'results/vanilla_extended/final_results.json',         None),
    ('no_dsr',         'results/ablation_ext_no_dsr/final_results.json',      None),
    ('no_attention',   'results/ablation_ext_no_attention/final_results.json', None),
    ('no_regime_λ=0.1','results/ablation_ext_no_regime_lambda_0p1/final_results.json', None),
    ('no_regime_λ=0.3','results/ablation_ext_no_regime_lambda_0p3/final_results.json', None),
    ('no_regime_λ=0.5','results/ablation_ext_no_regime_lambda_0p5/final_results.json', None),
]


def trailing_rolling_mean(values, window):
    """Mean of last `window` values up to and including index i.
    Returns NaN for the first window-1 entries (insufficient history).
    Trailing (not centered) because that's what you'd use online."""
    v = np.asarray(values, dtype=float)
    out = np.full_like(v, fill_value=np.nan)
    for i in range(window - 1, len(v)):
        out[i] = v[i - window + 1: i + 1].mean()
    return out


def load_history(primary_path, history_path):
    """Pull the (step, sharpe) sequence regardless of which file it lives in."""
    d = json.load(open(primary_path))
    if d.get('val_sharpe_history'):
        return d['val_sharpe_history']
    if history_path is not None and os.path.exists(history_path):
        h = json.load(open(history_path))
        return h.get('history')
    raise RuntimeError(f"No val_sharpe_history in {primary_path} or {history_path}")


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_dir, 'results', 'ablation_ext_summary')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes_flat = axes.flatten()

    for ax, (label, primary, hist_alt) in zip(axes_flat, SYSTEMS):
        primary_full = os.path.join(project_dir, primary)
        hist_alt_full = os.path.join(project_dir, hist_alt) if hist_alt else None
        try:
            hist = load_history(primary_full, hist_alt_full)
        except Exception as e:
            print(f"  SKIP {label}: {e}")
            ax.set_visible(False)
            continue

        steps   = np.array([h[0] for h in hist], dtype=int)
        sharpes = np.array([h[1] for h in hist], dtype=float)

        # ---- Single-eval peak (what we actually saved) ----
        single_idx    = int(np.argmax(sharpes))
        single_step   = int(steps[single_idx])
        single_sharpe = float(sharpes[single_idx])

        # ---- 5-eval trailing rolling mean ----
        rolling = trailing_rolling_mean(sharpes, WINDOW)
        # window-peak only valid where rolling is non-NaN
        valid = ~np.isnan(rolling)
        window_idx    = int(np.argmax(np.where(valid, rolling, -np.inf)))
        window_step   = int(steps[window_idx])
        window_sharpe = float(rolling[window_idx])

        # ---- Rolling mean AT the saved single-eval peak ----
        # (i.e. how much of a spike was the saved checkpoint?)
        rolling_at_saved = (
            float(rolling[single_idx]) if not np.isnan(rolling[single_idx])
            else float(np.mean(sharpes[max(0, single_idx - WINDOW + 1): single_idx + 1]))
        )
        spike_gap = single_sharpe - rolling_at_saved   # bigger gap = spikier

        # ---- Robustness verdict ----
        # We say it's "robust" if (a) the gap between single-peak and the
        # rolling-mean-at-that-step is < 0.10 Sharpe, AND (b) the window-
        # peak step is within ±2 evals (50k steps) of the saved step.
        robust = (spike_gap < 0.10) and (abs(window_step - single_step) <= 50_000)

        rows.append({
            'system': label,
            'saved_step (single-peak)':  single_step,
            'saved_sharpe':              round(single_sharpe, 4),
            'rolling_mean_at_saved':     round(rolling_at_saved, 4),
            'spike_gap':                 round(spike_gap, 4),
            'window_peak_step':          window_step,
            'window_peak_sharpe':        round(window_sharpe, 4),
            'same_choice':               window_step == single_step,
            'robust':                    bool(robust),
        })

        # ---- Plot ----
        ax.plot(steps, sharpes, color='#7f7f7f', marker='o', markersize=3,
                alpha=0.7, label='single-eval val Sharpe', linewidth=0.8)
        ax.plot(steps, rolling, color='#d62728', linewidth=2,
                label=f'{WINDOW}-eval rolling mean')
        ax.scatter([single_step], [single_sharpe], color='#1f77b4',
                   s=120, zorder=5, edgecolor='black',
                   label=f'saved (single-peak): {single_sharpe:.3f}')
        ax.scatter([window_step], [window_sharpe], color='#d62728',
                   s=120, zorder=5, marker='s', edgecolor='black',
                   label=f'window-peak: {window_sharpe:.3f}')
        verdict_txt = 'ROBUST ✓' if robust else 'SPIKY ⚠'
        ax.set_title(f'{label}   —   gap={spike_gap:+.3f}   '
                     f'[{verdict_txt}]',
                     fontsize=10)
        ax.set_xlabel('training step')
        ax.set_ylabel('val Sharpe')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=7)

    # Hide unused subplot if any
    for ax in axes_flat[len(SYSTEMS):]:
        ax.set_visible(False)

    plt.suptitle('Selection-criterion robustness — single-eval peak vs '
                 f'{WINDOW}-eval rolling-mean peak',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    chart_path = os.path.join(out_dir, 'robustness_check.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {chart_path}")

    # ---- Markdown table ----
    md = [
        '# Robustness check — single-eval vs rolling-mean selection\n',
        f'For each system, compare the **saved checkpoint** (chosen at the '
        f'single-eval peak val Sharpe) against the **{WINDOW}-eval trailing '
        f'rolling mean** at that same step.\n',
        '* `spike_gap`  = `saved_sharpe − rolling_mean_at_saved`. '
          'Larger gap → the saved peak was a transient spike, not a stable plateau.',
        '* `window_peak_step`  = step that would have been chosen under a '
          'rolling-mean criterion.',
        '* `same_choice`  = whether the rolling-mean criterion picks the same '
          'step as the single-peak criterion.',
        '* `robust` = `spike_gap < 0.10` AND `|saved_step − window_step| ≤ 50k`.\n',
        '| System | Saved step | Saved Sharpe | Rolling@saved | Spike gap | Window-peak step | Window-peak Sharpe | Same choice | Robust |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for r in rows:
        md.append(
            f"| {r['system']} | "
            f"{r['saved_step (single-peak)']:,} | "
            f"{r['saved_sharpe']:.4f} | "
            f"{r['rolling_mean_at_saved']:.4f} | "
            f"{r['spike_gap']:+.4f} | "
            f"{r['window_peak_step']:,} | "
            f"{r['window_peak_sharpe']:.4f} | "
            f"{'✓' if r['same_choice'] else '✗'} | "
            f"{'✓' if r['robust'] else '⚠'} |"
        )
    md_path = os.path.join(out_dir, 'robustness_check.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"  Wrote {md_path}")

    # ---- JSON ----
    json_path = os.path.join(out_dir, 'robustness_check.json')
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"  Wrote {json_path}")

    # ---- Console print ----
    print('\n' + '=' * 110)
    print(f' ROBUSTNESS CHECK — single-eval peak vs {WINDOW}-eval rolling mean')
    print('=' * 110)
    df = pd.DataFrame(rows)
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.float_format', '{:.4f}'.format):
        print(df.to_string(index=False))
    print('\n  Interpretation:')
    print('    spike_gap < 0.05 → very robust; saved model sits on a plateau')
    print('    spike_gap 0.05–0.15 → mildly spiky; some seed/eval noise')
    print('    spike_gap > 0.15 → genuinely spiky; saved checkpoint may not generalize')
    print('=' * 110 + '\n')


if __name__ == '__main__':
    main()
