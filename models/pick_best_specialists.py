"""
Re-select each multi-agent specialist from its per-eval checkpoints
using a REGIME-CORRECT criterion.

Each specialist was trained on regime-filtered data — bull spec only on
bull days, etc. The operational design routes each specialist only to
its own regime's days at inference time (via the HMM controller).

Therefore the right validation criterion is **the specialist's own
regime's val days**, not the full mixed-regime val. This file applies
that criterion:

  - BULL specialist     → 156 bull val days,    metric: Sharpe
  - SIDEWAYS specialist → 73  sideways val days, metric: Sharpe
  - BEAR specialist     → 23  bear val days,    metric: cumulative return
                                                  (Sharpe degenerates on
                                                   so few days after env
                                                   warmup; cumulative
                                                   return remains well-
                                                   defined)

Bear's tiny sample size is a known issue. Cumulative return is the
honest fallback because:
  - We want the specialist that loses the LEAST on bear val days
  - (final/initial - 1) is well-defined on any sample size > 1
  - This matches the design intent: "bear specialist should defend"

The bull/sideways criteria match exactly what the env-side val callback
would have used if it had access to all checkpoints — we just couldn't
apply it during training without per-eval checkpoint saving.

NO TEST DATA IS USED for selection.

Outputs:
  models/saved/multi_agent_ext/specialist_{bull,bear,sideways}.zip
      ← OVERWRITTEN with the regime-correct best checkpoint
  results/multi_agent_ext/specialist_selection.json
      — per-specialist ranking table
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv


SPECIALIST_CONFIG = [
    # (name, regime_id, selection_metric, sort_descending)
    ('bull',     0, 'sharpe_ratio',      True),   # higher Sharpe better
    ('bear',     1, 'cumulative_return', True),   # higher return = least loss
    ('sideways', 2, 'sharpe_ratio',      True),   # higher Sharpe better
]


def metrics_from_history(history, initial_balance=1_000_000):
    v = history['value'].values
    if len(v) < 2:
        return {'cumulative_return': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'annualized_volatility': 0.0,
                'final_value': float(v[0]) if len(v) else float(initial_balance)}
    final = v[-1]
    cum = (final - initial_balance) / initial_balance * 100
    dr = np.diff(v) / v[:-1]
    n = len(v)
    ann_ret = ((final / initial_balance) ** (252 / n) - 1) * 100
    ann_vol = float(np.std(dr) * np.sqrt(252) * 100)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(np.min((v - peak) / peak) * 100)
    return {'cumulative_return': round(cum, 4),
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown': round(mdd, 4),
            'annualized_volatility': round(ann_vol, 4),
            'final_value': round(float(final), 2)}


def evaluate_checkpoint(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()


def select_best_for_specialist(name, regime_id, metric, descending,
                               ckpt_dir, target_path, full_val_df):
    """Scan all checkpoints, evaluate on regime-filtered val, pick best."""
    # Regime-filtered val
    val_filtered = full_val_df[full_val_df['regime'] == regime_id].copy()
    n_days = val_filtered['date'].nunique()

    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.zip'))
    if not ckpts:
        print(f"  ❌ {name}: no checkpoints in {ckpt_dir}")
        return None

    val_env = PortfolioEnv(df=val_filtered, initial_balance=1_000_000)
    print(f"\n--- {name} specialist — {len(ckpts)} checkpoints on "
          f"{n_days} {name}-only val days  (criterion: {metric}) ---")

    rows = []
    for fname in ckpts:
        step = int(fname.replace('step_', '').replace('.zip', ''))
        path = os.path.join(ckpt_dir, fname.replace('.zip', ''))
        model = PPO.load(path, device='cpu')
        history = evaluate_checkpoint(model, val_env)
        m = metrics_from_history(history)
        rows.append({'step': step, **m})

    df = pd.DataFrame(rows).sort_values(metric, ascending=not descending)
    best = df.iloc[0]
    best_step = int(best['step'])

    src = os.path.join(ckpt_dir, f'step_{best_step:07d}.zip')
    dst = target_path + '.zip'
    shutil.copy(src, dst)
    print(f"  ★ Best: step {best_step:>7,}  "
          f"{metric}={best[metric]:>7.4f}  "
          f"(also: Return={best['cumulative_return']:>6.2f}%  "
          f"Sharpe={best['sharpe_ratio']:>7.4f}  "
          f"MDD={best['max_drawdown']:>6.2f}%)")
    print(f"  Top 5 by {metric}:")
    for _, r in df.head(5).iterrows():
        print(f"    step {int(r['step']):>7,}  "
              f"Return={r['cumulative_return']:>6.2f}%  "
              f"Sharpe={r['sharpe_ratio']:>7.4f}  "
              f"MDD={r['max_drawdown']:>6.2f}%")

    return {'criterion': metric,
            'regime_val_days': int(n_days),
            'selected_step': best_step,
            'selected_metrics': {k: float(best[k]) for k in
                                 ('cumulative_return', 'sharpe_ratio',
                                  'max_drawdown', 'annualized_volatility')},
            'all_checkpoints': rows}


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')

    print("=" * 72)
    print("   Specialist re-selection — REGIME-FILTERED criteria")
    print("=" * 72)

    # ---- Load full val (we'll filter per-specialist below) ----
    val_df = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    val_df['date'] = pd.to_datetime(val_df['date'])
    print(f"\n  Full val 2021: {val_df['date'].nunique()} days, "
          f"will filter per specialist's own regime")

    selections = {}
    for name, regime_id, metric, descending in SPECIALIST_CONFIG:
        ckpt_dir = os.path.join(base_dir, 'saved', 'multi_agent_ext',
                                'checkpoints', f'specialist_{name}')
        target  = os.path.join(base_dir, 'saved', 'multi_agent_ext',
                               f'specialist_{name}')
        if not os.path.isdir(ckpt_dir):
            print(f"\n  ⚠ Skip {name}: no checkpoint dir at {ckpt_dir}")
            continue
        sel = select_best_for_specialist(name, regime_id, metric, descending,
                                         ckpt_dir, target, val_df)
        if sel is not None:
            selections[name] = sel

    # ---- Save summary ----
    out_dir = os.path.join(project_dir, 'results', 'multi_agent_ext')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'specialist_selection.json')
    with open(out_path, 'w') as f:
        json.dump({'method': 'regime_filtered_val',
                   'specialists': selections}, f, indent=2, default=float)

    # ---- Final picks ----
    print("\n" + "=" * 72)
    print("   Final regime-correct selections")
    print("=" * 72)
    print(f"  {'Specialist':<11} {'Step':>8}  {'Criterion':>20}  "
          f"{'Return%':>8} {'Sharpe':>8} {'MDD%':>8}")
    for name, sel in selections.items():
        m = sel['selected_metrics']
        print(f"  {name:<11} {sel['selected_step']:>8,}  "
              f"{sel['criterion']:>20}  "
              f"{m['cumulative_return']:>8.2f} "
              f"{m['sharpe_ratio']:>8.4f} "
              f"{m['max_drawdown']:>8.2f}")
    print(f"\n  ✅ Summary → {out_path}")


if __name__ == '__main__':
    main()
