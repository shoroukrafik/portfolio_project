"""
Re-select the best bear specialist from its 40 per-eval checkpoints.

Why this exists:
  Bear's training-time val-Sharpe was 0.00 at every eval because the
  regime-filtered bear val (23 days in 2021) collapses after the env's
  20-day warmup window: too few portfolio values to compute a sensible
  std → Sharpe → 0 → no useful selection signal.

Fix:
  Re-evaluate each of the 40 per-eval bear checkpoints on the FULL
  252-day val (same criterion used for the personality agents). Pick
  the checkpoint with the best full-val Sharpe and copy it over the
  broken val-best save at models/saved/multi_agent_ext/specialist_bear.zip.

NOTE on methodology: the bear specialist is out-of-distribution on
bull/sideways days. Evaluating it on the full val is a "what does the
bear specialist do if it had to act every day in 2021" probe — not a
realistic deployment scenario (the controller routes by regime). But it
gives a meaningful relative ranking across the 40 candidate checkpoints,
which is all we need for selection. No test data is touched.

Outputs:
  models/saved/multi_agent_ext/specialist_bear.zip  ← OVERWRITTEN with picked checkpoint
  results/multi_agent_ext/bear_checkpoint_selection.json  — full table
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


def metrics_from_history(history, initial_balance=1_000_000):
    v = history['value'].values
    if len(v) < 2:
        return {'cumulative_return': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'annualized_volatility': 0.0}
    final = v[-1]
    cum = (final - initial_balance) / initial_balance * 100
    dr = np.diff(v) / v[:-1]
    n = len(v)
    ann_ret = ((final / initial_balance) ** (252 / n) - 1) * 100
    ann_vol = float(np.std(dr) * np.sqrt(252) * 100)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(np.min((v - peak) / peak) * 100)
    return {'cumulative_return': round(cum, 2),
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown': round(mdd, 2),
            'annualized_volatility': round(ann_vol, 2)}


def evaluate_checkpoint(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    ckpt_dir    = os.path.join(base_dir, 'saved', 'multi_agent_ext',
                               'checkpoints', 'specialist_bear')
    target_path = os.path.join(base_dir, 'saved', 'multi_agent_ext',
                               'specialist_bear')
    out_dir     = os.path.join(project_dir, 'results', 'multi_agent_ext')

    print("=" * 72)
    print("   Bear specialist — pick best from per-eval checkpoints")
    print("=" * 72)

    # ---- List checkpoints ----
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.zip'))
    if not ckpts:
        print(f"❌ No checkpoints in {ckpt_dir}")
        return
    print(f"\n  Found {len(ckpts)} checkpoints in {ckpt_dir}")

    # ---- Load full val 2021 env ----
    val_df = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    val_df['date'] = pd.to_datetime(val_df['date'])
    val_env = PortfolioEnv(df=val_df, initial_balance=1_000_000)
    print(f"  Eval set: full val 2021 ({val_df['date'].nunique()} days, "
          f"all regimes)")

    # ---- Score each checkpoint ----
    rows = []
    print()
    for fname in ckpts:
        step = int(fname.replace('step_', '').replace('.zip', ''))
        path = os.path.join(ckpt_dir, fname.replace('.zip', ''))
        model = PPO.load(path, device='cpu')
        history = evaluate_checkpoint(model, val_env)
        m = metrics_from_history(history)
        rows.append({'step': step, **m})
        print(f"  step {step:>7,}: "
              f"Return={m['cumulative_return']:>7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>7.4f}  "
              f"MDD={m['max_drawdown']:>7.2f}%")

    # ---- Pick the best ----
    df = pd.DataFrame(rows).sort_values('sharpe_ratio', ascending=False)
    best = df.iloc[0]
    print()
    print("=" * 72)
    print(f"  ★ BEST by full-val Sharpe:  step {int(best['step']):>7,}  "
          f"Sharpe={best['sharpe_ratio']:.4f}  "
          f"Return={best['cumulative_return']:.2f}%  "
          f"MDD={best['max_drawdown']:.2f}%")
    print("=" * 72)

    # ---- Copy that checkpoint over the val-best save ----
    best_step = int(best['step'])
    src = os.path.join(ckpt_dir, f'step_{best_step:07d}.zip')
    dst = target_path + '.zip'
    shutil.copy(src, dst)
    print(f"\n  ✅ Copied {src} → {dst}")

    # ---- Show top 5 for context ----
    print("\n  Top 5 checkpoints by full-val Sharpe:")
    for _, r in df.head(5).iterrows():
        print(f"    step {int(r['step']):>7,}  "
              f"Sharpe={r['sharpe_ratio']:>7.4f}  "
              f"Return={r['cumulative_return']:>7.2f}%  "
              f"MDD={r['max_drawdown']:>7.2f}%")

    # ---- Save the full table ----
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'bear_checkpoint_selection.json')
    with open(out_json, 'w') as f:
        json.dump({'criterion': 'full_val_2021_sharpe',
                   'selected_step': best_step,
                   'selected_metrics': {k: best[k] for k in
                                        ('cumulative_return',
                                         'sharpe_ratio',
                                         'max_drawdown',
                                         'annualized_volatility')},
                   'all_checkpoints': rows}, f, indent=2, default=float)
    print(f"  ✅ Full selection table → {out_json}")
    print()


if __name__ == '__main__':
    main()
