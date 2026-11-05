"""
Inspect portfolio concentration evolution for NDX 2017-variant across training.

For a selected set of checkpoints (early/mid/late), runs each through the
2024 bull test period and records:
  - Max single-stock weight per day
  - Top-1, top-5, top-10 weight (time-averaged)
  - Which stocks each checkpoint concentrates in (top 10 by avg weight)
  - Concentration HHI (sum of squared weights)

Output:
  results/ndx_2017_concentration.json
  results/ndx_2017_concentration_evolution.png
  results/ndx_2017_concentration_table.md
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
from portfolio_env import PortfolioEnv
from train_universal_v6_sp100 import V3_COLS_TO_DROP


CHECKPOINTS = [25000, 50000, 100000, 200000, 300000, 400000, 450000, 475000, 500000]


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def env_kwargs():
    return dict(
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: 0.0, 2: 0.0},
        risk_penalties={0: 0.1, 1: 0.5, 2: 0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
        use_markowitz_reward=False, use_hyperheuristic=False,
    )


def evaluate_concentration(step):
    """Run NDX 2017 checkpoint at given step on 2024 bull period, record weights."""
    df = load_df('test_bull_data_ndx_2017.csv')
    tickers = sorted(df['ticker'].unique())
    env = PortfolioEnv(df=df, **env_kwargs())
    model_path = f'models/saved/checkpoints/universal_v6_ndx_2017/step_{step:07d}.zip'
    model = PPO.load(model_path, device='cpu')

    obs, _ = env.reset()
    weights = []
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        weights.append(env.portfolio_weights.copy())
        done = term or trunc
    W = np.stack(weights)  # (n_days, n_stocks+1)
    return W, tickers


def concentration_metrics(W, tickers):
    """Compute concentration stats from weight matrix."""
    stock_W = W[:, :-1]
    cash_W  = W[:, -1]
    # Time-averaged per-stock weight
    avg = stock_W.mean(axis=0)
    sort_idx = np.argsort(avg)[::-1]
    top10 = [(tickers[i], float(avg[i])) for i in sort_idx[:10]]
    # Concentration measures
    top1 = float(avg[sort_idx[0]])
    top5 = float(avg[sort_idx[:5]].sum())
    top10_sum = float(avg[sort_idx[:10]].sum())
    # Daily peak top-1
    daily_top1 = stock_W.max(axis=1)
    # HHI (sum of squared weights), 1/n = 1/68 → uniform HHI ~0.0147
    daily_hhi = (stock_W ** 2).sum(axis=1)
    return {
        'top1_avg':      top1,
        'top5_avg':      top5,
        'top10_avg':     top10_sum,
        'top10_stocks':  top10,
        'daily_top1_mean':   float(daily_top1.mean()),
        'daily_top1_max':    float(daily_top1.max()),
        'daily_top1_median': float(np.median(daily_top1)),
        'hhi_mean':      float(daily_hhi.mean()),
        'hhi_max':       float(daily_hhi.max()),
        'cash_avg':      float(cash_W.mean()),
        'n_stocks':      stock_W.shape[1],
    }


def main():
    # Confirm we're on the right universe (NDX 2017 = 90 stocks)
    results = {}
    print(f"\n{'='*100}")
    print(f"   NDX 2017-variant — portfolio concentration across training steps")
    print(f"{'='*100}\n")
    for step in CHECKPOINTS:
        print(f"  Step {step:>7,} ...", flush=True)
        try:
            W, tickers = evaluate_concentration(step)
        except FileNotFoundError as e:
            print(f"    skip — checkpoint not found")
            continue
        m = concentration_metrics(W, tickers)
        results[step] = m

    # Markdown table
    md = [
        '# NDX 2017-variant — portfolio concentration evolution\n',
        'Evaluated on test_bull (2024). Equal-weight baseline = 1/90 = 1.11%.\n',
        '| Step | top1 avg | top5 avg | top10 avg | daily top1 mean | HHI mean | Cash avg |',
        '|---|---|---|---|---|---|---|',
    ]
    for step, m in results.items():
        md.append(f"| {step:,} | {m['top1_avg']*100:.2f}% | "
                  f"{m['top5_avg']*100:.2f}% | {m['top10_avg']*100:.2f}% | "
                  f"{m['daily_top1_mean']*100:.2f}% | {m['hhi_mean']:.4f} | "
                  f"{m['cash_avg']*100:.2f}% |")
    md.append('\n## Top 10 stocks by average weight at each checkpoint\n')
    for step, m in results.items():
        md.append(f"### Step {step:,}\n")
        stock_strs = ', '.join(f"{t} ({w*100:.2f}%)" for t, w in m['top10_stocks'])
        md.append(f"  {stock_strs}\n")

    with open('results/ndx_2017_concentration_table.md','w') as f:
        f.write('\n'.join(md))

    # Console table
    print(f"\n  {'Step':>9} {'top1':>7} {'top5':>7} {'top10':>7} {'daily top1':>11} {'HHI':>7} {'cash':>6}")
    print('  ' + '-'*60)
    for step, m in results.items():
        print(f"  {step:>9,} {m['top1_avg']*100:>6.2f}% {m['top5_avg']*100:>6.2f}% "
              f"{m['top10_avg']*100:>6.2f}% {m['daily_top1_mean']*100:>10.2f}% "
              f"{m['hhi_mean']:>7.4f} {m['cash_avg']*100:>5.2f}%")

    print(f"\n  Top 10 stocks by avg weight per checkpoint:")
    for step, m in results.items():
        stocks = ', '.join(f"{t}({w*100:.1f}%)" for t, w in m['top10_stocks'])
        print(f"    step {step:>7,}: {stocks}")

    # Save JSON
    with open('results/ndx_2017_concentration.json','w') as f:
        json.dump(results, f, indent=2, default=float)

    # PNG — two panels
    steps = list(results.keys())
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ax = axes[0]
    ax.plot(steps, [results[s]['top1_avg']*100 for s in steps],
            'o-', label='top-1 weight avg', color='#d62728')
    ax.plot(steps, [results[s]['top5_avg']*100 for s in steps],
            's-', label='top-5 weight sum avg', color='#ff7f0e')
    ax.plot(steps, [results[s]['top10_avg']*100 for s in steps],
            '^-', label='top-10 weight sum avg', color='#2ca02c')
    ax.axhline(1/90*100, color='gray', ls=':', label='equal weight (1/90 = 1.11%)')
    ax.axhline(5/90*100, color='gray', ls='--', alpha=0.6, label='5/90 = 5.56% (EW top-5)')
    ax.axhline(10/90*100, color='gray', ls='-.', alpha=0.6, label='10/90 = 11.11% (EW top-10)')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Weight (%)')
    ax.set_title('NDX 2017 — concentration on test_bull 2024 across training', weight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, [results[s]['hhi_mean'] for s in steps],
            'o-', color='#1f77b4', label='HHI (sum of squared weights)')
    ax.axhline(1/90, color='gray', ls=':', label='equal-weight HHI = 1/90 = 0.0111')
    ax.set_xlabel('Training step')
    ax.set_ylabel('HHI')
    ax.set_title('Herfindahl–Hirschman Index — concentration measure (higher = more concentrated)',
                  weight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/ndx_2017_concentration_evolution.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('\n✅ saved results/ndx_2017_concentration_evolution.png')
    print('✅ saved results/ndx_2017_concentration.json')
    print('✅ saved results/ndx_2017_concentration_table.md')


if __name__ == '__main__':
    main()
