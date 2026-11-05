"""
Markowitz Mean-Variance rolling baseline.

Classical 1952 portfolio optimization as a non-learned comparator. Two flavours:
  GMV  — long-only Global Minimum Variance (the safest classical baseline)
  MaxSharpe — long-only Tangency / max-Sharpe portfolio

Protocol:
  - Lookback window: 60 trading days
  - Rebalance frequency: 20 days (monthly)
  - Long-only, sum(w) = 1, max single-stock weight 30% (matches our hunter cap idea)
  - Use closing-price returns from the test-period CSV ITSELF (no leakage —
    the env clock provides time; we just read t-60..t-1 to estimate μ, Σ).
  - Transaction cost: 10 bps per turnover, matching env.

Outputs:
  results/markowitz_baseline.json
  results/markowitz_baseline.md
  results/markowitz_baseline.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

LOOKBACK = 60
REBAL_FREQ = 20
MAX_W = 0.30
TX_COST = 0.001    # 10 bps
ANNUALIZE = 252


def load_returns(df_path):
    """Pivot test CSV → wide returns DataFrame (rows=date, cols=ticker).

    Use the pre-computed `daily_return` column (cleaned/adjusted by data
    preparation) rather than raw close-price pct_change — the latter picks
    up unadjusted splits / dividends and explodes Markowitz weights.
    """
    df = pd.read_csv(f'data/{df_path}')
    df['date'] = pd.to_datetime(df['date'])
    rets = df.pivot(index='date', columns='ticker', values='daily_return')
    # Clip extreme outliers (anything >50% move in a day is data error for index stocks)
    rets = rets.clip(lower=-0.5, upper=0.5)
    # Drop first day (NaN by construction)
    rets = rets.dropna(how='all')
    return rets


def gmv_weights(R, max_w=MAX_W):
    """Long-only Global Minimum Variance with cap. Solves via projected gradient."""
    cov = R.cov().values
    n = cov.shape[0]
    # Closed-form GMV: w = Σ^{-1} 1 / (1' Σ^{-1} 1), then clip + renormalise.
    try:
        inv = np.linalg.pinv(cov + 1e-6 * np.eye(n))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov + 1e-4 * np.eye(n))
    ones = np.ones(n)
    w = inv @ ones
    w = w / w.sum()
    # Project: clip negatives to 0, cap at max_w, then renormalise iteratively
    for _ in range(50):
        w = np.clip(w, 0.0, max_w)
        s = w.sum()
        if s < 1e-9:
            w = np.ones(n) / n
            break
        w = w / s
        if (w <= max_w + 1e-9).all() and (w >= -1e-9).all():
            break
    return w


def max_sharpe_weights(R, max_w=MAX_W):
    """Long-only Max-Sharpe (tangency)."""
    mu = R.mean().values * ANNUALIZE
    cov = R.cov().values * ANNUALIZE
    n = len(mu)
    try:
        inv = np.linalg.pinv(cov + 1e-6 * np.eye(n))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov + 1e-4 * np.eye(n))
    w = inv @ mu
    # If all negative-expected-return, fall back to GMV
    if (mu <= 0).all() or w.sum() <= 0:
        return gmv_weights(R, max_w)
    w = w / w.sum()
    for _ in range(50):
        w = np.clip(w, 0.0, max_w)
        s = w.sum()
        if s < 1e-9:
            return gmv_weights(R, max_w)
        w = w / s
        if (w <= max_w + 1e-9).all() and (w >= -1e-9).all():
            break
    return w


def backtest(rets, strategy='gmv', lookback=LOOKBACK, rebal=REBAL_FREQ, init_value=1_000_000):
    """Walk-forward backtest. Returns portfolio value series."""
    R = rets.values
    dates = rets.index
    n_days, n_stocks = R.shape
    values = [init_value]
    w = np.ones(n_stocks) / n_stocks  # start equal-weighted before first rebal
    for t in range(1, n_days):
        if t >= lookback and (t - lookback) % rebal == 0:
            R_train = rets.iloc[t - lookback:t]
            if strategy == 'gmv':
                new_w = gmv_weights(R_train)
            else:
                new_w = max_sharpe_weights(R_train)
            # transaction cost on turnover
            turn = np.abs(new_w - w).sum() / 2.0
            values[-1] = values[-1] * (1.0 - turn * TX_COST)
            w = new_w
        # apply day t's returns
        r_t = np.nan_to_num(R[t], nan=0.0)
        port_r = float(w @ r_t)
        values.append(values[-1] * (1.0 + port_r))
    return np.array(values)


def metrics(values):
    """Match the project's official Sharpe formula used in
    train_final_extended.py calculate_metrics: ann_ret/ann_vol with
    geometric annualised return."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    r = v[1:] / v[:-1] - 1.0
    cum = float((v[-1] / v[0] - 1.0) * 100.0)
    ann_ret = float(((v[-1] / v[0]) ** (ANNUALIZE / n) - 1) * 100)
    ann_vol = float(r.std() * np.sqrt(ANNUALIZE) * 100)
    s = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    eq = v.copy()
    mdd = float(((eq / np.maximum.accumulate(eq)) - 1.0).min() * 100.0)
    return {'cum_ret': cum, 'sharpe': s, 'mdd': mdd}


def main():
    PERIODS_DJI = {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    }
    PERIODS_SP = {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    }
    out = {}
    for u, P in [('dji', PERIODS_DJI), ('sp100', PERIODS_SP)]:
        for p, csv in P.items():
            R = load_returns(csv)
            for strat in ['gmv', 'max_sharpe']:
                v = backtest(R, strategy=strat)
                m = metrics(v)
                key = f'{strat}|{u}|{p}'
                out[key] = m
                print(f"  {key:<28} → cum={m['cum_ret']:+7.2f}%  Sharpe={m['sharpe']:+.3f}  MDD={m['mdd']:+.2f}%")

    os.makedirs('results', exist_ok=True)
    with open('results/markowitz_baseline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n✅ saved results/markowitz_baseline.json")

    # Markdown table
    lines = ['# Markowitz Mean-Variance baselines (60-day lookback, 20-day rebalance, 30% cap)\n']
    lines.append('Long-only constraint, transaction cost 10 bps per turnover, walk-forward.\n')
    lines.append('| Strategy | Universe | Period | Cum.Ret % | Sharpe | MDD % |')
    lines.append('|---|---|---|---|---|---|')
    for k, v in out.items():
        s, u, p = k.split('|')
        lines.append(f"| {s} | {u.upper()} | {p} | {v['cum_ret']:+.2f} | {v['sharpe']:+.3f} | {v['mdd']:+.2f} |")
    with open('results/markowitz_baseline.md', 'w') as f:
        f.write('\n'.join(lines))
    print("✅ saved results/markowitz_baseline.md")

    # PNG — MV (both flavours) vs MAC v3 vs benchmarks per universe per period
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    periods = ['val', 'test_bear', 'test_2023', 'test_bull']
    for ax_idx, u in enumerate(['dji', 'sp100']):
        ax = axes[ax_idx]
        x = np.arange(len(periods))
        w = 0.18
        gmv_vals = [out[f'gmv|{u}|{p}']['cum_ret'] for p in periods]
        ms_vals  = [out[f'max_sharpe|{u}|{p}']['cum_ret'] for p in periods]
        # Load MAC v3 from existing sweep
        with open(f'results/mac_{u}_v3/checkpoint_sweep.json') as f:
            sw = json.load(f)
        mac = max(sw, key=lambda r: r['val_sharpe'])
        mac_vals = [mac['val_ret'], mac['test_bear_ret'], mac['test_2023_ret'], mac['test_bull_ret']]
        # Load universal v6
        with open(f'results/universal_v6_{u}/final_results.json') as f:
            v6 = json.load(f)
        v6_vals = [v6['periods'][p]['cumulative_return'] for p in periods]

        ax.bar(x - 1.5*w, mac_vals,  w, label='MAC v3 (ours)',     color='#1f77b4')
        ax.bar(x - 0.5*w, v6_vals,   w, label='Universal v6',       color='#d62728')
        ax.bar(x + 0.5*w, gmv_vals,  w, label='Markowitz GMV',      color='#2ca02c', hatch='//')
        ax.bar(x + 1.5*w, ms_vals,   w, label='Markowitz Max-Shp',  color='#9467bd', hatch='//')
        ax.set_xticks(x); ax.set_xticklabels(periods)
        ax.set_ylabel('Cumulative return (%)')
        ax.set_title(f'{u.upper()} — Learned (MAC v3, v6) vs Classical (Markowitz)')
        ax.axhline(0, color='gray', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper left' if u == 'dji' else 'lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig('results/markowitz_baseline.png', dpi=130, bbox_inches='tight')
    print("✅ saved results/markowitz_baseline.png")
    plt.close()


if __name__ == '__main__':
    main()
