"""
Three-year combined backtest (2022-2024) for Section 4.7.5.

Chains the three test periods (bear 2022, recovery 2023, bull 2024)
into one continuous wealth series, then computes combined cumulative
return, Sharpe and MDD on the pooled daily series. The chaining
preserves end-of-period wealth into the start of the next period, so
the cumulative return is the true compounded outcome of running the
strategy continuously across 2022-2024.

Strategies compared (per universe):
  - Canonical PPO agent (loaded from combined_backtest_2022_2024.json)
  - Buy-and-hold broad-market index (BH)
  - Daily-rebalanced equal-weight portfolio (EW)
  - Markowitz GMV  (60-day lookback, 20-day rebal, 30% cap, 10 bps cost)
  - Markowitz Max-Sharpe (same protocol)

Bootstrap CIs on combined Sharpe use stationary block bootstrap with
N = 1000 resamples and block length max(5, sqrt(T)).

Output: results/_canonical/combined_3yr_backtest.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark_metrics import bh_index_values, ew_values
from markowitz_baseline import load_returns, backtest as markowitz_backtest


PERIODS = ['test_bear', 'test_2023', 'test_bull']

UNIVERSES = {
    'DJI': {
        'csv_key': 'dji',
        'periods': {
            'test_bear': 'test_bear_data_ext.csv',
            'test_2023': 'test_2023_data_ext.csv',
            'test_bull': 'test_bull_data_ext.csv',
        },
    },
    'SP100': {
        'csv_key': 'sp100',
        'periods': {
            'test_bear': 'test_bear_data_sp100.csv',
            'test_2023': 'test_2023_data_sp100.csv',
            'test_bull': 'test_bull_data_sp100.csv',
        },
    },
    'NDX': {
        'csv_key': 'ndx',
        'periods': {
            'test_bear': 'test_bear_data_ndx_2017.csv',
            'test_2023': 'test_2023_data_ndx_2017.csv',
            'test_bull': 'test_bull_data_ndx_2017.csv',
        },
    },
}


def metrics_from_values(v):
    v = np.asarray(v, dtype=float)
    n = len(v)
    cum_ret = float((v[-1] / v[0] - 1) * 100)
    dr = np.diff(v) / v[:-1]
    ann_ret = float(((v[-1] / v[0]) ** (252 / n) - 1) * 100)
    ann_vol = float(dr.std() * np.sqrt(252) * 100)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(((v - peak) / peak).min() * 100)
    return {
        'cumulative_return': cum_ret,
        'sharpe_ratio': sharpe,
        'max_drawdown': mdd,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'n_days': n,
    }


def chain_values(period_values_list, init=1_000_000.0):
    """Chain per-period value series so wealth carries across boundaries.
    Each period's first value is dropped after the first (avoid duplicate)."""
    combined = []
    current_start = init
    for i, v in enumerate(period_values_list):
        v = np.asarray(v, dtype=float)
        normalised = v / v[0] * current_start
        if i == 0:
            combined.append(normalised)
        else:
            combined.append(normalised[1:])
        current_start = float(normalised[-1])
    return np.concatenate(combined)


def stationary_block_bootstrap_sharpe(daily_returns, n_iter=1000, seed=42):
    """Stationary block bootstrap of the annualised Sharpe.
    Returns (lower 2.5%, upper 97.5%)."""
    rng = np.random.default_rng(seed)
    r = np.asarray(daily_returns, dtype=float)
    n = len(r)
    block_len = max(5, int(np.sqrt(n)))
    samples = []
    for _ in range(n_iter):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            length = rng.geometric(1.0 / block_len)
            for j in range(length):
                idx.append((start + j) % n)
                if len(idx) >= n:
                    break
        rb = r[np.array(idx[:n])]
        mu = rb.mean()
        sigma = rb.std()
        if sigma > 0:
            samples.append(float(mu / sigma * np.sqrt(252)))
    samples = np.array(samples)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main():
    # === Load agent daily-return chains from existing combined file ===
    with open('results/_canonical/combined_backtest_2022_2024.json') as f:
        agent_data = json.load(f)

    out = {}

    for u, cfg in UNIVERSES.items():
        print(f"\n{'='*70}\n   {u}\n{'='*70}")
        period_csvs = cfg['periods']
        csv_key = cfg['csv_key']

        # ---- AGENT ----
        agent_daily = np.array(agent_data[u]['daily_returns'])
        agent_values = np.array(agent_data[u]['values'])
        m_agent = metrics_from_values(agent_values)
        lo, hi = stationary_block_bootstrap_sharpe(agent_daily[1:])
        print(f"  AGENT  cum={m_agent['cumulative_return']:+7.2f}%  "
              f"Sh={m_agent['sharpe_ratio']:+.3f} [{lo:+.2f}, {hi:+.2f}]  "
              f"MDD={m_agent['max_drawdown']:+6.2f}%  n={m_agent['n_days']}")

        # ---- BH ----
        bh_period_values = []
        for p in PERIODS:
            bh_period_values.append(bh_index_values(csv_key, period_csvs[p]))
        bh_chain = chain_values(bh_period_values)
        m_bh = metrics_from_values(bh_chain)
        bh_returns = np.diff(bh_chain) / bh_chain[:-1]
        lo_bh, hi_bh = stationary_block_bootstrap_sharpe(bh_returns)
        print(f"  BH     cum={m_bh['cumulative_return']:+7.2f}%  "
              f"Sh={m_bh['sharpe_ratio']:+.3f} [{lo_bh:+.2f}, {hi_bh:+.2f}]  "
              f"MDD={m_bh['max_drawdown']:+6.2f}%  n={m_bh['n_days']}")

        # ---- EW ----
        ew_period_values = []
        for p in PERIODS:
            ew_period_values.append(ew_values(period_csvs[p]))
        ew_chain = chain_values(ew_period_values)
        m_ew = metrics_from_values(ew_chain)
        ew_returns = np.diff(ew_chain) / ew_chain[:-1]
        lo_ew, hi_ew = stationary_block_bootstrap_sharpe(ew_returns)
        print(f"  EW     cum={m_ew['cumulative_return']:+7.2f}%  "
              f"Sh={m_ew['sharpe_ratio']:+.3f} [{lo_ew:+.2f}, {hi_ew:+.2f}]  "
              f"MDD={m_ew['max_drawdown']:+6.2f}%  n={m_ew['n_days']}")

        # ---- GMV ----
        gmv_period_values = []
        ms_period_values = []
        for p in PERIODS:
            R = load_returns(period_csvs[p])
            gmv_period_values.append(markowitz_backtest(R, strategy='gmv'))
            ms_period_values.append(markowitz_backtest(R, strategy='max_sharpe'))
        gmv_chain = chain_values(gmv_period_values)
        ms_chain  = chain_values(ms_period_values)
        m_gmv = metrics_from_values(gmv_chain)
        m_ms  = metrics_from_values(ms_chain)
        gmv_returns = np.diff(gmv_chain) / gmv_chain[:-1]
        ms_returns  = np.diff(ms_chain)  / ms_chain[:-1]
        lo_g, hi_g = stationary_block_bootstrap_sharpe(gmv_returns)
        lo_m, hi_m = stationary_block_bootstrap_sharpe(ms_returns)
        print(f"  GMV    cum={m_gmv['cumulative_return']:+7.2f}%  "
              f"Sh={m_gmv['sharpe_ratio']:+.3f} [{lo_g:+.2f}, {hi_g:+.2f}]  "
              f"MDD={m_gmv['max_drawdown']:+6.2f}%  n={m_gmv['n_days']}")
        print(f"  MaxSh  cum={m_ms['cumulative_return']:+7.2f}%  "
              f"Sh={m_ms['sharpe_ratio']:+.3f} [{lo_m:+.2f}, {hi_m:+.2f}]  "
              f"MDD={m_ms['max_drawdown']:+6.2f}%  n={m_ms['n_days']}")

        out[u] = {
            'AGENT': {**m_agent, 'sharpe_ci': [lo, hi]},
            'BH':    {**m_bh,    'sharpe_ci': [lo_bh, hi_bh]},
            'EW':    {**m_ew,    'sharpe_ci': [lo_ew, hi_ew]},
            'GMV':   {**m_gmv,   'sharpe_ci': [lo_g, hi_g]},
            'MaxSh': {**m_ms,    'sharpe_ci': [lo_m, hi_m]},
        }

    os.makedirs('results/_canonical', exist_ok=True)
    out_path = 'results/_canonical/combined_3yr_backtest.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Wrote {out_path}")


if __name__ == '__main__':
    main()
