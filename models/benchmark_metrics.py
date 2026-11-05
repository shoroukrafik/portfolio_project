"""
Per-universe benchmark metrics (BH + EW) on each evaluation period.

For each (universe, period):
  - BH  = buy-and-hold the broad-market index
            DJI universe   → DJI index
            SP100 universe → S&P 500 index
            NDX universe   → NASDAQ-100 index
  - EW  = equal-weight portfolio of the universe's constituents,
          rebalanced daily.

Computes cum return, annualised Sharpe (μ/σ × √252), and max drawdown
on the SAME period CSVs the canonical eval uses. Output JSON keyed by
{strategy}|{universe}|{period}.

Output: results/_canonical/benchmark_metrics.json
"""

import os
import sys
import json

import numpy as np
import pandas as pd


PERIOD_CSVS = {
    'dji': {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'sp100': {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    },
    'ndx': {
        'val':       'val_data_ndx_2017.csv',
        'test_bear': 'test_bear_data_ndx_2017.csv',
        'test_2023': 'test_2023_data_ndx_2017.csv',
        'test_bull': 'test_bull_data_ndx_2017.csv',
    },
}

INDEX_CSV = {
    'dji':   'dji_index_ext.csv',
    'sp100': 'sp500_index_ext.csv',
    'ndx':   'ndx_index_ext.csv',
}

OUT_PATH = 'results/_canonical/benchmark_metrics.json'


def metrics(values):
    """Match the project's official Sharpe formula used in
    train_final_extended.py calculate_metrics: ann_ret/ann_vol with
    geometric annualised return."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    cum_ret = float((v[-1] / v[0] - 1) * 100)
    dr = np.diff(v) / v[:-1]
    ann_ret = float(((v[-1] / v[0]) ** (252 / n) - 1) * 100)
    ann_vol = float(dr.std() * np.sqrt(252) * 100)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(((v - peak) / peak).min() * 100)
    return {'cumulative_return': cum_ret, 'sharpe_ratio': sharpe, 'max_drawdown': mdd}


def bh_index_values(universe, period_csv):
    """Buy-and-hold the broad-market index over the same date range."""
    period_df = pd.read_csv(f'data/{period_csv}')
    period_df['date'] = pd.to_datetime(period_df['date'])
    dates = sorted(period_df['date'].unique())
    start, end = dates[0], dates[-1]

    idx = pd.read_csv(f'data/{INDEX_CSV[universe]}')
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx[(idx['date'] >= start) & (idx['date'] <= end)].sort_values('date')
    return idx['close'].values


def ew_values(period_csv, tx_cost=0.001):
    """Daily-rebalanced equal-weight portfolio with 10bps transaction costs.

    Replicates the canonical project EW protocol:
      - Day 0: enter equal-weight (1/N each stock); no setup cost charged
      - Each day after market close: weights drift with stock returns,
        then rebalance back to 1/N. Turnover = sum |new_w - drifted_w|.
        Cost = turnover * 0.001 is deducted from portfolio value.

    This matches the EW definition used throughout the project, including
    the BENCH dictionary in models/eval_universal_v6_sweep.py.
    """
    df = pd.read_csv(f'data/{period_csv}')
    df['date'] = pd.to_datetime(df['date'])
    rets = df.pivot(index='date', columns='ticker', values='daily_return')
    rets = rets.dropna(how='all').fillna(0.0)
    n_stocks = rets.shape[1]
    target = np.full(n_stocks, 1.0 / n_stocks)

    weights = target.copy()
    portfolio_value = 1.0
    values = [portfolio_value]

    for t in range(rets.shape[0]):
        r_t = rets.iloc[t].values
        # Portfolio return today given current weights
        port_ret = float(np.dot(weights, r_t))
        # Update portfolio value (gross of cost)
        portfolio_value *= (1.0 + port_ret)
        # Compute drifted weights after market move
        if abs(1.0 + port_ret) > 1e-12:
            drifted = weights * (1.0 + r_t) / (1.0 + port_ret)
        else:
            drifted = target.copy()
        # Rebalance to equal weights, charge turnover cost
        turnover = float(np.abs(target - drifted).sum())
        cost = turnover * tx_cost
        portfolio_value *= (1.0 - cost)
        # Reset weights for next day
        weights = target.copy()
        values.append(portfolio_value)

    return np.array(values)


def main():
    out = {}
    for u, periods in PERIOD_CSVS.items():
        for p, csv in periods.items():
            bh = metrics(bh_index_values(u, csv))
            ew = metrics(ew_values(csv))
            out[f'BH|{u}|{p}'] = bh
            out[f'EW|{u}|{p}'] = ew
            print(f"  {u.upper():>5} {p:<10}  "
                  f"BH: cum={bh['cumulative_return']:+7.2f}%  Sh={bh['sharpe_ratio']:+.3f}  MDD={bh['max_drawdown']:+6.2f}%  |  "
                  f"EW: cum={ew['cumulative_return']:+7.2f}%  Sh={ew['sharpe_ratio']:+.3f}  MDD={ew['max_drawdown']:+6.2f}%")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
