"""
Markowitz baseline extension — NDX-2017 (90-stock variant).

Same protocol as models/markowitz_baseline.py (60-day lookback,
20-day rebalance, 30% cap, 10 bps tx cost, long-only walk-forward).
Adds NDX-2017 to the existing JSON without re-running DJI/SP100.

Output: appends NDX entries to results/_cross_analyses/markowitz_baseline/
        markowitz_baseline.json and refreshes the .md table.
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from markowitz_baseline import load_returns, backtest, metrics


PERIODS_NDX = {
    'val':       'val_data_ndx_2017.csv',
    'test_bear': 'test_bear_data_ndx_2017.csv',
    'test_2023': 'test_2023_data_ndx_2017.csv',
    'test_bull': 'test_bull_data_ndx_2017.csv',
}

OUT_JSON = 'results/_cross_analyses/markowitz_baseline/markowitz_baseline.json'
OUT_MD   = 'results/_cross_analyses/markowitz_baseline/markowitz_baseline.md'


def main():
    with open(OUT_JSON) as f:
        out = json.load(f)

    for p, csv in PERIODS_NDX.items():
        R = load_returns(csv)
        for strat in ['gmv', 'max_sharpe']:
            v = backtest(R, strategy=strat)
            m = metrics(v)
            key = f'{strat}|ndx|{p}'
            out[key] = m
            print(f"  {key:<28} → cum={m['cum_ret']:+7.2f}%  Sharpe={m['sharpe']:+.3f}  MDD={m['mdd']:+.2f}%")

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    lines = ['# Markowitz Mean-Variance baselines (60-day lookback, 20-day rebalance, 30% cap)\n']
    lines.append('Long-only constraint, transaction cost 10 bps per turnover, walk-forward.\n')
    lines.append('| Strategy | Universe | Period | Cum.Ret % | Sharpe | MDD % |')
    lines.append('|---|---|---|---|---|---|')
    for k, v in out.items():
        s, u, p = k.split('|')
        lines.append(f"| {s} | {u.upper()} | {p} | {v['cum_ret']:+.2f} | {v['sharpe']:+.3f} | {v['mdd']:+.2f} |")
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == '__main__':
    main()
