"""
Figure 4.1 — equity-curve overlays per universe per period.

3 universes (DJI, SP100, NDX) x 4 periods (val, bear, recovery, bull)
= 12 panels. Each panel shows the canonical agent + DJI/SP500/NDX
buy-and-hold + equal-weight portfolio, all rebased to 100 at period
start.

Agent runs use the canonical checkpoint and cash-floor:
  DJI 30        — step 800k + 70% bear cash floor
  S&P 100       — step 1.15M + 30% bear cash floor
  NASDAQ-100    — step 100k + no cash floor

Output: images/fig_equity_curves.pdf + images/fig_equity_curves.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from stable_baselines3 import PPO
from portfolio_env import PortfolioEnv
from train_universal_v6_sp100 import V3_COLS_TO_DROP


CANONICAL = {
    'DJI 30': {
        'ckpt':  'models/saved/checkpoints/universal_v6_dji_mom_2m/step_0800000.zip',
        'floor': 0.7,
        'index': 'dji_index_ext.csv',
        'periods': {
            'Validation (2021)':  'val_data_ext.csv',
            'Bear (2022)':        'test_bear_data_ext.csv',
            'Recovery (2023)':    'test_2023_data_ext.csv',
            'Bull (2024)':        'test_bull_data_ext.csv',
        },
    },
    'S&P 100': {
        'ckpt':  'models/saved/checkpoints/universal_v6_sp100_mom_2m/step_1150000.zip',
        'floor': 0.3,
        'index': 'sp500_index_ext.csv',
        'periods': {
            'Validation (2021)':  'val_data_sp100.csv',
            'Bear (2022)':        'test_bear_data_sp100.csv',
            'Recovery (2023)':    'test_2023_data_sp100.csv',
            'Bull (2024)':        'test_bull_data_sp100.csv',
        },
    },
    'NASDAQ-100': {
        'ckpt':  'models/saved/checkpoints/universal_v6_ndx_2017_mom/step_0100000.zip',
        'floor': 0.0,
        'index': 'ndx_index_ext.csv',
        'periods': {
            'Validation (2021)':  'val_data_ndx_2017.csv',
            'Bear (2022)':        'test_bear_data_ndx_2017.csv',
            'Recovery (2023)':    'test_2023_data_ndx_2017.csv',
            'Bull (2024)':        'test_bull_data_ndx_2017.csv',
        },
    },
}


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def agent_values(model, df, floor):
    env = PortfolioEnv(
        df=df,
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: float(floor), 2: 0.0},
        risk_penalties={0: 0.1, 1: 0.5, 2: 0.3},
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_markowitz_reward=False,
        use_hyperheuristic=False,
    )
    obs, _ = env.reset()
    values = [env.portfolio_value]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        values.append(env.portfolio_value)
        done = term or trunc
    return np.array(values)


def bh_values(period_csv, index_csv):
    period_df = pd.read_csv(f'data/{period_csv}')
    period_df['date'] = pd.to_datetime(period_df['date'])
    dates = sorted(period_df['date'].unique())
    start, end = dates[0], dates[-1]
    idx = pd.read_csv(f'data/{index_csv}')
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx[(idx['date'] >= start) & (idx['date'] <= end)].sort_values('date')
    return idx['close'].values


def ew_values(period_csv, tx_cost=0.001):
    df = pd.read_csv(f'data/{period_csv}')
    df['date'] = pd.to_datetime(df['date'])
    rets = df.pivot(index='date', columns='ticker', values='daily_return')
    rets = rets.dropna(how='all').fillna(0.0)
    n_stocks = rets.shape[1]
    target = np.full(n_stocks, 1.0 / n_stocks)
    weights = target.copy()
    pv = 1.0
    values = [pv]
    for t in range(rets.shape[0]):
        r_t = rets.iloc[t].values
        port_ret = float(np.dot(weights, r_t))
        pv *= (1.0 + port_ret)
        if abs(1.0 + port_ret) > 1e-12:
            drifted = weights * (1.0 + r_t) / (1.0 + port_ret)
        else:
            drifted = target.copy()
        turnover = float(np.abs(target - drifted).sum())
        cost = turnover * tx_cost
        pv *= (1.0 - cost)
        weights = target.copy()
        values.append(pv)
    return np.array(values)


def rebase_to_100(v):
    v = np.asarray(v, dtype=float)
    if v[0] == 0:
        return v
    return 100.0 * v / v[0]


def main():
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    colors = {'agent': '#1f3a5f', 'bh': '#777777', 'ew': '#c44e52'}

    for row_idx, (u_name, cfg) in enumerate(CANONICAL.items()):
        print(f"\n=== {u_name} (floor={cfg['floor']}) ===")
        model = PPO.load(cfg['ckpt'], device='cpu')
        for col_idx, (p_name, csv) in enumerate(cfg['periods'].items()):
            print(f"  {p_name} ...", flush=True)
            df = load_df(csv)
            ag = agent_values(model, df, cfg['floor'])
            bh = bh_values(csv, cfg['index'])
            ew = ew_values(csv)

            ax = axes[row_idx, col_idx]
            ax.plot(rebase_to_100(ag), color=colors['agent'], linewidth=1.4, label='Agent')
            ax.plot(rebase_to_100(bh), color=colors['bh'],    linewidth=1.0, label='BH', linestyle='--')
            ax.plot(rebase_to_100(ew), color=colors['ew'],    linewidth=1.0, label='EW', linestyle=':')
            ax.axhline(100, color='#bbbbbb', linewidth=0.4)
            ax.grid(True, axis='y', linestyle=':', linewidth=0.3, alpha=0.5)

            if row_idx == 0:
                ax.set_title(p_name, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(u_name, fontsize=10, fontweight='bold')
            if row_idx == 2:
                ax.set_xlabel('Trading day', fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    out_dir = 'images'
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, 'fig_equity_curves.pdf')
    png = os.path.join(out_dir, 'fig_equity_curves.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=180, bbox_inches='tight')
    print(f"\nWrote {pdf}")
    print(f"Wrote {png}")


if __name__ == '__main__':
    main()
