"""
Extended 2025 simulator that captures the FULL day-by-day trajectory
for the interactive investor-experience dashboard page.

For each universe + canonical agent it captures:
  - dates (list of YYYY-MM-DD strings)
  - portfolio values
  - daily returns
  - per-day stock weights (full vector, last entry is cash)
  - per-day regime label (0=bull, 1=bear, 2=sideways)
  - per-day cash %
  - tickers list (order matches the weights vector)

Also captures the BH index level and EW portfolio value per day on the
same calendar so the investor page can show comparisons.

Output: results/_canonical/simulator_2025_full.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))

from portfolio_env import PortfolioEnv                       # type: ignore
from train_final_extended import AttentionPortfolioExtractor  # noqa: F401
from train_universal_v6_sp100 import V3_COLS_TO_DROP         # type: ignore

from simulator_2025 import (
    CANONICAL, INDEX_TICKERS, EVAL_START, EVAL_END,
    HMM_TRAIN_START, DOWNLOAD_START, DOWNLOAD_END,
    download_indices, download_universe_stocks,
    build_hmm_features, fit_hmm, apply_hmm,
    assemble_universe_panel, metrics,
    bh_values, ew_values,
)

REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'sideways'}


def run_canonical_full(model_path, df, floor):
    """Run the canonical agent and capture FULL per-day trajectory."""
    df = df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])
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
    tickers = list(env.tickers)
    model = PPO.load(str(model_path), device='cpu')

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

    history = env.get_portfolio_history()
    history['date'] = pd.to_datetime(history['date'])

    # Extract per-day arrays
    dates = [d.strftime('%Y-%m-%d') for d in history['date']]
    values = history['value'].astype(float).tolist()
    weights_matrix = np.stack(history['weights'].tolist())   # (T, N+1)
    regimes = history['regime'].astype(int).tolist()
    returns = history['return'].fillna(0.0).astype(float).tolist() if 'return' in history.columns else [0.0] * len(dates)

    # Per-day cash and top holdings
    cash_pct = (weights_matrix[:, -1] * 100.0).tolist()
    stock_weights = weights_matrix[:, :-1]   # (T, N)

    # For each day: list of (ticker, weight) sorted by weight desc, top 10 only
    top_holdings = []
    for t in range(weights_matrix.shape[0]):
        w = stock_weights[t]
        ordered = sorted(zip(tickers, w.tolist()), key=lambda kv: kv[1], reverse=True)
        top_holdings.append([{'ticker': tk, 'weight': float(wt)}
                              for tk, wt in ordered[:10] if wt > 1e-4])

    return {
        'tickers':       tickers,
        'dates':         dates,
        'values':        values,
        'returns':       returns,
        'cash_pct':      cash_pct,
        'regimes':       regimes,
        'regime_names':  [REGIME_NAMES[r] for r in regimes],
        'top_holdings':  top_holdings,
    }


def bh_series(index_df, start, end):
    df = index_df[(index_df['date'] >= start) & (index_df['date'] <= end)].copy()
    df['date'] = pd.to_datetime(df['date'])
    return {
        'dates':  [d.strftime('%Y-%m-%d') for d in df['date']],
        'values': df['close'].astype(float).tolist(),
    }


def ew_series(panel, start, end, tx_cost=0.001):
    """Return list of {date, value} for daily-rebalanced equal-weight."""
    panel = panel[(panel['date'] >= start) & (panel['date'] <= end)].copy()
    panel['ret'] = panel.groupby('ticker')['close'].pct_change()
    panel = panel.dropna(subset=['ret'])
    wide = panel.pivot(index='date', columns='ticker', values='ret').fillna(0.0)
    if wide.empty:
        return {'dates': [], 'values': []}

    n = wide.shape[1]
    target = np.full(n, 1.0 / n)
    weights = target.copy()
    pv = 1_000_000.0
    dates = [wide.index[0].strftime('%Y-%m-%d')]
    values = [pv]
    for t in range(wide.shape[0]):
        r = wide.iloc[t].values
        port_r = float(weights @ r)
        pv *= (1 + port_r)
        drifted = weights * (1 + r) / (1 + port_r) if abs(1 + port_r) > 1e-12 else target.copy()
        turn = float(np.abs(target - drifted).sum())
        pv *= (1 - turn * tx_cost)
        weights = target.copy()
        dates.append(wide.index[t].strftime('%Y-%m-%d'))
        values.append(pv)
    return {'dates': dates, 'values': values}


def main():
    print('=' * 72)
    print('   2025 Forward Simulator — FULL trajectory capture')
    print('=' * 72)

    print('\n[1/4] Loading cached indices and stocks...')
    indices = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    stocks_data = {
        uname: download_universe_stocks(cfg['stock_list_file'], DOWNLOAD_START, DOWNLOAD_END)
        for uname, cfg in CANONICAL.items()
    }

    print('[2/4] Refitting HMMs and labelling 2025...')
    hmm_feats_dji   = build_hmm_features(indices['dji'],   indices['vix'])
    hmm_feats_sp500 = build_hmm_features(indices['sp500'], indices['vix'])
    hmm_dji   = fit_hmm(hmm_feats_dji,   seed=42)
    hmm_sp500 = fit_hmm(hmm_feats_sp500, seed=42)
    regime_dji   = apply_hmm(hmm_dji,   hmm_feats_dji)
    regime_sp500 = apply_hmm(hmm_sp500, hmm_feats_sp500)

    out = {}
    for uname, cfg in CANONICAL.items():
        print(f'\n[3-4/4] {uname}: assembling, running canonical agent with FULL capture...')
        panel = assemble_universe_panel(uname, cfg, stocks_data[uname], indices,
                                        regime_dji, regime_sp500)
        if panel.empty or panel['ticker'].nunique() < 5:
            print(f'  ERROR: insufficient data for {uname}')
            continue

        agent_full = run_canonical_full(cfg['ckpt'], panel, cfg['floor'])
        agent_m = metrics(agent_full['values'])
        print(f"  AGENT cum={agent_m['cumulative_return']:+7.2f}%  Sh={agent_m['sharpe_ratio']:+.3f}  n={agent_m['n_days']}")

        # BH and EW series (daily values)
        bh_key = 'ndx' if cfg['bh_ticker'] == '^NDX' else (
                 'sp500' if cfg['bh_ticker'] == '^GSPC' else 'dji')
        bh_ser = bh_series(indices[bh_key], EVAL_START, EVAL_END)
        ew_ser = ew_series(stocks_data[uname], EVAL_START, EVAL_END)
        bh_m = metrics(bh_ser['values'])
        ew_m = metrics(ew_ser['values'])

        out[uname] = {
            'universe_name': cfg['universe_name'],
            'eval_window':   f'{EVAL_START} to {EVAL_END}',
            'floor':         cfg['floor'],
            'agent':         {**agent_full, **agent_m},
            'BH':            {**bh_ser,     **bh_m, 'name': f"BH {cfg['bh_ticker'].lstrip('^')}"},
            'EW':            {**ew_ser,     **ew_m, 'name': f"EW {cfg['universe_name']}"},
        }

    out_path = PROJECT_ROOT / 'results' / '_canonical' / 'simulator_2025_full.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
