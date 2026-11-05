"""
Soft posterior-weighted variant of the 2025 full-trajectory simulator.

Same as models/simulator_2025_full.py but applies the soft posterior-
weighted cash floor at inference instead of the canonical hard binary
floor:
    floor_t = f_max * P_bear(t)
NDX has f_max = 0 so soft == hard; the NDX block is reused unchanged
from the canonical run.

Captures the same per-day trajectory (dates, values, returns, cash %,
regime, top holdings, tickers) so the dashboard can show day-by-day
investor experience under either floor mechanism.

Output: results/_canonical/simulator_2025_full_soft.json
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
from train_final_extended import AttentionPortfolioExtractor  # noqa
from train_universal_v6_sp100 import V3_COLS_TO_DROP

from simulator_2025 import (
    CANONICAL, HMM_TRAIN_START, DOWNLOAD_START, DOWNLOAD_END,
    download_indices, download_universe_stocks,
    build_hmm_features, fit_hmm, apply_hmm,
    assemble_universe_panel, metrics,
)
from simulator_2025_full import (
    REGIME_NAMES, bh_series, ew_series,
)

EVAL_START = '2025-01-01'
EVAL_END   = '2025-12-31'


def run_canonical_full_soft(model_path, df, f_max):
    """Run the canonical PPO checkpoint with SOFT posterior-weighted floor.

    Floor at day t = f_max * P_bear(t). Captures the full per-day trajectory
    in the same shape as simulator_2025_full.run_canonical_full.
    """
    df = df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])
    env = PortfolioEnv(
        df=df,
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: 0.0, 2: 0.0},   # disable static hard floor
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
        cur_step = env.current_step
        # Posterior-weighted floor injection (matches eval_soft_floor.run_soft)
        if env.regime_prob_array is not None:
            p_bear = float(env.regime_prob_array[cur_step, 1])
        else:
            p_bear = 1.0 if int(env.regime_id_array[cur_step]) == 1 else 0.0
        soft_floor = float(f_max) * p_bear
        cur_regime = int(env.regime_id_array[cur_step])
        env.cash_floors[cur_regime] = soft_floor

        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

    history = env.get_portfolio_history()
    history['date'] = pd.to_datetime(history['date'])

    dates = [d.strftime('%Y-%m-%d') for d in history['date']]
    values = history['value'].astype(float).tolist()
    weights_matrix = np.stack(history['weights'].tolist())
    regimes = history['regime'].astype(int).tolist()
    returns = (history['return'].fillna(0.0).astype(float).tolist()
               if 'return' in history.columns else [0.0] * len(dates))

    cash_pct = (weights_matrix[:, -1] * 100.0).tolist()
    stock_weights = weights_matrix[:, :-1]

    top_holdings = []
    for t in range(weights_matrix.shape[0]):
        w = stock_weights[t]
        ordered = sorted(zip(tickers, w.tolist()), key=lambda kv: kv[1], reverse=True)
        top_holdings.append([{'ticker': tk, 'weight': float(wt)}
                              for tk, wt in ordered[:10] if wt > 1e-4])

    return {
        'tickers':      tickers,
        'dates':        dates,
        'values':       values,
        'returns':      returns,
        'cash_pct':     cash_pct,
        'regimes':      regimes,
        'regime_names': [REGIME_NAMES[r] for r in regimes],
        'top_holdings': top_holdings,
    }


def main():
    print('=' * 72)
    print('   2025 Forward Simulator — SOFT posterior FULL trajectory')
    print('=' * 72)

    print('\n[1/4] Loading cached indices and stocks...')
    indices = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    stocks_data = {u: download_universe_stocks(c['stock_list_file'],
                                                DOWNLOAD_START, DOWNLOAD_END)
                   for u, c in CANONICAL.items()}

    print('[2/4] Refitting HMMs and labelling 2025...')
    hmm_feats_dji   = build_hmm_features(indices['dji'],   indices['vix'])
    hmm_feats_sp500 = build_hmm_features(indices['sp500'], indices['vix'])
    hmm_dji   = fit_hmm(hmm_feats_dji,   seed=42)
    hmm_sp500 = fit_hmm(hmm_feats_sp500, seed=42)
    regime_dji   = apply_hmm(hmm_dji,   hmm_feats_dji)
    regime_sp500 = apply_hmm(hmm_sp500, hmm_feats_sp500)

    out = {}
    for uname, cfg in CANONICAL.items():
        print(f'\n[3-4/4] {uname}: assembling panel, running SOFT-floor agent...')
        panel = assemble_universe_panel(uname, cfg, stocks_data[uname], indices,
                                        regime_dji, regime_sp500)
        if panel.empty or panel['ticker'].nunique() < 5:
            print(f'  ERROR: insufficient data for {uname}')
            continue

        agent_full = run_canonical_full_soft(cfg['ckpt'], panel, cfg['floor'])
        agent_m = metrics(agent_full['values'])
        print(f"  AGENT(soft) cum={agent_m['cumulative_return']:+7.2f}%  "
              f"Sh={agent_m['sharpe_ratio']:+.3f}  n={agent_m['n_days']}")

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

    out_path = PROJECT_ROOT / 'results' / '_canonical' / 'simulator_2025_full_soft.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
