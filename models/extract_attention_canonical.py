"""
Extract attention weights from the canonical mom_2m models.

For each canonical (DJI step 800k + 70% floor, SP100 step 1.15M + 30%
floor), runs the model through all four test periods and captures the
multi-head attention weights at every step, along with the HMM regime
label and the agent's chosen portfolio weights.

Produces:
  results/_canonical/attention/<universe>_<period>_summary.json
    contains top-15 stocks per regime + day counts
  results/_canonical/attention/<universe>_<period>_attention.npy
    raw mean attention matrix (n_stocks x n_stocks)
  results/_canonical/attention/<universe>_<period>_attention_by_regime.npy
    regime-averaged attention matrices (3 x n_stocks x n_stocks)

Output goes into results/_canonical/attention/ for use by the figure
and table generators downstream.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_final_extended import AttentionPortfolioExtractor  # required for PPO.load
from train_universal_v6_sp100 import V3_COLS_TO_DROP


CANONICAL = {
    'DJI': {
        'ckpt':  'models/saved/checkpoints/universal_v6_dji_mom_2m/step_0800000.zip',
        'floor': 0.7,
        'periods': {
            'val':       'val_data_ext.csv',
            'test_bear': 'test_bear_data_ext.csv',
            'test_2023': 'test_2023_data_ext.csv',
            'test_bull': 'test_bull_data_ext.csv',
        },
    },
    'SP100': {
        'ckpt':  'models/saved/checkpoints/universal_v6_sp100_mom_2m/step_1150000.zip',
        'floor': 0.3,
        'periods': {
            'val':       'val_data_sp100.csv',
            'test_bear': 'test_bear_data_sp100.csv',
            'test_2023': 'test_2023_data_sp100.csv',
            'test_bull': 'test_bull_data_sp100.csv',
        },
    },
}

REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'sideways'}
OUT_DIR = 'results/_canonical/attention'


def load_df(csv):
    df = pd.read_csv(f'data/{csv}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def run_episode(model, env):
    """Run one full episode capturing per-step attention, regime, weights, tickers."""
    obs, _ = env.reset()
    done = False
    regimes, attentions, weights = [], [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

        ext = model.policy.features_extractor
        if hasattr(ext, 'last_attention_weights') and ext.last_attention_weights is not None:
            attentions.append(ext.last_attention_weights[0])

        regimes.append(int(info.get('regime_id', 0)))
        weights.append(env.portfolio_weights.copy())
    return {
        'regimes':   np.asarray(regimes),
        'attention': np.asarray(attentions),   # (T, N, N)
        'weights':   np.asarray(weights),
        'tickers':   list(env.tickers),
    }


def summarize_period(out, universe, period):
    """Aggregate per-stock attention scores, top-k tickers, regime split."""
    A = out['attention']
    regimes = out['regimes']
    tickers = out['tickers']
    n_stocks = A.shape[1]

    # Average attention received per stock (column-mean of A, then mean over time)
    per_stock_attention = A.mean(axis=0).mean(axis=0)  # (N,)
    order = np.argsort(per_stock_attention)[::-1]
    top15 = [tickers[i] for i in order[:15]]
    top10 = [tickers[i] for i in order[:10]]

    # Per-regime average attention matrix
    reg_matrices = {}
    reg_top10 = {}
    reg_counts = {}
    for r_id, r_name in REGIME_NAMES.items():
        mask = regimes == r_id
        reg_counts[r_name] = int(mask.sum())
        if mask.sum() > 0:
            mat = A[mask].mean(axis=0)         # (N, N)
            reg_matrices[r_name] = mat
            per_stock_r = mat.mean(axis=0)     # (N,)
            order_r = np.argsort(per_stock_r)[::-1]
            reg_top10[r_name] = [tickers[i] for i in order_r[:10]]
        else:
            reg_matrices[r_name] = None
            reg_top10[r_name] = []

    summary = {
        'universe': universe,
        'period':   period,
        'n_steps':  int(A.shape[0]),
        'n_stocks': n_stocks,
        'regime_day_counts': reg_counts,
        'top_15_stocks_by_attention':  top15,
        'top_10_stocks_by_attention':  top10,
        'top_10_stocks_by_regime':     reg_top10,
    }

    # Save mean attention matrix (full period)
    np.save(os.path.join(OUT_DIR, f'{universe}_{period}_attention.npy'),
            A.mean(axis=0).astype(np.float32))
    # Save regime-averaged matrices stacked
    stacked = np.stack([
        reg_matrices['bull'] if reg_matrices['bull'] is not None else np.zeros((n_stocks, n_stocks)),
        reg_matrices['bear'] if reg_matrices['bear'] is not None else np.zeros((n_stocks, n_stocks)),
        reg_matrices['sideways'] if reg_matrices['sideways'] is not None else np.zeros((n_stocks, n_stocks)),
    ], axis=0).astype(np.float32)
    np.save(os.path.join(OUT_DIR, f'{universe}_{period}_attention_by_regime.npy'), stacked)
    # Save tickers list
    with open(os.path.join(OUT_DIR, f'{universe}_{period}_tickers.json'), 'w') as f:
        json.dump(tickers, f)
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_summaries = {}

    for universe, cfg in CANONICAL.items():
        print(f"\n=== {universe} canonical (floor={cfg['floor']}) ===")
        print(f"Loading {cfg['ckpt']}")
        model = PPO.load(cfg['ckpt'], device='cpu')
        all_summaries[universe] = {}
        for period, csv in cfg['periods'].items():
            print(f"  Extracting attention on {period} ...", flush=True)
            df = load_df(csv)
            env = PortfolioEnv(
                df=df,
                initial_balance=1_000_000,
                cash_floors={0: 0.0, 1: float(cfg['floor']), 2: 0.0},
                risk_penalties={0: 0.1, 1: 0.5, 2: 0.3},
                include_regime=True,
                include_regime_probs=True,
                include_vol_features=True,
                use_markowitz_reward=False,
                use_hyperheuristic=False,
            )
            out = run_episode(model, env)
            summary = summarize_period(out, universe, period)
            all_summaries[universe][period] = summary
            print(f"    steps={summary['n_steps']}  "
                  f"days bull/bear/side = "
                  f"{summary['regime_day_counts']['bull']}/"
                  f"{summary['regime_day_counts']['bear']}/"
                  f"{summary['regime_day_counts']['sideways']}")
            print(f"    top-5: {summary['top_10_stocks_by_attention'][:5]}")

    with open(os.path.join(OUT_DIR, 'all_summaries.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nWrote {OUT_DIR}/all_summaries.json plus per-period .npy and .json files")


if __name__ == '__main__':
    main()
