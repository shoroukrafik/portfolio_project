"""
Soft posterior-weighted cash-floor experiment.

For each canonical agent (DJI 70%, SP100 30%, NDX 0%), evaluate on both
  (a) the four thesis test periods (val 2021, bear 2022, recovery 2023, bull 2024)
  (b) the 2025 simulator window
under TWO floor regimes:
  HARD:  f_t = f_max  iff  HMM hard label == bear   (current canonical)
  SOFT:  f_t = f_max * P(bear | day t)               (new proposal)

The soft variant uses the HMM posterior probability that the current day
is bear. When the HMM is 100% confident bear, soft == hard. When the HMM
is uncertain (e.g., 40% bear posterior), the floor is 40% of canonical.

Output: results/_canonical/soft_floor_comparison.json
"""

import json
import os
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

# Reuse 2025 simulator data assembly
from simulator_2025 import (
    CANONICAL, INDEX_TICKERS,
    HMM_TRAIN_START, DOWNLOAD_START, DOWNLOAD_END,
    EVAL_START, EVAL_END,
    download_indices, download_universe_stocks,
    build_hmm_features, fit_hmm, apply_hmm,
    assemble_universe_panel, metrics,
)


# --- Map each canonical to the four thesis test CSVs ----------------------
THESIS_PERIODS = {
    'DJI':   {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'SP100': {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    },
    'NDX':   {
        'val':       'val_data_ndx_2017.csv',
        'test_bear': 'test_bear_data_ndx_2017.csv',
        'test_2023': 'test_2023_data_ndx_2017.csv',
        'test_bull': 'test_bull_data_ndx_2017.csv',
    },
}

DATA_DIR = PROJECT_ROOT / 'data'


def load_period_df(csv_name):
    df = pd.read_csv(DATA_DIR / csv_name)
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def make_env(df, floor):
    """Standard env init with the canonical floor on bear regime."""
    return PortfolioEnv(
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


def run_hard(model_path, df, f_max):
    """Standard run with the canonical hard floor."""
    env = make_env(df, f_max)
    model = PPO.load(str(model_path), device='cpu')
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()['value'].astype(float).tolist()


def run_soft(model_path, df, f_max):
    """Run with soft posterior-weighted floor.

    Implementation: disable the env's static hard floor (set all to 0)
    then before each step, override the floor for the current regime ID
    to f_max * P(bear | current day).
    """
    env = make_env(df, 0.0)  # disable static hard floor entirely
    model = PPO.load(str(model_path), device='cpu')
    obs, _ = env.reset()
    done = False
    while not done:
        cur_step = env.current_step
        # P(bear | current day). regime_prob_array is (n_days, 3): [bull, bear, side]
        if env.regime_prob_array is not None:
            p_bear = float(env.regime_prob_array[cur_step, 1])
        else:
            p_bear = 1.0 if int(env.regime_id_array[cur_step]) == 1 else 0.0
        soft_floor = float(f_max) * p_bear
        cur_regime = int(env.regime_id_array[cur_step])
        # Override the floor for the regime ID currently in play
        env.cash_floors[cur_regime] = soft_floor

        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()['value'].astype(float).tolist()


def evaluate_canonical(universe, cfg, periods_dfs, df_2025):
    """Run both HARD and SOFT on all 4 thesis periods + 2025."""
    f_max = float(cfg['floor'])
    print(f"\n=== {universe} (canonical f_max = {int(f_max*100)}%) ===")
    out = {'f_max': f_max, 'periods': {}}

    print(f"  {'Period':<14}{'HARD ret':>10}{'SOFT ret':>10}{'Δ ret':>10}"
          f"{'HARD Sh':>10}{'SOFT Sh':>10}"
          f"{'HARD MDD':>10}{'SOFT MDD':>10}")

    for pname, df in periods_dfs.items():
        if df is None or df.empty:
            continue
        hv = run_hard(cfg['ckpt'], df, f_max)
        sv = run_soft(cfg['ckpt'], df, f_max)
        hm = metrics(hv); sm = metrics(sv)
        delta_ret = sm['cumulative_return'] - hm['cumulative_return']
        out['periods'][pname] = {
            'hard': {**hm, 'values': hv},
            'soft': {**sm, 'values': sv},
            'delta_cum_ret': delta_ret,
        }
        print(f"  {pname:<14}"
              f"{hm['cumulative_return']:>+10.2f}"
              f"{sm['cumulative_return']:>+10.2f}"
              f"{delta_ret:>+10.2f}"
              f"{hm['sharpe_ratio']:>+10.3f}"
              f"{sm['sharpe_ratio']:>+10.3f}"
              f"{hm['max_drawdown']:>+10.2f}"
              f"{sm['max_drawdown']:>+10.2f}")

    # 2025
    if df_2025 is not None:
        hv = run_hard(cfg['ckpt'], df_2025, f_max)
        sv = run_soft(cfg['ckpt'], df_2025, f_max)
        hm = metrics(hv); sm = metrics(sv)
        delta_ret = sm['cumulative_return'] - hm['cumulative_return']
        out['periods']['sim_2025'] = {
            'hard': {**hm, 'values': hv},
            'soft': {**sm, 'values': sv},
            'delta_cum_ret': delta_ret,
        }
        print(f"  {'sim_2025':<14}"
              f"{hm['cumulative_return']:>+10.2f}"
              f"{sm['cumulative_return']:>+10.2f}"
              f"{delta_ret:>+10.2f}"
              f"{hm['sharpe_ratio']:>+10.3f}"
              f"{sm['sharpe_ratio']:>+10.3f}"
              f"{hm['max_drawdown']:>+10.2f}"
              f"{sm['max_drawdown']:>+10.2f}")

    return out


def main():
    print('=' * 78)
    print('   Soft posterior-weighted cash-floor experiment')
    print('=' * 78)

    # ---- Build 2025 panels (reuse simulator pipeline) ----
    print('\n[setup 1/3] Loading cached 2025 indices + stocks...')
    indices = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    stocks_data = {u: download_universe_stocks(c['stock_list_file'], DOWNLOAD_START, DOWNLOAD_END)
                   for u, c in CANONICAL.items()}

    print('[setup 2/3] Refitting HMMs and labelling 2025...')
    hmm_feats_dji   = build_hmm_features(indices['dji'],   indices['vix'])
    hmm_feats_sp500 = build_hmm_features(indices['sp500'], indices['vix'])
    hmm_dji   = fit_hmm(hmm_feats_dji,   seed=42)
    hmm_sp500 = fit_hmm(hmm_feats_sp500, seed=42)
    reg_dji   = apply_hmm(hmm_dji,   hmm_feats_dji)
    reg_sp500 = apply_hmm(hmm_sp500, hmm_feats_sp500)

    print('[setup 3/3] Assembling per-universe 2025 panels...')
    panels_2025 = {}
    for u, c in CANONICAL.items():
        p = assemble_universe_panel(u, c, stocks_data[u], indices, reg_dji, reg_sp500)
        panels_2025[u] = None if p.empty else p

    # ---- Evaluate each universe ----
    all_out = {}
    for u, cfg in CANONICAL.items():
        period_dfs = {pname: load_period_df(csv) for pname, csv in THESIS_PERIODS[u].items()}
        all_out[u] = evaluate_canonical(u, cfg, period_dfs, panels_2025.get(u))

    # ---- Save ----
    out_path = PROJECT_ROOT / 'results' / '_canonical' / 'soft_floor_comparison.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nWrote {out_path}")

    # ---- Final summary ----
    print('\n' + '=' * 78)
    print('   Summary: SOFT vs HARD floor (Δ cumulative return)')
    print('=' * 78)
    rows = []
    period_order = ['val', 'test_bear', 'test_2023', 'test_bull', 'sim_2025']
    header = f"  {'Universe':<8}" + ''.join(f"{p:>11}" for p in period_order)
    print(header)
    for u in CANONICAL:
        cells = []
        for p in period_order:
            d = all_out[u]['periods'].get(p)
            cells.append(f"{d['delta_cum_ret']:+10.2f}" if d else '       n/a')
        print(f"  {u:<8}" + ''.join(cells))


if __name__ == '__main__':
    main()
