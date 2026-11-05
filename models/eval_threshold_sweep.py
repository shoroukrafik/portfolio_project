"""
Threshold sweep for the soft posterior-weighted cash floor.

For each canonical agent (DJI 70%, SP100 30%, NDX 0%) we evaluate on all
five windows (val 2021, bear 2022, recovery 2023, bull 2024, sim 2025)
under FIVE floor formulations:
    HARD                      (canonical)
    PURE SOFT (tau=0)         (linear posterior weighting, every day)
    SOFT + tau=0.10           (gate posteriors below 0.10)
    SOFT + tau=0.20           (gate posteriors below 0.20)
    SOFT + tau=0.33           (uniform-prior threshold)
    SOFT + tau=0.50           (majority threshold)

Soft rule:
    if p_bear > tau:
        floor = f_max * p_bear
    else:
        floor = 0

Also prints the bear-posterior distribution per window so we know what
posterior values the agent actually saw.

Output: results/_canonical/threshold_sweep.json
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
from eval_soft_floor import THESIS_PERIODS, load_period_df, make_env


TAU_CANDIDATES = [0.0, 0.10, 0.20, 0.333, 0.50]


def run_with_threshold(model_path, df, f_max, tau):
    """Pure soft with low-posterior gate: floor = f_max*p_bear if p_bear>tau else 0."""
    env = make_env(df, 0.0)
    model = PPO.load(str(model_path), device='cpu')
    obs, _ = env.reset()
    done = False
    while not done:
        cur_step = env.current_step
        if env.regime_prob_array is not None:
            p_bear = float(env.regime_prob_array[cur_step, 1])
        else:
            p_bear = 1.0 if int(env.regime_id_array[cur_step]) == 1 else 0.0
        if p_bear > tau:
            soft_floor = float(f_max) * p_bear
        else:
            soft_floor = 0.0
        cur_regime = int(env.regime_id_array[cur_step])
        env.cash_floors[cur_regime] = soft_floor

        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()['value'].astype(float).tolist()


def run_hard(model_path, df, f_max):
    env = make_env(df, f_max)
    model = PPO.load(str(model_path), device='cpu')
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()['value'].astype(float).tolist()


def posterior_distribution(df):
    """Return basic statistics of p_bear across the dataframe's days."""
    if 'regime_bear_prob' in df.columns:
        # one row per (date, ticker), so pick first ticker per date
        per_day = df.groupby('date')['regime_bear_prob'].first().values
    else:
        return {'n_days': 0}
    p = np.asarray(per_day, dtype=float)
    return {
        'n_days':      int(len(p)),
        'mean':        float(p.mean()),
        'median':      float(np.median(p)),
        'p10':         float(np.percentile(p, 10)),
        'p25':         float(np.percentile(p, 25)),
        'p75':         float(np.percentile(p, 75)),
        'p90':         float(np.percentile(p, 90)),
        'frac_lt_010': float((p < 0.10).mean()),
        'frac_lt_020': float((p < 0.20).mean()),
        'frac_lt_033': float((p < 0.333).mean()),
        'frac_lt_050': float((p < 0.50).mean()),
        'frac_gt_080': float((p > 0.80).mean()),
    }


def main():
    print('=' * 80)
    print('   Soft posterior threshold sweep')
    print('=' * 80)

    print('\n[setup] Loading cached indices + stocks...')
    indices = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    stocks_data = {u: download_universe_stocks(c['stock_list_file'], DOWNLOAD_START, DOWNLOAD_END)
                   for u, c in CANONICAL.items()}

    print('[setup] Refitting HMMs and labelling 2025...')
    hmm_dji   = fit_hmm(build_hmm_features(indices['dji'],   indices['vix']), seed=42)
    hmm_sp500 = fit_hmm(build_hmm_features(indices['sp500'], indices['vix']), seed=42)
    reg_dji   = apply_hmm(hmm_dji,   build_hmm_features(indices['dji'],   indices['vix']))
    reg_sp500 = apply_hmm(hmm_sp500, build_hmm_features(indices['sp500'], indices['vix']))
    panels_2025 = {u: assemble_universe_panel(u, c, stocks_data[u], indices, reg_dji, reg_sp500)
                   for u, c in CANONICAL.items()}

    out = {}
    period_order = ['val', 'test_bear', 'test_2023', 'test_bull', 'sim_2025']

    for u, cfg in CANONICAL.items():
        f_max = float(cfg['floor'])
        print(f'\n=== {u} (f_max = {int(f_max*100)}%) ===')

        period_dfs = {pname: load_period_df(csv) for pname, csv in THESIS_PERIODS[u].items()}
        period_dfs['sim_2025'] = panels_2025[u]

        out[u] = {'f_max': f_max, 'periods': {}}

        # Per period: posterior distribution + all variant runs
        print(f'\n  {"Period":<12}{"P10":>6}{"med":>6}{"mean":>6}{"P90":>6}{"<0.20":>7}{"<0.33":>7}{"<0.50":>7}{">0.80":>7}')
        for pname in period_order:
            df = period_dfs.get(pname)
            if df is None or df.empty:
                continue
            d = posterior_distribution(df)
            print(f"  {pname:<12}{d['p10']:>6.2f}{d['median']:>6.2f}{d['mean']:>6.2f}{d['p90']:>6.2f}"
                  f"{d['frac_lt_020']:>7.2f}{d['frac_lt_033']:>7.2f}{d['frac_lt_050']:>7.2f}{d['frac_gt_080']:>7.2f}")
            out[u]['periods'][pname] = {'posterior_stats': d, 'variants': {}}

        # Run HARD baseline (only if f_max > 0; else degenerate)
        print(f'\n  HARD baseline (canonical):')
        for pname in period_order:
            df = period_dfs.get(pname)
            if df is None or df.empty:
                continue
            vals = run_hard(cfg['ckpt'], df, f_max) if f_max > 0 else None
            if vals is None:
                # NDX: f_max=0 — soft = hard = nothing
                m = {'cumulative_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'n_days': 0}
            else:
                m = metrics(vals)
            out[u]['periods'][pname]['variants']['hard'] = {**m, 'values': vals if vals else []}
            if vals:
                print(f"    {pname:<12} cum={m['cumulative_return']:+7.2f}%  Sh={m['sharpe_ratio']:+.3f}  MDD={m['max_drawdown']:+7.2f}%")

        # Run all threshold variants
        for tau in TAU_CANDIDATES:
            tag = f'soft_tau{tau:.3f}'
            print(f'\n  {tag} ({"pure soft" if tau == 0 else f"gate p>{tau:.2f}"}):')
            for pname in period_order:
                df = period_dfs.get(pname)
                if df is None or df.empty:
                    continue
                if f_max == 0:
                    # NDX: all variants degenerate to no floor; copy hard
                    out[u]['periods'][pname]['variants'][tag] = out[u]['periods'][pname]['variants']['hard']
                    continue
                vals = run_with_threshold(cfg['ckpt'], df, f_max, tau)
                m = metrics(vals)
                out[u]['periods'][pname]['variants'][tag] = {**m, 'values': vals}
                print(f"    {pname:<12} cum={m['cumulative_return']:+7.2f}%  Sh={m['sharpe_ratio']:+.3f}  MDD={m['max_drawdown']:+7.2f}%")

    out_path = PROJECT_ROOT / 'results' / '_canonical' / 'threshold_sweep.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
