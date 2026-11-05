"""
Walk-forward HMM with EXPANDING window (instead of rolling 10y).

For each evaluation year, train HMM on data from 2010-01-01 through
Dec 31 of (eval_year - 1). The 2010-2020 GFC anchor is preserved,
recent data is additive rather than replacing old.

Eval years vs training windows:
  2021 (val):       train 2010-2020 (11 years) — basically frozen
  2022 (test_bear): train 2010-2021 (12 years)
  2023:             train 2010-2022 (13 years)
  2024 (test_bull): train 2010-2023 (14 years)
  2025 (sim):       train 2010-2024 (15 years)

Output: results/_canonical/walkforward_expanding.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))

from simulator_2025 import (
    CANONICAL, HMM_TRAIN_START, DOWNLOAD_START, DOWNLOAD_END,
    download_indices, download_universe_stocks,
    build_hmm_features, fit_hmm, apply_hmm,
    assemble_universe_panel, metrics,
)
from eval_soft_floor import THESIS_PERIODS, load_period_df
from eval_advanced_floor import (
    PERIOD_TO_YEAR, label_walkforward,
    patch_regime_columns, run_soft_floor,
)


def fit_expanding_hmm(hmm_features, eval_year, anchor_start='2010-01-01', seed=42):
    """Fit HMM on expanding window from anchor_start to Dec 31 of (eval_year-1)."""
    train_start = pd.Timestamp(anchor_start)
    train_end   = pd.Timestamp(f"{eval_year - 1}-12-31")
    train_df = hmm_features[(hmm_features['date'] >= train_start) &
                            (hmm_features['date'] <= train_end)]
    from hmmlearn import hmm as _hmm
    X = train_df[['price_position', 'trend_signal', 'volatility_20d',
                  'return_40d', 'vix_level']].values
    best_model, best_score = None, -np.inf
    rng = np.random.default_rng(seed)
    for _ in range(20):
        try:
            m = _hmm.GaussianHMM(n_components=3, covariance_type='full',
                                 n_iter=1000, tol=1e-4,
                                 random_state=int(rng.integers(0, 1_000_000)))
            m.fit(X)
            s = m.score(X)
            if s > best_score:
                best_score, best_model = s, m
        except Exception:
            continue
    labels = best_model.predict(X)
    state_scores = np.zeros(3)
    feats = train_df[['price_position', 'trend_signal']].values
    for s in range(3):
        mask = labels == s
        if mask.sum() > 0:
            state_scores[s] = feats[mask, 0].mean() + 10 * feats[mask, 1].mean()
    order = np.argsort(state_scores)
    best_model._raw_to_named = {order[0]: 1, order[1]: 2, order[2]: 0}
    return best_model


def main():
    print('=' * 80)
    print('   Walk-forward HMM — EXPANDING window')
    print('=' * 80)

    print('\n[setup] Loading cached indices + stocks...')
    indices = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    stocks_data = {u: download_universe_stocks(c['stock_list_file'], DOWNLOAD_START, DOWNLOAD_END)
                   for u, c in CANONICAL.items()}

    hmm_feats_dji   = build_hmm_features(indices['dji'],   indices['vix'])
    hmm_feats_sp500 = build_hmm_features(indices['sp500'], indices['vix'])

    print('\n[setup] Fitting EXPANDING-window HMMs per year...')
    exp_hmms_dji   = {}
    exp_hmms_sp500 = {}
    for pname, year in PERIOD_TO_YEAR.items():
        print(f'  {pname} ({year}): training 2010 through {year-1}')
        exp_hmms_dji[year]   = fit_expanding_hmm(hmm_feats_dji,   year)
        exp_hmms_sp500[year] = fit_expanding_hmm(hmm_feats_sp500, year)

    exp_labels_dji = {y: label_walkforward(hmm_feats_dji, 'dji', y, exp_hmms_dji[y])
                      for y in PERIOD_TO_YEAR.values()}
    exp_labels_sp500 = {y: label_walkforward(hmm_feats_sp500, 'sp500', y, exp_hmms_sp500[y])
                        for y in PERIOD_TO_YEAR.values()}

    # Quick check: regime mix under expanding vs frozen for diagnosis
    print('\n  Expanding-window DJI HMM regime mix:')
    for y in PERIOD_TO_YEAR.values():
        wf = exp_labels_dji[y]
        counts = pd.Series(wf['regime']).map({0: 'bull', 1: 'bear', 2: 'side'}).value_counts().to_dict()
        print(f'    {y}: {counts}')

    print('\n[setup] Assembling 2025 panels with FROZEN HMM (will overwrite labels)...')
    hmm_dji_frozen   = fit_hmm(hmm_feats_dji,   seed=42)
    hmm_sp500_frozen = fit_hmm(hmm_feats_sp500, seed=42)
    reg_dji_frozen   = apply_hmm(hmm_dji_frozen,   hmm_feats_dji)
    reg_sp500_frozen = apply_hmm(hmm_sp500_frozen, hmm_feats_sp500)
    panels_2025 = {u: assemble_universe_panel(u, c, stocks_data[u], indices,
                                              reg_dji_frozen, reg_sp500_frozen)
                   for u, c in CANONICAL.items()}

    # Load previous results to compare against
    base = json.load(open(PROJECT_ROOT / 'results' / '_canonical' / 'advanced_floor_comparison.json'))

    out = {}
    period_order = ['val', 'test_bear', 'test_2023', 'test_bull', 'sim_2025']

    for u, cfg in CANONICAL.items():
        f_max = float(cfg['floor'])
        print(f'\n=== {u} (f_max = {int(f_max*100)}%) ===')
        out[u] = {'f_max': f_max, 'periods': {}}

        period_dfs = {pname: load_period_df(csv) for pname, csv in THESIS_PERIODS[u].items()}
        period_dfs['sim_2025'] = panels_2025[u]

        exp_labels_for_u = exp_labels_dji if cfg['broad_market'] == 'dji' else exp_labels_sp500

        for pname in period_order:
            df = period_dfs.get(pname)
            if df is None or df.empty:
                continue
            year = PERIOD_TO_YEAR[pname]

            exp_df = exp_labels_for_u[year]
            df_exp = patch_regime_columns(df, exp_df)
            if df_exp.empty:
                continue
            vals = run_soft_floor(cfg['ckpt'], df_exp, f_max)
            res = {**metrics(vals), 'values': vals}

            # Look up baselines
            base_soft = base[u]['periods'].get(pname, {}).get('soft_frozen', {}).get('cumulative_return')
            base_wf10 = base[u]['periods'].get(pname, {}).get('soft_walkforward', {}).get('cumulative_return')

            delta_vs_soft = res['cumulative_return'] - base_soft if base_soft is not None else 0
            delta_vs_wf10 = res['cumulative_return'] - base_wf10 if base_wf10 is not None else 0

            out[u]['periods'][pname] = {
                'expanding_walkfwd': res,
                'delta_vs_soft_frozen': delta_vs_soft,
                'delta_vs_rolling_walkfwd': delta_vs_wf10,
            }
            print(f"  {pname:<12} cum={res['cumulative_return']:+7.2f}%  "
                  f"Sh={res['sharpe_ratio']:+.3f}  MDD={res['max_drawdown']:+7.2f}%  "
                  f"vs SOFT: {delta_vs_soft:+5.2f}  vs WF-10y: {delta_vs_wf10:+5.2f}")

    out_path = PROJECT_ROOT / 'results' / '_canonical' / 'walkforward_expanding.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
