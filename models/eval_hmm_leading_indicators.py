"""
Extended HMM with leading-indicator features.

Adds two new features to the HMM observation to capture stress
DURATION (sustained vs transient):

  vix_60d_mean       — 60-day moving average of VIX. Captures the
                       quarterly stress level. A sustained bear
                       (2022) keeps this high; a 2025-style spike
                       barely moves it.
  vix_persistence_30d — fraction of last 30 days with VIX > 20.
                       Direct persistence signal. Near 1 for sustained
                       stress, near 0 for transient spikes.

Both are derived from existing VIX data (no new downloads).

The new HMM is fit on 2010-2020 (frozen) — same window as the
canonical. We compare it against:
  HARD (canonical 70/30/0 floor)
  PURE SOFT with the canonical HMM (locked refinement)
  PURE SOFT with the leading-indicator HMM (new experiment)

Output: results/_canonical/hmm_leading_indicators.json
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
    CANONICAL, HMM_TRAIN_START, HMM_TRAIN_END,
    DOWNLOAD_START, DOWNLOAD_END,
    download_indices, download_universe_stocks,
    build_hmm_features, fit_hmm, apply_hmm,
    assemble_universe_panel, metrics,
)
from eval_soft_floor import THESIS_PERIODS, load_period_df
from eval_advanced_floor import (
    label_walkforward, patch_regime_columns, run_soft_floor,
    PERIOD_TO_YEAR,
)


HMM_FEATURE_COLS = ['price_position', 'trend_signal', 'volatility_20d',
                    'return_40d', 'vix_level',
                    'vix_60d_mean', 'vix_persistence_30d']


def build_hmm_features_extended(index_df, vix_df):
    """5 canonical features + 2 leading-indicator features."""
    df = build_hmm_features(index_df, vix_df).copy()  # original 5 features
    vix = vix_df[['date', 'close']].copy().rename(columns={'close': 'vix_level'})
    vix['date'] = pd.to_datetime(vix['date'])
    vix = vix.sort_values('date').reset_index(drop=True)

    # 60-day moving average of VIX
    vix['vix_60d_mean'] = vix['vix_level'].rolling(60, min_periods=1).mean()
    # Fraction of last 30 days with VIX > 20
    vix['vix_above_20'] = (vix['vix_level'] > 20).astype(int)
    vix['vix_persistence_30d'] = vix['vix_above_20'].rolling(30, min_periods=1).mean()
    vix = vix[['date', 'vix_60d_mean', 'vix_persistence_30d']]

    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(vix, on='date', how='left')
    df = df.dropna().reset_index(drop=True)
    return df


def fit_hmm_extended(hmm_features, seed=42):
    """Fit a 3-state HMM on the extended 7-feature set, training on 2010-2020."""
    from hmmlearn import hmm as _hmm
    train = hmm_features[(hmm_features['date'] >= pd.Timestamp(HMM_TRAIN_START)) &
                         (hmm_features['date'] <= pd.Timestamp(HMM_TRAIN_END))]
    X = train[HMM_FEATURE_COLS].values
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
    if best_model is None:
        raise RuntimeError("Extended HMM training failed")

    labels = best_model.predict(X)
    # State scoring uses ORIGINAL 5 features so bull/bear/sideways
    # ordering matches the canonical convention.
    feats = train[['price_position', 'trend_signal']].values
    state_scores = np.zeros(3)
    for s in range(3):
        mask = labels == s
        if mask.sum() > 0:
            state_scores[s] = feats[mask, 0].mean() + 10 * feats[mask, 1].mean()
    order = np.argsort(state_scores)
    best_model._raw_to_named = {order[0]: 1, order[1]: 2, order[2]: 0}
    best_model._feature_cols = HMM_FEATURE_COLS
    return best_model


def apply_hmm_extended(model, hmm_features):
    X = hmm_features[HMM_FEATURE_COLS].values
    raw_labels = model.predict(X)
    posteriors = model.predict_proba(X)
    raw_to_named = model._raw_to_named
    named_labels = np.array([raw_to_named[r] for r in raw_labels])
    posteriors_named = np.zeros_like(posteriors)
    for raw, named in raw_to_named.items():
        posteriors_named[:, named] = posteriors[:, raw]
    out = hmm_features[['date']].copy().reset_index(drop=True)
    out['regime']               = named_labels
    out['regime_bull']          = (named_labels == 0).astype(int)
    out['regime_bear']          = (named_labels == 1).astype(int)
    out['regime_sideways']      = (named_labels == 2).astype(int)
    out['regime_bull_prob']     = posteriors_named[:, 0]
    out['regime_bear_prob']     = posteriors_named[:, 1]
    out['regime_sideways_prob'] = posteriors_named[:, 2]
    return out


def main():
    print('=' * 80)
    print('   Extended HMM — leading-indicator features')
    print('=' * 80)

    print('\n[setup 1/3] Loading indices, stocks...')
    indices = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    stocks_data = {u: download_universe_stocks(c['stock_list_file'], DOWNLOAD_START, DOWNLOAD_END)
                   for u, c in CANONICAL.items()}

    print('[setup 2/3] Building extended HMM-feature panels...')
    hmm_feats_dji_ext   = build_hmm_features_extended(indices['dji'],   indices['vix'])
    hmm_feats_sp500_ext = build_hmm_features_extended(indices['sp500'], indices['vix'])

    print('[setup 3/3] Fitting extended HMMs (frozen 2010-2020)...')
    hmm_dji_ext   = fit_hmm_extended(hmm_feats_dji_ext,   seed=42)
    hmm_sp500_ext = fit_hmm_extended(hmm_feats_sp500_ext, seed=42)
    reg_dji_ext   = apply_hmm_extended(hmm_dji_ext,   hmm_feats_dji_ext)
    reg_sp500_ext = apply_hmm_extended(hmm_sp500_ext, hmm_feats_sp500_ext)

    # Diagnostic: regime distribution per period under new HMM
    print('\n  Extended-HMM regime distribution per year (DJI HMM):')
    for pname, year in PERIOD_TO_YEAR.items():
        yr_df = reg_dji_ext[(reg_dji_ext['date'] >= pd.Timestamp(f'{year}-01-01')) &
                            (reg_dji_ext['date'] <= pd.Timestamp(f'{year}-12-31'))]
        counts = pd.Series(yr_df['regime']).map({0: 'bull', 1: 'bear', 2: 'side'}).value_counts().to_dict()
        # Also report posterior stats
        bear_p = yr_df['regime_bear_prob'].values
        print(f'    {year}: {counts}   p_bear: mean={bear_p.mean():.2f} '
              f'frac>0.8={float((bear_p>0.8).mean()):.2f} frac<0.2={float((bear_p<0.2).mean()):.2f}')

    # Assemble 2025 panels with EXTENDED HMM labels
    print('\n[setup] Assembling 2025 panels (frozen canonical pipeline, then patch with extended labels)...')
    hmm_dji_frozen   = fit_hmm(build_hmm_features(indices['dji'],   indices['vix']), seed=42)
    hmm_sp500_frozen = fit_hmm(build_hmm_features(indices['sp500'], indices['vix']), seed=42)
    reg_dji_frozen   = apply_hmm(hmm_dji_frozen,   build_hmm_features(indices['dji'],   indices['vix']))
    reg_sp500_frozen = apply_hmm(hmm_sp500_frozen, build_hmm_features(indices['sp500'], indices['vix']))
    panels_2025 = {u: assemble_universe_panel(u, c, stocks_data[u], indices,
                                              reg_dji_frozen, reg_sp500_frozen)
                   for u, c in CANONICAL.items()}

    # Load previous results for comparison
    adv = json.load(open(PROJECT_ROOT / 'results' / '_canonical' / 'advanced_floor_comparison.json'))

    out = {}
    period_order = ['val', 'test_bear', 'test_2023', 'test_bull', 'sim_2025']

    for u, cfg in CANONICAL.items():
        f_max = float(cfg['floor'])
        print(f'\n=== {u} (f_max = {int(f_max*100)}%) ===')
        out[u] = {'f_max': f_max, 'periods': {}}

        period_dfs = {pname: load_period_df(csv) for pname, csv in THESIS_PERIODS[u].items()}
        period_dfs['sim_2025'] = panels_2025[u]
        reg_for_u = reg_dji_ext if cfg['broad_market'] == 'dji' else reg_sp500_ext

        for pname in period_order:
            df = period_dfs.get(pname)
            if df is None or df.empty:
                continue
            year = PERIOD_TO_YEAR[pname]
            # Extract this period's window from the extended-regime panel
            wf_year = reg_for_u[(reg_for_u['date'] >= pd.Timestamp(f'{year}-01-01')) &
                                (reg_for_u['date'] <= pd.Timestamp(f'{year}-12-31'))]
            df_ext = patch_regime_columns(df, wf_year)
            if df_ext.empty:
                continue
            vals = run_soft_floor(cfg['ckpt'], df_ext, f_max)
            res = {**metrics(vals), 'values': vals}

            base_soft = adv[u]['periods'].get(pname, {}).get('soft_frozen', {}).get('cumulative_return')
            delta = res['cumulative_return'] - base_soft if base_soft is not None else 0

            out[u]['periods'][pname] = {
                'extended_hmm_soft': res,
                'delta_vs_soft_frozen': delta,
            }
            print(f"  {pname:<12} cum={res['cumulative_return']:+7.2f}%  "
                  f"Sh={res['sharpe_ratio']:+.3f}  MDD={res['max_drawdown']:+7.2f}%  "
                  f"Δ vs SOFT: {delta:+5.2f}")

    out_path = PROJECT_ROOT / 'results' / '_canonical' / 'hmm_leading_indicators.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
