"""
Two more refinements on top of the locked pure-soft posterior floor:

  (A) WALK-FORWARD HMM
      Refit the HMM yearly on a rolling 10-year window ending Dec 31 of the
      year before evaluation. So the 2025 evaluation uses an HMM trained on
      2015 through 2024 data; the 2024 evaluation uses 2014 through 2023, etc.
      This addresses the "stale HMM parameters" risk. The DJI agent uses a
      DJI-fitted walk-forward HMM; SP100 and NDX use SP500-fitted walk-forward
      HMMs (consistent with Section 3.2.6).
      Floor rule remains pure soft: floor = f_max * P_bear_walkforward.

  (B) VOLATILITY-CONDITIONED SOFT FLOOR
      Stay with the frozen (canonical) HMM. Weight the soft floor by a
      "vol persistence" score that measures how many of the last 10 days
      had above-baseline broad-market volatility:
          vol_persistence(t) = fraction of last 10 days where vol_20d_norm > 0
          floor(t) = f_max * P_bear(t) * vol_persistence(t)
      Sustained-high-volatility periods (real bears) get a full posterior-
      weighted floor; one-day vol spikes that revert get reduced floor.

Both refinements are evaluated on all 5 windows (val 2021, bear 2022,
recovery 2023, bull 2024, sim 2025) for all 3 canonical agents and
compared to (a) HARD baseline and (b) pure-soft locked baseline.

Output: results/_canonical/advanced_floor_comparison.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

warnings.filterwarnings('ignore')

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


# Periods are mapped to calendar years for walk-forward refitting
PERIOD_TO_YEAR = {
    'val':       2021,
    'test_bear': 2022,
    'test_2023': 2023,
    'test_bull': 2024,
    'sim_2025':  2025,
}


# ============================================================
# (A) WALK-FORWARD HMM
# ============================================================
def fit_walkforward_hmm_for_year(hmm_features, eval_year, train_window_years=10, seed=42):
    """Fit an HMM on a rolling window ending Dec 31 of (eval_year - 1).

    Returns a fitted hmm.GaussianHMM with _raw_to_named ordering attached.
    """
    train_start = pd.Timestamp(f"{eval_year - train_window_years}-01-01")
    train_end   = pd.Timestamp(f"{eval_year - 1}-12-31")
    train_df = hmm_features[(hmm_features['date'] >= train_start) &
                            (hmm_features['date'] <= train_end)]
    if len(train_df) < 100:
        raise RuntimeError(f"Not enough HMM training data for year {eval_year}: {len(train_df)} rows")

    # Re-use the fit_hmm helper from simulator_2025, but it expects the
    # standard 2010-2020 window; let's call it on the rolling subset.
    # The helper internally checks HMM_TRAIN_START / HMM_TRAIN_END; we
    # pass the relevant slice so it just trains on what we give it.
    # The fit_hmm in simulator_2025 selects internally; we replicate here
    # to avoid the date filter.
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
    if best_model is None:
        raise RuntimeError("HMM fitting failed for year " + str(eval_year))

    # State sort by (price_position + 10*trend_signal): bear lowest, bull highest
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


def label_walkforward(hmm_features, broad_market, eval_year, hmm_model):
    """Apply a walk-forward HMM to label the eval_year only."""
    eval_features = hmm_features[(hmm_features['date'] >= pd.Timestamp(f"{eval_year}-01-01")) &
                                 (hmm_features['date'] <= pd.Timestamp(f"{eval_year}-12-31"))]
    if eval_features.empty:
        return pd.DataFrame()
    X = eval_features[['price_position', 'trend_signal', 'volatility_20d',
                       'return_40d', 'vix_level']].values
    raw_labels = hmm_model.predict(X)
    posteriors = hmm_model.predict_proba(X)
    raw_to_named = hmm_model._raw_to_named
    named_labels = np.array([raw_to_named[r] for r in raw_labels])
    posteriors_named = np.zeros_like(posteriors)
    for raw, named in raw_to_named.items():
        posteriors_named[:, named] = posteriors[:, raw]
    out = eval_features[['date']].copy().reset_index(drop=True)
    out['regime'] = named_labels
    out['regime_bull']     = (named_labels == 0).astype(int)
    out['regime_bear']     = (named_labels == 1).astype(int)
    out['regime_sideways'] = (named_labels == 2).astype(int)
    out['regime_bull_prob']     = posteriors_named[:, 0]
    out['regime_bear_prob']     = posteriors_named[:, 1]
    out['regime_sideways_prob'] = posteriors_named[:, 2]
    return out


def patch_regime_columns(panel_df, walkforward_df):
    """Replace regime columns in a per-stock panel with walk-forward labels."""
    if walkforward_df.empty:
        return panel_df
    panel = panel_df.copy()
    panel['date'] = pd.to_datetime(panel['date'])
    wf = walkforward_df.copy()
    wf['date'] = pd.to_datetime(wf['date'])
    # Drop existing regime columns; re-merge from walk-forward
    regime_cols_to_drop = ['regime', 'regime_bull', 'regime_bear', 'regime_sideways',
                           'regime_bull_prob', 'regime_bear_prob', 'regime_sideways_prob']
    panel = panel.drop(columns=[c for c in regime_cols_to_drop if c in panel.columns])
    panel = panel.merge(wf, on='date', how='left')
    panel = panel.dropna(subset=['regime']).reset_index(drop=True)
    panel['regime'] = panel['regime'].astype(int)
    return panel


def run_soft_floor(model_path, df, f_max):
    """Pure soft posterior-weighted floor (the locked baseline rule)."""
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
        soft_floor = float(f_max) * p_bear
        cur_regime = int(env.regime_id_array[cur_step])
        env.cash_floors[cur_regime] = soft_floor
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()['value'].astype(float).tolist()


# ============================================================
# (B) VOLATILITY-CONDITIONED SOFT FLOOR
# ============================================================
def compute_vol_persistence(panel_df, window=10):
    """For each unique date in panel, compute fraction of last `window` days
    where the broad-market vol_20d_norm was positive (above-baseline)."""
    if 'vol_20d_norm' not in panel_df.columns:
        return None
    daily = panel_df.groupby('date')['vol_20d_norm'].first().sort_index()
    vol_high = (daily > 0).astype(int)
    persistence = vol_high.rolling(window=window, min_periods=1).mean()
    return persistence.reset_index().rename(columns={'vol_20d_norm': 'vol_persistence'})


def run_vol_conditioned_soft(model_path, df, f_max, vol_persistence_df):
    """Soft posterior floor weighted by vol persistence."""
    df = df.copy()
    if vol_persistence_df is not None and not vol_persistence_df.empty:
        df['date'] = pd.to_datetime(df['date'])
        vp = vol_persistence_df.copy()
        vp['date'] = pd.to_datetime(vp['date'])
        df = df.merge(vp, on='date', how='left')
        df['vol_persistence'] = df['vol_persistence'].fillna(0.5)

    env = make_env(df, 0.0)
    model = PPO.load(str(model_path), device='cpu')
    # Build a per-step vol_persistence array aligned with env.dates
    per_day_vp = (df.groupby('date')['vol_persistence'].first()
                  if 'vol_persistence' in df.columns else None)
    obs, _ = env.reset()
    done = False
    while not done:
        cur_step = env.current_step
        cur_date = pd.Timestamp(env.dates[cur_step])
        if env.regime_prob_array is not None:
            p_bear = float(env.regime_prob_array[cur_step, 1])
        else:
            p_bear = 1.0 if int(env.regime_id_array[cur_step]) == 1 else 0.0
        if per_day_vp is not None and cur_date in per_day_vp.index:
            vp_val = float(per_day_vp.loc[cur_date])
        else:
            vp_val = 1.0
        floor_value = float(f_max) * p_bear * vp_val
        cur_regime = int(env.regime_id_array[cur_step])
        env.cash_floors[cur_regime] = floor_value
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()['value'].astype(float).tolist()


# ============================================================
# Main pipeline
# ============================================================
def main():
    print('=' * 80)
    print('   Advanced floor experiments')
    print('=' * 80)

    print('\n[setup 1/3] Loading cached indices + stocks...')
    indices = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    stocks_data = {u: download_universe_stocks(c['stock_list_file'], DOWNLOAD_START, DOWNLOAD_END)
                   for u, c in CANONICAL.items()}

    print('[setup 2/3] Computing HMM-feature panels for DJI and SP500...')
    hmm_feats_dji   = build_hmm_features(indices['dji'],   indices['vix'])
    hmm_feats_sp500 = build_hmm_features(indices['sp500'], indices['vix'])

    print('[setup 3/3] Fitting WALK-FORWARD HMMs for each evaluation year...')
    # DJI HMM walk-forward (for DJI agent)
    wf_hmms_dji   = {}
    wf_hmms_sp500 = {}
    for pname, year in PERIOD_TO_YEAR.items():
        print(f'  {pname} ({year}): training on {year-10} to {year-1}')
        wf_hmms_dji[year]   = fit_walkforward_hmm_for_year(hmm_feats_dji,   year)
        wf_hmms_sp500[year] = fit_walkforward_hmm_for_year(hmm_feats_sp500, year)

    # Pre-build walk-forward label dataframes per year
    wf_labels_dji   = {y: label_walkforward(hmm_feats_dji,   'dji',   y, wf_hmms_dji[y])
                       for y in PERIOD_TO_YEAR.values()}
    wf_labels_sp500 = {y: label_walkforward(hmm_feats_sp500, 'sp500', y, wf_hmms_sp500[y])
                       for y in PERIOD_TO_YEAR.values()}

    # Quick sanity: distribution of regimes per year
    print('\n  Walk-forward regime distribution (DJI HMM) per year:')
    for y in PERIOD_TO_YEAR.values():
        wf = wf_labels_dji[y]
        counts = pd.Series(wf['regime']).map({0: 'bull', 1: 'bear', 2: 'side'}).value_counts().to_dict()
        print(f'    {y}: {counts}')

    # Build 2025 panels (frozen HMM versions, then patch with walk-forward)
    print('\n[setup] Assembling 2025 panels (frozen HMM)...')
    hmm_dji_frozen   = fit_hmm(hmm_feats_dji,   seed=42)
    hmm_sp500_frozen = fit_hmm(hmm_feats_sp500, seed=42)
    reg_dji_frozen   = apply_hmm(hmm_dji_frozen,   hmm_feats_dji)
    reg_sp500_frozen = apply_hmm(hmm_sp500_frozen, hmm_feats_sp500)
    panels_2025_frozen = {u: assemble_universe_panel(u, c, stocks_data[u], indices,
                                                     reg_dji_frozen, reg_sp500_frozen)
                          for u, c in CANONICAL.items()}

    # ======== Evaluate ========
    out = {}
    period_order = ['val', 'test_bear', 'test_2023', 'test_bull', 'sim_2025']

    for u, cfg in CANONICAL.items():
        f_max = float(cfg['floor'])
        print(f'\n=== {u} (f_max = {int(f_max*100)}%) ===')
        out[u] = {'f_max': f_max, 'periods': {}}

        # Thesis period dataframes (frozen HMM labels in CSV)
        period_dfs = {pname: load_period_df(csv) for pname, csv in THESIS_PERIODS[u].items()}
        period_dfs['sim_2025'] = panels_2025_frozen[u]

        wf_for_universe = wf_labels_dji if cfg['broad_market'] == 'dji' else wf_labels_sp500

        for pname in period_order:
            df = period_dfs.get(pname)
            if df is None or (hasattr(df, 'empty') and df.empty):
                continue
            year = PERIOD_TO_YEAR[pname]

            print(f'  -- {pname} ({year}) --')
            res = {}

            # (1) Pure soft (frozen HMM) — locked baseline
            soft_vals = run_soft_floor(cfg['ckpt'], df, f_max)
            res['soft_frozen'] = {**metrics(soft_vals), 'values': soft_vals}
            print(f"    soft_frozen     cum={res['soft_frozen']['cumulative_return']:+7.2f}%  Sh={res['soft_frozen']['sharpe_ratio']:+.3f}  MDD={res['soft_frozen']['max_drawdown']:+7.2f}%")

            # (2) Walk-forward HMM + soft
            wf_df = wf_for_universe[year]
            df_wf = patch_regime_columns(df, wf_df)
            if not df_wf.empty:
                wf_vals = run_soft_floor(cfg['ckpt'], df_wf, f_max)
                res['soft_walkforward'] = {**metrics(wf_vals), 'values': wf_vals}
                print(f"    soft_walkfwd    cum={res['soft_walkforward']['cumulative_return']:+7.2f}%  Sh={res['soft_walkforward']['sharpe_ratio']:+.3f}  MDD={res['soft_walkforward']['max_drawdown']:+7.2f}%")
            else:
                res['soft_walkforward'] = None

            # (3) Vol-conditioned soft (frozen HMM)
            vol_persistence = compute_vol_persistence(df)
            vol_vals = run_vol_conditioned_soft(cfg['ckpt'], df, f_max, vol_persistence)
            res['soft_vol_conditioned'] = {**metrics(vol_vals), 'values': vol_vals}
            print(f"    soft_vol_cond   cum={res['soft_vol_conditioned']['cumulative_return']:+7.2f}%  Sh={res['soft_vol_conditioned']['sharpe_ratio']:+.3f}  MDD={res['soft_vol_conditioned']['max_drawdown']:+7.2f}%")

            # (4) Walk-forward HMM + vol-conditioned (combo)
            if not df_wf.empty:
                vol_persistence_wf = compute_vol_persistence(df_wf)
                combo_vals = run_vol_conditioned_soft(cfg['ckpt'], df_wf, f_max, vol_persistence_wf)
                res['combo'] = {**metrics(combo_vals), 'values': combo_vals}
                print(f"    combo (wf+vol)  cum={res['combo']['cumulative_return']:+7.2f}%  Sh={res['combo']['sharpe_ratio']:+.3f}  MDD={res['combo']['max_drawdown']:+7.2f}%")
            else:
                res['combo'] = None

            out[u]['periods'][pname] = res

    # Save
    out_path = PROJECT_ROOT / 'results' / '_canonical' / 'advanced_floor_comparison.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
