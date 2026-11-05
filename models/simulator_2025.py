"""
2025 Forward Simulator for the three canonical regime-aware DRL agents.

Runs the canonical models (DJI mom_2m + 70% bear floor, SP100 mom_2m + 30%
bear floor, NDX 2017 + mom no floor) forward on 2025 market data, computes
benchmarks (broad-market BH, equal-weight EW), and saves per-universe
results to JSON.

Pipeline (self-contained, does not touch existing data files):
  1. Download fresh OHLCV for the universe stocks + indices + VIX from
     yfinance (2024-01-01 to 2025-12-31; 2024 history is buffer for
     252-day momentum features).
  2. Compute the same technical indicators as the v6 feature set
     (rsi, macd, bb, atr, sma, adx, cci, vol, momentum).
  3. Refit two HMMs on 2010 to 2020 broad-market features:
       HMM-A: DJI index + VIX  (for the DJI canonical)
       HMM-B: SP500 index + VIX (for the SP100 and NDX canonicals)
     Apply forward to 2025 to label each trading day.
  4. Apply normalisation using the existing per-stock training-period
     statistics in data/normalization_params_*.json.
  5. Run each canonical PPO checkpoint on the resulting 2025 dataframe
     via PortfolioEnv, capture portfolio history + regime allocation.
  6. Compute BH and EW benchmarks on the same window.
  7. Save results to results/_canonical/simulator_2025.json and print
     a summary table.

Output JSON shape:
{
  "DJI":   {"agent": {...metrics + values...}, "BH": {...}, "EW": {...}},
  "SP100": {...},
  "NDX":   {...}
}
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from hmmlearn import hmm
from stable_baselines3 import PPO

warnings.filterwarnings('ignore')


# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
RESULTS_DIR  = PROJECT_ROOT / 'results' / '_canonical'
CACHE_DIR    = PROJECT_ROOT / 'data' / '_simulator_2025_cache'
OUT_PATH     = RESULTS_DIR / 'simulator_2025.json'

# Make the project's own model code importable
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
from portfolio_env import PortfolioEnv                       # type: ignore
from train_final_extended import AttentionPortfolioExtractor  # noqa: F401 needed for PPO.load
from train_universal_v6_sp100 import V3_COLS_TO_DROP          # type: ignore


# --- Date windows --------------------------------------------------------
DOWNLOAD_START = '2024-01-01'   # 252 trading-day buffer before 2025-01-02
DOWNLOAD_END   = '2025-12-31'
EVAL_START     = '2025-01-01'
EVAL_END       = '2025-12-31'

# HMM training window (matches Methodology 3.2.3)
HMM_TRAIN_START = '2010-01-01'
HMM_TRAIN_END   = '2020-12-31'


# --- Canonical model configuration --------------------------------------
CANONICAL = {
    'DJI': {
        'ckpt':     PROJECT_ROOT / 'models/saved/checkpoints/universal_v6_dji_mom_2m/step_0800000.zip',
        'floor':    0.7,
        'norm_params': DATA_DIR / 'normalization_params_ext.json',
        'stock_list_file': DATA_DIR / 'stock_list_ext.txt',
        'use_long_mom':   True,        # canonical is mom_2m
        'index_ticker':   '^DJI',
        'broad_market':   'dji',       # which HMM to use
        'bh_ticker':      '^DJI',
        'universe_name':  'DJI 30',
    },
    'SP100': {
        'ckpt':     PROJECT_ROOT / 'models/saved/checkpoints/universal_v6_sp100_mom_2m/step_1150000.zip',
        'floor':    0.3,
        'norm_params': DATA_DIR / 'normalization_params_sp100.json',
        'stock_list_file': DATA_DIR / 'stock_list_sp100.txt',
        'use_long_mom':   True,
        'index_ticker':   '^GSPC',     # SP500 for canonical SP100 broad market
        'broad_market':   'sp500',
        'bh_ticker':      '^GSPC',     # SP500 BH benchmark (as in thesis)
        'universe_name':  'S&P 100',
    },
    'NDX': {
        'ckpt':     PROJECT_ROOT / 'models/saved/checkpoints/universal_v6_ndx_2017_mom/step_0100000.zip',
        'floor':    0.0,
        'norm_params': DATA_DIR / 'normalization_params_ndx_2017.json',
        'stock_list_file': DATA_DIR / 'stock_list_ndx_2017.txt',
        'use_long_mom':   True,
        'index_ticker':   '^NDX',
        'broad_market':   'sp500',     # NDX uses SP500 HMM (see Section 3.2.6)
        'bh_ticker':      '^NDX',
        'universe_name':  'NASDAQ-100 2017',
    },
}

INDEX_TICKERS = {
    'dji':   '^DJI',
    'sp500': '^GSPC',
    'ndx':   '^NDX',
    'vix':   '^VIX',
}


# === Step 1: download =====================================================
def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Robust yfinance download. Returns OHLCV dataframe indexed by date."""
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    df.index.name = 'date'
    return df.reset_index()


def download_universe_stocks(stock_list_path: Path, start: str, end: str) -> pd.DataFrame:
    """Download all stocks in a universe, return long-format dataframe."""
    cache_file = CACHE_DIR / f'stocks_{stock_list_path.stem}_{start}_{end}.csv'
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=['date'])

    tickers = [t.strip() for t in stock_list_path.read_text().splitlines() if t.strip()]
    print(f"  Downloading {len(tickers)} stocks for {stock_list_path.name}...")
    frames = []
    missing = []
    for i, ticker in enumerate(tickers, 1):
        df = download_ticker(ticker, start, end)
        if df is None or len(df) < 50:
            missing.append(ticker)
            continue
        df['ticker'] = ticker
        keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        df = df[[c for c in keep if c in df.columns]]
        frames.append(df)
        if i % 20 == 0:
            print(f"    {i}/{len(tickers)} done")
    if missing:
        print(f"  WARN: {len(missing)} tickers missing/short: {missing[:8]}...")
    if not frames:
        raise RuntimeError(f'No stock data downloaded for {stock_list_path.name}')

    out = pd.concat(frames, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_file, index=False)
    return out


def download_indices(start: str, end: str) -> dict:
    """Download all index series we need, return dict of dataframes."""
    out = {}
    for name, ticker in INDEX_TICKERS.items():
        cache = CACHE_DIR / f'index_{name}_{start}_{end}.csv'
        if cache.exists():
            out[name] = pd.read_csv(cache, parse_dates=['date'])
            continue
        df = download_ticker(ticker, start, end)
        if df is None:
            print(f"  WARN: index {ticker} download failed")
            continue
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        out[name] = df
    return out


# === Step 2: technical indicators =========================================
def compute_indicators_for_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Compute v6-feature-set technical indicators for one stock."""
    df = df.sort_values('date').reset_index(drop=True).copy()
    close = df['close']
    high  = df['high']
    low   = df['low']

    df['rsi']         = ta.momentum.rsi(close, window=14)
    macd              = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd']        = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff']   = macd.macd_diff()
    bb                = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_upper']    = bb.bollinger_hband()
    df['bb_lower']    = bb.bollinger_lband()
    df['bb_width']    = bb.bollinger_wband()
    df['atr']         = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df['sma_20']      = ta.trend.sma_indicator(close, window=20)
    df['sma_50']      = ta.trend.sma_indicator(close, window=50)
    df['adx']         = ta.trend.adx(high, low, close, window=14)
    df['cci']         = ta.trend.cci(high, low, close, window=20)
    df['daily_return']    = close.pct_change()
    df['volatility_20d']  = df['daily_return'].rolling(20).std()

    # Long-momentum: rolling 60/120/252-day cumulative returns, normalised across all stocks per day later
    df['mom_60d']  = close.pct_change(60)
    df['mom_120d'] = close.pct_change(120)
    df['mom_252d'] = close.pct_change(252)
    return df


def add_long_momentum_normalised(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score each long-momentum column ACROSS the panel per day."""
    panel = panel.copy()
    for raw, norm in [('mom_60d', 'mom_60d_norm'),
                      ('mom_120d', 'mom_120d_norm'),
                      ('mom_252d', 'mom_252d_norm')]:
        grp = panel.groupby('date')[raw]
        mean = grp.transform('mean')
        std  = grp.transform('std').replace(0, 1)
        panel[norm] = (panel[raw] - mean) / std
    return panel


# === Step 3: HMM regime detection =========================================
def build_hmm_features(index_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """5-dim regime feature vector: price_position, trend_signal,
    volatility_20d, return_40d, vix_level. Aligns to index dates."""
    df = index_df[['date', 'close']].copy().sort_values('date').reset_index(drop=True)
    df.columns = ['date', 'idx_close']

    df['return_1d'] = df['idx_close'].pct_change()
    df['return_40d'] = df['idx_close'].pct_change(40)
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    rolling_high = df['idx_close'].rolling(60).max()
    rolling_low  = df['idx_close'].rolling(60).min()
    df['price_position'] = (df['idx_close'] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)

    sma20 = df['idx_close'].rolling(20).mean()
    sma60 = df['idx_close'].rolling(60).mean()
    df['trend_signal'] = (sma20 - sma60) / sma60

    vix = vix_df[['date', 'close']].rename(columns={'close': 'vix_level'})
    df = df.merge(vix, on='date', how='left')

    df = df[['date', 'price_position', 'trend_signal', 'volatility_20d', 'return_40d', 'vix_level']]
    return df.dropna().reset_index(drop=True)


def fit_hmm(hmm_features_df: pd.DataFrame, n_restarts: int = 20, seed: int = 42) -> hmm.GaussianHMM:
    """Fit a 3-state Gaussian HMM with random restarts, keep best log-likelihood."""
    train = hmm_features_df[(hmm_features_df['date'] >= HMM_TRAIN_START) &
                            (hmm_features_df['date'] <= HMM_TRAIN_END)]
    X = train[['price_position', 'trend_signal', 'volatility_20d',
               'return_40d', 'vix_level']].values

    best_model, best_score = None, -np.inf
    rng = np.random.default_rng(seed)
    for r in range(n_restarts):
        try:
            m = hmm.GaussianHMM(
                n_components=3, covariance_type='full',
                n_iter=1000, tol=1e-4,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            m.fit(X)
            s = m.score(X)
            if s > best_score:
                best_score, best_model = s, m
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError('HMM training failed across all restarts')

    # Sort states by (price_position + 10*trend_signal), low = bear, high = bull
    state_scores = np.zeros(3)
    labels_train = best_model.predict(X)
    feats = train[['price_position', 'trend_signal']].values
    for s in range(3):
        mask = labels_train == s
        if mask.sum() > 0:
            state_scores[s] = feats[mask, 0].mean() + 10 * feats[mask, 1].mean()
    # Lowest = bear (1), highest = bull (0), middle = sideways (2)
    order = np.argsort(state_scores)
    raw_to_named = {order[0]: 1, order[1]: 2, order[2]: 0}  # bear=1, side=2, bull=0
    best_model._raw_to_named = raw_to_named
    return best_model


def apply_hmm(model: hmm.GaussianHMM, hmm_features_df: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted HMM to a feature dataframe, return labels + posteriors."""
    X = hmm_features_df[['price_position', 'trend_signal', 'volatility_20d',
                         'return_40d', 'vix_level']].values
    raw_labels = model.predict(X)
    posteriors = model.predict_proba(X)
    raw_to_named = getattr(model, '_raw_to_named', {0: 0, 1: 1, 2: 2})

    named_labels = np.array([raw_to_named[r] for r in raw_labels])
    # Reorder posteriors to match named regime indices
    posteriors_named = np.zeros_like(posteriors)
    for raw, named in raw_to_named.items():
        posteriors_named[:, named] = posteriors[:, raw]

    name_map = {0: 'bull', 1: 'bear', 2: 'sideways'}
    out = hmm_features_df[['date']].copy()
    out['regime'] = named_labels
    out['regime_name'] = [name_map[r] for r in named_labels]
    out['regime_bull']     = (named_labels == 0).astype(int)
    out['regime_bear']     = (named_labels == 1).astype(int)
    out['regime_sideways'] = (named_labels == 2).astype(int)
    out['regime_bull_prob']     = posteriors_named[:, 0]
    out['regime_bear_prob']     = posteriors_named[:, 1]
    out['regime_sideways_prob'] = posteriors_named[:, 2]
    return out


# === Step 4: normalisation ================================================
def apply_normalisation(panel: pd.DataFrame, norm_params: dict) -> pd.DataFrame:
    """Per-ticker z-score using saved training-period mean/std per feature."""
    panel = panel.copy()
    cols_to_norm = ['open', 'high', 'low', 'close', 'volume',
                    'rsi', 'macd', 'macd_signal', 'macd_diff',
                    'bb_upper', 'bb_lower', 'bb_width',
                    'atr', 'sma_20', 'sma_50', 'adx', 'cci',
                    'volatility_20d']
    for ticker, grp in panel.groupby('ticker'):
        if ticker not in norm_params:
            continue
        params = norm_params[ticker]
        idx = panel['ticker'] == ticker
        for col in cols_to_norm:
            if col in params and col in panel.columns:
                mean = params[col]['mean']
                std  = params[col]['std']
                if std > 0:
                    panel.loc[idx, col] = (panel.loc[idx, col] - mean) / std
    return panel


# === Step 5: broad-market vol features ====================================
def add_broad_market_vol_features(per_stock: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Add the 3 broad-market vol features used by v6."""
    idx = index_df[['date', 'close']].copy().sort_values('date').reset_index(drop=True)
    idx['idx_ret']     = idx['close'].pct_change()
    idx['vol_20d']     = idx['idx_ret'].rolling(20).std()
    idx['vol_60d']     = idx['idx_ret'].rolling(60).std()
    idx['vol_ratio']   = idx['vol_20d'] / idx['vol_60d'].replace(0, np.nan)
    # Z-score across the full series (proxy for training-period normalisation)
    for c in ['vol_20d', 'vol_60d', 'vol_ratio']:
        s = idx[c]
        m, sd = s.mean(), s.std()
        idx[c + '_norm'] = (s - m) / (sd if sd > 0 else 1)
    keep = ['date', 'vol_20d_norm', 'vol_60d_norm', 'vol_ratio_norm']
    return per_stock.merge(idx[keep], on='date', how='left')


# === Step 6: agent inference ==============================================
def run_canonical_agent(model_path: Path, df: pd.DataFrame, floor: float) -> dict:
    """Run a canonical PPO checkpoint on the 2025 dataframe, return history."""
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
    model = PPO.load(str(model_path), device='cpu')
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

    history = env.get_portfolio_history()
    return {
        'dates':  [str(d)[:10] for d in history['date'].tolist()],
        'values': history['value'].astype(float).tolist(),
    }


# === Step 7: benchmarks ===================================================
def metrics(values, n_days_year=252):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2 or v[0] <= 0:
        return {'cumulative_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'n_days': n}
    cum_ret = float((v[-1] / v[0] - 1) * 100)
    dr      = np.diff(v) / v[:-1]
    ann_ret = float(((v[-1] / v[0]) ** (n_days_year / n) - 1) * 100)
    ann_vol = float(dr.std() * np.sqrt(n_days_year) * 100)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak    = np.maximum.accumulate(v)
    mdd     = float(((v - peak) / peak).min() * 100)
    return {'cumulative_return': cum_ret, 'sharpe_ratio': sharpe,
            'max_drawdown': mdd, 'n_days': n}


def bh_values(index_df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    df = index_df[(index_df['date'] >= start) & (index_df['date'] <= end)]
    return df['close'].values


def ew_values(panel: pd.DataFrame, start: str, end: str, tx_cost: float = 0.001) -> np.ndarray:
    """Daily-rebalanced equal-weight portfolio, matching benchmark_metrics.py."""
    panel = panel[(panel['date'] >= start) & (panel['date'] <= end)].copy()
    panel['ret'] = panel.groupby('ticker')['close'].pct_change()
    panel = panel.dropna(subset=['ret'])
    wide = panel.pivot(index='date', columns='ticker', values='ret').fillna(0.0)
    n = wide.shape[1]
    if n == 0:
        return np.array([1_000_000.0])
    target = np.full(n, 1.0 / n)
    weights = target.copy()
    pv = 1_000_000.0
    values = [pv]
    for t in range(wide.shape[0]):
        r = wide.iloc[t].values
        port_r = float(weights @ r)
        pv *= (1 + port_r)
        drifted = weights * (1 + r) / (1 + port_r) if abs(1 + port_r) > 1e-12 else target.copy()
        turn = float(np.abs(target - drifted).sum())
        pv *= (1 - turn * tx_cost)
        weights = target.copy()
        values.append(pv)
    return np.array(values)


# === Step 8: dataframe assembly ===========================================
def assemble_universe_panel(
    universe: str,
    cfg: dict,
    stocks_raw: pd.DataFrame,
    indices: dict,
    regime_dji: pd.DataFrame,
    regime_sp500: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the final per-day, per-stock dataframe in env format,
    truncated to 2025-01-01 onward."""
    print(f"  Computing indicators for {universe} ({stocks_raw['ticker'].nunique()} stocks)...")
    stocks = stocks_raw.groupby('ticker', group_keys=False).apply(compute_indicators_for_stock)
    stocks = add_long_momentum_normalised(stocks)
    stocks = stocks.dropna(subset=['close', 'rsi', 'mom_60d_norm']).reset_index(drop=True)

    # Normalise stock features
    with open(cfg['norm_params']) as f:
        norm_params = json.load(f)
    stocks = apply_normalisation(stocks, norm_params)

    # Add broad-market vol from the same index used in the original pipeline
    bm_index = indices[cfg['broad_market'] if cfg['broad_market'] != 'dji' else 'dji']
    stocks = add_broad_market_vol_features(stocks, bm_index)

    # Attach regime columns
    regime_df = regime_dji if cfg['broad_market'] == 'dji' else regime_sp500
    stocks = stocks.merge(regime_df, on='date', how='left')

    # Filter to 2025 eval window
    stocks['date'] = pd.to_datetime(stocks['date'])
    stocks = stocks[(stocks['date'] >= EVAL_START) & (stocks['date'] <= EVAL_END)]
    stocks = stocks.dropna(subset=['regime', 'regime_bull_prob', 'vol_20d_norm']).reset_index(drop=True)

    # Match expected column ordering / minimal column set
    base_cols = [
        'date', 'open', 'high', 'low', 'close', 'volume', 'ticker',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_upper', 'bb_lower', 'bb_width',
        'atr', 'sma_20', 'sma_50', 'adx', 'cci',
        'daily_return', 'volatility_20d',
        'mom_60d_norm',
    ]
    if cfg['use_long_mom']:
        base_cols += ['mom_120d_norm', 'mom_252d_norm']
    regime_cols = ['regime', 'regime_name',
                   'regime_bull', 'regime_bear', 'regime_sideways',
                   'regime_bull_prob', 'regime_bear_prob', 'regime_sideways_prob',
                   'vol_20d_norm', 'vol_60d_norm', 'vol_ratio_norm']
    stocks = stocks[[c for c in base_cols + regime_cols if c in stocks.columns]]
    return stocks


# === Main =================================================================
def main():
    print('=' * 72)
    print('   2025 Forward Simulator — three canonical agents')
    print('=' * 72)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Download data ----
    print('\n[1/7] Downloading indices and VIX...')
    indices = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    for name in ['dji', 'sp500', 'ndx', 'vix']:
        if name not in indices:
            raise RuntimeError(f'Required index {name} not downloaded')

    print('[2/7] Downloading universe stocks...')
    stocks_data = {}
    for uname, cfg in CANONICAL.items():
        stocks_data[uname] = download_universe_stocks(
            cfg['stock_list_file'], DOWNLOAD_START, DOWNLOAD_END
        )

    # ---- Fit HMMs ----
    print('[3/7] Fitting HMMs on 2010-2020 broad-market features...')
    hmm_features_dji   = build_hmm_features(indices['dji'],   indices['vix'])
    hmm_features_sp500 = build_hmm_features(indices['sp500'], indices['vix'])
    hmm_dji   = fit_hmm(hmm_features_dji,   seed=42)
    hmm_sp500 = fit_hmm(hmm_features_sp500, seed=42)
    print('  DJI HMM:   log-lik =', hmm_dji.score(hmm_features_dji[['price_position', 'trend_signal',
                                                                     'volatility_20d', 'return_40d',
                                                                     'vix_level']].values))
    print('  SP500 HMM: log-lik =', hmm_sp500.score(hmm_features_sp500[['price_position', 'trend_signal',
                                                                         'volatility_20d', 'return_40d',
                                                                         'vix_level']].values))

    # ---- Label 2025 regimes ----
    print('[4/7] Labelling 2025 trading days...')
    regime_dji   = apply_hmm(hmm_dji,   hmm_features_dji)
    regime_sp500 = apply_hmm(hmm_sp500, hmm_features_sp500)
    for name, df in [('DJI', regime_dji), ('SP500', regime_sp500)]:
        eval_only = df[(df['date'] >= EVAL_START) & (df['date'] <= EVAL_END)]
        counts = eval_only['regime_name'].value_counts().to_dict()
        print(f'  {name} 2025 regime mix: {counts}')

    # ---- Run agents + benchmarks ----
    out = {}
    for uname, cfg in CANONICAL.items():
        print(f'\n[5-7/7] {uname}: assembling features, running agent, computing benchmarks...')
        panel = assemble_universe_panel(uname, cfg, stocks_data[uname], indices,
                                        regime_dji, regime_sp500)
        if panel.empty or panel['ticker'].nunique() < 5:
            print(f'  ERROR: insufficient data for {uname} (n_stocks={panel["ticker"].nunique()})')
            continue

        # Agent
        try:
            agent_hist = run_canonical_agent(cfg['ckpt'], panel, cfg['floor'])
            agent_m = metrics(agent_hist['values'])
            print(f"  AGENT   cum={agent_m['cumulative_return']:+7.2f}%  Sh={agent_m['sharpe_ratio']:+.3f}  MDD={agent_m['max_drawdown']:+6.2f}%  n={agent_m['n_days']}")
        except Exception as e:
            print(f'  AGENT FAILED: {e}')
            agent_hist = {'dates': [], 'values': []}
            agent_m = {'cumulative_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'n_days': 0, 'error': str(e)}

        # BH
        bh = indices['ndx' if cfg['bh_ticker'] == '^NDX' else
                     ('sp500' if cfg['bh_ticker'] == '^GSPC' else 'dji')]
        bh_vals = bh_values(bh, EVAL_START, EVAL_END)
        bh_m = metrics(bh_vals)
        print(f"  BH      cum={bh_m['cumulative_return']:+7.2f}%  Sh={bh_m['sharpe_ratio']:+.3f}  MDD={bh_m['max_drawdown']:+6.2f}%  n={bh_m['n_days']}")

        # EW
        ew_vals = ew_values(stocks_data[uname], EVAL_START, EVAL_END)
        ew_m = metrics(ew_vals)
        print(f"  EW      cum={ew_m['cumulative_return']:+7.2f}%  Sh={ew_m['sharpe_ratio']:+.3f}  MDD={ew_m['max_drawdown']:+6.2f}%  n={ew_m['n_days']}")

        out[uname] = {
            'universe_name': cfg['universe_name'],
            'eval_window': f'{EVAL_START} to {EVAL_END}',
            'agent': {**agent_m, 'dates': agent_hist['dates'], 'values': agent_hist['values']},
            'BH':    {**bh_m, 'values': bh_vals.tolist()},
            'EW':    {**ew_m, 'values': ew_vals.tolist()},
        }

    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
