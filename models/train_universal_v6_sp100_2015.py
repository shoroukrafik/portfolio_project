"""
SP100 2015-cutoff (NO LONG-MOMENTUM) — the ablation paired with
train_universal_v6_sp100_2015_mom.py.

Completes the 2×2 design that isolates the effect of long-momentum features
from the effect of shorter training window + expanded universe.

                       Window 2007-2020         Window 2015-2020
                       (14yr, 90 stocks)        (6yr, 99 stocks)
─────────────────────────────────────────────────────────────────────
NO long-mom            universal_v6_sp100       universal_v6_sp100_2015  ←
+ long-mom             universal_v6_sp100_mom   universal_v6_sp100_2015_mom

This script trains the bottom-left cell:
  - Train window: 2015-2020 (6 years instead of 14)
  - Universe:     ~99 stocks (vs production SP100's 90)
  - Features:     V6 baseline ONLY (no mom_120d_norm, no mom_252d_norm)
  - Steps:        500k (matches the +mom variant for clean comparison)

Output: ppo_universal_v6_sp100_2015.zip
"""

import os
import sys
import json
import argparse
import pandas as pd
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_final_extended import (
    AttentionPortfolioExtractor,
    DetailedCallback,
    ValidationCheckpointCallback,
    calculate_metrics,
    evaluate_agent,
)
from train_universal_v6_sp100 import (
    V6_CASH_FLOORS, V6_RISK_PENALTIES, V6_NET_ARCH,
    V6_FEATURE_COLUMNS, V3_COLS_TO_DROP,
)

# NOTE: V6_FEATURE_COLUMNS (no mom_120d/mom_252d) — that's the whole point.
# To prevent PortfolioEnv from auto-picking up those columns from the data,
# we drop them from the loaded dataframes before constructing the env.
LONG_MOM_COLS_TO_DROP = ['mom_120d_norm', 'mom_252d_norm']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_steps', type=int, default=500_000)
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    name_tag = '_2015'
    if args.seed != 42:
        name_tag += f'_s{args.seed}'
    results_dir = os.path.join(project_dir, 'results', f'universal_v6_sp100{name_tag}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 80)
    print('   SP100 2015-cutoff (NO LONG-MOMENTUM) — ablation pair for _mom variant')
    print('   Train window: 2015-2020 (6yr)   Universe: relaxed (~99 stocks)')
    print('   Features:     V6 baseline (mom_60d only — NO mom_120d/mom_252d)')
    print('=' * 80)

    def load(p):
        df = pd.read_csv(os.path.join(data_dir, p))
        df['date'] = pd.to_datetime(df['date'])
        drop_cols = [c for c in V3_COLS_TO_DROP if c in df.columns]
        drop_cols += [c for c in LONG_MOM_COLS_TO_DROP if c in df.columns]
        return df.drop(columns=drop_cols)

    train_df     = load('train_data_sp100_2015.csv')
    val_df       = load('val_data_sp100_2015.csv')
    test_bear_df = load('test_bear_data_sp100_2015.csv')
    test_2023_df = load('test_2023_data_sp100_2015.csv')
    test_bull_df = load('test_bull_data_sp100_2015.csv')

    # Sanity-check that the long-mom cols are gone
    assert 'mom_120d_norm' not in train_df.columns, "long-mom still in data!"
    assert 'mom_252d_norm' not in train_df.columns, "long-mom still in data!"

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"\n   Stocks: {n_stocks}  Features per stock: {n_features}")
    print(f"   Features: {feature_columns}")

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=V6_CASH_FLOORS,
        risk_penalties=V6_RISK_PENALTIES,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_markowitz_reward=False,
        use_hyperheuristic=False,
    )
    train_env     = PortfolioEnv(df=train_df,     **env_kwargs)
    val_env       = PortfolioEnv(df=val_df,       **env_kwargs)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kwargs)
    test_2023_env = PortfolioEnv(df=test_2023_df, **env_kwargs)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kwargs)

    policy_kwargs = dict(
        features_extractor_class=AttentionPortfolioExtractor,
        features_extractor_kwargs=dict(
            n_stocks=n_stocks, n_features=n_features,
            embed_dim=64, n_heads=4, features_dim=128,
        ),
        net_arch=V6_NET_ARCH,
    )
    hp_path = os.path.join(project_dir, 'results', 'best_hyperparams.json')
    with open(hp_path) as f:
        hp = json.load(f)
    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        ent_coef=hp['ent_coef'],
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=args.seed, device='cpu',
    )

    print(f"\n--- Training {args.total_steps:,} steps ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=args.total_steps)
    best_path = os.path.join(saved_dir, f'ppo_universal_v6_sp100{name_tag}')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', f'universal_v6_sp100{name_tag}')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_universal_v6_sp100{name_tag}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} at step {val_cb.best_step:,}")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({'history': val_cb.history,
                   'best_step': val_cb.best_step,
                   'best_val_sharpe': val_cb.best_val_sharpe}, f, indent=2)

    model = PPO.load(best_path, env=train_env)
    print(f"\n--- Evaluating BEST model on all periods ---\n")
    results = {}
    for name, env, desc in [
        ('train',     train_env,     'Training (2015-2020)'),
        ('val',       val_env,       'Validation (2021)'),
        ('test_bear', test_bear_env, 'Test Bear (2022)'),
        ('test_2023', test_2023_env, 'Test Recovery (2023)'),
        ('test_bull', test_bull_env, 'Test Bull (2024)'),
    ]:
        history = evaluate_agent(model, env)
        m = calculate_metrics(history)
        results[name] = m
        print(f"  {desc}: Return={m['cumulative_return']:>+7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>+7.4f}  MDD={m['max_drawdown']:>+7.2f}%")

    save = {
        'method': f'universal_v6_sp100{name_tag}',
        'universe': f'SP100 2015-cutoff ({n_stocks} stocks) NO long-mom',
        'experimental_changes': {
            'train_window':  '2015-2020 (6yr vs production 14yr)',
            'cutoff':        '2015-01-15 (vs production 2007-01-15)',
            'features':      'V6 baseline (no mom_120d/mom_252d) — ablation pair',
            'total_steps':   args.total_steps,
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step':       val_cb.best_step,
        'periods':         results,
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")

    print()
    print('=' * 80)
    print('   2×2 ABLATION COMPARISON')
    print('=' * 80)
    print('   Cell                                    val      bear      2023     bull')
    print('   ' + '-' * 70)
    try:
        prod = json.load(open('results/universal_v6_sp100/final_results.json'))['periods']
        print(f"   SP100 2007-2020 no-mom (production):   "
              f"{prod['val']['cumulative_return']:>+6.2f}  {prod['test_bear']['cumulative_return']:>+6.2f}  "
              f"{prod['test_2023']['cumulative_return']:>+6.2f}  {prod['test_bull']['cumulative_return']:>+6.2f}")
    except FileNotFoundError:
        pass
    try:
        mom_prod = json.load(open('results/universal_v6_sp100_mom/final_results.json'))['periods']
        print(f"   SP100 2007-2020 + long-mom:            "
              f"{mom_prod['val']['cumulative_return']:>+6.2f}  {mom_prod['test_bear']['cumulative_return']:>+6.2f}  "
              f"{mom_prod['test_2023']['cumulative_return']:>+6.2f}  {mom_prod['test_bull']['cumulative_return']:>+6.2f}")
    except FileNotFoundError:
        pass
    print(f"   SP100 2015 no-mom (THIS RUN):          "
          f"{results['val']['cumulative_return']:>+6.2f}  {results['test_bear']['cumulative_return']:>+6.2f}  "
          f"{results['test_2023']['cumulative_return']:>+6.2f}  {results['test_bull']['cumulative_return']:>+6.2f}")
    try:
        m2015 = json.load(open('results/universal_v6_sp100_2015_mom/final_results.json'))['periods']
        print(f"   SP100 2015 + long-mom:                 "
              f"{m2015['val']['cumulative_return']:>+6.2f}  {m2015['test_bear']['cumulative_return']:>+6.2f}  "
              f"{m2015['test_2023']['cumulative_return']:>+6.2f}  {m2015['test_bull']['cumulative_return']:>+6.2f}")
    except FileNotFoundError:
        print(f"   SP100 2015 + long-mom: (not done yet)")
    print('=' * 80)


if __name__ == '__main__':
    main()
