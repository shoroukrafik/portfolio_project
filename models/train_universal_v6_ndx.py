"""
Universal v6 — NDX (NASDAQ-100). Identical design to v6 DJI / SP100 (clean
hybrid). See train_universal_v6_sp100.py for full rationale.

This is the THIRD-MARKET universality test:
  - DJI 30 (existing)        — large-cap blue-chip financial / industrial mix
  - SP100 (existing)         — large-cap broad-market mega-cap
  - NDX (this script)        — large-cap tech-heavy / growth-tilted

Same recipe across all three (universal-design discipline). Single-agent
test only — no MAC retrain. Outputs feed into the cross-market comparison.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_regime', action='store_true',
                        help='Ablation: remove ALL regime mechanisms.')
    parser.add_argument('--seed', type=int, default=42,
                        help='PPO seed. Default 42. Non-42 → _s{seed} suffix.')
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    name_tag = '_noregime' if args.no_regime else ''
    if args.seed != 42:
        name_tag += f'_s{args.seed}'
    results_dir = os.path.join(project_dir, 'results', f'universal_v6_ndx{name_tag}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    total_timesteps = 1_000_000

    print("=" * 72)
    print("   UNIVERSAL v6 — NDX (NASDAQ-100, 3rd-market universality test)")
    print("=" * 72)
    print(f"\n   Cash floor:    {V6_CASH_FLOORS}")
    print(f"   λ regime:      {V6_RISK_PENALTIES}")

    def load(p):
        df = pd.read_csv(os.path.join(data_dir, p))
        df['date'] = pd.to_datetime(df['date'])
        return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])

    train_df     = load('train_data_ndx.csv')
    val_df       = load('val_data_ndx.csv')
    test_bear_df = load('test_bear_data_ndx.csv')
    test_2023_df = load('test_2023_data_ndx.csv')
    test_bull_df = load('test_bull_data_ndx.csv')

    required = ['mom_60d_norm', 'vol_20d_norm', 'regime_bull_prob']
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. "
                           "Run data/add_universal_features.py first.")

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"   Stocks: {n_stocks}  Features per stock: {n_features}")

    if args.no_regime:
        env_kwargs = dict(
            initial_balance=1_000_000,
            cash_floors={0: 0.0, 1: 0.0, 2: 0.0},
            risk_penalties={0: 0.3, 1: 0.3, 2: 0.3},
            include_regime=False,
            include_regime_probs=False,
            include_vol_features=False,
            use_markowitz_reward=False, use_hyperheuristic=False,
        )
    else:
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

    print(f"\n--- Training {total_timesteps:,} steps (~1-2 hours on CPU) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, f'ppo_universal_v6_ndx{name_tag}')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', f'universal_v6_ndx{name_tag}')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_universal_v6_ndx{name_tag}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({'history': val_cb.history,
                   'best_step': val_cb.best_step,
                   'best_val_sharpe': val_cb.best_val_sharpe}, f, indent=2)

    model = PPO.load(best_path, env=train_env)
    print(f"\n--- Evaluating BEST model on all periods ---\n")
    results = {}
    for name, env, desc in [
        ('train',     train_env,     'Training (2007-2020)'),
        ('val',       val_env,       'Validation (2021)'),
        ('test_bear', test_bear_env, 'Test Bear (2022)'),
        ('test_2023', test_2023_env, 'Test Recovery (2023)'),
        ('test_bull', test_bull_env, 'Test Bull (2024)'),
    ]:
        history = evaluate_agent(model, env)
        m = calculate_metrics(history)
        results[name] = m
        print(f"  {desc}: Return={m['cumulative_return']:>7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>7.4f}  "
              f"MDD={m['max_drawdown']:>7.2f}%")

    save = {
        'method': 'universal_v6_ndx',
        'universe': 'NASDAQ-100 historical subset',
        'v6_settings': {
            'cash_floors': V6_CASH_FLOORS,
            'risk_penalties': V6_RISK_PENALTIES,
            'include_regime_probs': True,
            'include_vol_features': True,
            'net_arch': V6_NET_ARCH,
            'feature_columns': feature_columns,
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")
    print("=" * 72)


if __name__ == '__main__':
    main()
