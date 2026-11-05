"""
Universal v4 — hyper-heuristic agent on DJI 30 (Paper 8 framework).

Same architecture, same data, same features as v2 — but the action space
becomes a DISCRETE choice over 8 pre-defined strategies from
models.strategies.STRATEGY_REGISTRY. The agent's job is now:
"given today's market state, which of 8 strategies do I deploy?"

Architecture summary:
  • Same data + v3 features (mom_60d, lags, trend cats, vol features, regime probs)
  • Same AttentionPortfolioExtractor (encodes 29 features × n_stocks)
  • PPO actor/critic on the small head [128, 64]
  • Action space: Discrete(8)
  • Reward: DSR + raw_return − 0.1 × max(0, −return) − cost
    (v2-style; uniform λ=0.1; no Markowitz overfitting risk)
  • Cash floor: bear=50%, sideways=0%, bull=0% (same as v2)

Outputs:
  models/saved/ppo_v4_hh_dji.zip
  results/v4_hh_dji/final_results.json
"""

import os
import sys
import json
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


# v4 settings — shared with SP100 trainer (Universal across universes)
V4_CASH_FLOORS = {0: 0.0, 1: 0.5, 2: 0.0}     # same as v2
V4_LAMBDA      = 0.1                            # uniform downside penalty
V4_NET_ARCH    = dict(pi=[128, 64], vf=[128, 64])   # head can shrink — discrete 8-way

V4_FEATURE_COLUMNS = [
    'close','open','high','low','volume',
    'rsi','macd','macd_signal','macd_diff',
    'bb_upper','bb_lower','bb_width',
    'atr','sma_20','sma_50','adx','cci',
    'daily_return','volatility_20d',
    'mom_60d_norm',
    'r_lag_1_norm','r_lag_5_norm','r_lag_10_norm',
    'r_lag_20_norm','r_lag_40_norm','r_lag_60_norm',
    'is_uptrend','is_sideways','is_downtrend',
]


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', 'v4_hh_dji')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    total_timesteps = 1_000_000

    print("=" * 72)
    print("   UNIVERSAL v4 (hyper-heuristic) — DJI 30")
    print("=" * 72)
    print("\n   Action space: Discrete(8)  (Paper 8 Cui et al. 2024)")
    print("   Reward: DSR + raw_return − 0.1×downside − cost  (v2-style)")
    print(f"   Cash floor: {V4_CASH_FLOORS}")

    # ---- Load DJI data (already has v2 + v3 features) ----
    print("\n--- Loading DJI extended data ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_ext.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_ext.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_ext.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_ext.csv'))
    for df in (train_df, val_df, test_bear_df, test_2023_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    required = ['mom_60d_norm', 'r_lag_60_norm', 'is_uptrend',
                'vol_20d_norm', 'regime_bull_prob']
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Run "
                           "data/add_universal_features.py then "
                           "data/add_v3_features.py first.")

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V4_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"   Stocks: {n_stocks}  Features per stock: {n_features}")

    # ---- Envs (HYPER-HEURISTIC mode) ----
    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=V4_CASH_FLOORS,
        risk_lambda_fixed=V4_LAMBDA,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_hyperheuristic=True,    # ← THE KEY FLAG
    )
    train_env     = PortfolioEnv(df=train_df,     **env_kwargs)
    val_env       = PortfolioEnv(df=val_df,       **env_kwargs)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kwargs)
    test_2023_env = PortfolioEnv(df=test_2023_df, **env_kwargs)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kwargs)

    # ---- Build PPO ----
    policy_kwargs = dict(
        features_extractor_class=AttentionPortfolioExtractor,
        features_extractor_kwargs=dict(
            n_stocks=n_stocks, n_features=n_features,
            embed_dim=64, n_heads=4, features_dim=128,
        ),
        net_arch=V4_NET_ARCH,
    )
    hp_path = os.path.join(project_dir, 'results', 'best_hyperparams.json')
    with open(hp_path) as f:
        hp = json.load(f)
    print(f"   Hyperparameters: lr={hp['learning_rate']}, "
          f"n_steps={hp['n_steps']}, batch={hp['batch_size']}, "
          f"ent_coef={hp['ent_coef']}\n")

    # Higher entropy coefficient for the DISCRETE 8-action space.
    # Default 0.01 is calibrated for continuous control; on small discrete
    # action spaces it lets the policy collapse onto 1-2 actions too early
    # (action mode collapse). 0.05 keeps the policy exploratory long
    # enough for PPO to discover the value of all 8 strategies.
    V4_ENT_COEF = 0.05
    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        ent_coef=V4_ENT_COEF,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=42, device='cpu',
    )

    print(f"--- Training {total_timesteps:,} steps "
          f"(~15-20 min on CPU; discrete action = faster training) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_v4_hh_dji')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'v4_hh_dji')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_v4_hh_dji_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({'history': val_cb.history,
                   'best_step': val_cb.best_step,
                   'best_val_sharpe': val_cb.best_val_sharpe}, f, indent=2)

    # ---- Evaluate ----
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
        'method': 'v4_hyperheuristic_dji',
        'universe': 'DJI 30',
        'v4_settings': {
            'cash_floors': V4_CASH_FLOORS,
            'risk_lambda_fixed': V4_LAMBDA,
            'use_hyperheuristic': True,
            'net_arch': V4_NET_ARCH,
            'feature_columns': feature_columns,
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")

    # ---- Compare against FULL + v2 ----
    full_path = os.path.join(project_dir, 'results', 'extended', 'final_results.json')
    v2_path = os.path.join(project_dir, 'results', 'universal_v2_dji', 'final_results.json')
    t2023_path = os.path.join(project_dir, 'results', 'test_2023_summary.json')
    if os.path.exists(full_path) and os.path.exists(v2_path):
        full = json.load(open(full_path))
        v2 = json.load(open(v2_path))
        t2023 = (json.load(open(t2023_path)) if os.path.exists(t2023_path) else {}
                ).get('results', {}).get('FULL (baseline)', {})
        print("\n" + "=" * 80)
        print("   v4 hyper-heuristic DJI  vs  v2 DJI  vs  DJI FULL")
        print("=" * 80)
        print(f"  {'Period':<12} {'System':<28} {'Ret%':>8} {'Sharpe':>8} {'MDD%':>8}")
        for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
            if period == 'test_2023':
                full_m = t2023
            else:
                full_m = full.get(f'agent_{period}', {})
            v2_m = v2.get('periods', {}).get(period, {})
            v4_m = results[period]
            for label, m in [('DJI FULL', full_m), ('Universal v2 DJI', v2_m),
                              ('v4 hyper-heuristic DJI', v4_m)]:
                if not m: continue
                print(f"  {period:<12} {label:<28} "
                      f"{m.get('cumulative_return', 0):>8.2f} "
                      f"{m.get('sharpe_ratio', 0):>8.4f} "
                      f"{m.get('max_drawdown', 0):>8.2f}")
            print()

    if detailed.rewards_history:
        steps   = [x[0] for x in detailed.rewards_history]
        rewards = [x[1] for x in detailed.rewards_history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rewards, color='blue', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title(f'v4 hyper-heuristic DJI ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 72)
    print("   v4 HYPER-HEURISTIC DJI COMPLETE ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
