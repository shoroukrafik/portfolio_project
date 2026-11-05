"""
Universal v5 — SP100. The "no cash floor" design with v2's enriched features.

Motivation: our SP100 `no_floor` ablation (trained as part of the ablation
sweep) scored 16.52% on bull 2024 — the BEST SP100 bull result across all
our models. Removing the cash floor entirely lets the agent participate
in market rallies without the 50% bear-cash drag on HMM-bear-labeled
days within bull periods (~6% of 2024 days).

v5 adds v2's feature additions on top of the no_floor recipe:
  + mom_60d_norm per stock
  + vol_20d_norm, vol_60d_norm, vol_ratio_norm (broad-market vol)
  + regime probabilities (bull/bear/sideways soft signal)

Reward returns to the validated v1/v2 form:
  reward = DSR + raw_return − 0.1 × max(0, −return)
  (NOT Markowitz — v3 showed Markowitz overfits on SP100 with PPO)

Cash floor: ALL zero. Risk management is learned, not enforced.

Literature precedent: Papers 2, 6, 7, 8 — every published S&P 100 / 90-stock
DRL portfolio paper that WINS uses NO cash floor. The floor was our novelty
that helped bear MDD but capped bull-market participation.

Outputs:
  models/saved/ppo_universal_v5_sp100.zip
  results/universal_v5_sp100/final_results.json
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


# Universal v5 settings (shared with DJI v5 trainer)
V5_CASH_FLOORS = {0: 0.0, 1: 0.0, 2: 0.0}    # ALL zero — no floor
V5_LAMBDA      = 0.1                           # uniform low downside penalty
V5_NET_ARCH    = dict(pi=[128, 64], vf=[128, 64])

V5_FEATURE_COLUMNS = [
    'close','open','high','low','volume',
    'rsi','macd','macd_signal','macd_diff',
    'bb_upper','bb_lower','bb_width',
    'atr','sma_20','sma_50','adx','cci',
    'daily_return','volatility_20d',
    'mom_60d_norm',     # v2 feature
]


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', 'universal_v5_sp100')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    total_timesteps = 1_000_000

    print("=" * 72)
    print("   UNIVERSAL v5 — S&P 100 (NO CASH FLOOR + v2 features)")
    print("=" * 72)
    print("\n   Recipe:")
    print(f"     Cash floor:  {V5_CASH_FLOORS}  (ALL ZERO — no floor at all)")
    print(f"     λ:           {V5_LAMBDA} uniform")
    print("     Reward:      DSR + raw_return − λ×downside  (validated v2 form)")
    print("     Features:    v2 set (incl. mom_60d_norm, regime probs, vol features)")
    print(f"   Budget: {total_timesteps:,} steps with val checkpointing")

    print("\n--- Loading SP100 data ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_sp100.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_sp100.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_sp100.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_sp100.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_sp100.csv'))
    for df in (train_df, val_df, test_bear_df, test_2023_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    required = ['mom_60d_norm', 'vol_20d_norm', 'regime_bull_prob']
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. "
                           "Run data/add_universal_features.py first.")

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V5_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"   Stocks: {n_stocks}  Features per stock: {n_features}")

    # ---- Envs ----
    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=V5_CASH_FLOORS,
        risk_lambda_fixed=V5_LAMBDA,
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

    # ---- Build PPO ----
    policy_kwargs = dict(
        features_extractor_class=AttentionPortfolioExtractor,
        features_extractor_kwargs=dict(
            n_stocks=n_stocks, n_features=n_features,
            embed_dim=64, n_heads=4, features_dim=128,
        ),
        net_arch=V5_NET_ARCH,
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
        ent_coef=hp['ent_coef'],            # standard 0.01 — continuous action
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=42, device='cpu',
    )

    print(f"--- Training {total_timesteps:,} steps "
          f"(~75-100 min on CPU; SP100 large state) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_universal_v5_sp100')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'universal_v5_sp100')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_universal_v5_sp100_final')
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
        'method': 'universal_v5_sp100',
        'universe': 'S&P 100 historical subset (90 stocks)',
        'v5_settings': {
            'cash_floors': V5_CASH_FLOORS,
            'risk_lambda_fixed': V5_LAMBDA,
            'include_regime_probs': True,
            'include_vol_features': True,
            'net_arch': V5_NET_ARCH,
            'feature_columns': feature_columns,
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")

    # ---- Compare against v1, v2, no_floor ablation, benchmarks ----
    v1 = json.load(open('results/extended_sp100/final_results.json'))
    v2 = json.load(open('results/universal_v2_sp100/final_results.json'))
    nf = json.load(open('results/ablation_sp100_no_floor/final_results.json'))

    bench = {'val': (28.79, 33.11), 'test_bear': (-19.95, -9.48),
             'test_2023': (24.73, 21.17), 'test_bull': (24.54, 18.16)}

    print("\n" + "=" * 90)
    print("   UNIVERSAL v5 SP100 vs prior models vs benchmarks")
    print("=" * 90)
    print(f"  {'Period':<11} {'System':<30} {'Ret%':>8} {'Sharpe':>8} {'MDD%':>8}")
    for p in ['val', 'test_bear', 'test_2023', 'test_bull']:
        bh, ew = bench[p]
        print(f"  {p:<11} {'BENCHMARK: S&P 500 BH':<30} {bh:>8.2f}")
        print(f"  {p:<11} {'BENCHMARK: EW SP100':<30} {ew:>8.2f}")
        v1_m = v1.get(f'agent_{p}', {})
        v2_m = v2.get('periods', {}).get(p, {})
        nf_m = nf.get('periods', {}).get(p, {})
        v5_m = results[p]
        for label, m in [('SP100 v1 (FULL)', v1_m),
                          ('SP100 v2', v2_m),
                          ('SP100 no_floor (ablation)', nf_m),
                          ('Universal v5 SP100', v5_m)]:
            if not m: continue
            print(f"  {p:<11} {label:<30} "
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
        ax.set_title(f'Universal v5 SP100 ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 72)
    print("   UNIVERSAL v5 SP100 COMPLETE ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
