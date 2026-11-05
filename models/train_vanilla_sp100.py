"""
Vanilla baseline for the SP100 universe — the most stripped-down PPO
that still trains on the same data + same training protocol as SP100
FULL v1, but with ALL thesis novelties removed at once.

Use it as the reference for the cumulative contribution of the FULL
design on SP100. The gap between vanilla SP100 and SP100 v1 quantifies
"how much do all the thesis additions buy us, in aggregate, on this
universe?"

==========================================================================
REMOVED COMPARED TO SP100 FULL v1
==========================================================================
  - HMM regime detection             (not consulted at all)
  - Regime one-hot in observation    (include_regime=False)
  - Regime-conditioned cash floor    (ablation_mode='vanilla' disables it)
  - λ downside penalty               (risk_lambda_fixed=0.0)
  - DSR reward                       (raw return reward instead)
  - Multi-head attention extractor   (replaced with MLP)

==========================================================================
KEPT (the irreducible PPO baseline)
==========================================================================
  - PPO algorithm + actor-critic with MLP heads [128, 64]
  - MLP feature extractor (Linear stack → 128 dims)
  - 19 technical features per stock (z-score normalized, train-period stats)
  - Softmax action over 91 outputs (90 stocks + cash)
  - 0.1% transaction cost
  - Reward = raw daily portfolio return
  - 1M training steps, val-checkpoint selection by val Sharpe
  - Per-eval checkpoint saving
  - Standard hyperparameters (lr=5e-5, n_steps=1024, batch=64, ent_coef=0.01)
  - seed=42

Outputs (parallel — does NOT touch DJI vanilla or SP100 v1):
  models/saved/ppo_vanilla_sp100.zip
  models/saved/ppo_vanilla_sp100_final.zip
  models/saved/checkpoints/vanilla_sp100/step_*.zip
  results/vanilla_sp100/final_results.json
  results/vanilla_sp100/training_curve.png
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
from train_ablation import MLPPortfolioExtractor
from train_final_extended import (
    DetailedCallback,
    ValidationCheckpointCallback,
    calculate_metrics,
    evaluate_agent,
)


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', 'vanilla_sp100')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    total_timesteps = 1_000_000

    print("=" * 70)
    print("   VANILLA baseline — S&P 100 universe (no thesis novelties)")
    print("=" * 70)
    print("\n   REMOVED: HMM regime, cash floor, λ penalty, DSR, attention")
    print("   KEPT:    PPO + MLP + raw return reward + 19 features")
    print(f"   Budget:  {total_timesteps:,} steps with val checkpointing")

    # ---- Load S&P 100 data ----
    print("\n--- Loading S&P 100 data ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_sp100.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_sp100.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_sp100.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_sp100.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_sp100.csv'))
    for df in (train_df, val_df, test_bear_df, test_2023_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [
        'close','open','high','low','volume',
        'rsi','macd','macd_signal','macd_diff',
        'bb_upper','bb_lower','bb_width',
        'atr','sma_20','sma_50','adx','cci',
        'daily_return','volatility_20d',
    ]
    feature_columns = [f for f in feature_columns if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"   Stocks: {n_stocks}  Features: {n_features}")
    print(f"   Train: {train_df['date'].nunique()} days  "
          f"Val: {val_df['date'].nunique()}  "
          f"Bear: {test_bear_df['date'].nunique()}  "
          f"2023: {test_2023_df['date'].nunique()}  "
          f"Bull: {test_bull_df['date'].nunique()}")

    # ---- Envs (vanilla settings) ----
    # ablation_mode='vanilla' inside the env:
    #   - disables the regime cash floor (cash_floors → all 0)
    #   - swaps DSR for raw daily return as the reward signal
    # include_regime=False removes the 3-dim regime one-hot from the obs.
    # risk_lambda_fixed=0.0 zeros out the downside penalty entirely.
    print("\n--- Creating environments (vanilla settings) ---\n")
    env_kwargs = dict(
        initial_balance=1_000_000,
        ablation_mode='vanilla',
        risk_lambda_fixed=0.0,
        include_regime=False,
    )
    train_env     = PortfolioEnv(df=train_df,     **env_kwargs)
    val_env       = PortfolioEnv(df=val_df,       **env_kwargs)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kwargs)
    test_2023_env = PortfolioEnv(df=test_2023_df, **env_kwargs)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kwargs)

    # ---- Build PPO with MLP extractor (NO attention) ----
    policy_kwargs = dict(
        features_extractor_class=MLPPortfolioExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    hp_path = os.path.join(project_dir, 'results', 'best_hyperparams.json')
    with open(hp_path) as f:
        hp = json.load(f)
    print(f"   Hyperparameters: lr={hp['learning_rate']}, "
          f"n_steps={hp['n_steps']}, batch={hp['batch_size']}, "
          f"ent_coef={hp['ent_coef']}\n")

    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        ent_coef=hp['ent_coef'],
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=42, device='cpu',
    )

    # ---- Train ----
    print(f"--- Training {total_timesteps:,} steps "
          f"(~50-90 min on CPU; SP100 has 3× DJI state space) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_vanilla_sp100')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'vanilla_sp100')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_vanilla_sp100_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")
    print(f"   Best  → {best_path}.zip")

    # ---- Reload BEST and evaluate ----
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
        'method': 'sp100_vanilla',
        'universe': 'S&P 100 historical subset (90 stocks)',
        'components_removed': [
            'differential_sharpe_ratio_reward',
            'regime_cash_floor',
            'regime_conditioned_lambda',
            'attention_feature_extractor',
            'regime_one_hot_observation',
            'hmm_regime_detection_consumed',
        ],
        'env_ablation_mode': 'vanilla',
        'risk_lambda_fixed': 0.0,
        'include_regime': False,
        'use_attention': False,
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
        'val_sharpe_history': val_cb.history,
        'rewards_history': detailed.rewards_history,
    }
    out = os.path.join(results_dir, 'final_results.json')
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {out}")

    # ---- Compare against SP100 FULL v1 ----
    sp100_full_path = os.path.join(project_dir, 'results', 'extended_sp100',
                                    'final_results.json')
    if os.path.exists(sp100_full_path):
        with open(sp100_full_path) as f:
            full = json.load(f)
        print("\n" + "=" * 80)
        print("   VANILLA SP100 vs SP100 FULL v1  "
              "(cumulative contribution of all thesis novelties)")
        print("=" * 80)
        print(f"  {'Period':<12} {'System':<24} {'Return%':>9} "
              f"{'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*70}")
        for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
            bk = f'agent_{period}'
            if bk not in full:
                continue
            b = full[bk]
            m = results[period]
            print(f"  {period:<12} {'SP100 FULL v1':<24} "
                  f"{b['cumulative_return']:>9.2f} "
                  f"{b['sharpe_ratio']:>8.4f} "
                  f"{b['max_drawdown']:>9.2f}")
            print(f"  {period:<12} {'VANILLA SP100':<24} "
                  f"{m['cumulative_return']:>9.2f} "
                  f"{m['sharpe_ratio']:>8.4f} "
                  f"{m['max_drawdown']:>9.2f}")
            d_ret = b['cumulative_return'] - m['cumulative_return']
            d_shp = b['sharpe_ratio']      - m['sharpe_ratio']
            d_mdd = b['max_drawdown']      - m['max_drawdown']
            sgn = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
            print(f"  {'':12} {'  Δ (FULL - vanilla)':<24} "
                  f"{sgn(d_ret):>9} {sgn(d_shp):>8} {sgn(d_mdd):>9}")
            print()

    # ---- Training curve ----
    if detailed.rewards_history:
        steps   = [x[0] for x in detailed.rewards_history]
        rewards = [x[1] for x in detailed.rewards_history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rewards, color='blue', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title(f'Vanilla SP100 ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 70)
    print("   VANILLA SP100 COMPLETE ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
