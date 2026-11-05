"""
Universal v2 — applied to the S&P 100 universe (90 stocks).

Identical design to train_universal_v2_dji.py — the whole point of
Universal v2 is to use the SAME design on both universes. Only the data
files (and saved model paths) differ.

See models/train_universal_v2_dji.py for the full rationale of each
refinement (5 changes vs FULL, all backed by lit + ablations).

Outputs (parallel — does NOT overwrite SP100 v1):
  models/saved/ppo_universal_v2_sp100.zip
  models/saved/ppo_universal_v2_sp100_final.zip
  models/saved/checkpoints/universal_v2_sp100/step_*.zip
  results/universal_v2_sp100/final_results.json
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
from train_universal_v2_dji import (
    UNIVERSAL_V2_CASH_FLOORS,
    UNIVERSAL_V2_LAMBDA,
    UNIVERSAL_V2_FEATURE_COLUMNS,
)


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', 'universal_v2_sp100')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    total_timesteps = 1_000_000

    print("=" * 72)
    print("   UNIVERSAL v2 — S&P 100 (90 stocks)")
    print("=" * 72)
    print("\n   IDENTICAL design to Universal v2 DJI. Only the data differs.")
    print("     + mom_60d, vol_20d/60/ratio, regime probs (added obs)")
    print(f"     ~ λ uniform = {UNIVERSAL_V2_LAMBDA}, cash_floors = {UNIVERSAL_V2_CASH_FLOORS}")
    print(f"   Budget: {total_timesteps:,} steps with val checkpointing")

    # ---- Load SP100 data ----
    print("\n--- Loading SP100 data (with universal v2 features) ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_sp100.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_sp100.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_sp100.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_sp100.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_sp100.csv'))
    for df in (train_df, val_df, test_bear_df, test_2023_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    required = ['mom_60d_norm', 'vol_20d_norm', 'vol_60d_norm', 'vol_ratio_norm',
                'regime_bull_prob', 'regime_bear_prob', 'regime_sideways_prob']
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing universal v2 columns: {missing}. "
                           "Run data/add_universal_features.py first.")

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in UNIVERSAL_V2_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"   Stocks: {n_stocks} (SP100 historical subset)")
    print(f"   Features per stock: {n_features}  (incl. mom_60d_norm)")
    print(f"   Train: {train_df['date'].nunique()} days  "
          f"Val: {val_df['date'].nunique()}  "
          f"Bear: {test_bear_df['date'].nunique()}  "
          f"2023: {test_2023_df['date'].nunique()}  "
          f"Bull: {test_bull_df['date'].nunique()}")

    # ---- Envs ----
    print("\n--- Creating environments (Universal v2 settings) ---\n")
    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=UNIVERSAL_V2_CASH_FLOORS,
        risk_lambda_fixed=UNIVERSAL_V2_LAMBDA,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
    )
    train_env     = PortfolioEnv(df=train_df,     **env_kwargs)
    val_env       = PortfolioEnv(df=val_df,       **env_kwargs)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kwargs)
    test_2023_env = PortfolioEnv(df=test_2023_df, **env_kwargs)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kwargs)

    # ---- Build PPO ----
    print("\n--- Building PPO ---\n")
    policy_kwargs = dict(
        features_extractor_class=AttentionPortfolioExtractor,
        features_extractor_kwargs=dict(
            n_stocks=n_stocks, n_features=n_features,
            embed_dim=64, n_heads=4, features_dim=128,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
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
        verbose=0, seed=42, device='cpu',
    )

    # ---- Train ----
    print(f"--- Training {total_timesteps:,} steps "
          f"(~60-90 min on CPU; SP100 has 3× DJI state space) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_universal_v2_sp100')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'universal_v2_sp100')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_universal_v2_sp100_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({
            'history': val_cb.history,
            'best_step': val_cb.best_step,
            'best_val_sharpe': val_cb.best_val_sharpe,
        }, f, indent=2)

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
        'method': 'universal_v2_sp100',
        'universe': 'S&P 100 historical subset (90 stocks)',
        'universal_v2_settings': {
            'cash_floors': UNIVERSAL_V2_CASH_FLOORS,
            'risk_lambda_fixed': UNIVERSAL_V2_LAMBDA,
            'include_regime_probs': True,
            'include_vol_features': True,
            'feature_columns': feature_columns,
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
    }
    out = os.path.join(results_dir, 'final_results.json')
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {out}")

    # ---- Compare against SP100 v1 ----
    v1_path = os.path.join(project_dir, 'results', 'extended_sp100',
                            'final_results.json')
    if os.path.exists(v1_path):
        v1 = json.load(open(v1_path))
        print("\n" + "=" * 80)
        print("   UNIVERSAL v2 SP100 vs SP100 v1 (FULL)")
        print("=" * 80)
        print(f"  {'Period':<12} {'System':<28} {'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
        for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
            v1_m = v1.get(f'agent_{period}', {})
            v2_m = results[period]
            if not v1_m: continue
            print(f"  {period:<12} {'SP100 v1 (FULL)':<28} "
                  f"{v1_m['cumulative_return']:>9.2f} "
                  f"{v1_m['sharpe_ratio']:>8.4f} "
                  f"{v1_m['max_drawdown']:>9.2f}")
            print(f"  {period:<12} {'Universal v2 SP100':<28} "
                  f"{v2_m['cumulative_return']:>9.2f} "
                  f"{v2_m['sharpe_ratio']:>8.4f} "
                  f"{v2_m['max_drawdown']:>9.2f}")
            d_ret = v2_m['cumulative_return'] - v1_m['cumulative_return']
            d_shp = v2_m['sharpe_ratio'] - v1_m['sharpe_ratio']
            sgn = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
            print(f"  {'':12} {'  Δ (v2 - v1)':<28} {sgn(d_ret):>9} {sgn(d_shp):>8}")
            print()

    if detailed.rewards_history:
        steps   = [x[0] for x in detailed.rewards_history]
        rewards = [x[1] for x in detailed.rewards_history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rewards, color='blue', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title(f'Universal v2 SP100 ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 72)
    print("   UNIVERSAL v2 SP100 COMPLETE ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
