"""
Universal v2 — applied to the DJI 30 universe.

Same architecture and methodology as DJI FULL, with five literature- and
ablation-informed refinements applied to BOTH DJI and SP100 universes
identically:

  1. + per-stock 60-day momentum feature (mom_60d_norm)
       — Paper 6 J.P. Morgan 60-day lookback precedent
  2. + broad-market vol features (vol_20d_norm, vol_60d_norm,
       vol_ratio_norm = vol_20d/vol_60d)
       — Paper 6 J.P. Morgan vol-regime indicator
  3. + HMM regime probabilities (regime_bull_prob, regime_bear_prob)
       — Paper 2 fuzzy ensemble 3-dim trend vector analog
  4. Reward: uniform λ = 0.1 (was regime-conditioned {0.1, 0.5, 0.3})
       — DJI ablations showed uniform λ=0.1 outperforms regime-conditioned;
         SP100 ablations show uniform λ=0.1 ties FULL (no penalty either way)
  5. Cash floor: sideways floor relaxed to 0% (was 10%)
       — bear floor 50% PRESERVED (the validated defender)
       — sideways floor was pure drag on SP100 2023

Everything else is unchanged:
  • AttentionPortfolioExtractor + PPO + MLP heads [128, 64]
  • DSR reward (validated bull-enabler on DJI, neutral on SP100)
  • Standard hyperparameters (lr=5e-5, n_steps=1024, batch=64)
  • 1M training steps, val-checkpoint selection, per-eval checkpoints
  • seed=42

Outputs (parallel — does NOT overwrite DJI FULL):
  models/saved/ppo_universal_v2_dji.zip
  models/saved/ppo_universal_v2_dji_final.zip
  models/saved/checkpoints/universal_v2_dji/step_*.zip
  results/universal_v2_dji/final_results.json
  results/universal_v2_dji/training_curve.png
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


# ---- Universal v2 settings (identical across DJI and SP100 trainers) ----
UNIVERSAL_V2_CASH_FLOORS = {0: 0.0, 1: 0.5, 2: 0.0}   # bull=0, bear=50%, sideways=0
UNIVERSAL_V2_LAMBDA      = 0.1                         # uniform
UNIVERSAL_V2_FEATURE_COLUMNS = [
    'close','open','high','low','volume',
    'rsi','macd','macd_signal','macd_diff',
    'bb_upper','bb_lower','bb_width',
    'atr','sma_20','sma_50','adx','cci',
    'daily_return','volatility_20d',
    'mom_60d_norm',          # ← NEW Universal v2 feature
]


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', 'universal_v2_dji')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    total_timesteps = 1_000_000

    print("=" * 72)
    print("   UNIVERSAL v2 — DJI 30")
    print("=" * 72)
    print("\n   Refinements vs FULL:")
    print("     + mom_60d per stock (Paper 6 JPM lookback)")
    print("     + vol_20d, vol_60d, vol_ratio (Paper 6 JPM vol-regime)")
    print("     + regime probs in obs (Paper 2 fuzzy analog)")
    print("     ~ λ uniform = 0.1 (was regime-conditioned)")
    print("     ~ sideways floor 10% → 0% (bear floor 50% KEPT)")
    print(f"   Budget: {total_timesteps:,} steps with val checkpointing")

    # ---- Load DJI extended data ----
    print("\n--- Loading DJI extended data (with universal v2 features) ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_ext.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_ext.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_ext.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_ext.csv'))
    for df in (train_df, val_df, test_bear_df, test_2023_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    # Verify v2 features are present
    required = ['mom_60d_norm', 'vol_20d_norm', 'vol_60d_norm', 'vol_ratio_norm',
                'regime_bull_prob', 'regime_bear_prob', 'regime_sideways_prob']
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing universal v2 columns: {missing}. "
                           "Run data/add_universal_features.py first.")

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in UNIVERSAL_V2_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"   Stocks: {n_stocks} (DJI 30)")
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

    # ---- Build PPO with attention extractor ----
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
    print(f"--- Training {total_timesteps:,} steps (~20-30 min on CPU) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_universal_v2_dji')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'universal_v2_dji')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_universal_v2_dji_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")

    # ---- Save histories ----
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
        'method': 'universal_v2_dji',
        'universe': 'DJI 30',
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

    # ---- Compare against DJI FULL ----
    full_path = os.path.join(project_dir, 'results', 'extended', 'final_results.json')
    t2023_path = os.path.join(project_dir, 'results', 'test_2023_summary.json')
    if os.path.exists(full_path):
        full = json.load(open(full_path))
        t2023 = (json.load(open(t2023_path)) if os.path.exists(t2023_path) else {}
                ).get('results', {}).get('FULL (baseline)', {})
        print("\n" + "=" * 80)
        print("   UNIVERSAL v2 DJI vs DJI FULL")
        print("=" * 80)
        print(f"  {'Period':<12} {'System':<24} {'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
        for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
            if period == 'test_2023':
                full_m = t2023
            else:
                full_m = full.get(f'agent_{period}', {})
            if not full_m: continue
            v2_m = results[period]
            print(f"  {period:<12} {'DJI FULL':<24} "
                  f"{full_m.get('cumulative_return',0):>9.2f} "
                  f"{full_m.get('sharpe_ratio',0):>8.4f} "
                  f"{full_m.get('max_drawdown',0):>9.2f}")
            print(f"  {period:<12} {'Universal v2 DJI':<24} "
                  f"{v2_m['cumulative_return']:>9.2f} "
                  f"{v2_m['sharpe_ratio']:>8.4f} "
                  f"{v2_m['max_drawdown']:>9.2f}")
            d_ret = v2_m['cumulative_return'] - full_m.get('cumulative_return', 0)
            d_shp = v2_m['sharpe_ratio'] - full_m.get('sharpe_ratio', 0)
            sgn = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
            print(f"  {'':12} {'  Δ (v2 - FULL)':<24} {sgn(d_ret):>9} "
                  f"{sgn(d_shp):>8}")
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
        ax.set_title(f'Universal v2 DJI ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 72)
    print("   UNIVERSAL v2 DJI COMPLETE ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
