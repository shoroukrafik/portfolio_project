"""
SP30 v2 — FULL design with universe-calibrated env constants.

Same architecture, same hyperparameters, same training budget as DJI 30
FULL and as the original SP30 experiment. ONLY the env's regime-conditional
constants (cash floor percentages + λ values) differ — calibrated to fit
SP30's more growth-tilted character.

Rationale (from the SP30 v1 diagnostic):
  - SP30 v1 used DJI's calibration (cash floor 50%/10%, λ 0.1/0.5/0.3)
  - Val Sharpe spiked to 3.45 but test results lost across the board
  - Diagnostic confirmed not a selection artifact — the over-defensive
    DJI calibration suppressed SP30's growth-stock upside
  - SP30 has more growth names (NFLX, GOOGL, AMD, COST, ADBE, ISRG, INTU)
    that need different risk treatment than DJI mega-blue-chips

Calibration differences from DJI 30 / SP30 v1:
                       DJI 30 / SP30 v1   →  SP30 v2 (calibrated)
   Bear cash floor      50%                30%   (less over-defense)
   Sideways cash floor  10%                 5%   (more participation)
   Bull cash floor       0%                 0%   (unchanged)
   Bull λ              0.10               0.05   (more aggressive in bull)
   Bear λ              0.50               0.50   (unchanged — bear stays defensive)
   Sideways λ          0.30               0.20   (slightly less defensive)

Everything else identical:
  - Same AttentionPortfolioExtractor (embed_dim=64, n_heads=4, features_dim=128)
  - Same hyperparameters from results/best_hyperparams.json
  - Same 1M timesteps + val-checkpointing every 25k
  - Same seed (42)
  - Same train/val/test data splits (the SP30 universe + retuned HMM)
  - Per-eval checkpoints saved for diagnostic

Outputs (parallel — does NOT touch DJI 30 OR original SP30):
  models/saved/ppo_best_extended_sp30_v2.zip
  models/saved/ppo_final_extended_sp30_v2.zip
  models/saved/checkpoints/full_extended_sp30_v2/
  results/extended_sp30_v2/final_results.json
  results/extended_sp30_v2/training_curve.png
  results/extended_sp30_v2/val_sharpe_history.json
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


# Universe-calibrated env constants for SP30 (growth-tilted universe)
SP30_CASH_FLOORS    = {0: 0.0,  1: 0.30, 2: 0.05}   # bull / bear / sideways
SP30_RISK_PENALTIES = {0: 0.05, 1: 0.50, 2: 0.20}


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', 'extended_sp30_v2')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 72)
    print("   SP30 v2 — FULL design with calibrated env constants")
    print("=" * 72)
    print(f"\n   Cash floors:    bull={SP30_CASH_FLOORS[0]}, "
          f"bear={SP30_CASH_FLOORS[1]}, "
          f"sideways={SP30_CASH_FLOORS[2]}  (was 0.0/0.5/0.1)")
    print(f"   Risk penalties: bull={SP30_RISK_PENALTIES[0]}, "
          f"bear={SP30_RISK_PENALTIES[1]}, "
          f"sideways={SP30_RISK_PENALTIES[2]}  (was 0.1/0.5/0.3)")

    # ---- Load SP30 data ----
    print("\n--- Loading SP30 data ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_sp30.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_sp30.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_sp30.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_sp30.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_sp30.csv'))
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
    print(f"   Stocks: {n_stocks}, Features: {n_features}")

    # ---- Envs (with calibrated cash_floors + risk_penalties) ----
    print("\n--- Creating environments with SP30 calibration ---\n")
    env_kwargs = dict(initial_balance=1_000_000,
                      cash_floors=SP30_CASH_FLOORS,
                      risk_penalties=SP30_RISK_PENALTIES)
    train_env     = PortfolioEnv(df=train_df,     **env_kwargs)
    val_env       = PortfolioEnv(df=val_df,       **env_kwargs)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kwargs)
    test_2023_env = PortfolioEnv(df=test_2023_df, **env_kwargs)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kwargs)

    # ---- PPO — SAME hyperparameters and architecture ----
    print("\n--- Building PPO (identical hyperparameters to DJI 30 / SP30 v1) ---\n")
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
    print(f"   lr={hp['learning_rate']}, n_steps={hp['n_steps']}, "
          f"batch={hp['batch_size']}, ent_coef={hp['ent_coef']}")

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
    total_timesteps = 1_000_000
    print(f"\n--- Training {total_timesteps:,} steps "
          f"(~{int(total_timesteps/225_000 * 12)} min on CPU) ---\n")

    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_best_extended_sp30_v2')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'full_extended_sp30_v2')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_final_extended_sp30_v2')
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
    with open(os.path.join(results_dir, 'training_rewards.json'), 'w') as f:
        json.dump(detailed.rewards_history, f, indent=2)

    # ---- Reload BEST and evaluate ----
    model = PPO.load(best_path, env=train_env)
    print(f"\n--- Evaluating BEST model on all periods ---\n")
    results = {}
    for label, env, desc in [
        ('train',     train_env,     'Training (2007-2020)'),
        ('val',       val_env,       'Validation (2021)'),
        ('test_bear', test_bear_env, 'Test Bear (2022)'),
        ('test_2023', test_2023_env, 'Test Recovery (2023)'),
        ('test_bull', test_bull_env, 'Test Bull (2024)'),
    ]:
        history = evaluate_agent(model, env)
        m = calculate_metrics(history)
        results[label] = m
        print(f"  {desc}: Return={m['cumulative_return']:>7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>7.4f}  "
              f"MDD={m['max_drawdown']:>7.2f}%")

    # ---- Save ----
    save = {
        'method': 'FULL_extended_sp30_v2',
        'universe': 'S&P-30 non-DJI mega-caps',
        'config': 'identical PPO hyperparameters and architecture to DJI 30 FULL; '
                  'env-level cash_floors + risk_penalties calibrated for SP30 universe',
        'cash_floors':    SP30_CASH_FLOORS,
        'risk_penalties': SP30_RISK_PENALTIES,
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'agent_train':     results['train'],
        'agent_val':       results['val'],
        'agent_test_bear': results['test_bear'],
        'agent_test_2023': results['test_2023'],
        'agent_test_bull': results['test_bull'],
    }
    out_path = os.path.join(results_dir, 'final_results.json')
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2, default=float)
    print(f"\n✅ Results → {out_path}")

    # ---- Three-way comparison: SP30 v2 vs SP30 v1 vs DJI 30 ----
    sp30_v1_path = os.path.join(project_dir, 'results', 'extended_sp30',
                                 'final_results.json')
    dji_path = os.path.join(project_dir, 'results', 'extended',
                            'final_results.json')
    t2023_path = os.path.join(project_dir, 'results', 'test_2023_summary.json')

    sp30_v1 = json.load(open(sp30_v1_path)) if os.path.exists(sp30_v1_path) else None
    dji     = json.load(open(dji_path))     if os.path.exists(dji_path)     else None
    dji_2023 = {}
    if os.path.exists(t2023_path):
        dji_2023 = json.load(open(t2023_path)).get('results', {}).get('FULL (baseline)', {})

    if sp30_v1 or dji:
        print("\n" + "=" * 90)
        print("   SP30 v2 (calibrated)  vs  SP30 v1 (DJI-calibrated)  vs  DJI 30 FULL")
        print("=" * 90)
        print(f"  {'Period':<10} {'System':<28} "
              f"{'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*64}")
        period_map_v1 = {
            'val':       'agent_val',
            'test_bear': 'agent_test_bear',
            'test_2023': 'agent_test_2023',
            'test_bull': 'agent_test_bull',
        }
        period_map_dji = {
            'val':       ('agent_val', dji),
            'test_bear': ('agent_test_bear', dji),
            'test_2023': (None, dji_2023),
            'test_bull': ('agent_test_bull', dji),
        }
        for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
            if dji:
                key, src = period_map_dji[period]
                dji_m = src.get(key) if key else src
                if dji_m and 'cumulative_return' in dji_m:
                    print(f"  {period:<10} {'DJI 30 FULL':<28} "
                          f"{dji_m['cumulative_return']:>9.2f} "
                          f"{dji_m['sharpe_ratio']:>8.4f} "
                          f"{dji_m['max_drawdown']:>9.2f}")
            if sp30_v1:
                k = period_map_v1[period]
                v1 = sp30_v1.get(k)
                if v1:
                    print(f"  {period:<10} {'SP30 v1 (DJI-calibrated)':<28} "
                          f"{v1['cumulative_return']:>9.2f} "
                          f"{v1['sharpe_ratio']:>8.4f} "
                          f"{v1['max_drawdown']:>9.2f}")
            v2 = results[period]
            print(f"  {period:<10} {'SP30 v2 (calibrated) ★':<28} "
                  f"{v2['cumulative_return']:>9.2f} "
                  f"{v2['sharpe_ratio']:>8.4f} "
                  f"{v2['max_drawdown']:>9.2f}")
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
        ax.set_title(f'SP30 v2 (calibrated) — {total_timesteps:,} steps')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 72)
    print("   SP30 v2 (calibrated) COMPLETE ✅")
    print("=" * 72)


if __name__ == '__main__':
    main()
