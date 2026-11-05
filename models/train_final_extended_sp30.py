"""
FULL extended baseline — applied to the S&P-30 universe.

Identical architecture, hyperparameters, training budget, and val-
checkpoint protocol to the DJI 30 FULL extended baseline. Only the
data changes: the agent trains on 30 non-DJI mega-caps from the S&P 500
(see data/stock_list_sp30.txt) instead of the DJI 30.

This is an EXTERNAL VALIDITY test: does the regime-aware design
generalize beyond DJI 30 when nothing else changes?

NO PPO hyperparameter retuning was performed for this experiment. The
same `results/best_hyperparams.json` and the same architecture
(`AttentionPortfolioExtractor` with embed_dim=64, n_heads=4,
features_dim=128) are used as for DJI 30.

Test periods include 2023 directly (not retrofitted like the DJI
pipeline) so we get bear-2022, recovery-2023, bull-2024 from one run.

Outputs (parallel — does NOT touch the DJI 30 baseline):
  models/saved/ppo_best_extended_sp30.zip
  models/saved/ppo_final_extended_sp30.zip
  results/extended_sp30/final_results.json
  results/extended_sp30/training_curve.png
  results/extended_sp30/val_sharpe_history.json
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
# Reuse the architecture + callbacks + metrics from the DJI 30 trainer.
from train_final_extended import (
    AttentionPortfolioExtractor,
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
    results_dir = os.path.join(project_dir, 'results', 'extended_sp30')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("   FULL extended — S&P-30 universe (external validity test)")
    print("=" * 70)

    # ---- Load SP-30 data ----
    print("\n--- Loading S&P-30 data ---\n")
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

    print(f"   Stocks: {n_stocks} (S&P-30, non-DJI)")
    print(f"   Features: {n_features}")
    print(f"   Train: {train_df['date'].nunique()} days  "
          f"Val: {val_df['date'].nunique()}  "
          f"Bear: {test_bear_df['date'].nunique()}  "
          f"2023: {test_2023_df['date'].nunique()}  "
          f"Bull: {test_bull_df['date'].nunique()}")

    # Per-regime breakdown (train)
    train_regimes = (train_df[['date','regime']].drop_duplicates()
                     .groupby('regime').size().to_dict())
    name = {0:'bull', 1:'bear', 2:'sideways'}
    print(f"\n   Train regime breakdown:")
    for rid, cnt in sorted(train_regimes.items()):
        pct = cnt / sum(train_regimes.values()) * 100
        print(f"     {name.get(int(rid),'?'):<10} {cnt:>5} days ({pct:.1f}%)")

    # ---- Envs ----
    print("\n--- Creating environments ---\n")
    train_env     = PortfolioEnv(df=train_df,     initial_balance=1_000_000)
    val_env       = PortfolioEnv(df=val_df,       initial_balance=1_000_000)
    test_bear_env = PortfolioEnv(df=test_bear_df, initial_balance=1_000_000)
    test_2023_env = PortfolioEnv(df=test_2023_df, initial_balance=1_000_000)
    test_bull_env = PortfolioEnv(df=test_bull_df, initial_balance=1_000_000)

    # ---- Build PPO — SAME hyperparameters as DJI 30 FULL ----
    print("\n--- Building PPO (identical config to DJI 30 FULL) ---\n")
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
    print(f"   Hyperparameters (reused from DJI 30 tuning): "
          f"lr={hp['learning_rate']}, n_steps={hp['n_steps']}, "
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
    best_path = os.path.join(saved_dir, 'ppo_best_extended_sp30')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'full_extended_sp30')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_final_extended_sp30')
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
        'method': 'FULL_extended_sp30',
        'universe': 'S&P-30 non-DJI mega-caps',
        'config': 'identical to DJI 30 FULL (no retuning)',
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
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {out_path}")

    # ---- Compare to DJI 30 FULL ----
    dji_path = os.path.join(project_dir, 'results', 'extended', 'final_results.json')
    t2023_path = os.path.join(project_dir, 'results', 'test_2023_summary.json')
    if os.path.exists(dji_path):
        dji = json.load(open(dji_path))
        # Pull DJI's 2023 number from the consolidated 2023 file
        dji_2023 = {}
        if os.path.exists(t2023_path):
            dji_2023 = json.load(open(t2023_path)).get('results', {}).get('FULL (baseline)', {})

        print("\n" + "=" * 78)
        print("   FULL on DJI 30  vs  FULL on S&P 30  (SAME PPO config, different stocks)")
        print("=" * 78)
        print(f"  {'Period':<12} {'Universe':<14} "
              f"{'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*64}")
        period_map = {
            'val':       ('agent_val', dji),
            'test_bear': ('agent_test_bear', dji),
            'test_2023': (None, dji_2023),
            'test_bull': ('agent_test_bull', dji),
        }
        for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
            key, src = period_map[period]
            dji_metrics = src.get(key) if key else src
            sp_metrics = results[period]
            if dji_metrics and 'cumulative_return' in dji_metrics:
                print(f"  {period:<12} {'DJI 30':<14} "
                      f"{dji_metrics['cumulative_return']:>9.2f} "
                      f"{dji_metrics['sharpe_ratio']:>8.4f} "
                      f"{dji_metrics['max_drawdown']:>9.2f}")
            print(f"  {period:<12} {'S&P 30':<14} "
                  f"{sp_metrics['cumulative_return']:>9.2f} "
                  f"{sp_metrics['sharpe_ratio']:>8.4f} "
                  f"{sp_metrics['max_drawdown']:>9.2f}")
            if dji_metrics and 'cumulative_return' in dji_metrics:
                d_ret = sp_metrics['cumulative_return'] - dji_metrics['cumulative_return']
                d_shp = sp_metrics['sharpe_ratio']      - dji_metrics['sharpe_ratio']
                d_mdd = sp_metrics['max_drawdown']      - dji_metrics['max_drawdown']
                sgn = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
                print(f"  {'':12} {'  Δ (SP-DJI)':<14} "
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
        ax.set_title(f'FULL extended — S&P-30 universe ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 70)
    print("   FULL extended on S&P-30 COMPLETE ✅")
    print("=" * 70)


if __name__ == '__main__':
    main()
