"""
MTL v2 trainer — four architectural fixes vs v1:

  1. Deeper regime heads (3-layer MLP each, was 1 Linear+ReLU)
  2. Pairwise cosine diversity loss between head outputs
  3. No auxiliary regime-classification loss (removed)
  4. Wider features_dim (192 vs 128) for more head expressivity

Everything else (env, hyperparameters, training budget, val-checkpoint
protocol, per-eval checkpoint saving) is identical to v1 so the
comparison isolates the architecture changes.

Outputs (parallel to v1, no overwrite):
  models/saved/ppo_mtl_v2_extended.zip
  models/saved/ppo_mtl_v2_extended_final.zip
  models/saved/checkpoints/mtl_v2_extended/    (per-eval checkpoints)
  results/mtl_v2_extended/final_results.json
  results/mtl_v2_extended/training_curve.png
  results/mtl_v2_extended/val_sharpe_history.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from mtl_components_v2 import AttentionMTLExtractorV2, MultiTaskPPOv2
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
    results_dir = os.path.join(project_dir, 'results', 'mtl_v2_extended')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("   MTL v2 — deeper heads + diversity loss + no aux + wider features")
    print("=" * 60)
    print("\n   Fixes targeting v1's head-collapse finding:")
    print("     • Deep heads (3-layer MLP each, ~5x more params per head)")
    print("     • Pairwise cosine diversity loss (heads pushed apart)")
    print("     • No aux regime loss (trunk freed for downstream specialisation)")
    print("     • features_dim=192 (was 128)")

    # ---- Load data ----
    print("\n--- Loading extended data with regime probabilities ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_ext.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_ext.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_ext.csv'))
    for df in (train_df, val_df, test_bear_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    PROB_COLS = ['regime_bull_prob', 'regime_bear_prob', 'regime_sideways_prob']
    for name, df in [('train', train_df), ('val', val_df),
                     ('bear', test_bear_df), ('bull', test_bull_df)]:
        missing = [c for c in PROB_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{name} data missing {missing}")
    print("   ✅ regime probability columns present in all four periods")

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
    print(f"   Train: {train_df['date'].nunique()} days  "
          f"Val: {val_df['date'].nunique()}  "
          f"Bear: {test_bear_df['date'].nunique()}  "
          f"Bull: {test_bull_df['date'].nunique()}")

    # ---- Envs ----
    print("\n--- Creating environments (include_regime_probs=True) ---\n")
    env_kw = dict(initial_balance=1_000_000, include_regime_probs=True)
    train_env     = PortfolioEnv(df=train_df,     **env_kw)
    val_env       = PortfolioEnv(df=val_df,       **env_kw)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kw)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kw)

    # ---- Build MultiTaskPPOv2 with v2 extractor ----
    print("\n--- Building MultiTaskPPOv2 ---\n")
    policy_kwargs = dict(
        features_extractor_class=AttentionMTLExtractorV2,
        features_extractor_kwargs=dict(
            n_stocks=n_stocks, n_features=n_features,
            embed_dim=64, n_heads=4,
            features_dim=192,
            head_hidden_dim=256,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
    hp_path = os.path.join(project_dir, 'results', 'best_hyperparams.json')
    with open(hp_path) as f:
        hp = json.load(f)
    print(f"   Hyperparameters: lr={hp['learning_rate']}, "
          f"n_steps={hp['n_steps']}, batch={hp['batch_size']}, "
          f"ent_coef={hp['ent_coef']}, diversity_coef=0.05")

    model = MultiTaskPPOv2(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        ent_coef=hp['ent_coef'],
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        diversity_coef=0.05,
        verbose=0, seed=42, device='cpu',
    )

    # ---- Train ----
    total_timesteps = 1_000_000
    print(f"\n--- Training {total_timesteps:,} steps "
          f"(~{int(total_timesteps/225_000 * 12)} min on CPU) ---\n")

    detailed = DetailedCallback(eval_freq=50_000,
                                total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_mtl_v2_extended')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'mtl_v2_extended')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_mtl_v2_extended_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")
    print(f"   Best  → {best_path}.zip")
    print(f"   Per-eval checkpoints in {checkpoint_dir}")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({
            'history': val_cb.history,
            'best_step': val_cb.best_step,
            'best_val_sharpe': val_cb.best_val_sharpe,
        }, f, indent=2)
    with open(os.path.join(results_dir, 'training_rewards.json'), 'w') as f:
        json.dump(detailed.rewards_history, f, indent=2)

    # ---- Reload BEST and evaluate ----
    from stable_baselines3 import PPO as _PPO
    model = _PPO.load(best_path, env=train_env,
                      custom_objects={'features_extractor_class':
                                      AttentionMTLExtractorV2})
    print(f"\n--- Evaluating BEST model on all periods ---\n")
    results = {}
    for name, env, desc in [
        ('train',     train_env,     'Training (2007-2020)'),
        ('val',       val_env,       'Validation (2021)'),
        ('test_bear', test_bear_env, 'Test Bear (2022)'),
        ('test_bull', test_bull_env, 'Test Bull (2024)'),
    ]:
        history = evaluate_agent(model, env)
        m = calculate_metrics(history)
        results[name] = m
        print(f"  {desc}: Return={m['cumulative_return']:>7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>7.4f}  "
              f"MDD={m['max_drawdown']:>7.2f}%")

    # ---- Save ----
    save = {
        'method': 'mtl_v2_multi_agent',
        'architecture': {
            'extractor': 'AttentionMTLExtractorV2',
            'n_regime_heads': 3,
            'deep_heads': True,
            'features_dim': 192,
            'head_hidden_dim': 256,
            'aux_loss': False,
            'diversity_coef': 0.05,
            'mixing': 'soft_hmm_probabilities',
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'agent_train':     results['train'],
        'agent_val':       results['val'],
        'agent_test_bear': results['test_bear'],
        'agent_test_bull': results['test_bull'],
    }
    out = os.path.join(results_dir, 'final_results.json')
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {out}")

    # ---- Compare against FULL + MTL v1 ----
    ext_path = os.path.join(project_dir, 'results', 'extended',
                            'final_results.json')
    v1_path = os.path.join(project_dir, 'results', 'mtl_extended',
                           'final_results.json')
    refs = {}
    if os.path.exists(ext_path):
        refs['FULL'] = json.load(open(ext_path))
    if os.path.exists(v1_path):
        refs['MTL v1'] = json.load(open(v1_path))

    if refs:
        print("\n" + "=" * 84)
        print("   MTL v2  vs  MTL v1  vs  FULL extended baseline")
        print("=" * 84)
        print(f"  {'Period':<12} {'System':<24} {'Return%':>9} "
              f"{'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*70}")
        for period in ['val', 'test_bear', 'test_bull']:
            for label, ref in refs.items():
                bk = f'agent_{period}'
                if bk not in ref:
                    continue
                b = ref[bk]
                print(f"  {period:<12} {label:<24} "
                      f"{b['cumulative_return']:>9.2f} "
                      f"{b['sharpe_ratio']:>8.4f} "
                      f"{b['max_drawdown']:>9.2f}")
            m = results[period]
            print(f"  {period:<12} {'MTL v2':<24} "
                  f"{m['cumulative_return']:>9.2f} "
                  f"{m['sharpe_ratio']:>8.4f} "
                  f"{m['max_drawdown']:>9.2f}")
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
        ax.set_title(f'MTL v2 ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 60)
    print("   MTL v2 COMPLETE ✅")
    print("=" * 60)


if __name__ == '__main__':
    main()
