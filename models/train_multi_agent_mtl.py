"""
Phase 1 — Multi-task learning multi-agent (MTL).

Single neural network with:
  - Shared attention backbone (same as FULL)
  - 3 regime-specialised projection heads (bull / bear / sideways)
  - Soft mixing of head outputs by per-day HMM probabilities
  - 1 auxiliary regime-prediction head trained with cross-entropy

Reward, cash floor, regime-conditioned λ, extended training data,
hyperparameters and val-checkpointing protocol are byte-identical to
the FULL extended baseline — only the feature extractor and PPO loss
differ.

Outputs (parallel to results/extended/, no overwrite):
  models/saved/ppo_mtl_extended.zip                  — val-checkpointed best
  models/saved/ppo_mtl_extended_final.zip            — end-of-training
  results/mtl_extended/final_results.json
  results/mtl_extended/training_curve.png
  results/mtl_extended/val_sharpe_history.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from mtl_components import AttentionMTLExtractor, MultiTaskPPO
# Reuse helpers from the FULL trainer (callbacks, metrics)
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
    results_dir = os.path.join(project_dir, 'results', 'mtl_extended')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("   PHASE 1 — MTL multi-agent (3 regime heads + aux loss)")
    print("=" * 60)
    print("\n   Architecture:")
    print("     • Shared attention backbone (same as FULL)")
    print("     • 3 regime-specialised projection heads (bull / bear / sideways)")
    print("     • Soft mixing of head outputs by HMM probabilities")
    print("     • Auxiliary regime-classification head (CE loss)")

    # ---- Load extended data (with HMM probability columns) ----
    print("\n--- Loading extended data with regime probabilities ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_ext.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_ext.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_ext.csv'))
    for df in (train_df, val_df, test_bear_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    # Sanity: probability columns must be present
    PROB_COLS = ['regime_bull_prob', 'regime_bear_prob', 'regime_sideways_prob']
    for name, df in [('train', train_df), ('val', val_df),
                     ('bear', test_bear_df), ('bull', test_bull_df)]:
        missing = [c for c in PROB_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"{name} data missing {missing}. "
                f"Run data/add_regime_probs_to_data.py first."
            )
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

    # ---- Envs (include_regime_probs=True to expose HMM probs to extractor) ----
    print("\n--- Creating environments (include_regime_probs=True) ---\n")
    env_kw = dict(initial_balance=1_000_000, include_regime_probs=True)
    train_env     = PortfolioEnv(df=train_df,     **env_kw)
    val_env       = PortfolioEnv(df=val_df,       **env_kw)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kw)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kw)

    # ---- Build MultiTaskPPO with MTL extractor ----
    print("\n--- Building MultiTaskPPO (same hyperparameters as FULL) ---\n")
    policy_kwargs = dict(
        features_extractor_class=AttentionMTLExtractor,
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
          f"ent_coef={hp['ent_coef']}, aux_loss_coef=0.1")

    model = MultiTaskPPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        ent_coef=hp['ent_coef'],
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        aux_loss_coef=0.1,
        verbose=0, seed=42, device='cpu',
    )

    # ---- Train 1M steps with val checkpointing ----
    total_timesteps = 1_000_000
    print(f"\n--- Training {total_timesteps:,} steps "
          f"(~{int(total_timesteps/225_000 * 12)} min on CPU) ---\n")

    detailed = DetailedCallback(eval_freq=50_000,
                                total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_mtl_extended')
    # MTL has more parameters than FULL → susceptible to val-spike overfitting.
    # Save every eval checkpoint so we can apply a rolling-mean selection
    # criterion post-hoc instead of single-eval val-best.
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'mtl_extended')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_mtl_extended_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")
    print(f"   Best  → {best_path}.zip")

    # Save val + reward histories
    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({
            'history': val_cb.history,
            'best_step': val_cb.best_step,
            'best_val_sharpe': val_cb.best_val_sharpe,
        }, f, indent=2)
    with open(os.path.join(results_dir, 'training_rewards.json'), 'w') as f:
        json.dump(detailed.rewards_history, f, indent=2)

    # ---- Reload BEST model and evaluate ----
    # Note: PPO.load (parent class) creates a plain PPO; we want MultiTaskPPO
    # to keep the custom train(), but for inference plain PPO is identical.
    # Inference doesn't trigger train() so the aux loss is irrelevant here.
    from stable_baselines3 import PPO as _PPO
    model = _PPO.load(best_path, env=train_env,
                      custom_objects={'features_extractor_class':
                                      AttentionMTLExtractor})
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

    # ---- Save (same structure as FULL's final_results.json) ----
    save = {
        'method': 'mtl_multi_agent_phase1',
        'architecture': {
            'extractor': 'AttentionMTLExtractor',
            'n_regime_heads': 3,
            'aux_loss': True,
            'aux_loss_coef': 0.1,
            'mixing': 'soft_hmm_probabilities',
        },
        'env': {
            'include_regime_probs': True,
            'cash_floor': True,
            'dsr_reward': True,
            'regime_lambda': True,
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

    # ---- Compare against extended FULL baseline ----
    ext_path = os.path.join(project_dir, 'results', 'extended',
                            'final_results.json')
    if os.path.exists(ext_path):
        with open(ext_path) as f:
            ext = json.load(f)
        print("\n" + "=" * 76)
        print("   MTL multi-agent (Phase 1)  vs  FULL extended baseline")
        print("=" * 76)
        print(f"  {'Period':<12} {'System':<22} {'Return%':>9} "
              f"{'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*70}")
        for period in ['val', 'test_bear', 'test_bull']:
            bk = f'agent_{period}'
            if bk not in ext:
                continue
            b = ext[bk]
            m = results[period]
            print(f"  {period:<12} {'FULL (baseline)':<22} "
                  f"{b['cumulative_return']:>9.2f} "
                  f"{b['sharpe_ratio']:>8.4f} "
                  f"{b['max_drawdown']:>9.2f}")
            print(f"  {period:<12} {'MTL (Phase 1)':<22} "
                  f"{m['cumulative_return']:>9.2f} "
                  f"{m['sharpe_ratio']:>8.4f} "
                  f"{m['max_drawdown']:>9.2f}")
            d_ret = m['cumulative_return'] - b['cumulative_return']
            d_shp = m['sharpe_ratio']      - b['sharpe_ratio']
            d_mdd = m['max_drawdown']      - b['max_drawdown']
            sgn = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
            print(f"  {'':12} {'  Δ (MTL − FULL)':<22} "
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
        ax.set_title(f'MTL multi-agent (Phase 1, {total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 60)
    print("   PHASE 1 MTL COMPLETE ✅")
    print("=" * 60)


if __name__ == '__main__':
    main()
