"""
Universal v5 — DJI 30. Same recipe as v5 SP100 (NO cash floor + v2 features).

Maintains the "single design, applied identically to both universes" goal.
DJI's bear protection comes from learned stock selection (low-vol picks
during HMM-bear days) rather than artificial cash floor. Expected: bear
MDD slightly worse than v2 (which had bear floor=50%) but val/2023/bull
performance preserved or improved.

See train_universal_v5_sp100.py for full rationale.
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
from train_universal_v5_sp100 import (
    V5_CASH_FLOORS, V5_LAMBDA, V5_NET_ARCH, V5_FEATURE_COLUMNS,
)


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', 'universal_v5_dji')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    total_timesteps = 1_000_000

    print("=" * 72)
    print("   UNIVERSAL v5 — DJI 30 (NO CASH FLOOR + v2 features)")
    print("=" * 72)
    print(f"\n   Cash floor: {V5_CASH_FLOORS}  (ALL ZERO)")
    print(f"   λ uniform: {V5_LAMBDA}")
    print("   Reward: DSR + raw_return − λ×downside  (v2-style)")

    print("\n--- Loading DJI data ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_ext.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_ext.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_ext.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_ext.csv'))
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
        ent_coef=hp['ent_coef'],
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=42, device='cpu',
    )

    print(f"--- Training {total_timesteps:,} steps (~25 min on CPU) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, 'ppo_universal_v5_dji')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', 'universal_v5_dji')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_universal_v5_dji_final')
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
        'method': 'universal_v5_dji',
        'universe': 'DJI 30',
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

    # ---- Compare against FULL + v2 + benchmarks ----
    full = json.load(open('results/extended/final_results.json'))
    v2 = json.load(open('results/universal_v2_dji/final_results.json'))
    t2023 = json.load(open('results/test_2023_summary.json')).get('results', {}).get('FULL (baseline)', {})
    dji_bh = {'val': 20.23, 'test_bear': -9.40, 'test_2023': 13.74, 'test_bull': 12.88}
    ew_dji = {'val': 25.09, 'test_bear': -10.25, 'test_bull': 22.04}

    print("\n" + "=" * 90)
    print("   UNIVERSAL v5 DJI vs prior models vs benchmarks")
    print("=" * 90)
    print(f"  {'Period':<11} {'System':<30} {'Ret%':>8} {'Sharpe':>8} {'MDD%':>8}")
    for p in ['val', 'test_bear', 'test_2023', 'test_bull']:
        print(f"  {p:<11} {'BENCHMARK: DJI BH':<30} {dji_bh[p]:>8.2f}")
        if p in ew_dji:
            print(f"  {p:<11} {'BENCHMARK: EW DJI':<30} {ew_dji[p]:>8.2f}")
        if p == 'test_2023':
            full_m = t2023
        else:
            full_m = full.get(f'agent_{p}', {})
        v2_m = v2.get('periods', {}).get(p, {})
        v5_m = results[p]
        for label, m in [('DJI FULL', full_m),
                          ('Universal v2 DJI', v2_m),
                          ('Universal v5 DJI', v5_m)]:
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
        ax.set_title(f'Universal v5 DJI ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 72)
    print("   UNIVERSAL v5 DJI COMPLETE ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
