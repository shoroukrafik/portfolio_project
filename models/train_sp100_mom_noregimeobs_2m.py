"""
Experimental SP100 variant for the asymmetric-regime hypothesis test.

Configuration (the "strong" version of the no-regime hypothesis):
  - REMOVE: regime one-hot from observation        (include_regime=False)
  - REMOVE: regime posterior from observation      (include_regime_probs=False)
  - REMOVE: broad-market vol features              (include_vol_features=False)
  - KEEP:   regime-conditioned lambda in reward    (risk_penalties=V6_RISK_PENALTIES)
  - KEEP:   long-momentum features (mom_120d, mom_252d)
  - 2M training steps (same as canonical)
  - Cash floors all zero at training (canonical setup)

What this isolates: the contribution of removing the 9 regime-related
OBSERVATION features while keeping the regime-conditioned REWARD shaping
and long-momentum features.

Output:
  models/saved/ppo_sp100_mom_noregimeobs_2m.zip               (best by val Sharpe)
  models/saved/ppo_sp100_mom_noregimeobs_2m_final.zip         (final checkpoint)
  models/saved/checkpoints/sp100_mom_noregimeobs_2m/step_*.zip
  results/sp100_mom_noregimeobs_2m/val_sharpe_history.json
  results/sp100_mom_noregimeobs_2m/final_results.json

Estimated runtime: ~3 hours on a modern Apple Silicon CPU.
After training, run eval_sp100_mom_noregimeobs_defensive.py to apply
the DEFENSIVE rule and floor sweep.
"""

import os
import sys
import json
import argparse
import pandas as pd
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
from train_universal_v6_sp100 import (
    V6_CASH_FLOORS, V6_RISK_PENALTIES, V6_NET_ARCH,
    V6_FEATURE_COLUMNS, V3_COLS_TO_DROP,
)

V6_FEATURE_COLUMNS_MOM = V6_FEATURE_COLUMNS + ['mom_120d_norm', 'mom_252d_norm']

TAG = 'sp100_mom_noregimeobs_2m'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', TAG)
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 80)
    print(f'   EXPERIMENT: {TAG}')
    print('   Recipe: keep regime-conditioned lambda + long-mom, REMOVE regime obs')
    print(f'   Training: {args.total_steps:,} steps')
    print('=' * 80)

    def load(p):
        df = pd.read_csv(os.path.join(data_dir, p))
        df['date'] = pd.to_datetime(df['date'])
        return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])

    train_df     = load('train_data_sp100.csv')
    val_df       = load('val_data_sp100.csv')
    test_bear_df = load('test_bear_data_sp100.csv')
    test_2023_df = load('test_2023_data_sp100.csv')
    test_bull_df = load('test_bull_data_sp100.csv')

    required = ['mom_60d_norm', 'mom_120d_norm', 'mom_252d_norm']
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}.")

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS_MOM if f in train_df.columns]
    n_features = len(feature_columns)
    print(f'   Stocks: {n_stocks}  Features per stock: {n_features}')
    print(f'   Regime obs: OFF  |  Regime probs: OFF  |  Vol features: OFF')
    print(f'   Regime-conditioned lambda: ON  ({V6_RISK_PENALTIES})')
    print(f'   Long-momentum features: ON  (mom_120d, mom_252d)')

    # KEY CONFIGURATION: regime obs OFF, regime-lambda ON, long-mom ON
    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=V6_CASH_FLOORS,
        risk_penalties=V6_RISK_PENALTIES,
        include_regime=False,
        include_regime_probs=False,
        include_vol_features=False,
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
        net_arch=V6_NET_ARCH,
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
        verbose=0, seed=args.seed, device='cpu',
    )

    print(f"\n--- Training {args.total_steps:,} steps (~3 hours) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=args.total_steps)
    best_path = os.path.join(saved_dir, f'ppo_{TAG}')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', TAG)
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_{TAG}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} at step {val_cb.best_step:,}")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({'history': val_cb.history,
                   'best_step': val_cb.best_step,
                   'best_val_sharpe': val_cb.best_val_sharpe}, f, indent=2)

    # Quick eval at the best-by-val-Sharpe checkpoint (DEFENSIVE rule applied separately)
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
        print(f"  {desc}: Return={m['cumulative_return']:>+7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>+7.4f}  MDD={m['max_drawdown']:>+7.2f}%")

    save = {
        'method': TAG,
        'universe': 'SP100 + long-momentum + regime-lambda kept + regime obs REMOVED',
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
        'config': {
            'include_regime': False,
            'include_regime_probs': False,
            'include_vol_features': False,
            'risk_penalties': V6_RISK_PENALTIES,
            'cash_floors_train': V6_CASH_FLOORS,
            'long_mom_features': True,
            'total_steps': args.total_steps,
        },
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")
    print(f"\n   Next step: run eval_sp100_mom_noregimeobs_defensive.py")
    print(f"   for the DEFENSIVE rule + cash-floor sweep.")


if __name__ == '__main__':
    main()
