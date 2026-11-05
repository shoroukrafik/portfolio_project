"""
LR experiment C: DJI Universal v6 with LINEAR LR SCHEDULE (3e-4 → 5e-5).

Standard PPO best practice: start at higher LR for fast initial progress,
decay toward lower LR for stable convergence. Bridges the choice between
literature default (3e-4) and our current production (5e-5).

Identical to train_universal_v6_dji.py EXCEPT learning_rate is a linear
schedule from 3e-4 (start) to 5e-5 (end of training).

Output: ppo_universal_v6_dji_lrsched.zip
        results/universal_v6_dji_lrsched/
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


LR_START = 3e-4   # Stable-Baselines3 PPO default
LR_END   = 5e-5   # Our current production LR (matches at end of training)


def linear_lr_schedule(lr_start: float, lr_end: float):
    """
    Return an SB3-compatible learning rate schedule function.
    SB3 passes progress_remaining in [1.0, 0.0]: 1.0 at start, 0.0 at end.
    """
    def fn(progress_remaining: float) -> float:
        # Linear: at start (pr=1.0) returns lr_start, at end (pr=0.0) returns lr_end
        return lr_end + (lr_start - lr_end) * progress_remaining
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    name_tag = '_lrsched'
    if args.seed != 42:
        name_tag += f'_s{args.seed}'
    results_dir = os.path.join(project_dir, 'results', f'universal_v6_dji{name_tag}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 80)
    print('   DJI Universal v6 — LR EXPERIMENT C: LINEAR SCHEDULE 3e-4 → 5e-5')
    print('   Standard PPO best practice — fast initial progress + stable convergence')
    print('=' * 80)

    def load(p):
        df = pd.read_csv(os.path.join(data_dir, p))
        df['date'] = pd.to_datetime(df['date'])
        return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])

    train_df     = load('train_data_ext.csv')
    val_df       = load('val_data_ext.csv')
    test_bear_df = load('test_bear_data_ext.csv')
    test_2023_df = load('test_2023_data_ext.csv')
    test_bull_df = load('test_bull_data_ext.csv')

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"\n   Stocks: {n_stocks}  Features per stock: {n_features}")
    print(f"   LR schedule: linear  {LR_START} → {LR_END}  over {args.total_steps:,} steps")

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=V6_CASH_FLOORS,
        risk_penalties=V6_RISK_PENALTIES,
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
        net_arch=V6_NET_ARCH,
    )
    hp_path = os.path.join(project_dir, 'results', 'best_hyperparams.json')
    with open(hp_path) as f:
        hp = json.load(f)

    lr_schedule = linear_lr_schedule(LR_START, LR_END)

    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,            # ← THE EXPERIMENT: callable schedule
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        ent_coef=hp['ent_coef'],
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=args.seed, device='cpu',
    )

    print(f"\n--- Training {args.total_steps:,} steps with LR schedule {LR_START} → {LR_END} ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=args.total_steps)
    best_path = os.path.join(saved_dir, f'ppo_universal_v6_dji{name_tag}')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', f'universal_v6_dji{name_tag}')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_universal_v6_dji{name_tag}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} at step {val_cb.best_step:,}")

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
        print(f"  {desc}: Return={m['cumulative_return']:>+7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>+7.4f}  MDD={m['max_drawdown']:>+7.2f}%")

    save = {
        'method': f'universal_v6_dji{name_tag}',
        'universe': 'DJI 30 (LR experiment C: linear schedule 3e-4 → 5e-5)',
        'experimental_changes': {
            'learning_rate_schedule':  f'linear {LR_START} → {LR_END} over {args.total_steps} steps',
            'rationale':                'Standard PPO best practice — bridges SB3 default (3e-4) with our production (5e-5)',
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step':       val_cb.best_step,
        'periods':         results,
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")


if __name__ == '__main__':
    main()
