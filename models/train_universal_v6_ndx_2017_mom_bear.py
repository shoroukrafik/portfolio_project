"""
NDX 2017 + LONG-MOMENTUM + BEAR-DEFENSE (CVaR penalty only — asymmetric).

Built on top of NDX 2017 + long-momentum, adds the SINGLE bear-defense knob
that's asymmetric by construction:

  use_cvar_penalty = True
  cvar_lambda      = 0.5
  cvar_window      = 60
  cvar_alpha       = 0.05

CVaR penalty by definition only fires when realized tail loss exceeds the
5% VaR threshold. On bull-period days when no tail loss occurs, the term
is zero — theoretically zero bull trade-off.

Same training setup as NDX 2017 + long-momentum (500k steps).

Output: ppo_universal_v6_ndx_2017_mom_bear.zip
"""

import os
import sys
import json
import argparse
import numpy as np
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

# Same extended feature set as long-momentum variant
V6_FEATURE_COLUMNS_MOM = V6_FEATURE_COLUMNS + ['mom_120d_norm', 'mom_252d_norm']

# CVaR penalty config (defender's secret sauce, asymmetric tail-risk reward)
CVAR_CONFIG = dict(
    use_cvar_penalty=True,
    cvar_lambda=0.5,
    cvar_window=60,
    cvar_alpha=0.05,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_steps', type=int, default=500_000)
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    name_tag = '_2017_mom_bear'
    if args.seed != 42:
        name_tag += f'_s{args.seed}'
    results_dir = os.path.join(project_dir, 'results', f'universal_v6_ndx{name_tag}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 80)
    print('   NDX 2017 + LONG-MOMENTUM + BEAR-DEFENSE (CVaR penalty)')
    print('   Hypothesis: asymmetric tail-risk reward improves bear without')
    print('               touching bull (CVaR only fires on tail losses)')
    print('=' * 80)
    print(f"   CVaR settings: {CVAR_CONFIG}")

    def load(p):
        df = pd.read_csv(os.path.join(data_dir, p))
        df['date'] = pd.to_datetime(df['date'])
        return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])

    train_df     = load('train_data_ndx_2017.csv')
    val_df       = load('val_data_ndx_2017.csv')
    test_bear_df = load('test_bear_data_ndx_2017.csv')
    test_2023_df = load('test_2023_data_ndx_2017.csv')
    test_bull_df = load('test_bull_data_ndx_2017.csv')

    required = ['mom_60d_norm', 'mom_120d_norm', 'mom_252d_norm']
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}.")

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS_MOM if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"\n   Stocks: {n_stocks}  Features per stock: {n_features}")

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=V6_CASH_FLOORS,
        risk_penalties=V6_RISK_PENALTIES,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_markowitz_reward=False,
        use_hyperheuristic=False,
        **CVAR_CONFIG,
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

    print(f"\n--- Training {args.total_steps:,} steps ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=args.total_steps)
    best_path = os.path.join(saved_dir, f'ppo_universal_v6_ndx{name_tag}')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', f'universal_v6_ndx{name_tag}')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_universal_v6_ndx{name_tag}_final')
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
        ('train',     train_env,     'Training (2017-2020)'),
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
        'method': f'universal_v6_ndx{name_tag}',
        'universe': f'NDX 2017 + LONG-MOM + CVaR bear defense ({n_stocks} stocks)',
        'experimental_changes': {
            'features':         'V6 + mom_120d_norm + mom_252d_norm',
            'cvar':              CVAR_CONFIG,
            'total_steps':       args.total_steps,
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step':       val_cb.best_step,
        'periods':         results,
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")

    # Comparison vs NDX 2017 + long-momentum (no bear defense)
    print()
    print('=' * 80)
    print('   COMPARISON vs NDX 2017 + long-momentum (no bear defense)')
    print('=' * 80)
    try:
        with open('results/universal_v6_ndx_2017_mom/final_results.json') as f:
            base = json.load(f)
        for p in ['val','test_bear','test_2023','test_bull']:
            bm = base['periods'][p]
            mm = results[p]
            d_ret = mm['cumulative_return'] - bm['cumulative_return']
            print(f"  {p:<11} | mom-only {bm['cumulative_return']:>+7.2f} | "
                  f"+CVaR {mm['cumulative_return']:>+7.2f} | Δ {d_ret:>+6.2f} pp")
    except FileNotFoundError:
        print("  (baseline mom-only not found)")
    print('=' * 80)


if __name__ == '__main__':
    main()
