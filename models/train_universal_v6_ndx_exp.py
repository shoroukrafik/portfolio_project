"""
NDX EXPERIMENTAL VARIANTS — out of universal-recipe rules (sandbox only).

The production NDX v6 baseline (ppo_universal_v6_ndx.zip) stays untouched.
This file produces SEPARATE experimental artifacts to address the bull-capture
limit observed on NDX 2024 (~+8% captured of the +28% ^NDX benchmark).

Variants (one knob changed at a time vs production v6 for clean attribution):

  exp1 — LOWER BULL PENALTY ONLY
    risk_penalties: {bull: 0.01, bear: 0.5, sideways: 0.3}  (bull 0.1 → 0.01)
    Everything else unchanged. Tests: does loosening just bull risk help capture
    a concentrated bull rally while preserving bear defense?

  exp2 — CONCENTRATION BONUS ONLY
    use_concentration_bonus=True, top_k=5, threshold=0.05, lambda=0.05  (LIGHT)
    Risk penalties unchanged. NO max_weight penalty (we learned from hunter v2-v4
    that combining concentration + max-weight penalty causes net-zero / collapse).
    Tests: does directly rewarding concentration on top-5 stocks help?

  exp3 — PURE LOG-RETURN REWARD (abandon DSR entirely)
    use_markowitz_reward=True, markowitz_beta=0.0
    REPLACES DSR + downside penalty with raw r_t − cost. Tests: is DSR the
    structural constraint on concentration? Bear defense WILL likely degrade —
    that collapse is itself informative about DSR's role.

  exp4 — LOWER PPO ENTROPY (less exploration → more committed policy)
    ent_coef: 0.01 → 0.001  (10× lower). Env unchanged from production v6.
    Tests: is PPO's entropy regularizer the constraint that prevents
    concentration? Lower entropy lets the policy commit harder to its
    top-stock picks at training time. Risk: PPO instability in 70D action
    space if entropy is too low (model can collapse to a single suboptimal
    stock if early gradients are noisy).

Each variant defaults to 500k steps (quick check). To extend, use --total_steps 1000000.

Output files:
  models/saved/ppo_universal_v6_ndx_exp1.zip  (or _exp2)
  models/saved/checkpoints/universal_v6_ndx_exp1/
  results/universal_v6_ndx_exp1/...
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
from train_universal_v6_sp100 import V6_NET_ARCH, V6_FEATURE_COLUMNS, V3_COLS_TO_DROP


# Production v6 baseline (unchanged knobs)
PROD_CASH_FLOORS    = {0: 0.0, 1: 0.0, 2: 0.0}
PROD_RISK_PENALTIES = {0: 0.1, 1: 0.5, 2: 0.3}


def variant_config(variant):
    """Returns (env_kwargs_delta, ppo_kwargs_delta) for each variant.
    Everything else stays at production v6 defaults."""
    env, ppo = {}, {}
    if variant == 'exp1':
        env = dict(
            risk_penalties={0: 0.01, 1: 0.5, 2: 0.3},   # bull 0.1 → 0.01
            use_concentration_bonus=False,
        )
    elif variant == 'exp2':
        env = dict(
            risk_penalties=PROD_RISK_PENALTIES,
            use_concentration_bonus=True,
            concentration_top_k=5,
            concentration_threshold=0.05,
            concentration_lambda=0.05,                  # LIGHT (half of 0.1, vs hunter v2's 0.3)
        )
    elif variant == 'exp3':
        env = dict(
            risk_penalties=PROD_RISK_PENALTIES,
            use_markowitz_reward=True,                  # REPLACES DSR + regime-conditioned downside
            markowitz_beta=0.0,                          # raw log-return reward
            use_concentration_bonus=False,
        )
    elif variant == 'exp4':
        # No env change — only PPO entropy coefficient
        env = dict(
            risk_penalties=PROD_RISK_PENALTIES,
            use_concentration_bonus=False,
        )
        ppo = dict(ent_coef=0.001)                       # 10× lower than production 0.01
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return env, ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['exp1','exp2','exp3','exp4'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_steps', type=int, default=500_000)
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    name_tag = f'_{args.variant}'
    if args.seed != 42:
        name_tag += f'_s{args.seed}'
    results_dir = os.path.join(project_dir, 'results', f'universal_v6_ndx{name_tag}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"   NDX EXPERIMENTAL ({args.variant}) — OUT of universal-recipe rules")
    print("=" * 80)
    env_delta, ppo_delta = variant_config(args.variant)
    print("   env_kwargs deltas:")
    for k, v in env_delta.items():
        print(f"     {k}: {v}")
    if ppo_delta:
        print("   ppo_kwargs deltas:")
        for k, v in ppo_delta.items():
            print(f"     {k}: {v}")
    print(f"   total_steps: {args.total_steps:,}")
    print(f"   Production NDX v6 baseline (ppo_universal_v6_ndx.zip): UNTOUCHED")

    def load(p):
        df = pd.read_csv(os.path.join(data_dir, p))
        df['date'] = pd.to_datetime(df['date'])
        return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])

    train_df     = load('train_data_ndx.csv')
    val_df       = load('val_data_ndx.csv')
    test_bear_df = load('test_bear_data_ndx.csv')
    test_2023_df = load('test_2023_data_ndx.csv')
    test_bull_df = load('test_bull_data_ndx.csv')

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"\n   Stocks: {n_stocks}  Features per stock: {n_features}")

    # Base v6 env_kwargs, then overlay variant deltas
    base_env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=PROD_CASH_FLOORS,
        risk_penalties=PROD_RISK_PENALTIES,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_markowitz_reward=False,
        use_hyperheuristic=False,
        use_concentration_bonus=False,
    )
    env_kwargs = {**base_env_kwargs, **env_delta}

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
    # Resolve PPO ent_coef: ppo_delta override > hp default
    eff_ent_coef = ppo_delta.get('ent_coef', hp['ent_coef'])
    if eff_ent_coef != hp['ent_coef']:
        print(f"\n   ⚙ PPO ent_coef override: {hp['ent_coef']} → {eff_ent_coef}")
    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        ent_coef=eff_ent_coef,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=args.seed, device='cpu',
    )

    print(f"\n--- Training {args.total_steps:,} steps (~45-60 min on CPU) ---\n")
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
    top1_avg = None
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
              f"Sharpe={m['sharpe_ratio']:>+7.4f}  "
              f"MDD={m['max_drawdown']:>+7.2f}%")

    save = {
        'method': f'universal_v6_ndx{name_tag}',
        'universe': 'NASDAQ-100 (EXPERIMENTAL — out of universal rules)',
        'variant': args.variant,
        'recipe_modifications': {'env_delta': env_delta, 'ppo_delta': ppo_delta},
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")

    print()
    print('='*80)
    print(f'   COMPARISON vs PRODUCTION NDX v6 baseline (+8.03 bull, val 2.07)')
    print('='*80)
    try:
        with open('results/universal_v6_ndx/final_results.json') as f:
            prod = json.load(f)
        for p in ['val','test_bear','test_2023','test_bull']:
            pm = prod['periods'][p]
            em = results[p]
            d_ret = em['cumulative_return'] - pm['cumulative_return']
            print(f"  {p:<11} | prod {pm['cumulative_return']:>+7.2f} | "
                  f"{args.variant} {em['cumulative_return']:>+7.2f} | "
                  f"Δret {d_ret:>+6.2f} pp")
    except FileNotFoundError:
        print("  (production NDX results not found — skip comparison)")
    print('='*80)


if __name__ == '__main__':
    main()
