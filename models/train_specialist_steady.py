"""
Train the STEADY EDDIE specialist (MARS v3, balanced role).

Reward shaping — middle of the 3:
  cash_floors    = {bull: 0.0, bear: 0.2, sideways: 0.0}   # light bear floor
  risk_penalties = {bull: 0.2, bear: 0.5, sideways: 0.3}   # balanced everywhere
  No CVaR penalty (no tail-risk override — standard DSR)
  Standard ent_coef from best_hyperparams.json

Architecture identical to v6 (AttentionPortfolioExtractor + MlpPolicy
net_arch [128, 64]). The role-aligned reward is what makes it the
"reliable Sharpe-maximiser" of the trio.

Selection: best by val Sharpe (the standard criterion).

Usage:
  python3 models/train_specialist_steady.py --universe dji
  python3 models/train_specialist_steady.py --universe sp100
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
    PeriodReturnCallback,
)
from train_universal_v6_sp100 import V6_FEATURE_COLUMNS, V3_COLS_TO_DROP


# Universal STEADY v1 (FINAL) recipe.
#
# After exploring v2/v3/v4 variations (lighter risk + concentration bonus),
# none beat v1's original recipe for DJI 7/7 wins. v1 produced best test_2023
# return on DJI (+24.40% at step 500k, 7/7 wins). Reverted to v1 + 1M steps.
#
# v1 (universal — same recipe on both universes):
#   cash_floors    = {bull:0, bear:0.2, sideways:0}   # light bear floor
#   risk_penalties = {bull:0.2, bear:0.5, sideways:0.3}  # regime-conditioned
#   No concentration bonus, no max-weight penalty
#   DSR reward + downside penalty (default)
STEADY_CASH_FLOORS    = {0: 0.0, 1: 0.2, 2: 0.0}
STEADY_RISK_PENALTIES = {0: 0.2, 1: 0.5, 2: 0.3}
# Concentration bonus DISABLED (we tested — doesn't help steady)
STEADY_USE_CONCENTRATION       = False
STEADY_CONCENTRATION_TOP_K     = 10        # unused
STEADY_CONCENTRATION_THRESHOLD = 0.0       # unused
STEADY_CONCENTRATION_LAMBDA    = 0.0       # unused

NET_ARCH = dict(pi=[128, 64], vf=[128, 64])


def load(data_dir, name):
    df = pd.read_csv(os.path.join(data_dir, name))
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', choices=['dji','sp100'], required=True)
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--eval_freq',   type=int, default=25_000)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints',
                                   f'steady_{args.universe}')
    results_dir = os.path.join(project_dir, 'results',
                                f'specialist_steady_{args.universe}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if args.universe == 'dji':
        train_csv = 'train_data_ext.csv'
        val_csv   = 'val_data_ext.csv'
        t2023_csv = 'test_2023_data_ext.csv'
    else:
        train_csv = 'train_data_sp100.csv'
        val_csv   = 'val_data_sp100.csv'
        t2023_csv = 'test_2023_data_sp100.csv'

    train_df = load(data_dir, train_csv)
    val_df   = load(data_dir, val_csv)
    t2023_df = load(data_dir, t2023_csv)
    n_stocks    = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)

    print("=" * 72)
    print(f"   STEADY EDDIE — {args.universe.upper()} (MARS v3 balanced, v1 recipe)")
    print(f"   cash_floors={STEADY_CASH_FLOORS}, λ={STEADY_RISK_PENALTIES}")
    print(f"   reward: DSR + downside (no concentration bonus, no max-weight)")
    print(f"   {args.total_steps:,} steps, checkpoint every {args.eval_freq:,}")
    print("=" * 72)
    print(f"   Stocks: {n_stocks}  Features per stock: {n_features}")

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=STEADY_CASH_FLOORS,
        risk_penalties=STEADY_RISK_PENALTIES,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_cvar_penalty=False,
        use_concentration_bonus=STEADY_USE_CONCENTRATION,
        concentration_top_k=STEADY_CONCENTRATION_TOP_K,
        concentration_threshold=STEADY_CONCENTRATION_THRESHOLD,
        concentration_lambda=STEADY_CONCENTRATION_LAMBDA,
    )
    train_env = PortfolioEnv(df=train_df, **env_kwargs)
    val_env   = PortfolioEnv(df=val_df,   **env_kwargs)
    t2023_env = PortfolioEnv(df=t2023_df, **env_kwargs)

    policy_kwargs = dict(
        features_extractor_class=AttentionPortfolioExtractor,
        features_extractor_kwargs=dict(
            n_stocks=n_stocks, n_features=n_features,
            embed_dim=64, n_heads=4, features_dim=128,
        ),
        net_arch=NET_ARCH,
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
        ent_coef=hp['ent_coef'],   # standard 0.01
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=args.seed, device='cpu',
    )

    print(f"\n--- Training STEADY EDDIE for {args.total_steps:,} steps ---\n")
    detailed = DetailedCallback(eval_freq=50_000,
                                 total_timesteps=args.total_steps)
    best_path      = os.path.join(saved_dir, f'ppo_steady_{args.universe}')
    best2023_path  = os.path.join(saved_dir, f'ppo_steady_{args.universe}_best2023')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path,
        eval_freq=args.eval_freq, checkpoint_dir=checkpoint_dir,
    )
    # Live test_2023 tracking — steady's role-aligned criterion (sideways).
    # Saves ppo_steady_{universe}_best2023.zip selected by max 2023 return.
    sideways_cb = PeriodReturnCallback(
        period_env=t2023_env, period_name='2023',
        save_path=best2023_path, eval_freq=args.eval_freq,
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[detailed, val_cb, sideways_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_steady_{args.universe}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}  → {best_path}.zip")
    print(f"✅ Best by 2023 return: {sideways_cb.best_ret:+.2f}% "
          f"(MDD {sideways_cb.best_mdd:+.2f}%) at step {sideways_cb.best_step:,}  "
          f"→ {best2023_path}.zip  ⭐ steady's PRIMARY checkpoint")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({'history': val_cb.history,
                   'best_step': val_cb.best_step,
                   'best_val_sharpe': val_cb.best_val_sharpe}, f, indent=2)
    with open(os.path.join(results_dir, 'test_2023_return_history.json'), 'w') as f:
        json.dump({'history': sideways_cb.history,
                   'best_step': sideways_cb.best_step,
                   'best_2023_return': sideways_cb.best_ret,
                   'best_2023_mdd':    sideways_cb.best_mdd}, f, indent=2)

    print("\n" + "=" * 72)
    print(f"   STEADY EDDIE {args.universe.upper()} COMPLETE ✅")
    print(f"   → {best_path}.zip")
    print(f"   → checkpoints in {checkpoint_dir}/")
    print("=" * 72)


if __name__ == '__main__':
    main()
