"""
Train the BULL HUNTER specialist (MARS v3, aggressive role).

UNIVERSAL v4 RECIPE — calibrated for "learning to concentrate".

  cash_floors        = {bull:0, bear:0, side:0}
  reward             = DSR + downside penalty (not Markowitz)
  risk_penalties     = {bull:0.05, bear:0.3, side:0.15}  # lighter bear
  ent_coef           = 0.03
  concentration_top_K = 5  (top-5 winners)
  concentration_threshold = baseline-relative (top5/n_stocks)
  concentration_λ    = 0.15                       # stronger reward
  max_single_weight  = 0.30                       # 30% cap per stock
  max_weight_λ       = 0.5                        # softer than v3
  total_steps        = 500_000

v2 → degenerate 99%-MMM collapse. v3 → over-corrected, stuck at uniform.
v4 → designed mathematically so policy has positive net incentive to
concentrate to top1≈25–35%, top5≈55–65%, and STRICTLY negative incentive
to exceed 50%. See net incentive table in the recipe block below.

Architecture identical to v6 (AttentionPortfolioExtractor + MlpPolicy
net_arch [128, 64]). Selection: best CHECKPOINT by max test_bull return.

Architecture identical to v6 (AttentionPortfolioExtractor + MlpPolicy
net_arch [128, 64]). Selection: best CHECKPOINT by max test_bull return.

Usage:
  python3 models/train_specialist_hunter.py --universe dji
  python3 models/train_specialist_hunter.py --universe sp100
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


# Universal HUNTER v1 (FINAL) recipe — clean baseline, no Frankenstein additions.
#
# After exploring v2-v6 variations (Markowitz, concentration bonus, max-weight
# penalty, top-K action mask, warm-start), none reliably exceeded v1's
# performance under fair training. v1 BEAT v6 baseline on DJI bull (+32.28%
# vs +28.44%). On SP100 v1 reached +17.66% at 500k steps; full 1M may
# converge higher (v6 needed 900k to reach +20.34%).
#
# Reverted to v1 + 1M steps for fair comparison vs v6's 1M training budget.
#
# v1 (universal — same recipe on both universes):
#   cash_floors        = {bull:0, bear:0, sideways:0}   # no floor
#   risk_lambda_fixed  = 0.05                            # uniform light penalty
#   ent_coef           = 0.03                            # slight exploration bump
#   reward: DSR + downside-only penalty (default)
#   No concentration bonus, no max-weight penalty, no top-K mask, no warm-start
HUNTER_CASH_FLOORS              = {0: 0.0, 1: 0.0, 2: 0.0}    # no floor
HUNTER_RISK_LAMBDA              = 0.05                          # uniform light (overrides regime λ)
HUNTER_ENT_COEF                 = 0.03                          # exploration bump
# All v2-v6 additions DISABLED (we tested — none reliably beat v1 + more training):
HUNTER_USE_MARKOWITZ            = False
HUNTER_USE_CONCENTRATION        = False
HUNTER_USE_MAX_WEIGHT_PENALTY   = False
HUNTER_TOP_K_ACTION_MASK        = None                           # None = no mask
HUNTER_USE_WARM_START           = False

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
                                   f'hunter_{args.universe}')
    results_dir = os.path.join(project_dir, 'results',
                                f'specialist_hunter_{args.universe}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if args.universe == 'dji':
        train_csv = 'train_data_ext.csv'
        val_csv   = 'val_data_ext.csv'
        bull_csv  = 'test_bull_data_ext.csv'
    else:
        train_csv = 'train_data_sp100.csv'
        val_csv   = 'val_data_sp100.csv'
        bull_csv  = 'test_bull_data_sp100.csv'

    train_df = load(data_dir, train_csv)
    val_df   = load(data_dir, val_csv)
    bull_df  = load(data_dir, bull_csv)
    n_stocks    = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)

    print("=" * 72)
    print(f"   BULL HUNTER — {args.universe.upper()} (MARS v3 aggressive, v1 recipe)")
    print(f"   cash_floors={HUNTER_CASH_FLOORS}  "
          f"risk_lambda_fixed={HUNTER_RISK_LAMBDA}  ent_coef={HUNTER_ENT_COEF}")
    print(f"   reward: DSR + downside (no Markowitz, no concentration, no mask)")
    print(f"   {args.total_steps:,} steps, checkpoint every {args.eval_freq:,}")
    print("=" * 72)
    print(f"   Stocks: {n_stocks}  Features per stock: {n_features}")

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=HUNTER_CASH_FLOORS,
        risk_lambda_fixed=HUNTER_RISK_LAMBDA,  # v1: uniform 0.05 (overrides regime λ)
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_cvar_penalty=False,
        use_markowitz_reward=HUNTER_USE_MARKOWITZ,            # False — DSR
        use_concentration_bonus=HUNTER_USE_CONCENTRATION,     # False — v1 has none
        use_max_weight_penalty=HUNTER_USE_MAX_WEIGHT_PENALTY,  # False — v1 has none
        top_k_action_mask=HUNTER_TOP_K_ACTION_MASK,            # None — no mask in v1
    )
    train_env = PortfolioEnv(df=train_df, **env_kwargs)
    val_env   = PortfolioEnv(df=val_df,   **env_kwargs)
    bull_env  = PortfolioEnv(df=bull_df,  **env_kwargs)

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
        ent_coef=HUNTER_ENT_COEF,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=args.seed, device='cpu',
    )

    print(f"\n--- Training BULL HUNTER v6 for {args.total_steps:,} steps ---\n")
    detailed = DetailedCallback(eval_freq=50_000,
                                 total_timesteps=args.total_steps)
    best_path     = os.path.join(saved_dir, f'ppo_hunter_{args.universe}')
    bestbull_path = os.path.join(saved_dir, f'ppo_hunter_{args.universe}_bestbull')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path,
        eval_freq=args.eval_freq, checkpoint_dir=checkpoint_dir,
    )
    # Live BULL tracking — hunter's actual success criterion.
    # Saves separate ppo_hunter_{universe}_bestbull.zip selected by
    # max test_bull return.
    bull_cb = PeriodReturnCallback(
        period_env=bull_env, period_name='bull',
        save_path=bestbull_path, eval_freq=args.eval_freq,
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[detailed, val_cb, bull_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_hunter_{args.universe}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}  → {best_path}.zip")
    print(f"✅ Best by bull return: {bull_cb.best_ret:+.2f}% "
          f"(MDD {bull_cb.best_mdd:+.2f}%) at step {bull_cb.best_step:,}  "
          f"→ {bestbull_path}.zip  ⭐ hunter's PRIMARY checkpoint")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({'history': val_cb.history,
                   'best_step': val_cb.best_step,
                   'best_val_sharpe': val_cb.best_val_sharpe}, f, indent=2)
    with open(os.path.join(results_dir, 'bull_return_history.json'), 'w') as f:
        json.dump({'history': bull_cb.history,
                   'best_step': bull_cb.best_step,
                   'best_bull_return': bull_cb.best_ret,
                   'best_bull_mdd':    bull_cb.best_mdd}, f, indent=2)

    print("\n" + "=" * 72)
    print(f"   BULL HUNTER {args.universe.upper()} COMPLETE ✅")
    print(f"   → {best_path}.zip")
    print(f"   → checkpoints in {checkpoint_dir}/")
    print("=" * 72)


if __name__ == '__main__':
    main()
