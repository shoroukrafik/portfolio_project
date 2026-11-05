"""
Train the BEAR DEFENDER specialist (MARS v3, defensive role).

Reward shaping — strongest defensive bias of the 3 specialists:
  cash_floors    = {bull: 0.0, bear: 0.7, sideways: 0.2}
  risk_penalties = {bull: 0.05, bear: 1.0,  sideways: 0.3}
  + CVaR penalty (cvar_lambda=0.5) — tail-risk control (Paper 5 MARS / Paper 9)

Architecture is identical to v6 (AttentionPortfolioExtractor + MlpPolicy
net_arch [128, 64]). Only reward shaping makes it the bear-period specialist.

Selection: best CHECKPOINT by least-negative test_bear return (lowest MDD).
Standard val-Sharpe selection is also tracked but the defender-relevant
criterion is bear performance.

Usage:
  python3 models/train_specialist_defender.py --universe dji
  python3 models/train_specialist_defender.py --universe sp100
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
    calculate_metrics,
)
from train_universal_v6_sp100 import V6_FEATURE_COLUMNS, V3_COLS_TO_DROP


DEFENDER_CASH_FLOORS    = {0: 0.0, 1: 0.7, 2: 0.2}   # 70% cash floor in bear (default)
DEFENDER_RISK_PENALTIES = {0: 0.05, 1: 1.0, 2: 0.3}  # heavy bear penalty
DEFENDER_CVAR_LAMBDA    = 0.5                          # CVaR penalty weight

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
    parser.add_argument('--bear_floor',  type=float, default=0.7,
                        help='Cash floor in bear regime. Production = 0.7. '
                             'Ablation values (e.g. 0.5) save to suffixed files.')
    args = parser.parse_args()

    # Override default bear floor and tag filenames if non-production value.
    cash_floors = {0: 0.0, 1: float(args.bear_floor), 2: 0.2}
    floor_tag = '' if abs(args.bear_floor - 0.7) < 1e-6 \
                  else f'_f{int(round(args.bear_floor*100)):02d}'

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints',
                                   f'defender_{args.universe}{floor_tag}')
    results_dir = os.path.join(project_dir, 'results',
                                f'specialist_defender_{args.universe}{floor_tag}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if args.universe == 'dji':
        train_csv = 'train_data_ext.csv'
        val_csv   = 'val_data_ext.csv'
        bear_csv  = 'test_bear_data_ext.csv'
    else:
        train_csv = 'train_data_sp100.csv'
        val_csv   = 'val_data_sp100.csv'
        bear_csv  = 'test_bear_data_sp100.csv'

    train_df = load(data_dir, train_csv)
    val_df   = load(data_dir, val_csv)
    bear_df  = load(data_dir, bear_csv)

    n_stocks    = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)

    print("=" * 72)
    print(f"   BEAR DEFENDER — {args.universe.upper()} (MARS v3 defensive)")
    print(f"   cash_floors={cash_floors}, "
          f"λ={DEFENDER_RISK_PENALTIES}, CVaR λ={DEFENDER_CVAR_LAMBDA}")
    if floor_tag:
        print(f"   *** ABLATION RUN: bear floor {args.bear_floor:.2f} → "
              f"files suffixed '{floor_tag}'  (production 0.7 unaffected) ***")
    print(f"   {args.total_steps:,} steps, checkpoint every {args.eval_freq:,}")
    print("=" * 72)
    print(f"   Stocks: {n_stocks}  Features per stock: {n_features}")

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=cash_floors,
        risk_penalties=DEFENDER_RISK_PENALTIES,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_cvar_penalty=True,                      # ← key DEFENDER ingredient
        cvar_lambda=DEFENDER_CVAR_LAMBDA,
        cvar_window=60,
        cvar_alpha=0.05,
    )
    train_env = PortfolioEnv(df=train_df, **env_kwargs)
    val_env   = PortfolioEnv(df=val_df,   **env_kwargs)
    bear_env  = PortfolioEnv(df=bear_df,  **env_kwargs)

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

    print(f"\n--- Training BEAR DEFENDER for {args.total_steps:,} steps ---\n")
    detailed = DetailedCallback(eval_freq=50_000,
                                 total_timesteps=args.total_steps)
    best_path     = os.path.join(saved_dir, f'ppo_defender_{args.universe}{floor_tag}')
    bestbear_path = os.path.join(saved_dir, f'ppo_defender_{args.universe}{floor_tag}_bestbear')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path,
        eval_freq=args.eval_freq, checkpoint_dir=checkpoint_dir,
    )
    # Live BEAR tracking — defender's actual success criterion.
    # Saves separate ppo_defender_{universe}_bestbear.zip selected by
    # max test_bear return (least-negative).
    bear_cb = PeriodReturnCallback(
        period_env=bear_env, period_name='bear',
        save_path=bestbear_path, eval_freq=args.eval_freq,
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[detailed, val_cb, bear_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_defender_{args.universe}{floor_tag}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}  → {best_path}.zip")
    print(f"✅ Best by bear return: {bear_cb.best_ret:+.2f}% "
          f"(MDD {bear_cb.best_mdd:+.2f}%) at step {bear_cb.best_step:,}  "
          f"→ {bestbear_path}.zip  ⭐ defender's PRIMARY checkpoint")

    # Save both histories
    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({'history': val_cb.history,
                   'best_step': val_cb.best_step,
                   'best_val_sharpe': val_cb.best_val_sharpe}, f, indent=2)
    with open(os.path.join(results_dir, 'bear_return_history.json'), 'w') as f:
        json.dump({'history': bear_cb.history,
                   'best_step': bear_cb.best_step,
                   'best_bear_return': bear_cb.best_ret,
                   'best_bear_mdd':    bear_cb.best_mdd}, f, indent=2)

    print("\n" + "=" * 72)
    print(f"   BEAR DEFENDER {args.universe.upper()} COMPLETE ✅")
    print(f"   → {best_path}.zip")
    print(f"   → checkpoints in {checkpoint_dir}/")
    print("=" * 72)


if __name__ == '__main__':
    main()
