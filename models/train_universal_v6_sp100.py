"""
Universal v6 — SP100. The "clean hybrid" design:

  • v2's features (19 base + mom_60d_norm = 20 per stock)
  • v1's regime-conditioned λ {bull:0.1, bear:0.5, sideways:0.3}
  • NO cash floor (all zero — first principle from v5/no_floor ablation)
  • DSR + raw_return − λ_regime × downside − cost  reward
  • Same PPO/attention architecture, 1M steps, val-checkpoint
  • Same HMM regime detection

Motivation: when the cash floor is removed (which we found is the right
move for SP100), the soft penalty has to do the defensive work. v2 used
a uniform λ=0.1 which is fine WITH a floor but under-defends WITHOUT
one. The SP100 no_floor ablation (which kept regime-conditioned λ) is
empirically the best SP100 single-checkpoint design. v6 reproduces that
ablation's recipe but WITH mom_60d_norm added on top.

Outputs:
  models/saved/ppo_universal_v6_sp100.zip
  results/universal_v6_sp100/final_results.json
"""

import os
import sys
import json
import argparse
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


# v6 settings — applied identically to DJI and SP100
V6_CASH_FLOORS    = {0: 0.0, 1: 0.0, 2: 0.0}    # no floor
V6_RISK_PENALTIES = {0: 0.1, 1: 0.5, 2: 0.3}     # regime-conditioned λ (v1 form)
V6_NET_ARCH       = dict(pi=[128, 64], vf=[128, 64])

# v2-style feature set: 19 base + mom_60d_norm. Excludes v3 lags / trend cats.
V6_FEATURE_COLUMNS = [
    'close','open','high','low','volume',
    'rsi','macd','macd_signal','macd_diff',
    'bb_upper','bb_lower','bb_width',
    'atr','sma_20','sma_50','adx','cci',
    'daily_return','volatility_20d',
    'mom_60d_norm',     # v2 feature
]

# v3 columns to DROP from df so env auto-detects only v6's 20 features
V3_COLS_TO_DROP = [
    'r_lag_1_norm','r_lag_5_norm','r_lag_10_norm',
    'r_lag_20_norm','r_lag_40_norm','r_lag_60_norm',
    'is_uptrend','is_sideways','is_downtrend',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_regime', action='store_true',
                        help='Ablation H2: remove ALL regime mechanisms — '
                             'no regime_probs/vol in obs, no regime-conditioned '
                             'risk penalties (uniform 0.3), include_regime=False')
    parser.add_argument('--seed', type=int, default=42,
                        help='PPO seed. Default 42 = production. '
                             '!=42 → output files tagged _s{seed} (no overwrite).')
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    name_tag = '_noregime' if args.no_regime else ''
    if args.seed != 42:
        name_tag += f'_s{args.seed}'
    results_dir = os.path.join(project_dir, 'results', f'universal_v6_sp100{name_tag}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    total_timesteps = 1_000_000

    print("=" * 72)
    if args.no_regime:
        print("   UNIVERSAL v6 — S&P 100  *** H2 ABLATION: NO REGIME ARCHITECTURE ***")
        print("   regime obs OFF, vol obs OFF, regime risk-λ OFF (uniform 0.3)")
    else:
        print("   UNIVERSAL v6 — S&P 100 (NO floor + regime-λ + v2 features)")
    print("=" * 72)
    if not args.no_regime:
        print(f"\n   Cash floor:    {V6_CASH_FLOORS}  (ALL ZERO)")
        print(f"   λ regime:      {V6_RISK_PENALTIES}  (regime-conditioned)")
        print(f"   Features:      v2 set (19 base + mom_60d_norm = 20 per stock)")
        print(f"   Reward:        DSR + raw_return − λ_regime × downside − cost")
        print(f"   Architecture:  PPO + attention + {V6_NET_ARCH['pi']} head")

    print("\n--- Loading SP100 data ---\n")
    def load(p):
        df = pd.read_csv(os.path.join(data_dir, p))
        df['date'] = pd.to_datetime(df['date'])
        return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])
    train_df     = load('train_data_sp100.csv')
    val_df       = load('val_data_sp100.csv')
    test_bear_df = load('test_bear_data_sp100.csv')
    test_2023_df = load('test_2023_data_sp100.csv')
    test_bull_df = load('test_bull_data_sp100.csv')

    required = ['mom_60d_norm', 'vol_20d_norm', 'regime_bull_prob']
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. "
                           "Run data/add_universal_features.py first.")

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [f for f in V6_FEATURE_COLUMNS if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"   Stocks: {n_stocks}  Features per stock: {n_features}")

    # ---- Envs — regime-conditioned λ via risk_penalties (NO risk_lambda_fixed) ----
    if args.no_regime:
        env_kwargs = dict(
            initial_balance=1_000_000,
            cash_floors={0: 0.0, 1: 0.0, 2: 0.0},
            risk_penalties={0: 0.3, 1: 0.3, 2: 0.3},     # uniform — no regime conditioning
            include_regime=False,
            include_regime_probs=False,
            include_vol_features=False,
            use_markowitz_reward=False,
            use_hyperheuristic=False,
        )
    else:
        env_kwargs = dict(
            initial_balance=1_000_000,
            cash_floors=V6_CASH_FLOORS,
            risk_penalties=V6_RISK_PENALTIES,   # regime-conditioned λ
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

    # ---- PPO ----
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

    print(f"\n--- Training {total_timesteps:,} steps (~75 min on CPU) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, f'ppo_universal_v6_sp100{name_tag}')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints', f'universal_v6_sp100{name_tag}')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_universal_v6_sp100{name_tag}_final')
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
        'method': 'universal_v6_sp100',
        'universe': 'S&P 100 historical subset (90 stocks)',
        'v6_settings': {
            'cash_floors': V6_CASH_FLOORS,
            'risk_penalties': V6_RISK_PENALTIES,
            'include_regime_probs': True,
            'include_vol_features': True,
            'net_arch': V6_NET_ARCH,
            'feature_columns': feature_columns,
        },
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {results_dir}/final_results.json")

    # Skip the v1/v2/v5/no_floor comparison block when running the H2 ablation
    # — those baselines are not the relevant comparator for the no-regime test.
    if args.no_regime:
        print(f"\n✅ H2 (no-regime) ablation complete → {results_dir}/")
        return

    # Compare to v1, v2, v5, no_floor, benchmarks
    v1 = json.load(open('results/extended_sp100/final_results.json'))
    v2 = json.load(open('results/universal_v2_sp100/final_results.json'))
    v5 = json.load(open('results/universal_v5_sp100/final_results.json'))
    nf = json.load(open('results/ablation_sp100_no_floor/final_results.json'))

    bench = {'val':(28.79,29.33,33.11), 'test_bear':(-19.95,-22.99,-9.48),
             'test_2023':(24.73,31.39,21.17), 'test_bull':(24.54,30.90,18.16)}

    print("\n" + "=" * 92)
    print("   v6 SP100 vs v1, v2, v5, no_floor — and all 3 benchmarks")
    print("=" * 92)
    print(f"  {'Period':<12} {'System':<32} {'Ret%':>8} {'Sharpe':>8} {'MDD%':>8}")
    for p in ['val','test_bear','test_2023','test_bull']:
        sp, oe, ew = bench[p]
        print(f"\n  {p:<12} {'S&P 500 BH':<32} {sp:>8.2f}")
        print(f"  {p:<12} {'S&P 100 BH (OEX)':<32} {oe:>8.2f}")
        print(f"  {p:<12} {'EW SP100':<32} {ew:>8.2f}")
        v1_m = v1.get(f'agent_{p}', {})
        v2_m = v2.get('periods', {}).get(p, {})
        v5_m = v5.get('periods', {}).get(p, {})
        nf_m = nf.get('periods', {}).get(p, {})
        v6_m = results[p]
        for label, m in [('SP100 v1 (FULL)', v1_m),
                          ('SP100 v2', v2_m),
                          ('SP100 no_floor ablation', nf_m),
                          ('SP100 v5', v5_m),
                          ('SP100 v6 ⭐', v6_m)]:
            if not m: continue
            print(f"  {p:<12} {label:<32} "
                  f"{m.get('cumulative_return', 0):>8.2f} "
                  f"{m.get('sharpe_ratio', 0):>8.4f} "
                  f"{m.get('max_drawdown', 0):>8.2f}")

    if detailed.rewards_history:
        steps = [x[0] for x in detailed.rewards_history]
        rewards = [x[1] for x in detailed.rewards_history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rewards, color='blue', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title(f'Universal v6 SP100 ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 72)
    print("   UNIVERSAL v6 SP100 COMPLETE ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
