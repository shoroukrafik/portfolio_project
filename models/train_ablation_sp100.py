# ============================================
# Ablation study on the S&P 100 FULL extended baseline
# ============================================
# Mirrors models/train_ablation_extended.py but operates on the 90-stock
# S&P 100 universe. Same architecture / hyperparameters / training budget
# as SP100 FULL v1 — only ONE component changed per run.
#
# Ablation choices (parallel to DJI extended ablations):
#   --ablation no_floor             cash floor disabled
#   --ablation no_dsr               reward uses raw daily return (no DSR)
#   --ablation no_attention         MLP feature extractor instead of attention
#   --ablation no_regime_lambda     replace regime-conditioned λ with a single
#                                   fixed λ (use --fixed-lambda to set value;
#                                   0.1 / 0.3 / 0.5 are the canonical sweep)
#
# Each run uses the SAME 2007-2020 SP100 training data and the SAME SP30 HMM
# regime labels as SP100 FULL v1.
#
# Outputs:
#   models/saved/ppo_ablation_sp100_<run_tag>.zip
#   models/saved/checkpoints/ablation_sp100_<run_tag>/step_*.zip
#   results/ablation_sp100_<run_tag>/final_results.json
#
# Phase 5 originals + extended baseline + SP100 v1 are NOT touched.
# ============================================

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_final_extended import AttentionPortfolioExtractor
from train_ablation import MLPPortfolioExtractor


ABLATION_CHOICES = ['no_floor', 'no_dsr', 'no_attention', 'no_regime_lambda']


# ============================================
# CALLBACKS — same as train_final_extended.py
# ============================================

class DetailedCallback(BaseCallback):
    def __init__(self, eval_freq=50_000, total_timesteps=1_000_000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.total_timesteps = total_timesteps
        self.rewards_history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                recent = [ep['r'] for ep in self.model.ep_info_buffer]
                mean_r = float(np.mean(recent))
                self.rewards_history.append((self.n_calls, mean_r))
                print(f"  Step {self.n_calls:>8,} / {self.total_timesteps:,} "
                      f"| Mean Reward: {mean_r:>8.4f} "
                      f"| Episodes seen: {len(recent)}")
        return True


class ValidationCheckpointCallback(BaseCallback):
    def __init__(self, val_env, save_path, eval_freq=25_000, verbose=1,
                 checkpoint_dir=None):
        super().__init__(verbose)
        self.val_env = val_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_val_sharpe = -np.inf
        self.best_step = 0
        self.history = []
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def _evaluate(self):
        obs, _ = self.val_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = self.val_env.step(action)
            done = term or trunc
        h = self.val_env.get_portfolio_history()
        v = h['value'].values
        if len(v) < 2:
            return 0.0
        dr = np.diff(v) / v[:-1]
        n = len(v)
        ann_ret = ((v[-1] / v[0]) ** (252 / n) - 1) * 100
        ann_vol = float(np.std(dr) * np.sqrt(252) * 100)
        return ann_ret / ann_vol if ann_vol > 0 else 0.0

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            s = self._evaluate()
            self.history.append((self.n_calls, s))
            if s > self.best_val_sharpe:
                self.best_val_sharpe = s
                self.best_step = self.n_calls
                self.model.save(self.save_path)
                print(f"  ★ Step {self.n_calls:>7,}: val Sharpe={s:>7.4f} "
                      f"(NEW BEST — saved)")
            else:
                print(f"    Step {self.n_calls:>7,}: val Sharpe={s:>7.4f} "
                      f"(best={self.best_val_sharpe:.4f} @ step "
                      f"{self.best_step:,})")
            if self.checkpoint_dir is not None:
                ck = os.path.join(self.checkpoint_dir,
                                  f'step_{self.n_calls:07d}')
                self.model.save(ck)
        return True


def calculate_metrics(history, initial_balance=1_000_000):
    values = history['value'].values
    final_value = values[-1]
    cum_ret = (final_value - initial_balance) / initial_balance * 100
    daily_ret = np.diff(values) / values[:-1]
    n = len(values)
    ann_ret = ((final_value/initial_balance)**(252/n) - 1) * 100
    ann_vol = np.std(daily_ret) * np.sqrt(252) * 100
    sharpe = ann_ret/ann_vol if ann_vol > 0 else 0
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    mdd = np.min(dd) * 100
    return {
        'cumulative_return': round(cum_ret, 2),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(mdd, 2),
        'annualized_volatility': round(ann_vol, 2),
        'final_value': round(final_value, 2),
    }


def evaluate_agent(model, env):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()


# ============================================
# MAIN
# ============================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Ablation study on SP100 FULL v1 (1M val-checkpointed)')
    p.add_argument('--ablation', required=True, choices=ABLATION_CHOICES)
    p.add_argument('--timesteps', type=int, default=1_000_000)
    p.add_argument('--fixed-lambda', type=float, default=0.3,
                   help='Fixed λ value when ablation=no_regime_lambda. '
                        'Use 0.1 / 0.3 / 0.5 to cover the canonical sweep.')
    return p.parse_args()


def main():
    args = parse_args()
    ablation = args.ablation
    total_timesteps = args.timesteps

    # Map ablation flag → env mode + which feature extractor
    env_ablation_mode = ablation if ablation in ('no_floor', 'no_dsr') else None
    use_attention = (ablation != 'no_attention')
    fixed_lambda = args.fixed_lambda if ablation == 'no_regime_lambda' else None
    if ablation == 'no_regime_lambda':
        run_tag = f'no_regime_lambda_{str(fixed_lambda).replace(".", "p")}'
    else:
        run_tag = ablation

    print("=" * 60)
    print(f"   SP100 FULL — ABLATION = {ablation}")
    print("=" * 60)
    print(f"\n   Env ablation mode  : {env_ablation_mode}")
    print(f"   Feature extractor  : "
          f"{'AttentionPortfolioExtractor' if use_attention else 'MLPPortfolioExtractor'}")
    if fixed_lambda is not None:
        print(f"   Fixed λ (all regimes): {fixed_lambda}  "
              f"(replaces bull=0.1/bear=0.5/sideways=0.3)")
    print(f"   Training budget    : {total_timesteps:,} steps "
          f"with val checkpointing")

    # ---- Paths ----
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', f'ablation_sp100_{run_tag}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---- Load SP100 data ----
    print("\n--- Loading SP100 data ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_sp100.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_sp100.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_sp100.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_sp100.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_sp100.csv'))
    for df in (train_df, val_df, test_bear_df, test_2023_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

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
          f"2023: {test_2023_df['date'].nunique()}  "
          f"Bull: {test_bull_df['date'].nunique()}")

    # ---- Envs ----
    print("\n--- Creating environments ---\n")
    env_kwargs = dict(initial_balance=1_000_000,
                      ablation_mode=env_ablation_mode,
                      risk_lambda_fixed=fixed_lambda)
    train_env     = PortfolioEnv(df=train_df,     **env_kwargs)
    val_env       = PortfolioEnv(df=val_df,       **env_kwargs)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kwargs)
    test_2023_env = PortfolioEnv(df=test_2023_df, **env_kwargs)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kwargs)

    # ---- Build PPO with right extractor ----
    if use_attention:
        policy_kwargs = dict(
            features_extractor_class=AttentionPortfolioExtractor,
            features_extractor_kwargs=dict(
                n_stocks=n_stocks, n_features=n_features,
                embed_dim=64, n_heads=4, features_dim=128,
            ),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=MLPPortfolioExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )

    hp_path = os.path.join(project_dir, 'results', 'best_hyperparams.json')
    with open(hp_path) as f:
        hp = json.load(f)
    print(f"   Hyperparameters: lr={hp['learning_rate']}, "
          f"n_steps={hp['n_steps']}, batch={hp['batch_size']}, "
          f"ent_coef={hp['ent_coef']}\n")

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

    # ---- Train ----
    print(f"--- Training {total_timesteps:,} steps "
          f"(~50-90 min on CPU; SP100 has 3× DJI state space) ---\n")
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    best_path = os.path.join(saved_dir, f'ppo_ablation_sp100_{run_tag}')
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints',
                                  f'ablation_sp100_{run_tag}')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_ablation_sp100_{run_tag}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")
    print(f"   Best  → {best_path}.zip")

    # ---- Reload BEST and evaluate ----
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

    # ---- Save ----
    save = {
        'method': f'sp100_ablation_{run_tag}',
        'universe': 'S&P 100 historical subset (90 stocks)',
        'ablation': ablation,
        'env_ablation_mode': env_ablation_mode,
        'use_attention': use_attention,
        'risk_lambda_fixed': fixed_lambda,
        'best_val_sharpe': val_cb.best_val_sharpe,
        'best_step': val_cb.best_step,
        'periods': results,
        'val_sharpe_history': val_cb.history,
        'rewards_history': detailed.rewards_history,
    }
    out = os.path.join(results_dir, 'final_results.json')
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n✅ Results → {out}")

    # ---- Compare against SP100 FULL v1 ----
    sp100_full_path = os.path.join(project_dir, 'results', 'extended_sp100',
                                    'final_results.json')
    if os.path.exists(sp100_full_path):
        with open(sp100_full_path) as f:
            full = json.load(f)
        print("\n" + "=" * 80)
        print(f"   SP100 ABLATION '{ablation}' vs SP100 FULL v1")
        print("=" * 80)
        print(f"  {'Period':<12} {'System':<28} {'Return%':>9} "
              f"{'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*74}")
        for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
            bk = f'agent_{period}'
            if bk not in full:
                continue
            b = full[bk]
            m = results[period]
            print(f"  {period:<12} {'SP100 FULL v1':<28} "
                  f"{b['cumulative_return']:>9.2f} "
                  f"{b['sharpe_ratio']:>8.4f} "
                  f"{b['max_drawdown']:>9.2f}")
            print(f"  {period:<12} {f'Ablation: {run_tag}':<28} "
                  f"{m['cumulative_return']:>9.2f} "
                  f"{m['sharpe_ratio']:>8.4f} "
                  f"{m['max_drawdown']:>9.2f}")
            d_ret = m['cumulative_return'] - b['cumulative_return']
            d_shp = m['sharpe_ratio']      - b['sharpe_ratio']
            d_mdd = m['max_drawdown']      - b['max_drawdown']
            sgn = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
            print(f"  {'':12} {'  Δ (ablation - full)':<28} "
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
        ax.set_title(f'SP100 ablation: {ablation} ({total_timesteps:,} steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 60)
    print(f"   SP100 ABLATION '{ablation}' COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
