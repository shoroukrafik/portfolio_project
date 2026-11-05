# ============================================
# PHASE 5b: Ablation Study (1M steps each)
# ============================================
# Runs ONE ablation per invocation. Each ablation removes one of the
# three thesis novelties and re-trains from scratch with the same
# hyperparameters as the Phase 5 baseline (best_hyperparams.json).
#
# Usage:
#   python3 train_ablation.py --ablation no_floor
#   python3 train_ablation.py --ablation no_attention
#   python3 train_ablation.py --ablation no_dsr
#
# Outputs (per ablation):
#   models/saved/ppo_best_<name>.zip
#   models/saved/ppo_final_<name>.zip
#   results/ablation_<name>/final_results.json
#   results/ablation_<name>/training_curve.png
#   results/ablation_<name>/val_sharpe_history.json
#
# Baseline files (results/final_results.json, models/saved/ppo_best.zip)
# are NEVER overwritten.
# ============================================

import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv


# ============================================
# FEATURE EXTRACTORS
# ============================================

class AttentionPortfolioExtractor(BaseFeaturesExtractor):
    """Full model: multi-head self-attention over 30 stocks."""

    def __init__(self, observation_space, n_stocks=30, n_features=19,
                 embed_dim=64, n_heads=4, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.n_stocks = n_stocks
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.stock_feat_size = n_stocks * n_features
        self.extra_feat_size = 3 + n_stocks + 1

        self.stock_encoder = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.extra_encoder = nn.Sequential(
            nn.Linear(self.extra_feat_size, 32),
            nn.ReLU(),
        )
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim + 32, features_dim),
            nn.ReLU(),
        )
        self.last_attention_weights = None

    def forward(self, observations):
        batch_size = observations.shape[0]
        stock_features = observations[:, :self.stock_feat_size]
        extra_features = observations[:, self.stock_feat_size:]

        if extra_features.shape[1] != self.extra_feat_size:
            if extra_features.shape[1] < self.extra_feat_size:
                padding = torch.zeros(
                    batch_size,
                    self.extra_feat_size - extra_features.shape[1],
                    device=observations.device
                )
                extra_features = torch.cat([extra_features, padding], dim=1)
            else:
                extra_features = extra_features[:, :self.extra_feat_size]

        stock_features = stock_features.reshape(
            batch_size, self.n_stocks, self.n_features)
        stock_embeddings = self.stock_encoder(stock_features)
        attn_output, attn_weights = self.attention(
            stock_embeddings, stock_embeddings, stock_embeddings)
        self.last_attention_weights = attn_weights.detach().cpu().numpy()
        pooled = attn_output.mean(dim=1)
        extra_encoded = self.extra_encoder(extra_features)
        combined = torch.cat([pooled, extra_encoded], dim=1)
        return self.combiner(combined)


class MLPPortfolioExtractor(BaseFeaturesExtractor):
    """
    Ablation: replaces the attention block with a plain MLP that sees the
    same flat observation. Same final feature_dim so the policy/value heads
    (net_arch=[128,64]) are identical — only the front-end differs.
    """

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, features_dim),
            nn.ReLU(),
        )
        # Kept for compatibility with evaluate_agent() — never populated.
        self.last_attention_weights = None

    def forward(self, observations):
        return self.mlp(observations)


# ============================================
# CALLBACKS (same as train_final.py)
# ============================================

class DetailedCallback(BaseCallback):
    def __init__(self, eval_freq=50000, total_timesteps=225_000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.total_timesteps = total_timesteps
        self.rewards_history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                recent = [ep['r'] for ep in self.model.ep_info_buffer]
                mean_r = np.mean(recent)
                self.rewards_history.append((self.n_calls, mean_r))
                print(f"  Step {self.n_calls:>8,} / {self.total_timesteps:,} "
                      f"| Mean Reward: {mean_r:>8.4f} "
                      f"| Episodes seen: {len(recent)}")
        return True


class ValidationLoggingCallback(BaseCallback):
    """
    Logs validation Sharpe every `eval_freq` steps for the diagnostic plot.
    Does NOT save models — under the budget-matched control protocol, we
    only use the FINAL model (at step `total_timesteps`) for evaluation,
    not best-of-N. This avoids the selection bias of cherry-picking the
    favourable snapshot.
    """

    def __init__(self, val_env, eval_freq=25_000, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.history = []

    def _evaluate_validation(self):
        obs, _ = self.val_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = self.val_env.step(action)
            done = term or trunc
        history = self.val_env.get_portfolio_history()
        values = history['value'].values
        if len(values) < 2:
            return 0.0
        daily_ret = np.diff(values) / values[:-1]
        n = len(values)
        ann_ret = ((values[-1] / values[0]) ** (252 / n) - 1) * 100
        ann_vol = float(np.std(daily_ret) * np.sqrt(252) * 100)
        if ann_vol <= 0:
            return 0.0
        return ann_ret / ann_vol

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            val_sharpe = self._evaluate_validation()
            self.history.append((self.n_calls, val_sharpe))
            print(f"    Step {self.n_calls:>7,}: val Sharpe = {val_sharpe:>7.4f}")
        return True


# ============================================
# METRICS / EVAL HELPERS (same as train_final.py)
# ============================================

def calculate_metrics(history, initial_balance=1_000_000):
    values = history['value'].values
    final_value = values[-1]
    cumulative_return = (final_value - initial_balance) / initial_balance * 100
    daily_returns = np.diff(values) / values[:-1]
    n_days = len(values)
    annualized_return = ((final_value/initial_balance)**(252/n_days)-1)*100
    annualized_vol = np.std(daily_returns) * np.sqrt(252) * 100
    sharpe_ratio = annualized_return/annualized_vol if annualized_vol > 0 else 0
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_drawdown = np.min(drawdown) * 100
    calmar_ratio = annualized_return/abs(max_drawdown) if abs(max_drawdown) > 0 else 0
    return {
        'cumulative_return': round(cumulative_return, 2),
        'annualized_return': round(annualized_return, 2),
        'annualized_volatility': round(annualized_vol, 2),
        'sharpe_ratio': round(sharpe_ratio, 4),
        'max_drawdown': round(max_drawdown, 2),
        'calmar_ratio': round(calmar_ratio, 4),
        'final_value': round(final_value, 2),
    }


def evaluate_agent(model, env):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_portfolio_history()


# ============================================
# MAIN
# ============================================

ABLATION_CHOICES = ['no_floor', 'no_attention', 'no_dsr']


def parse_args():
    p = argparse.ArgumentParser(description='Run one ablation study')
    p.add_argument('--ablation', required=True, choices=ABLATION_CHOICES,
                   help='Which component to ablate')
    p.add_argument('--timesteps', type=int, default=225_000,
                   help='Training budget. Default 225k = baseline\'s peak step '
                        '(budget-matched comparison). Final model at this step '
                        'is used for evaluation (no best-of-N selection).')
    return p.parse_args()


def main():
    args = parse_args()
    ablation = args.ablation
    total_timesteps = args.timesteps

    # Map ablation flag → which env knob and which feature extractor to use
    env_ablation_mode = ablation if ablation in ('no_floor', 'no_dsr') else None
    use_attention = (ablation != 'no_attention')

    print("=" * 60)
    print(f"   PHASE 5b: ABLATION = {ablation}")
    print("=" * 60)
    print(f"\n  Env ablation_mode  : {env_ablation_mode}")
    print(f"  Feature extractor  : "
          f"{'AttentionPortfolioExtractor' if use_attention else 'MLPPortfolioExtractor'}")
    print(f"  Training budget    : {total_timesteps:,} steps")
    print(f"  Eval protocol      : FINAL model at step {total_timesteps:,} "
          f"(budget-matched control, no best-of-N selection)")

    # Paths
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    results_dir = os.path.join(project_dir, 'results', f'ablation_{ablation}')
    saved_dir   = os.path.join(base_dir, 'saved')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(saved_dir, exist_ok=True)

    # ---- Load Data ----
    print("\n--- Loading data ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data.csv'))
    for df in (train_df, val_df, test_bear_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    n_stocks = train_df['ticker'].nunique()
    feature_columns = [
        'close','open','high','low','volume',
        'rsi','macd','macd_signal','macd_diff',
        'bb_upper','bb_lower','bb_width',
        'atr','sma_20','sma_50','adx','cci',
        'daily_return','volatility_20d'
    ]
    feature_columns = [f for f in feature_columns if f in train_df.columns]
    n_features = len(feature_columns)
    print(f"✅ Data loaded — {n_stocks} stocks × {n_features} features")

    # ---- Create Environments (with ablation mode applied) ----
    print("\n--- Creating environments ---\n")
    train_env     = PortfolioEnv(df=train_df,     initial_balance=1_000_000,
                                 ablation_mode=env_ablation_mode)
    val_env       = PortfolioEnv(df=val_df,       initial_balance=1_000_000,
                                 ablation_mode=env_ablation_mode)
    test_bear_env = PortfolioEnv(df=test_bear_df, initial_balance=1_000_000,
                                 ablation_mode=env_ablation_mode)
    test_bull_env = PortfolioEnv(df=test_bull_df, initial_balance=1_000_000,
                                 ablation_mode=env_ablation_mode)

    # ---- Build Model ----
    print("\n--- Building PPO model ---\n")
    if use_attention:
        policy_kwargs = dict(
            features_extractor_class=AttentionPortfolioExtractor,
            features_extractor_kwargs=dict(
                n_stocks=n_stocks,
                n_features=n_features,
                embed_dim=64,
                n_heads=4,
                features_dim=128,
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
    print(f"Loaded best hyperparameters: {hp['name']}")
    print(f"  lr={hp['learning_rate']}, n_steps={hp['n_steps']}, "
          f"batch={hp['batch_size']}, ent_coef={hp['ent_coef']}")

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        ent_coef=hp['ent_coef'],
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=42,
        device='cpu',
    )
    print("✅ Model configured")

    # ---- Train (budget-matched, no best-of-N) ----
    # Approx wall-clock on CPU: ~12 min per 225k steps
    est_min = max(1, int(total_timesteps / 225_000 * 12))
    print(f"\n--- Training for {total_timesteps:,} steps ---")
    print(f"Expected time: ~{est_min} minutes on CPU\n")

    detailed_callback = DetailedCallback(
        eval_freq=max(25_000, total_timesteps // 10),
        total_timesteps=total_timesteps,
    )
    val_callback = ValidationLoggingCallback(
        val_env=val_env,
        eval_freq=25_000,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[detailed_callback, val_callback],
        progress_bar=True,
    )

    print("\n✅ Training complete!")

    # Save the FINAL model — this is what we evaluate on (budget-matched control)
    final_path = os.path.join(saved_dir, f'ppo_ablation_{ablation}')
    model.save(final_path)
    print(f"✅ Final model (step {total_timesteps:,}) → {final_path}.zip")

    val_history_path = os.path.join(results_dir, 'val_sharpe_history.json')
    with open(val_history_path, 'w') as f:
        json.dump({
            'history': val_callback.history,
            'note': ('Logged for diagnostic purposes only. The FINAL model '
                     f'(step {total_timesteps:,}) is used for evaluation, '
                     'not best-of-N selection.'),
        }, f, indent=2)

    # Diagnostic only: log whether val Sharpe was still rising. Under the
    # budget-matched protocol we DO NOT extend — every variant is evaluated
    # at the agreed step count regardless of where its trajectory was headed.
    if len(val_callback.history) >= 3:
        last_three = [v for _, v in val_callback.history[-3:]]
        trend = "rising" if last_three[-1] > max(last_three[:-1]) else \
                "falling" if last_three[-1] < min(last_three[:-1]) else "flat"
        print(f"\n  Val Sharpe trajectory at end: {[round(v, 3) for v in last_three]} ({trend})")
        print(f"  (Diagnostic only — model NOT extended; budget-matched protocol)")

    # No reload — we evaluate the model in-memory at its final state.

    # ---- Evaluate on all 4 periods ----
    print("\n--- Evaluating on all periods ---\n")
    results = {}
    for name, env, desc in [
        ('train',     train_env,     'Training (2016-2020)'),
        ('val',       val_env,       'Validation (2021)'),
        ('test_bear', test_bear_env, 'Test Bear (2022)'),
        ('test_bull', test_bull_env, 'Test Bull (2024)'),
    ]:
        print(f"  {desc}...")
        history = evaluate_agent(model, env)
        metrics = calculate_metrics(history)
        results[name] = {'metrics': metrics, 'description': desc}
        print(f"    Return: {metrics['cumulative_return']}% | "
              f"Sharpe: {metrics['sharpe_ratio']} | "
              f"MDD: {metrics['max_drawdown']}%")

    # ---- Save results JSON ----
    save_data = {
        'ablation': ablation,
        'training_budget': total_timesteps,
        'eval_protocol': 'final_model_at_budget_step',
        'val_sharpe_history': val_callback.history,
    }
    for k, v in results.items():
        save_data[f'agent_{k}'] = v['metrics']

    results_path = os.path.join(results_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✅ Results → {results_path}")

    # ---- Training curve ----
    if detailed_callback.rewards_history:
        steps  = [x[0] for x in detailed_callback.rewards_history]
        rewards = [x[1] for x in detailed_callback.rewards_history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rewards, color='blue', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title(f'Ablation: {ablation} — Training Curve ({total_timesteps:,} steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = os.path.join(results_dir, 'training_curve.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Training curve → {curve_path}")

    # ---- Compare against baseline (if available) ----
    baseline_path = os.path.join(project_dir, 'results', 'final_results.json')
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            base = json.load(f)

        def _row(label, src):
            m = src
            return (f"  {label:<22} "
                    f"{m['cumulative_return']:>9.2f} "
                    f"{m['sharpe_ratio']:>8.4f} "
                    f"{m['max_drawdown']:>9.2f}")

        print("\n" + "=" * 60)
        print("   ABLATION vs BASELINE")
        print("=" * 60)
        print(f"  {'Period':<22} {'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*52}")
        for period in ['train', 'val', 'test_bear', 'test_bull']:
            base_key = f'agent_{period}'
            if base_key in base:
                print(_row(f"baseline {period}", base[base_key]))
                print(_row(f"{ablation} {period}", results[period]['metrics']))
                print()
    else:
        print("\n(no baseline final_results.json found — run train_final.py first "
              "to enable side-by-side comparison)")

    print("\n" + "=" * 60)
    print(f"   ABLATION '{ablation}' COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
