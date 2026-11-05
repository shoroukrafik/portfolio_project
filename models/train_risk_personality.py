# ============================================
# MULTI-AGENT v2: RISK-PERSONALITY agents
# ============================================
# Three independent PPO agents, all trained on the SAME full data
# (1210 days, all 3 regimes mixed) but with DIFFERENT fixed risk
# penalty λ. Each develops a distinct trading personality.
#
#   Conservative agent  λ = 0.8   (heavy downside aversion)
#   Balanced agent      λ = 0.3   (moderate)
#   Aggressive agent    λ = 0.05  (chase upside)
#
# At test time, an HMM-driven controller routes/blends:
#   bear regime / high bear prob   → conservative agent
#   sideways regime / moderate     → balanced agent
#   bull regime / high bull prob   → aggressive agent
#
# Why this works (and the v1 regime-filtered design didn't):
#   - All agents see the FULL data — no regime-filtered scarcity
#   - Each agent has 1210 days of training (vs v1 bear's 76 days)
#   - Differentiation is in the REWARD (well-known ML technique),
#     not the data (which caused v1's overfitting)
#
# Usage:
#   python3 train_risk_personality.py                          # all 3 (~2.5 hr)
#   python3 train_risk_personality.py --personality conservative   # one
#   python3 train_risk_personality.py --timesteps 500000           # shorter
#
# Outputs:
#   models/saved/risk_personality_ext/personality_{conservative,balanced,aggressive}.zip
#   results/risk_personality_ext/training_summary.json
#
# (Original Phase 5 results in results/risk_personality/ are preserved as
# historical "failed-experiment" reference and are NOT overwritten.)
# ============================================

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv

# Risk personalities — name → fixed λ (downside penalty)
PERSONALITIES = {
    'conservative': 0.8,
    'balanced':     0.3,
    'aggressive':   0.05,
}


# ============================================
# ATTENTION FEATURE EXTRACTOR — same as Phase 5 baseline
# ============================================

class AttentionPortfolioExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_stocks=30, n_features=19,
                 embed_dim=64, n_heads=4, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.n_stocks = n_stocks
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.stock_feat_size = n_stocks * n_features
        self.extra_feat_size = 3 + n_stocks + 1
        self.stock_encoder = nn.Sequential(
            nn.Linear(n_features, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True,
        )
        self.extra_encoder = nn.Sequential(
            nn.Linear(self.extra_feat_size, 32), nn.ReLU(),
        )
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim + 32, features_dim), nn.ReLU(),
        )
        self.last_attention_weights = None

    def forward(self, observations):
        b = observations.shape[0]
        sf = observations[:, :self.stock_feat_size]
        ef = observations[:, self.stock_feat_size:]
        if ef.shape[1] != self.extra_feat_size:
            if ef.shape[1] < self.extra_feat_size:
                pad = torch.zeros(b, self.extra_feat_size - ef.shape[1],
                                  device=observations.device)
                ef = torch.cat([ef, pad], dim=1)
            else:
                ef = ef[:, :self.extra_feat_size]
        sf = sf.reshape(b, self.n_stocks, self.n_features)
        emb = self.stock_encoder(sf)
        attn_out, attn_w = self.attention(emb, emb, emb)
        self.last_attention_weights = attn_w.detach().cpu().numpy()
        pooled = attn_out.mean(dim=1)
        ee = self.extra_encoder(ef)
        return self.combiner(torch.cat([pooled, ee], dim=1))


# ============================================
# CALLBACKS
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
    """
    Standard val-Sharpe-best checkpointing on FULL val (252 days, all
    regimes mixed). Each personality agent gets val-checkpointed:
    none of them face the v1-bear "no val data" problem because
    each is trained on full data and evaluated on full val.
    """

    def __init__(self, val_env, save_path, eval_freq=25_000, verbose=1,
                 checkpoint_dir=None):
        super().__init__(verbose)
        self.val_env = val_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_val_sharpe = -np.inf
        self.best_step = 0
        self.history = []
        # When set, save every eval step's model so the best can be
        # re-selected post-hoc by any criterion.
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


# ============================================
# TRAIN ONE PERSONALITY
# ============================================

def train_personality(name, lambda_value, train_df, val_df,
                      total_timesteps, hp, n_stocks, n_features, paths):
    print("\n" + "=" * 60)
    print(f"   Training {name.upper()} personality (λ = {lambda_value})")
    print("=" * 60)
    print(f"   Train data: {train_df['date'].nunique()} days (FULL — no filter)")
    print(f"   Val data:   {val_df['date'].nunique()} days (FULL)")

    # Both train and val envs use the same fixed λ — the agent's
    # personality is constant whether learning or being evaluated.
    train_env = PortfolioEnv(df=train_df, initial_balance=1_000_000,
                             risk_lambda_fixed=lambda_value)
    val_env   = PortfolioEnv(df=val_df,   initial_balance=1_000_000,
                             risk_lambda_fixed=lambda_value)

    policy_kwargs = dict(
        features_extractor_class=AttentionPortfolioExtractor,
        features_extractor_kwargs=dict(
            n_stocks=n_stocks, n_features=n_features,
            embed_dim=64, n_heads=4, features_dim=128,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
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

    # Callbacks
    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=total_timesteps)
    save_path = os.path.join(paths['saved'], f'personality_{name}')
    checkpoint_dir = os.path.join(paths['saved'], 'checkpoints',
                                  f'personality_{name}')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=save_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"\n   Training {total_timesteps:,} steps "
          f"(~{int(total_timesteps/225_000 * 12)} min on CPU)...\n")
    model.learn(total_timesteps=total_timesteps,
                callback=[detailed, val_cb], progress_bar=True)

    print(f"\n   ✅ Best-by-val saved → personality_{name}.zip "
          f"(val Sharpe {val_cb.best_val_sharpe:.4f} at step "
          f"{val_cb.best_step:,})")

    return {
        'personality': name,
        'lambda': float(lambda_value),
        'train_days': int(train_df['date'].nunique()),
        'val_days': int(val_df['date'].nunique()),
        'best_val_sharpe': float(val_cb.best_val_sharpe),
        'best_step': int(val_cb.best_step),
        'rewards_history': detailed.rewards_history,
        'val_sharpe_history': val_cb.history,
    }


# ============================================
# MAIN
# ============================================

def parse_args():
    p = argparse.ArgumentParser(description='Train risk-personality agents')
    p.add_argument('--personality', default='all',
                   choices=['all'] + list(PERSONALITIES.keys()))
    p.add_argument('--timesteps', type=int, default=1_000_000,
                   help='Training budget per personality (default 1M)')
    return p.parse_args()


def main():
    args = parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    # Extended-data retraining — outputs go to NEW dirs so original Phase 5
    # results (in results/risk_personality/) are preserved as "failed" reference.
    saved_dir   = os.path.join(base_dir, 'saved', 'risk_personality_ext')
    results_dir = os.path.join(project_dir, 'results', 'risk_personality_ext')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    paths = {'saved': saved_dir, 'results': results_dir}

    print("=" * 60)
    print("   MULTI-AGENT v2: RISK-PERSONALITY TRAINING")
    print("=" * 60)
    print(f"\n   Saved dir:   {saved_dir}")
    print(f"   Results dir: {results_dir}")

    # Load full data (NO filtering — that was v1's mistake)
    # Extended pipeline: 2007-2020 train, 2010-retuned HMM regime labels.
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data_ext.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    train_df['date'] = pd.to_datetime(train_df['date'])
    val_df['date']   = pd.to_datetime(val_df['date'])

    feature_columns = [
        'close','open','high','low','volume',
        'rsi','macd','macd_signal','macd_diff',
        'bb_upper','bb_lower','bb_width',
        'atr','sma_20','sma_50','adx','cci',
        'daily_return','volatility_20d',
    ]
    feature_columns = [f for f in feature_columns if f in train_df.columns]
    n_features = len(feature_columns)
    n_stocks   = train_df['ticker'].nunique()

    print(f"\n  Stocks: {n_stocks}, Features: {n_features}")
    print(f"  Train: {train_df['date'].nunique()} days (FULL)")
    print(f"  Val:   {val_df['date'].nunique()} days (FULL)")

    with open(os.path.join(project_dir, 'results', 'best_hyperparams.json')) as f:
        hp = json.load(f)
    print(f"\n  Hyperparameters: lr={hp['learning_rate']}, "
          f"n_steps={hp['n_steps']}, batch={hp['batch_size']}, "
          f"ent_coef={hp['ent_coef']}")

    # Determine which personalities to train
    if args.personality == 'all':
        names_to_train = list(PERSONALITIES.keys())
    else:
        names_to_train = [args.personality]

    print(f"\n  Training: {names_to_train}")
    print(f"  λ values: {[PERSONALITIES[n] for n in names_to_train]}")

    summaries = []
    for name in names_to_train:
        s = train_personality(
            name=name,
            lambda_value=PERSONALITIES[name],
            train_df=train_df, val_df=val_df,
            total_timesteps=args.timesteps,
            hp=hp,
            n_stocks=n_stocks, n_features=n_features,
            paths=paths,
        )
        summaries.append(s)

    # Save / merge training summary
    summary_path = os.path.join(results_dir, 'training_summary.json')
    if os.path.exists(summary_path) and args.personality != 'all':
        with open(summary_path) as f:
            prev = json.load(f)
        prev = [x for x in prev if x['personality'] not in
                [s['personality'] for s in summaries]]
        summaries = prev + summaries
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\n✅ Training summary → {summary_path}")

    print("\n" + "=" * 60)
    print("   RISK-PERSONALITY TRAINING COMPLETE ✅")
    print("=" * 60)
    for s in summaries:
        print(f"  {s['personality']:<14} λ={s['lambda']:<5} "
              f"best val Sharpe={s['best_val_sharpe']:.4f} "
              f"at step {s['best_step']:,}")
    print(f"\nNext step: python3 eval_risk_personality.py --mode soft")


if __name__ == "__main__":
    main()
