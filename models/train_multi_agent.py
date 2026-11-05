# ============================================
# MULTI-AGENT: Train regime specialists
# ============================================
# Trains three independent PPO policies, each on a regime-filtered
# subset of the training data. Hybrid checkpointing protocol (Option A):
#
#   Bull specialist     → 1M steps, val-checkpointed on bull val days
#   Sideways specialist → 1M steps, val-checkpointed on sideways val days
#   Bear specialist     → 1M steps, FINAL model saved (val 2021 has no
#                         bear days under our HMM, so val-checkpointing
#                         is not possible — documented as a limitation)
#
# Usage:
#   python3 train_multi_agent.py                         # all 3
#   python3 train_multi_agent.py --specialist bear       # just bear
#   python3 train_multi_agent.py --timesteps 500000      # shorter run
#
# Outputs:
#   models/saved/multi_agent_ext/specialist_{bull,bear,sideways}.zip
#   results/multi_agent_ext/training_summary.json
#
# (Original Phase 5 results in results/multi_agent/ are preserved as
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

REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'sideways'}
NAME_TO_ID = {v: k for k, v in REGIME_NAMES.items()}

# A specialist needs at least this many regime-filtered val days to
# trust val checkpointing. Below this we fall back to "final model".
MIN_VAL_DAYS_FOR_CHECKPOINT = 30


# ============================================
# ARCHITECTURE — same as Phase 5 baseline
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
            nn.Linear(embed_dim, embed_dim),  nn.ReLU(),
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
    Val-Sharpe-best checkpointing on a regime-filtered val env. With the
    extended pipeline, this is now used for ALL specialists (bull, bear,
    sideways). Bear's val signal is noisy (few bear days in 2021), but we
    val-checkpoint anyway and ALSO save every per-eval checkpoint so the
    best model can be re-selected post-hoc by any criterion.
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
        # When set, every eval step's model is also saved under
        # checkpoint_dir/step_<n>.zip so the best can be re-selected later.
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
            # Always save per-eval checkpoint (for post-hoc selection)
            if self.checkpoint_dir is not None:
                ck = os.path.join(self.checkpoint_dir,
                                  f'step_{self.n_calls:07d}')
                self.model.save(ck)
        return True


# ============================================
# CORE: train one specialist
# ============================================

def train_specialist(regime_id, train_df, val_df, total_timesteps,
                     hp, n_stocks, n_features, paths):
    """
    Train one regime specialist.

    Bull/Sideways: val-checkpointed (saves best by val Sharpe on
                   regime-filtered val data).
    Bear:          no val data → save final model at total_timesteps.
    """
    name = REGIME_NAMES[regime_id]
    print("\n" + "=" * 60)
    print(f"   Training {name.upper()} specialist (regime_id={regime_id})")
    print("=" * 60)

    # Filter training data — only this regime's days
    train_filtered = train_df[train_df['regime'] == regime_id].copy()
    n_train_days = train_filtered['date'].nunique()
    print(f"   Train days: {n_train_days} (filtered from "
          f"{train_df['date'].nunique()} total)")

    # Val set is regime-filtered (judge each specialist on its OWN regime's
    # val days). Bear may have few val days in 2021 → noisy val-Sharpe, but
    # we val-checkpoint anyway and ALSO save every per-eval checkpoint so the
    # best model can be re-selected post-hoc by any criterion.
    val_filtered = val_df[val_df['regime'] == regime_id].copy()
    n_val_days = val_filtered['date'].nunique()
    print(f"   Val days:   {n_val_days} (regime-filtered to '{name}')")

    # Always val-checkpoint. With few val days the signal is noisy but at
    # least we save SOME checkpoint based on per-regime val Sharpe.
    can_val_checkpoint = True
    print(f"   Protocol:   1M + val checkpointing every 25k steps "
          f"+ all checkpoints saved for post-hoc selection")

    # Build envs
    train_env = PortfolioEnv(df=train_filtered, initial_balance=1_000_000)
    val_env   = PortfolioEnv(df=val_filtered,   initial_balance=1_000_000)

    # PPO model — same architecture and hyperparams as Phase 5 baseline
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
    callbacks = [detailed]

    save_path = os.path.join(paths['saved'], f'specialist_{name}')
    checkpoint_dir = os.path.join(paths['saved'], 'checkpoints',
                                  f'specialist_{name}')
    val_callback = ValidationCheckpointCallback(
        val_env=val_env, save_path=save_path, eval_freq=25_000,
        checkpoint_dir=checkpoint_dir,
    )
    callbacks.append(val_callback)

    # Train
    print(f"\n   Training {total_timesteps:,} steps "
          f"(~{int(total_timesteps/225_000 * 12)} min on CPU)...\n")
    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                progress_bar=True)

    # All specialists now val-checkpoint (regime-filtered val), AND all
    # per-eval checkpoints are saved under checkpoints/specialist_<name>/
    # so the best can be re-selected post-hoc by any criterion.
    print(f"\n   ✅ Best-by-val saved → specialist_{name}.zip "
          f"(val Sharpe {val_callback.best_val_sharpe:.4f} "
          f"at step {val_callback.best_step:,})")
    print(f"   ✅ Per-eval checkpoints in {checkpoint_dir}")

    return {
        'regime_id': regime_id,
        'regime_name': name,
        'train_days': int(n_train_days),
        'val_days': int(n_val_days),
        'val_checkpointed': True,
        'best_val_sharpe': float(val_callback.best_val_sharpe),
        'best_step': int(val_callback.best_step),
        'rewards_history': detailed.rewards_history,
        'val_sharpe_history': val_callback.history,
        'checkpoint_dir': checkpoint_dir,
    }


# ============================================
# MAIN
# ============================================

def parse_args():
    p = argparse.ArgumentParser(description='Train regime specialists')
    p.add_argument('--specialist', default='all',
                   choices=['all', 'bull', 'bear', 'sideways'],
                   help='Which specialist to train (default: all 3)')
    p.add_argument('--timesteps', type=int, default=1_000_000,
                   help='Training budget per specialist (default 1M)')
    return p.parse_args()


def main():
    args = parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    # Extended-data retraining — outputs go to NEW dirs so original Phase 5
    # results (in results/multi_agent/) are preserved as "failed" reference.
    saved_dir   = os.path.join(base_dir, 'saved', 'multi_agent_ext')
    results_dir = os.path.join(project_dir, 'results', 'multi_agent_ext')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    paths = {'saved': saved_dir, 'results': results_dir}

    print("=" * 60)
    print("   MULTI-AGENT TRAINING — regime specialists")
    print("=" * 60)
    print(f"\n   Saved dir:   {saved_dir}")
    print(f"   Results dir: {results_dir}")

    # Load data
    # Extended pipeline: 2007-2020 train, 2010-retuned HMM regime labels
    # baked in. Just switching to the _ext.csv files automatically uses
    # the retuned HMM since the CSVs were regenerated with the new HMM.
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

    # Print regime breakdown so user sees what each specialist gets
    print("\n--- Data summary ---")
    print(f"  Stocks: {n_stocks}, Features: {n_features}")
    train_counts = (train_df[['date','regime']].drop_duplicates()
                    .groupby('regime').size().to_dict())
    val_counts = (val_df[['date','regime']].drop_duplicates()
                  .groupby('regime').size().to_dict())
    print(f"  Train regimes: ", end='')
    for rid, name in REGIME_NAMES.items():
        c = int(train_counts.get(rid, 0))
        print(f"{name}={c}  ", end='')
    print()
    print(f"  Val   regimes: ", end='')
    for rid, name in REGIME_NAMES.items():
        c = int(val_counts.get(rid, 0))
        print(f"{name}={c}  ", end='')
    print()

    # Load best hyperparams (same as Phase 5 baseline)
    with open(os.path.join(project_dir, 'results', 'best_hyperparams.json')) as f:
        hp = json.load(f)
    print(f"\n  Hyperparameters: lr={hp['learning_rate']}, "
          f"n_steps={hp['n_steps']}, batch={hp['batch_size']}, "
          f"ent_coef={hp['ent_coef']}")

    # Decide which specialists to train
    if args.specialist == 'all':
        regime_ids_to_train = [0, 1, 2]
    else:
        regime_ids_to_train = [NAME_TO_ID[args.specialist]]

    summaries = []
    for rid in regime_ids_to_train:
        s = train_specialist(
            regime_id=rid,
            train_df=train_df, val_df=val_df,
            total_timesteps=args.timesteps,
            hp=hp,
            n_stocks=n_stocks, n_features=n_features,
            paths=paths,
        )
        summaries.append(s)

    # Save training summary
    summary_path = os.path.join(results_dir, 'training_summary.json')
    # Append/merge if a per-specialist run is being made
    if os.path.exists(summary_path) and args.specialist != 'all':
        with open(summary_path) as f:
            prev = json.load(f)
        # Drop any prior entry for the regime we just retrained
        prev = [x for x in prev if x['regime_id'] not in
                [s['regime_id'] for s in summaries]]
        summaries = prev + summaries
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\n✅ Training summary → {summary_path}")

    print("\n" + "=" * 60)
    print("   MULTI-AGENT TRAINING COMPLETE ✅")
    print("=" * 60)
    for s in summaries:
        print(f"  {s['regime_name'].upper():<10}: "
              f"train_days={s['train_days']:<5} "
              f"val_days={s['val_days']:<3} "
              f"best_val_sharpe={s['best_val_sharpe']:.4f} "
              f"@ step {s['best_step']:,}")
    print(f"\nNext step: python3 eval_multi_agent.py")


if __name__ == "__main__":
    main()
