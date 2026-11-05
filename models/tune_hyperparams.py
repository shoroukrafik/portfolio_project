# ============================================
# PHASE 5 - Script 1: Hyperparameter Tuning
# ============================================
# We try different combinations of settings
# and pick whichever gets the best Sharpe ratio
# on the VALIDATION set (2021 data)
# ============================================

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv

print("=" * 60)
print("   PHASE 5 - Script 1: Hyperparameter Tuning")
print("=" * 60)

# ============================================
# COPY: Attention Network (same as Phase 4)
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


# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_metrics(history, initial_balance=1_000_000):
    values = history['value'].values
    final_value = values[-1]
    cumulative_return = (final_value - initial_balance) / initial_balance * 100
    daily_returns = np.diff(values) / values[:-1]
    n_days = len(values)
    annualized_return = ((final_value / initial_balance) ** (252/n_days) - 1) * 100
    annualized_vol = np.std(daily_returns) * np.sqrt(252) * 100
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_drawdown = np.min(drawdown) * 100
    calmar_ratio = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0
    return {
        'cumulative_return': round(cumulative_return, 2),
        'annualized_return': round(annualized_return, 2),
        'annualized_volatility': round(annualized_vol, 2),
        'sharpe_ratio': round(sharpe_ratio, 4),
        'max_drawdown': round(max_drawdown, 2),
        'calmar_ratio': round(calmar_ratio, 4),
    }

def evaluate_on_validation(model, val_env):
    obs, info = val_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = val_env.step(action)
        done = terminated or truncated
    history = val_env.get_portfolio_history()
    metrics = calculate_metrics(history)
    return metrics['sharpe_ratio'], metrics


# ============================================
# HYPERPARAMETER SEARCH
# ============================================

def run_hyperparameter_search(train_df, val_df, n_stocks, n_features):
    
    # Configurations to try. After fixing the reward (DSR + softmax +
    # downside-only), the old "best" of lr=3e-4 caused late-training
    # collapse. lr=1e-4 was stable but possibly too conservative on bull.
    # We search around lr=1e-4 with neighbours in both directions, plus
    # variants in n_steps / batch / ent_coef.
    configs = [
        # (learning_rate, n_steps, batch_size, ent_coef, timesteps, name)
        (1e-4, 1024, 64,  0.01,  200_000, "baseline_1e-4"),       # known-good
        (2e-4, 1024, 64,  0.01,  200_000, "lr_2e-4"),              # more aggressive
        (5e-5, 1024, 64,  0.01,  200_000, "lr_5e-5"),              # very conservative
        (2e-4, 2048, 128, 0.01,  200_000, "lr2e4_n2048_b128"),     # bigger rollouts
        (1e-4, 1024, 64,  0.005, 200_000, "lower_entropy"),        # less exploration
    ]
    
    results = []
    best_sharpe = -np.inf
    best_config = None
    
    print(f"\nTrying {len(configs)} configurations on validation set...")
    print("Each takes ~10-11 minutes. Total: ~50-55 minutes.\n")

    for i, (lr, n_steps, batch_size, ent_coef, timesteps, name) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Config: {name}")
        print(f"  lr={lr}, n_steps={n_steps}, batch_size={batch_size}, "
              f"ent_coef={ent_coef}, steps={timesteps:,}")
        
        # Create fresh environments for each trial
        train_env = PortfolioEnv(df=train_df, initial_balance=1_000_000)
        val_env   = PortfolioEnv(df=val_df,   initial_balance=1_000_000)
        
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
        
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            verbose=0,
            seed=42,
            device='cpu',
        )
        
        # Train
        model.learn(total_timesteps=timesteps, progress_bar=True)
        
        # Evaluate on validation
        sharpe, metrics = evaluate_on_validation(model, val_env)
        
        print(f"  → Val Sharpe: {sharpe:.4f} | "
              f"Return: {metrics['cumulative_return']}% | "
              f"MDD: {metrics['max_drawdown']}%")
        
        results.append({
            'name': name,
            'learning_rate': lr,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'ent_coef': ent_coef,
            'timesteps': timesteps,
            'val_sharpe': sharpe,
            'val_return': metrics['cumulative_return'],
            'val_mdd': metrics['max_drawdown'],
        })

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_config = {
                'name': name,
                'learning_rate': lr,
                'n_steps': n_steps,
                'batch_size': batch_size,
                'ent_coef': ent_coef,
                'timesteps': timesteps,
            }
            print(f"  ★ NEW BEST! Sharpe = {sharpe:.4f}")
        
        print()
    
    return results, best_config


# ============================================
# MAIN
# ============================================

def main():
    # Paths
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("\n--- Loading data ---\n")
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    train_df['date'] = pd.to_datetime(train_df['date'])
    val_df = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
    val_df['date'] = pd.to_datetime(val_df['date'])

    n_stocks   = train_df['ticker'].nunique()
    feature_columns = [
        'close','open','high','low','volume',
        'rsi','macd','macd_signal','macd_diff',
        'bb_upper','bb_lower','bb_width',
        'atr','sma_20','sma_50','adx','cci',
        'daily_return','volatility_20d'
    ]
    feature_columns = [f for f in feature_columns if f in train_df.columns]
    n_features = len(feature_columns)

    print(f"Stocks: {n_stocks}, Features: {n_features}")

    # Run search
    results, best_config = run_hyperparameter_search(
        train_df, val_df, n_stocks, n_features)

    # Print summary table
    print("\n" + "=" * 70)
    print("   HYPERPARAMETER SEARCH RESULTS")
    print("=" * 70)
    print(f"{'Config':<22} {'Val Sharpe':>12} {'Val Return%':>12} {'Val MDD%':>10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['val_sharpe'], reverse=True):
        print(f"{r['name']:<22} {r['val_sharpe']:>12.4f} "
              f"{r['val_return']:>12.2f} {r['val_mdd']:>10.2f}")
    print("-" * 70)

    print(f"\n★ BEST CONFIG: {best_config['name']}")
    print(f"  learning_rate = {best_config['learning_rate']}")
    print(f"  n_steps       = {best_config['n_steps']}")
    print(f"  batch_size    = {best_config['batch_size']}")
    print(f"  ent_coef      = {best_config['ent_coef']}")

    # Save best config
    config_path = os.path.join(results_dir, 'best_hyperparams.json')
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\n✅ Best config saved to {config_path}")
    print("\nNow run: python3 train_final.py")


if __name__ == "__main__":
    main()