# ============================================
# PATH X — Extended baseline, NO CASH FLOOR (1M steps, 2007-2020 train)
# ============================================
# Same as train_final_extended.py but cash floor is DISABLED throughout
# training and evaluation. Tests whether the agent's learned policy
# develops bear defense from the extended training data + regime-
# conditioned λ alone, without the hard cash-floor constraint.
#
# All envs created with ablation_mode='no_floor', so cash_floors become
# {0:0, 1:0, 2:0}. The DSR reward + regime-conditioned downside penalty
# (λ_bear=0.5) remain active.
#
# Output goes to:
#   models/saved/ppo_best_extended_nofloor.zip
#   models/saved/ppo_final_extended_nofloor.zip
#   results/extended_nofloor/final_results.json
#   results/extended_nofloor/training_curve.png
#   results/extended_nofloor/final_performance.png
#
# Originals (Phase 5, ext+floor) are NOT touched.
# ============================================

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

print("=" * 60)
print("   PATH X: Extended baseline, NO CASH FLOOR (1M steps)")
print("=" * 60)


# ============================================
# ATTENTION FEATURE EXTRACTOR (same as Phase 5)
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
    def __init__(self, val_env, save_path, eval_freq=25_000, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_val_sharpe = -np.inf
        self.best_step = 0
        self.history = []

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
        return True


# ============================================
# METRICS / EVAL HELPERS
# ============================================

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
        'annualized_return': round(ann_ret, 2),
        'annualized_volatility': round(ann_vol, 2),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(mdd, 2),
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


def compute_benchmark_equal_weight(df, initial_balance=1_000_000):
    dates = sorted(df['date'].unique())
    tickers = sorted(df['ticker'].unique())
    pivot = df.pivot(index='date', columns='ticker', values='close')
    pivot = pivot[tickers].dropna()
    daily_ret = pivot.pct_change().fillna(0)
    port_ret = daily_ret.mean(axis=1)
    values = [initial_balance]
    for r in port_ret:
        values.append(values[-1] * (1 + r))
    values = values[1:]
    return pd.DataFrame({'date': pivot.index, 'value': values})


def compute_benchmark_dji(index_df, initial_balance=1_000_000):
    idx = index_df.copy().sort_values('date')
    start = idx['close'].iloc[0]
    idx['value'] = (idx['close'] / start) * initial_balance
    return idx[['date','value']].reset_index(drop=True)


# ============================================
# MAIN
# ============================================

def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    results_dir = os.path.join(project_dir, 'results', 'extended_nofloor')
    saved_dir   = os.path.join(base_dir, 'saved')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(saved_dir, exist_ok=True)

    # ---- Load EXTENDED data ----
    print("\n--- Loading extended data (2007-2024 with V→MA) ---\n")
    train_df     = pd.read_csv(os.path.join(data_dir, 'train_data_ext.csv'))
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_ext.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_ext.csv'))
    train_orig     = pd.read_csv(os.path.join(data_dir, 'train_data_original_ext.csv'))
    val_orig       = pd.read_csv(os.path.join(data_dir, 'val_data_original_ext.csv'))
    test_bear_orig = pd.read_csv(os.path.join(data_dir, 'test_bear_data_original_ext.csv'))
    test_bull_orig = pd.read_csv(os.path.join(data_dir, 'test_bull_data_original_ext.csv'))
    test_bear_idx = pd.read_csv(os.path.join(data_dir, 'test_bear_index_ext.csv'))
    test_bull_idx = pd.read_csv(os.path.join(data_dir, 'test_bull_index_ext.csv'))

    for df in [train_df, val_df, test_bear_df, test_bull_df,
               train_orig, val_orig, test_bear_orig, test_bull_orig,
               test_bear_idx, test_bull_idx]:
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
    tickers = sorted(train_df['ticker'].unique())

    print(f"✅ Loaded extended data")
    print(f"   Stocks: {n_stocks}, Features: {n_features}")
    print(f"   Train: {train_df['date'].nunique()} days "
          f"(was 1210 in original — now {train_df['date'].nunique()})")
    print(f"   Val:   {val_df['date'].nunique()} days")
    print(f"   Bear:  {test_bear_df['date'].nunique()} days")
    print(f"   Bull:  {test_bull_df['date'].nunique()} days")

    # Per-regime breakdown so we can see the change vs original
    train_regimes = (train_df[['date','regime']].drop_duplicates()
                     .groupby('regime').size().to_dict())
    name = {0:'bull', 1:'bear', 2:'sideways'}
    print(f"\n   Train regime breakdown:")
    for rid, cnt in sorted(train_regimes.items()):
        rid_int = int(rid)
        pct = cnt / sum(train_regimes.values()) * 100
        print(f"     {name.get(rid_int,'?'):<10} {cnt:>5} days ({pct:.1f}%)")

    # ---- Create environments — cash floor DISABLED everywhere ----
    print("\n--- Creating environments (ablation_mode='no_floor') ---\n")
    train_env     = PortfolioEnv(df=train_df,     initial_balance=1_000_000,
                                 ablation_mode='no_floor')
    val_env       = PortfolioEnv(df=val_df,       initial_balance=1_000_000,
                                 ablation_mode='no_floor')
    test_bear_env = PortfolioEnv(df=test_bear_df, initial_balance=1_000_000,
                                 ablation_mode='no_floor')
    test_bull_env = PortfolioEnv(df=test_bull_df, initial_balance=1_000_000,
                                 ablation_mode='no_floor')

    # ---- Build PPO with same hyperparameters as Phase 5 ----
    print("\n--- Building PPO model (Phase 5 hyperparameters) ---\n")
    policy_kwargs = dict(
        features_extractor_class=AttentionPortfolioExtractor,
        features_extractor_kwargs=dict(
            n_stocks=n_stocks, n_features=n_features,
            embed_dim=64, n_heads=4, features_dim=128,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
    hp_path = os.path.join(project_dir, 'results', 'best_hyperparams.json')
    with open(hp_path) as f:
        hp = json.load(f)
    print(f"  Hyperparameters: lr={hp['learning_rate']}, "
          f"n_steps={hp['n_steps']}, batch={hp['batch_size']}, "
          f"ent_coef={hp['ent_coef']}")

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
    print("✅ Model configured (same architecture as Phase 5 baseline)")

    # ---- Train 1M steps with val checkpointing ----
    print("\n--- Training 1,000,000 steps ---")
    print("Expected time: ~50-90 min on CPU\n")

    detailed = DetailedCallback(eval_freq=50_000, total_timesteps=1_000_000)
    best_path = os.path.join(saved_dir, 'ppo_best_extended_nofloor')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path, eval_freq=25_000,
    )
    model.learn(total_timesteps=1_000_000,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, 'ppo_final_extended_nofloor')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} at step {val_cb.best_step:,}")
    print(f"   Best  → {best_path}.zip")
    print(f"   Final → {final_path}.zip")

    # Save val + reward histories for thesis training-curve plot
    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({
            'history': val_cb.history,
            'best_step': val_cb.best_step,
            'best_val_sharpe': val_cb.best_val_sharpe,
        }, f, indent=2)
    with open(os.path.join(results_dir, 'training_rewards.json'), 'w') as f:
        json.dump(detailed.rewards_history, f, indent=2)

    # ---- Reload best model and evaluate ----
    print("\n--- Loading BEST model for evaluation ---")
    model = PPO.load(best_path, env=train_env)

    agent_results = {}
    datasets = {
        'train':     (train_env,     'Training (2007-2020)'),
        'val':       (val_env,       'Validation (2021)'),
        'test_bear': (test_bear_env, 'Test Bear (2022)'),
        'test_bull': (test_bull_env, 'Test Bull (2024)'),
    }
    for name, (env, desc) in datasets.items():
        print(f"  {desc}...")
        history = evaluate_agent(model, env)
        m = calculate_metrics(history)
        agent_results[name] = {'metrics': m, 'history': history, 'description': desc}
        print(f"    Return: {m['cumulative_return']}% | "
              f"Sharpe: {m['sharpe_ratio']} | MDD: {m['max_drawdown']}%")

    # ---- Benchmarks ----
    benchmarks = {}
    for name, orig_df, desc in [
        ('train',     train_orig,     'Train 2007-2020'),
        ('val',       val_orig,       'Val 2021'),
        ('test_bear', test_bear_orig, 'Bear 2022'),
        ('test_bull', test_bull_orig, 'Bull 2024'),
    ]:
        h = compute_benchmark_equal_weight(orig_df)
        benchmarks[f'equal_{name}'] = {
            'metrics': calculate_metrics(h), 'history': h, 'description': desc,
        }

    for name, idx_df, desc in [
        ('test_bear', test_bear_idx, 'Bear 2022 DJI'),
        ('test_bull', test_bull_idx, 'Bull 2024 DJI'),
    ]:
        h = compute_benchmark_dji(idx_df)
        benchmarks[f'dji_{name}'] = {
            'metrics': calculate_metrics(h), 'history': h, 'description': desc,
        }

    # ---- Save results ----
    save_data = {'method': 'extended_data_baseline_nofloor',
                 'best_val_sharpe': val_cb.best_val_sharpe,
                 'best_step': val_cb.best_step}
    for k, v in agent_results.items():
        save_data[f'agent_{k}'] = v['metrics']
    for k, v in benchmarks.items():
        save_data[f'benchmark_{k}'] = v['metrics']
    results_path = os.path.join(results_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✅ Results → {results_path}")

    # ---- Comparison table vs all prior systems + extended-with-floor ----
    base_path = os.path.join(project_dir, 'results', 'final_results.json')
    if os.path.exists(base_path):
        with open(base_path) as f:
            base = json.load(f)
        print("\n" + "=" * 86)
        print("   EXTENDED NO-FLOOR vs ALL PRIOR SYSTEMS")
        print("=" * 86)
        print(f"  {'Period':<12} {'System':<32} {'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*82}")

        # Load v1, v2, ext+floor
        v1_path = os.path.join(project_dir, 'results', 'multi_agent', 'final_results_hard.json')
        v2_path = os.path.join(project_dir, 'results', 'risk_personality', 'final_results_soft.json')
        ef_path = os.path.join(project_dir, 'results', 'extended', 'final_results.json')
        v1 = json.load(open(v1_path)) if os.path.exists(v1_path) else None
        v2 = json.load(open(v2_path)) if os.path.exists(v2_path) else None
        ef = json.load(open(ef_path)) if os.path.exists(ef_path) else None

        for period in ['val', 'test_bear', 'test_bull']:
            bk = f'agent_{period}'
            ext_m = agent_results[period]['metrics']

            # Phase 5 baseline
            if bk in base:
                b = base[bk]
                print(f"  {period:<12} {'Phase 5 baseline':<32} "
                      f"{b['cumulative_return']:>9.2f} "
                      f"{b['sharpe_ratio']:>8.4f} "
                      f"{b['max_drawdown']:>9.2f}")
            # Multi-agent v1
            if v1 and v1['periods'].get(period):
                v = v1['periods'][period]
                print(f"  {period:<12} {'Multi-agent v1 (regime)':<32} "
                      f"{v['cumulative_return']:>9.2f} "
                      f"{v['sharpe_ratio']:>8.4f} "
                      f"{v['max_drawdown']:>9.2f}")
            # Multi-agent v2
            if v2 and v2['periods'].get(period):
                v = v2['periods'][period]
                print(f"  {period:<12} {'Multi-agent v2 (risk pers.)':<32} "
                      f"{v['cumulative_return']:>9.2f} "
                      f"{v['sharpe_ratio']:>8.4f} "
                      f"{v['max_drawdown']:>9.2f}")
            # Extended WITH floor
            if ef and bk in ef:
                f5 = ef[bk]
                print(f"  {period:<12} {'Extended + floor':<32} "
                      f"{f5['cumulative_return']:>9.2f} "
                      f"{f5['sharpe_ratio']:>8.4f} "
                      f"{f5['max_drawdown']:>9.2f}")
            # Extended NO floor (this run)
            print(f"  {period:<12} {'Extended NO FLOOR (this)':<32} "
                  f"{ext_m['cumulative_return']:>9.2f} "
                  f"{ext_m['sharpe_ratio']:>8.4f} "
                  f"{ext_m['max_drawdown']:>9.2f}")
            print()

    # ---- Charts ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, period, title in [
        (axes[0], 'test_bear', 'Bear 2022'),
        (axes[1], 'test_bull', 'Bull 2024'),
    ]:
        ah = agent_results[period]['history']
        eh = benchmarks[f'equal_{period}']['history']
        dh = benchmarks[f'dji_{period}']['history']
        ax.plot(pd.to_datetime(ah['date']), ah['value']/1e6,
                label='Extended baseline', color='darkgreen', linewidth=2)
        ax.plot(pd.to_datetime(eh['date']), eh['value']/1e6,
                label='Equal Weight', color='orange', linewidth=1.5, linestyle='--')
        ax.plot(pd.to_datetime(dh['date']), dh['value']/1e6,
                label='DJI Index', color='gray', linewidth=1.5, linestyle=':')
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='$1M')
        m = agent_results[period]['metrics']
        ax.set_title(f'{title}\nReturn: {m["cumulative_return"]}% | '
                     f'Sharpe: {m["sharpe_ratio"]} | MDD: {m["max_drawdown"]}%')
        ax.set_ylabel('Portfolio Value ($M)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.suptitle('Extended-Data Baseline (trained 2007-2020)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_performance.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    if detailed.rewards_history:
        steps   = [x[0] for x in detailed.rewards_history]
        rewards = [x[1] for x in detailed.rewards_history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rewards, color='darkgreen', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Extended-baseline training curve (1M steps)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 60)
    print("   EXTENDED-DATA TRAINING COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
