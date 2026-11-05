# ============================================
# Attention explainability — Extended Baseline
# ============================================
# Loads ppo_best_extended.zip (Extended + Floor, the final thesis system)
# and re-evaluates it on val 2021, bear 2022, and bull 2024 — capturing
# the multi-head attention weights at every step.
#
# Produces:
#   1. Bar charts: top-15 stocks by average attention weight per period
#   2. Regime-conditional attention: how attention shifts in bear vs
#      sideways vs bull regimes WITHIN each test set (the most thesis-
#      relevant chart — shows the agent's attention RESPONDS to regime).
#   3. Attention-vs-portfolio-weight scatter: does the agent actually
#      bet on the stocks it attends to?
#
# Saves to results/extended/attention_*.png
# ============================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv

# We need the AttentionPortfolioExtractor class to be loadable
# (PPO.load reconstructs the policy using this class)
from train_final_extended import AttentionPortfolioExtractor  # noqa: F401


REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'sideways'}


def evaluate_with_attention(model, env):
    """
    Run one full episode, capturing attention weights at every step
    along with the regime label and chosen portfolio weights.

    Returns dict with arrays:
        regimes:   (n_steps,)         — regime id per step
        attention: (n_steps, 30, 30)  — attention[i,j] = how much
                   stock i attended to stock j (averaged over heads)
        weights:   (n_steps, 31)      — agent's chosen portfolio weights
                   (30 stocks + cash)
    """
    obs, info = env.reset()
    done = False
    regimes, attentions, weights, dates = [], [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

        ext = model.policy.features_extractor
        if hasattr(ext, 'last_attention_weights') and ext.last_attention_weights is not None:
            # SB3 batch dim is 1 in eval — take [0]; shape (30, 30)
            attentions.append(ext.last_attention_weights[0])

        regimes.append(int(info.get('regime_id', 0)))
        # snapshot the env's CURRENT portfolio weights for this step
        weights.append(env.portfolio_weights.copy())
        dates.append(info.get('date'))

    return {
        'regimes':   np.asarray(regimes),
        'attention': np.asarray(attentions),
        'weights':   np.asarray(weights),
        'dates':     dates,
    }


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    results_dir = os.path.join(project_dir, 'results', 'extended')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("   Attention extraction — Extended Baseline")
    print("=" * 60)

    # ---- Load model ----
    model_path = os.path.join(saved_dir, 'ppo_best_extended')
    print(f"\nLoading model: {model_path}.zip")
    model = PPO.load(model_path, device='cpu')

    # ---- Load eval data ----
    eval_sets = [
        ('val',       'val_data_ext.csv',       'Validation 2021'),
        ('test_bear', 'test_bear_data_ext.csv', 'Bear 2022'),
        ('test_bull', 'test_bull_data_ext.csv', 'Bull 2024'),
    ]

    period_results = {}
    tickers = None
    for name, fn, _label in eval_sets:
        df = pd.read_csv(os.path.join(data_dir, fn))
        df['date'] = pd.to_datetime(df['date'])
        if tickers is None:
            tickers = sorted(df['ticker'].unique())
        env = PortfolioEnv(df=df, initial_balance=1_000_000)
        print(f"\n  {name}: {df['date'].nunique()} days, "
              f"capturing attention…")
        out = evaluate_with_attention(model, env)
        period_results[name] = out
        # Sanity print
        attn = out['attention']
        if len(attn) == 0:
            print(f"    ⚠ No attention captured (model may not have an "
                  f"AttentionPortfolioExtractor)")
            continue
        # Average over query dim to get "per-stock" attention received
        per_stock_attn = attn.mean(axis=0).mean(axis=0)  # shape (30,)
        regime_counts = pd.Series(out['regimes']).value_counts().to_dict()
        print(f"    Captured {len(attn)} steps  "
              f"regimes={ {REGIME_NAMES.get(k,k): v for k,v in regime_counts.items()} }")

    # ============================================
    # Chart 1: top-15 attention bar charts (one per period)
    # ============================================
    print("\n--- Chart 1: top-15 stocks by attention (per period) ---")
    for name, color, label in [
        ('test_bull', 'steelblue', 'Bull 2024'),
        ('test_bear', 'indianred', 'Bear 2022'),
        ('val',       'darkgreen', 'Val 2021'),
    ]:
        attn = period_results[name]['attention']
        if len(attn) == 0:
            continue
        avg_attn = attn.mean(axis=0).mean(axis=0)  # (30,)
        order = np.argsort(avg_attn)[::-1]
        sorted_tickers = [tickers[i] for i in order]
        sorted_attn    = avg_attn[order]

        fig, ax = plt.subplots(figsize=(11, 6))
        top = 15
        ax.barh(range(top), sorted_attn[:top][::-1], color=color)
        ax.set_yticks(range(top))
        ax.set_yticklabels(sorted_tickers[:top][::-1])
        ax.set_xlabel('Average attention weight (received)')
        ax.set_title(f'Top 15 Stocks by Attention — {label}\n'
                     f'(Extended + Floor model)')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        out_path = os.path.join(results_dir, f'attention_{name}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {out_path}")

    # ============================================
    # Chart 2: regime-conditional attention (the thesis-relevant one)
    # Within bear 2022 AND bull 2024, how does attention DIFFER on
    # bear-labeled days vs sideways-labeled days vs bull-labeled days?
    # ============================================
    def _plot_regime_conditional(period_name, period_label, out_filename):
        d = period_results[period_name]
        if len(d['attention']) == 0:
            return None
        regimes = d['regimes']
        attn    = d['attention']  # (T, 30, 30)
        per_day_attn = attn.mean(axis=1)  # (T, 30) — attention received per stock per day
        cond = {}
        for rid in (0, 1, 2):
            mask = (regimes == rid)
            if mask.sum() > 0:
                cond[rid] = per_day_attn[mask].mean(axis=0)
            else:
                cond[rid] = None

        fig, ax = plt.subplots(figsize=(13, 7))
        x = np.arange(len(tickers))
        bar_w = 0.27
        for i, (rid, color) in enumerate([(0, 'green'), (2, 'orange'), (1, 'red')]):
            if cond[rid] is None:
                continue
            offset = (i - 1) * bar_w
            ax.bar(x + offset, cond[rid], bar_w,
                   label=f"{REGIME_NAMES[rid]} days "
                         f"({(regimes == rid).sum()})",
                   color=color, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=90, fontsize=8)
        ax.set_ylabel('Average attention weight (received)')
        ax.set_title(f'Regime-conditional attention within {period_label}\n'
                     f"(How does the agent's attention shift by regime?)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        out_path = os.path.join(results_dir, out_filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        return out_path

    print("\n--- Chart 2a: regime-conditional attention within Bear 2022 ---")
    p = _plot_regime_conditional('test_bear', 'Bear 2022',
                                 'attention_regime_conditional_bear.png')
    if p: print(f"  ✅ {p}")

    print("\n--- Chart 2b: regime-conditional attention within Bull 2024 ---")
    p = _plot_regime_conditional('test_bull', 'Bull 2024',
                                 'attention_regime_conditional_bull.png')
    if p: print(f"  ✅ {p}")

    # ============================================
    # Chart 3: bear vs bull attention DIFFERENCE
    # Which stocks get MORE attention in bear vs bull?
    # Negative bars = stocks the agent attends to MORE in bear (defensive).
    # Positive bars = stocks the agent attends to MORE in bull (aggressive).
    # ============================================
    print("\n--- Chart 3: bear vs bull attention difference ---")
    bull_attn = period_results['test_bull']['attention']
    bear_attn = period_results['test_bear']['attention']
    if len(bull_attn) > 0 and len(bear_attn) > 0:
        bull_avg = bull_attn.mean(axis=0).mean(axis=0)
        bear_avg = bear_attn.mean(axis=0).mean(axis=0)
        diff = bull_avg - bear_avg  # +ve → more attended in bull
        order = np.argsort(diff)
        sorted_tickers = [tickers[i] for i in order]
        sorted_diff    = diff[order]

        fig, ax = plt.subplots(figsize=(13, 6))
        colors = ['indianred' if d < 0 else 'steelblue' for d in sorted_diff]
        ax.barh(range(len(sorted_tickers)), sorted_diff, color=colors)
        ax.set_yticks(range(len(sorted_tickers)))
        ax.set_yticklabels(sorted_tickers, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Bull 2024 attention − Bear 2022 attention')
        ax.set_title('Stocks the agent attends to MORE in BULL (right) vs BEAR (left)\n'
                     '(Extended + Floor model)')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        out_path = os.path.join(results_dir, 'attention_bull_vs_bear_diff.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {out_path}")

    # ============================================
    # Chart 4: attention vs portfolio weight scatter
    # Does the agent invest in the stocks it attends to?
    # ============================================
    print("\n--- Chart 4: attention vs portfolio weight (does it invest where it looks?) ---")
    for name, label in [('test_bull', 'Bull 2024'), ('test_bear', 'Bear 2022')]:
        d = period_results[name]
        if len(d['attention']) == 0:
            continue
        avg_attn = d['attention'].mean(axis=0).mean(axis=0)  # (30,)
        avg_w    = d['weights'][:, :30].mean(axis=0)         # (30,) — drop cash
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(avg_attn, avg_w, color='purple', s=60, alpha=0.7)
        for i, t in enumerate(tickers):
            ax.annotate(t, (avg_attn[i], avg_w[i]), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
        # Linear fit
        if len(avg_attn) > 1:
            slope, intercept = np.polyfit(avg_attn, avg_w, 1)
            xs = np.linspace(avg_attn.min(), avg_attn.max(), 50)
            ax.plot(xs, slope*xs + intercept, color='gray',
                    linestyle='--', alpha=0.5, label=f'fit: slope={slope:.2f}')
            ax.legend()
        ax.set_xlabel('Average attention received')
        ax.set_ylabel('Average portfolio weight')
        ax.set_title(f'Attention vs portfolio weight — {label}\n'
                     f'(Does the agent BET on what it ATTENDS to?)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(results_dir, f'attention_vs_weight_{name}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {out_path}")

    # ============================================
    # Save the raw attention data so you can explore later
    # ============================================
    summary = {}
    for name, d in period_results.items():
        if len(d['attention']) == 0:
            continue
        summary[name] = {
            'top_15_stocks_by_attention':
                [tickers[i] for i in
                 np.argsort(d['attention'].mean(axis=0).mean(axis=0))[::-1][:15]],
            'regime_day_counts': {
                REGIME_NAMES[k]: int(np.sum(d['regimes'] == k))
                for k in (0, 1, 2)
            },
            'n_steps_captured': int(len(d['attention'])),
        }
    with open(os.path.join(results_dir, 'attention_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary → {os.path.join(results_dir, 'attention_summary.json')}")

    print("\n" + "=" * 60)
    print("   Attention extraction COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
