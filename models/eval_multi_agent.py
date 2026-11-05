# ============================================
# MULTI-AGENT: Evaluate the regime-routed system
# ============================================
# Loads the three trained specialists, builds a RegimeController,
# walks through each test set (val, bear, bull) using the controller
# to pick the active specialist per day. Compares to the frozen
# Phase 5 single-agent baseline.
#
# Outputs:
#   results/multi_agent/final_results.json
#   results/multi_agent/specialist_usage.json
#   results/multi_agent/comparison.png
# ============================================

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from regime_controller import RegimeController, REGIME_NAMES


# ============================================
# METRICS — identical formula to train_final.py
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


# ============================================
# CORE: walk through env using controller
# ============================================

def evaluate_multi_agent(controller, env, mode='hard', prob_lookup=None):
    """
    Run one full episode in `env` using `controller`.

    mode='hard': pick one specialist per day based on argmax regime label
    mode='soft': blend all 3 specialists weighted by HMM probabilities

    For soft mode, prob_lookup must be a dict {pd.Timestamp -> [p_bull, p_bear, p_side]}.
    The env exposes today's date as info['date'] (string), which we use to look up probs.

    Returns (history_df, specialist_usage_pct_dict).
    """
    obs, info = env.reset()
    controller.reset_counts()
    done = False
    while not done:
        if mode == 'hard':
            regime_id = info.get('regime_id', 0)
            action, _ = controller.predict(obs, regime_id, deterministic=True)
        elif mode == 'soft':
            today = pd.to_datetime(info.get('date'))
            if today not in prob_lookup:
                # Defensive fallback — should not happen if probs were merged
                regime_id = info.get('regime_id', 0)
                probs = np.zeros(3, dtype=np.float32)
                probs[regime_id] = 1.0
            else:
                probs = prob_lookup[today]
            action, _ = controller.predict_soft(obs, probs, deterministic=True)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_portfolio_history(), controller.usage_summary()


# ============================================
# MAIN
# ============================================

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate the multi-agent system')
    p.add_argument('--mode', default='hard', choices=['hard', 'soft'],
                   help="hard: argmax regime → one specialist. "
                        "soft: HMM probabilities → weighted blend of all 3.")
    return p.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    # Extended pipeline — reads from the new dirs created by the retrain.
    saved_dir   = os.path.join(base_dir, 'saved', 'multi_agent_ext')
    results_dir = os.path.join(project_dir, 'results', 'multi_agent_ext')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"   MULTI-AGENT EVALUATION (mode={mode})")
    print("=" * 60)

    # ---- Load specialists ----
    print("\n--- Loading specialists ---")
    specialists = {}
    for regime_id, name in REGIME_NAMES.items():
        path = os.path.join(saved_dir, f'specialist_{name}')
        if not os.path.exists(path + '.zip'):
            raise FileNotFoundError(
                f"Missing {path}.zip — run train_multi_agent.py first.")
        specialists[regime_id] = PPO.load(path, device='cpu')
        print(f"  ✅ {name:<8} specialist loaded ({path}.zip)")

    controller = RegimeController(specialists)

    # ---- Load probability lookup if soft mode ----
    prob_lookup = None
    if mode == 'soft':
        prob_path = os.path.join(data_dir, 'regime_probabilities_ext.csv')
        if not os.path.exists(prob_path):
            raise FileNotFoundError(
                f"Need {prob_path} for soft mode. Run "
                f"`python3 data/add_regime_probabilities.py` first.")
        pdf = pd.read_csv(prob_path)
        pdf['date'] = pd.to_datetime(pdf['date'])
        prob_lookup = {
            row['date']: np.array([row['regime_bull_prob'],
                                   row['regime_bear_prob'],
                                   row['regime_sideways_prob']],
                                  dtype=np.float32)
            for _, row in pdf.iterrows()
        }
        print(f"\n  Loaded {len(prob_lookup)} dated probability rows for soft blend")

    # ---- Load eval data ----
    eval_sets = {}
    for name, fn in [('val',       'val_data_ext.csv'),
                     ('test_bear', 'test_bear_data_ext.csv'),
                     ('test_bull', 'test_bull_data_ext.csv')]:
        df = pd.read_csv(os.path.join(data_dir, fn))
        df['date'] = pd.to_datetime(df['date'])
        eval_sets[name] = df

    # ---- Evaluate on each ----
    print("\n--- Evaluating multi-agent system ---\n")
    results = {}
    for name, df in eval_sets.items():
        print(f"  {name}...")
        env = PortfolioEnv(df=df, initial_balance=1_000_000)
        history, usage = evaluate_multi_agent(
            controller, env, mode=mode, prob_lookup=prob_lookup,
        )
        metrics = calculate_metrics(history)
        results[name] = {
            'metrics': metrics,
            'specialist_usage_pct': usage,
            'history': history,
        }
        print(f"     Return: {metrics['cumulative_return']:>7.2f}% | "
              f"Sharpe: {metrics['sharpe_ratio']:>6.4f} | "
              f"MDD: {metrics['max_drawdown']:>6.2f}%")
        print(f"     Specialist usage: "
              f"bull={usage[0]}%  bear={usage[1]}%  sideways={usage[2]}%")

    # ---- Save results JSON ----
    save_data = {
        'method': f'multi_agent_{mode}',
        'periods': {},
        'specialist_usage_pct': {},
    }
    for k, v in results.items():
        save_data['periods'][k] = v['metrics']
        save_data['specialist_usage_pct'][k] = v['specialist_usage_pct']
    # Mode-specific filename so hard and soft don't overwrite each other
    save_path = os.path.join(results_dir, f'final_results_{mode}.json')
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✅ Results → {save_path}")

    # ---- Compare to FULL EXTENDED baseline (not Phase 5) ----
    baseline_path = os.path.join(project_dir, 'results', 'extended', 'final_results.json')
    if not os.path.exists(baseline_path):
        print("\n(no FULL extended baseline found — skipping comparison)")
        return

    with open(baseline_path) as f:
        base = json.load(f)

    print("\n" + "=" * 76)
    print(f"   MULTI-AGENT ({mode}) vs FULL EXTENDED BASELINE")
    print("=" * 76)
    print(f"  {'Period':<12} {'Method':<24} {'Return%':>9} "
          f"{'Sharpe':>8} {'MDD%':>9}")
    print(f"  {'-'*70}")
    label_multi = f'Multi-agent ({mode})'
    for period in ['val', 'test_bear', 'test_bull']:
        bk = f'agent_{period}'
        if bk not in base:
            continue
        b = base[bk]
        m = results[period]['metrics']
        print(f"  {period:<12} {'FULL extended (single)':<24} "
              f"{b['cumulative_return']:>9.2f} "
              f"{b['sharpe_ratio']:>8.4f} "
              f"{b['max_drawdown']:>9.2f}")
        print(f"  {period:<12} {label_multi:<24} "
              f"{m['cumulative_return']:>9.2f} "
              f"{m['sharpe_ratio']:>8.4f} "
              f"{m['max_drawdown']:>9.2f}")

        # Δ row
        d_ret  = m['cumulative_return'] - b['cumulative_return']
        d_shp  = m['sharpe_ratio']      - b['sharpe_ratio']
        d_mdd  = m['max_drawdown']      - b['max_drawdown']
        sign = lambda x, suffix='': f"{'+' if x >= 0 else ''}{x:.2f}{suffix}"
        print(f"  {'':12} {'  Δ (multi - base)':<24} "
              f"{sign(d_ret):>9} "
              f"{sign(d_shp):>8} "
              f"{sign(d_mdd):>9}")
        print()

    # Also print hard-vs-soft comparison if both result files exist
    other_mode = 'soft' if mode == 'hard' else 'hard'
    other_path = os.path.join(results_dir, f'final_results_{other_mode}.json')
    if os.path.exists(other_path):
        with open(other_path) as f:
            other = json.load(f)
        print("=" * 76)
        print(f"   MULTI-AGENT: {mode} vs {other_mode}")
        print("=" * 76)
        print(f"  {'Period':<12} {'Method':<24} {'Return%':>9} "
              f"{'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*70}")
        for period in ['val', 'test_bear', 'test_bull']:
            cur = results[period]['metrics']
            oth = other['periods'].get(period)
            if oth is None:
                continue
            print(f"  {period:<12} {f'Multi-agent ({other_mode})':<24} "
                  f"{oth['cumulative_return']:>9.2f} "
                  f"{oth['sharpe_ratio']:>8.4f} "
                  f"{oth['max_drawdown']:>9.2f}")
            print(f"  {period:<12} {f'Multi-agent ({mode})':<24} "
                  f"{cur['cumulative_return']:>9.2f} "
                  f"{cur['sharpe_ratio']:>8.4f} "
                  f"{cur['max_drawdown']:>9.2f}")
            print()

    # ---- Plot: portfolio value curves on bear & bull ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, period, title in [
        (axes[0], 'test_bear', 'Bear Market 2022'),
        (axes[1], 'test_bull', 'Bull Market 2024'),
    ]:
        h = results[period]['history']
        ax.plot(pd.to_datetime(h['date']), h['value']/1e6,
                label='Multi-agent', color='green', linewidth=2)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3,
                   label='Initial $1M')
        m = results[period]['metrics']
        ax.set_title(f'{title}\nReturn: {m["cumulative_return"]}% | '
                     f'Sharpe: {m["sharpe_ratio"]} | MDD: {m["max_drawdown"]}%')
        ax.set_ylabel('Portfolio Value ($M)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.suptitle('Multi-Agent Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(results_dir, 'comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Chart → {chart_path}")

    print("\n" + "=" * 60)
    print("   MULTI-AGENT EVALUATION COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
