# ============================================
# Evaluate the RISK-PERSONALITY multi-agent system (v2)
# ============================================
# Loads the three trained personality agents:
#   personality_aggressive    (λ=0.05) → routed to in BULL regimes
#   personality_balanced      (λ=0.3)  → routed to in SIDEWAYS regimes
#   personality_conservative  (λ=0.8)  → routed to in BEAR regimes
#
# Two modes:
#   hard: regime label argmax → one personality
#   soft: HMM regime probabilities → weighted blend of all 3
#
# Compares results to:
#   1. Phase 5 single-agent baseline
#   2. Multi-agent v1 (regime specialists) — if those results exist
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


# Mapping: regime_id → personality name (for the controller)
#   regime_id 0 (bull)     → aggressive    (low risk aversion, ride upside)
#   regime_id 1 (bear)     → conservative  (high risk aversion, defend)
#   regime_id 2 (sideways) → balanced      (moderate)
REGIME_TO_PERSONALITY = {
    0: 'aggressive',
    1: 'conservative',
    2: 'balanced',
}


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


def evaluate_system(controller, env, mode='hard', prob_lookup=None):
    """One full episode. Returns (history_df, usage_pct_dict)."""
    obs, info = env.reset()
    controller.reset_counts()
    done = False
    while not done:
        if mode == 'hard':
            regime_id = info.get('regime_id', 0)
            action, _ = controller.predict(obs, regime_id, deterministic=True)
        else:  # soft
            today = pd.to_datetime(info.get('date'))
            if today not in prob_lookup:
                regime_id = info.get('regime_id', 0)
                probs = np.zeros(3, dtype=np.float32)
                probs[regime_id] = 1.0
            else:
                probs = prob_lookup[today]
            action, _ = controller.predict_soft(obs, probs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_portfolio_history(), controller.usage_summary()


# ============================================
# MAIN
# ============================================

def parse_args():
    p = argparse.ArgumentParser(description='Eval risk-personality multi-agent')
    p.add_argument('--mode', default='soft', choices=['hard', 'soft'],
                   help='Routing mode (default: soft, since each agent has '
                        'a distinct continuous personality)')
    return p.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    # Extended pipeline — reads from the new dirs created by the retrain.
    saved_dir   = os.path.join(base_dir, 'saved', 'risk_personality_ext')
    results_dir = os.path.join(project_dir, 'results', 'risk_personality_ext')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"   MULTI-AGENT v2 (RISK-PERSONALITY) EVALUATION (mode={mode})")
    print("=" * 60)

    # ---- Load personality agents ----
    print("\n--- Loading risk-personality agents ---")
    specialists = {}
    for regime_id, personality in REGIME_TO_PERSONALITY.items():
        path = os.path.join(saved_dir, f'personality_{personality}')
        if not os.path.exists(path + '.zip'):
            raise FileNotFoundError(
                f"Missing {path}.zip — run "
                f"train_risk_personality.py first.")
        specialists[regime_id] = PPO.load(path, device='cpu')
        regime_name = REGIME_NAMES[regime_id]
        print(f"  ✅ regime={regime_name:<8} → personality_{personality} loaded")

    controller = RegimeController(specialists)

    # ---- Load HMM probabilities if soft mode ----
    prob_lookup = None
    if mode == 'soft':
        prob_path = os.path.join(data_dir, 'regime_probabilities_ext.csv')
        if not os.path.exists(prob_path):
            raise FileNotFoundError(
                f"Need {prob_path} — run "
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
        print(f"  Loaded {len(prob_lookup)} probability rows for soft blend")

    # ---- Load eval data ----
    eval_sets = {}
    for name, fn in [('val',       'val_data_ext.csv'),
                     ('test_bear', 'test_bear_data_ext.csv'),
                     ('test_bull', 'test_bull_data_ext.csv')]:
        df = pd.read_csv(os.path.join(data_dir, fn))
        df['date'] = pd.to_datetime(df['date'])
        eval_sets[name] = df

    # ---- Evaluate on each ----
    print("\n--- Evaluating risk-personality system ---\n")
    results = {}
    for name, df in eval_sets.items():
        print(f"  {name}...")
        env = PortfolioEnv(df=df, initial_balance=1_000_000)
        history, usage = evaluate_system(
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
        # Translate regime usage → personality usage for clarity
        agg_pct = usage[0]   # regime_id=0 → aggressive
        cons_pct = usage[1]  # regime_id=1 → conservative
        bal_pct = usage[2]   # regime_id=2 → balanced
        print(f"     Personality usage: aggressive={agg_pct:.1f}%  "
              f"conservative={cons_pct:.1f}%  balanced={bal_pct:.1f}%")

    # ---- Save results JSON ----
    save_data = {
        'method': f'risk_personality_{mode}',
        'lambda_values': {'aggressive': 0.05, 'balanced': 0.3, 'conservative': 0.8},
        'periods': {},
        'personality_usage_pct': {},
    }
    for k, v in results.items():
        save_data['periods'][k] = v['metrics']
        save_data['personality_usage_pct'][k] = {
            'aggressive':   v['specialist_usage_pct'][0],
            'conservative': v['specialist_usage_pct'][1],
            'balanced':     v['specialist_usage_pct'][2],
        }
    save_path = os.path.join(results_dir, f'final_results_{mode}.json')
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✅ Results → {save_path}")

    # ---- Compare to FULL EXTENDED baseline (not Phase 5) ----
    baseline_path = os.path.join(project_dir, 'results', 'extended', 'final_results.json')
    base = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            base = json.load(f)

    print("\n" + "=" * 76)
    print(f"   RISK-PERSONALITY ({mode}) vs FULL EXTENDED BASELINE")
    print("=" * 76)
    print(f"  {'Period':<12} {'Method':<26} {'Return%':>9} "
          f"{'Sharpe':>8} {'MDD%':>9}")
    print(f"  {'-'*72}")
    label_v2 = f'Risk-personality ({mode})'
    for period in ['val', 'test_bear', 'test_bull']:
        m = results[period]['metrics']
        bk = f'agent_{period}'
        if base and bk in base:
            b = base[bk]
            print(f"  {period:<12} {'FULL extended (single)':<26} "
                  f"{b['cumulative_return']:>9.2f} "
                  f"{b['sharpe_ratio']:>8.4f} "
                  f"{b['max_drawdown']:>9.2f}")
        print(f"  {period:<12} {label_v2:<26} "
              f"{m['cumulative_return']:>9.2f} "
              f"{m['sharpe_ratio']:>8.4f} "
              f"{m['max_drawdown']:>9.2f}")
        if base and bk in base:
            b = base[bk]
            d_ret = m['cumulative_return'] - b['cumulative_return']
            d_shp = m['sharpe_ratio']      - b['sharpe_ratio']
            d_mdd = m['max_drawdown']      - b['max_drawdown']
            sign = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
            print(f"  {'':12} {'  Δ (v2 - baseline)':<26} "
                  f"{sign(d_ret):>9} "
                  f"{sign(d_shp):>8} "
                  f"{sign(d_mdd):>9}")
        print()

    # ---- Compare to multi-agent v1 (regime specialists) if available ----
    v1_path = os.path.join(project_dir, 'results', 'multi_agent',
                           f'final_results_{mode}.json')
    if not os.path.exists(v1_path):
        # Fall back to whichever v1 exists
        v1_path = os.path.join(project_dir, 'results', 'multi_agent',
                               'final_results_hard.json')
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1 = json.load(f)
        print("=" * 76)
        v1_mode = v1.get('method', 'multi_agent_hard').replace('multi_agent_', '')
        print(f"   RISK-PERSONALITY ({mode}) vs MULTI-AGENT v1 ({v1_mode}) "
              f"[regime specialists]")
        print("=" * 76)
        print(f"  {'Period':<12} {'Method':<26} {'Return%':>9} "
              f"{'Sharpe':>8} {'MDD%':>9}")
        print(f"  {'-'*72}")
        for period in ['val', 'test_bear', 'test_bull']:
            m = results[period]['metrics']
            v1m = v1['periods'].get(period)
            if v1m is None:
                continue
            print(f"  {period:<12} {f'v1 regime specialists':<26} "
                  f"{v1m['cumulative_return']:>9.2f} "
                  f"{v1m['sharpe_ratio']:>8.4f} "
                  f"{v1m['max_drawdown']:>9.2f}")
            print(f"  {period:<12} {f'v2 risk personalities':<26} "
                  f"{m['cumulative_return']:>9.2f} "
                  f"{m['sharpe_ratio']:>8.4f} "
                  f"{m['max_drawdown']:>9.2f}")
            d_ret = m['cumulative_return'] - v1m['cumulative_return']
            d_shp = m['sharpe_ratio']      - v1m['sharpe_ratio']
            d_mdd = m['max_drawdown']      - v1m['max_drawdown']
            sign = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
            print(f"  {'':12} {'  Δ (v2 - v1)':<26} "
                  f"{sign(d_ret):>9} "
                  f"{sign(d_shp):>8} "
                  f"{sign(d_mdd):>9}")
            print()

    # ---- Plot: portfolio value curves ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, period, title in [
        (axes[0], 'test_bear', 'Bear Market 2022'),
        (axes[1], 'test_bull', 'Bull Market 2024'),
    ]:
        h = results[period]['history']
        ax.plot(pd.to_datetime(h['date']), h['value']/1e6,
                label=f'Risk-personality ({mode})', color='purple', linewidth=2)
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
    plt.suptitle(f'Risk-Personality Multi-Agent v2 ({mode})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(results_dir, f'comparison_{mode}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Chart → {chart_path}")

    print("\n" + "=" * 60)
    print(f"   v2 EVALUATION COMPLETE ({mode}) ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
