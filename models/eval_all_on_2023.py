"""
Evaluate every saved model on the 2023 test period (recovery year).

Covers:
  - FULL extended baseline
  - Vanilla DRL baseline
  - All component ablations (no_dsr, no_attention, no_regime_λ sweep)
  - no_floor (eval-time only — uses FULL model with ablation_mode='no_floor')
  - MTL v2 (the more honest multi-agent variant)
  - Multi-agent specialists (soft routing)
  - Risk-personality system   (soft routing)
  - Benchmarks: DJI buy-and-hold, equal-weight portfolio

Reads:
  models/saved/...                   (every saved model)
  data/test_2023_data_ext.csv         (built by data/build_test_2023.py)
  data/test_2023_index_ext.csv        (DJI for benchmark)
  data/regime_probabilities_ext.csv   (for soft routing in MA / RP)

Writes:
  results/test_2023_summary.json      — every system's metrics for 2023
  results/test_2023_table.md          — markdown table for thesis
  Prints a summary table.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from regime_controller import RegimeController


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
SAVED_DIR   = os.path.join(PROJECT_DIR, 'models', 'saved')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')


# -------- Metrics ----------

def calc_metrics(history, initial_balance=1_000_000):
    v = history['value'].values if hasattr(history, 'values') else np.asarray(history)
    if len(v) < 2:
        return {'cumulative_return': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'annualized_volatility': 0.0}
    final = v[-1]
    cum = (final - initial_balance) / initial_balance * 100
    dr = np.diff(v) / v[:-1]
    n = len(v)
    ann_ret = ((final / initial_balance) ** (252 / n) - 1) * 100
    ann_vol = float(np.std(dr) * np.sqrt(252) * 100)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(np.min((v - peak) / peak) * 100)
    return {'cumulative_return': round(cum, 2),
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown': round(mdd, 2),
            'annualized_volatility': round(ann_vol, 2)}


# -------- Single-agent evaluation ----------

def eval_single_agent(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()


def eval_simple_model(label, model_path, df_2023, env_kwargs):
    """Load + evaluate a single-agent model."""
    print(f"\n  → {label}")
    if not os.path.exists(model_path + '.zip'):
        print(f"    SKIP — missing: {model_path}.zip")
        return None
    model = PPO.load(model_path, device='cpu')
    env = PortfolioEnv(df=df_2023, initial_balance=1_000_000, **env_kwargs)
    history = eval_single_agent(model, env)
    m = calc_metrics(history)
    print(f"    Return={m['cumulative_return']:>7.2f}%  "
          f"Sharpe={m['sharpe_ratio']:>7.4f}  "
          f"MDD={m['max_drawdown']:>7.2f}%")
    return m


# -------- Multi-agent / risk-personality evaluation ----------

def eval_controller_system(label, regime_to_name_zip,
                           df_2023, prob_lookup, mode='soft'):
    """Load 3 agents, route via HMM, evaluate."""
    print(f"\n  → {label} ({mode})")
    specialists = {}
    for regime_id, name in regime_to_name_zip.items():
        path = os.path.join(SAVED_DIR, name)
        if not os.path.exists(path + '.zip'):
            print(f"    SKIP — missing: {path}.zip")
            return None
        specialists[regime_id] = PPO.load(path, device='cpu')
    controller = RegimeController(specialists)
    env = PortfolioEnv(df=df_2023, initial_balance=1_000_000)
    obs, info = env.reset()
    done = False
    while not done:
        today = pd.Timestamp(env.dates[env.current_step])
        if mode == 'hard':
            regime_id = info.get('regime_id', 0)
            action, _ = controller.predict(obs, regime_id, deterministic=True)
        else:  # soft
            if today not in prob_lookup:
                regime_id = info.get('regime_id', 0)
                probs = np.zeros(3, dtype=np.float32)
                probs[regime_id] = 1.0
            else:
                probs = prob_lookup[today]
            action, _ = controller.predict_soft(obs, probs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    history = env.get_portfolio_history()
    m = calc_metrics(history)
    print(f"    Return={m['cumulative_return']:>7.2f}%  "
          f"Sharpe={m['sharpe_ratio']:>7.4f}  "
          f"MDD={m['max_drawdown']:>7.2f}%")
    return m


# -------- Benchmarks ----------

def dji_buy_hold(idx_2023_df, initial_balance=1_000_000):
    start_close = idx_2023_df['close'].iloc[0]
    values = idx_2023_df['close'].values / start_close * initial_balance
    return calc_metrics(values, initial_balance)


def equal_weight_daily_rebal(df_2023, initial_balance=1_000_000):
    """1/30 across stocks, rebalanced daily. Uses raw daily_return column."""
    daily = (df_2023.groupby('date')['daily_return'].mean().sort_index())
    rets = daily.values
    values = np.empty(len(rets))
    values[0] = initial_balance
    for i in range(1, len(rets)):
        values[i] = values[i - 1] * (1.0 + rets[i])
    return calc_metrics(values, initial_balance)


# -------- Main ----------

def main():
    print("=" * 78)
    print("   2023 test eval — every saved model + benchmarks")
    print("=" * 78)

    # ---- Load 2023 data ----
    df_2023 = pd.read_csv(os.path.join(DATA_DIR, 'test_2023_data_ext.csv'))
    df_2023['date'] = pd.to_datetime(df_2023['date'])
    idx_2023 = pd.read_csv(os.path.join(DATA_DIR, 'test_2023_index_ext.csv'))
    idx_2023['date'] = pd.to_datetime(idx_2023['date'])
    print(f"\n  Test period: {df_2023['date'].min().date()} → "
          f"{df_2023['date'].max().date()}  "
          f"({df_2023['date'].nunique()} trading days)")

    # ---- HMM probability lookup (for soft routing) ----
    probs_df = pd.read_csv(os.path.join(DATA_DIR, 'regime_probabilities_ext.csv'))
    probs_df['date'] = pd.to_datetime(probs_df['date'])
    prob_lookup = {
        row['date']: np.array([row['regime_bull_prob'],
                               row['regime_bear_prob'],
                               row['regime_sideways_prob']], dtype=np.float32)
        for _, row in probs_df.iterrows()
    }

    results = {}

    # ---- Benchmarks ----
    print("\n--- Benchmarks ---")
    print("  → DJI buy-and-hold")
    m = dji_buy_hold(idx_2023)
    print(f"    Return={m['cumulative_return']:>7.2f}%  "
          f"Sharpe={m['sharpe_ratio']:>7.4f}  "
          f"MDD={m['max_drawdown']:>7.2f}%")
    results['BENCHMARK: DJI'] = m
    print("  → Equal-weight daily rebal")
    m = equal_weight_daily_rebal(df_2023)
    print(f"    Return={m['cumulative_return']:>7.2f}%  "
          f"Sharpe={m['sharpe_ratio']:>7.4f}  "
          f"MDD={m['max_drawdown']:>7.2f}%")
    results['BENCHMARK: EW'] = m

    # ---- Single-agent models ----
    print("\n--- Single-agent models ---")

    # FULL extended
    m = eval_simple_model(
        'FULL (baseline)',
        os.path.join(SAVED_DIR, 'ppo_best_extended'),
        df_2023, env_kwargs={})
    if m: results['FULL (baseline)'] = m

    # Vanilla (regime-blind, no floor, no DSR, λ=0, MLP)
    m = eval_simple_model(
        'VANILLA',
        os.path.join(SAVED_DIR, 'ppo_vanilla_extended'),
        df_2023, env_kwargs={'ablation_mode': 'vanilla',
                             'risk_lambda_fixed': 0.0,
                             'include_regime': False})
    if m: results['VANILLA'] = m

    # no_floor (eval-time only — uses FULL model with floor off)
    m = eval_simple_model(
        'no_floor (eval)',
        os.path.join(SAVED_DIR, 'ppo_best_extended'),
        df_2023, env_kwargs={'ablation_mode': 'no_floor'})
    if m: results['no_floor (eval)'] = m

    # no_dsr ablation
    m = eval_simple_model(
        'no_dsr',
        os.path.join(SAVED_DIR, 'ppo_ablation_ext_no_dsr'),
        df_2023, env_kwargs={'ablation_mode': 'no_dsr'})
    if m: results['no_dsr'] = m

    # no_attention ablation
    m = eval_simple_model(
        'no_attention',
        os.path.join(SAVED_DIR, 'ppo_ablation_ext_no_attention'),
        df_2023, env_kwargs={})
    if m: results['no_attention'] = m

    # no_regime_lambda variants (0.1, 0.3, 0.5)
    for tag, lam in [('0p1', 0.1), ('0p3', 0.3), ('0p5', 0.5)]:
        m = eval_simple_model(
            f'no_regime_λ={lam}',
            os.path.join(SAVED_DIR, f'ppo_ablation_ext_no_regime_lambda_{tag}'),
            df_2023, env_kwargs={'risk_lambda_fixed': lam})
        if m: results[f'no_regime_λ={lam}'] = m

    # MTL v2
    try:
        from mtl_components_v2 import AttentionMTLExtractorV2
        custom_obj = {'features_extractor_class': AttentionMTLExtractorV2}
    except ImportError:
        custom_obj = {}
    print(f"\n  → MTL v2")
    mtl_path = os.path.join(SAVED_DIR, 'ppo_mtl_v2_extended')
    if os.path.exists(mtl_path + '.zip'):
        model = PPO.load(mtl_path, device='cpu', custom_objects=custom_obj)
        env = PortfolioEnv(df=df_2023, initial_balance=1_000_000,
                           include_regime_probs=True)
        history = eval_single_agent(model, env)
        m = calc_metrics(history)
        print(f"    Return={m['cumulative_return']:>7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>7.4f}  "
              f"MDD={m['max_drawdown']:>7.2f}%")
        results['MTL v2'] = m

    # ---- Multi-agent specialists (soft routing) ----
    print("\n--- Controller systems ---")
    m = eval_controller_system(
        'Multi-agent specialists',
        {0: 'multi_agent_ext/specialist_bull',
         1: 'multi_agent_ext/specialist_bear',
         2: 'multi_agent_ext/specialist_sideways'},
        df_2023, prob_lookup, mode='soft')
    if m: results['Multi-agent (soft)'] = m

    # ---- Risk personalities (soft routing) ----
    m = eval_controller_system(
        'Risk personalities',
        {0: 'risk_personality_ext/personality_aggressive',
         1: 'risk_personality_ext/personality_conservative',
         2: 'risk_personality_ext/personality_balanced'},
        df_2023, prob_lookup, mode='soft')
    if m: results['Risk-personality (soft)'] = m

    # ---- Save JSON ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, 'test_2023_summary.json')
    with open(json_path, 'w') as f:
        json.dump({'test_period': '2023', 'days': int(df_2023['date'].nunique()),
                   'results': results}, f, indent=2)

    # ---- Markdown table ----
    md_path = os.path.join(RESULTS_DIR, 'test_2023_table.md')
    md = ['# 2023 test period — all systems\n',
          f'250 trading days; DJI +13.74%. Recovery year between bear-2022 and bull-2024.\n',
          '| System | Cumulative Return (%) | Sharpe | Max Drawdown (%) |',
          '|---|---:|---:|---:|']
    # Sort by Sharpe descending for legibility
    for k, m in sorted(results.items(),
                       key=lambda kv: kv[1]['sharpe_ratio'], reverse=True):
        md.append(f"| {k} | {m['cumulative_return']:.2f} | "
                  f"{m['sharpe_ratio']:.4f} | {m['max_drawdown']:.2f} |")
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')

    # ---- Console summary, sorted by Sharpe ----
    print("\n" + "=" * 78)
    print(f"   2023 RESULTS — sorted by Sharpe")
    print("=" * 78)
    print(f"  {'System':<28} {'Return %':>10} {'Sharpe':>10} {'MDD %':>10}")
    print("  " + "-" * 64)
    for k, m in sorted(results.items(),
                       key=lambda kv: kv[1]['sharpe_ratio'], reverse=True):
        flag = ''
        if k == 'FULL (baseline)':       flag = ' ★ FULL'
        elif k.startswith('BENCHMARK'):  flag = ' ◇'
        print(f"  {k:<28} {m['cumulative_return']:>10.2f} "
              f"{m['sharpe_ratio']:>10.4f} {m['max_drawdown']:>10.2f}{flag}")
    print(f"\n  ✅ JSON → {json_path}")
    print(f"  ✅ Markdown table → {md_path}\n")


if __name__ == '__main__':
    main()
