"""
Evaluate MARS-Pareto v1 (rule-based soft routing) on both universes.

Loads 3 v6 specialists per universe (defensive/balanced/aggressive) and
runs the soft-routed ensemble through val / bear / 2023 / bull periods.
Compares results to each universe's benchmarks AND to the best single
v6 checkpoint.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_final_extended import calculate_metrics
from mars_pareto_controller import build_controller

# Variant catalogue: name → (description, temperature, blend_mode, use_max_bull)
VARIANTS = {
    'v1':       {'temperature': 1.0, 'blend_mode': 'logits',  'use_max_bull': False, 'description': 'baseline 3-specialist routing (T=1.0)'},
    'v1a':      {'temperature': 0.5, 'blend_mode': 'logits',  'use_max_bull': False, 'description': 'sharpened routing (T=0.5)'},
    'v1a3':     {'temperature': 0.3, 'blend_mode': 'logits',  'use_max_bull': False, 'description': 'very sharp routing (T=0.3)'},
    'v1a7':     {'temperature': 0.7, 'blend_mode': 'logits',  'use_max_bull': False, 'description': 'mildly sharpened routing (T=0.7)'},
    'v1d':      {'temperature': 1.0, 'blend_mode': 'weights', 'use_max_bull': False, 'description': 'post-softmax weight blend (T=1.0)'},
    'v1d_sharp':{'temperature': 0.5, 'blend_mode': 'weights', 'use_max_bull': False, 'description': 'post-softmax weight blend + sharpened (T=0.5)'},
    'v1b':      {'temperature': 1.0, 'blend_mode': 'logits',  'use_max_bull': True,  'description': '4-specialist routing — max-bull for high bull confidence'},
    'v1b_w':    {'temperature': 1.0, 'blend_mode': 'weights', 'use_max_bull': True,  'description': '4-specialist routing + post-softmax blend'},
}

# Drop v3 cols so env exposes v6's 20-feature obs (matches what specialists expect)
V3_DROP = ['r_lag_1_norm','r_lag_5_norm','r_lag_10_norm',
           'r_lag_20_norm','r_lag_40_norm','r_lag_60_norm',
           'is_uptrend','is_sideways','is_downtrend']


def load(p):
    df = pd.read_csv(f'data/{p}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_DROP if c in df.columns])


def run_mars_pareto(env, controller):
    """Step an env using MARS-Pareto controller — regime probs from env."""
    obs, _ = env.reset()
    done = False
    records = []
    controller.reset_log()
    while not done:
        step = env.current_step
        date = env.dates[step]
        # HMM regime probs for today (already in env.regime_prob_array)
        regime_probs = env.regime_prob_array[step].copy()
        # Get blended action from controller
        action, _ = controller.predict(obs, regime_probs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        records.append({
            'date': pd.Timestamp(date).isoformat()[:10],
            'value': float(env.portfolio_value),
            'regime_id': int(env.regime_id_array[step]),
            'bull_prob': float(regime_probs[0]),
            'bear_prob': float(regime_probs[1]),
            'sideways_prob': float(regime_probs[2]),
            'weights': [float(w) for w in env.portfolio_weights],
            'cash_weight': float(env.portfolio_weights[-1]),
            'reward': float(reward),
        })
        done = term or trunc
    return records


def evaluate_universe(universe, period_files, env_kwargs, label='MARS-Pareto',
                       temperature=1.0, blend_mode='logits'):
    print(f"\n{'='*78}")
    print(f"   {label} — {universe}  (T={temperature}, blend={blend_mode})")
    print(f"{'='*78}\n")
    controller = build_controller(universe, temperature=temperature,
                                   blend_mode=blend_mode)
    results = {}
    for period_key, csv_name in period_files.items():
        df = load(csv_name)
        env = PortfolioEnv(df=df, **env_kwargs)
        records = run_mars_pareto(env, controller)
        values = [r['value'] for r in records]
        m = calculate_metrics(pd.DataFrame({'value': values}))
        results[period_key] = m
        usage = controller.usage_summary()
        print(f"  {period_key:<12} "
              f"Ret={m['cumulative_return']:>+7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:>+7.4f}  "
              f"MDD={m['max_drawdown']:>+7.2f}%")
        print(f"              specialist weights — "
              f"defensive={usage['defensive_mean_weight']:.2f}  "
              f"balanced={usage['balanced_mean_weight']:.2f}  "
              f"aggressive={usage['aggressive_mean_weight']:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='v1', choices=list(VARIANTS.keys()),
                        help='Which MARS-Pareto variant to evaluate')
    args = parser.parse_args()
    variant_cfg = VARIANTS[args.variant]
    temperature = variant_cfg['temperature']
    blend_mode  = variant_cfg.get('blend_mode', 'logits')

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base, 'results', f'mars_pareto_{args.variant}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*78}")
    print(f"   MARS-Pareto {args.variant.upper()}: "
          f"{variant_cfg['description']}")
    print(f"{'='*78}")

    # Periods (same for both universes)
    DJI_PERIODS = {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    }
    SP100_PERIODS = {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    }

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: 0.0, 2: 0.0},
        risk_penalties={0: 0.1, 1: 0.5, 2: 0.3},
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
    )

    dji_results   = evaluate_universe('DJI 30', DJI_PERIODS, env_kwargs,
                                       temperature=temperature,
                                       blend_mode=blend_mode)
    sp100_results = evaluate_universe('SP100',  SP100_PERIODS, env_kwargs,
                                       temperature=temperature,
                                       blend_mode=blend_mode)

    # ============================================================
    # COMPARISON TABLES
    # ============================================================
    DJI_BENCH = {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    }
    SP100_BENCH = {
        'val':       {'SP500 BH': 28.79, 'OEX BH': 29.33, 'EW SP100': 33.11},
        'test_bear': {'SP500 BH': -19.95, 'OEX BH': -22.99, 'EW SP100': -9.48},
        'test_2023': {'SP500 BH': 24.73, 'OEX BH': 31.39, 'EW SP100': 21.17},
        'test_bull': {'SP500 BH': 24.54, 'OEX BH': 30.90, 'EW SP100': 18.16},
    }

    DJI_FULL = {'val': 27.20, 'test_bear': -7.36, 'test_2023': 27.16, 'test_bull': 24.35}
    SP100_V6_BEST = {  # v6 step 250k for DJI, step 900k for SP100 (current production)
        'val': 27.34, 'test_bear': -10.06, 'test_2023': 15.09, 'test_bull': 20.34,
    }

    def print_universe_comparison(name, results, bench, references):
        print(f"\n{'='*92}")
        print(f"   MARS-Pareto on {name} — vs benchmarks AND reference single-checkpoint")
        print(f"{'='*92}")
        wins = 0; total = 0
        for p in ['val','test_bear','test_2023','test_bull']:
            print(f"\n  {p}")
            for b, val in bench[p].items():
                print(f"    {b:<14} {val:>+8.2f}")
                total += 1
            for ref_name, ref in references.items():
                if p in ref:
                    print(f"    {ref_name:<14} {ref[p]:>+8.2f}")
            mars = results[p]['cumulative_return']
            mars_sharpe = results[p]['sharpe_ratio']
            print(f"    MARS-Pareto    {mars:>+8.2f}  (Sharpe {mars_sharpe:+.3f})")
            for b, val in bench[p].items():
                better = (mars > val)
                mark = "✅" if better else "❌"
                d = mars - val
                print(f"    vs {b:<14}: {mark}  Δ {d:+.2f}pp")
                if better: wins += 1
        print(f"\n  → MARS-Pareto on {name}: {wins}/{total} benchmark head-to-head wins")
        return wins, total

    dji_wins, dji_total = print_universe_comparison(
        'DJI 30', dji_results, DJI_BENCH, {'DJI FULL': DJI_FULL})
    sp_wins, sp_total = print_universe_comparison(
        'SP100', sp100_results, SP100_BENCH,
        {'v6 main (step 900k)': SP100_V6_BEST})

    print(f"\n{'='*78}")
    print(f"   TOTAL across both universes: "
          f"{dji_wins + sp_wins} / {dji_total + sp_total} wins")
    print(f"{'='*78}")

    # Save
    out = {
        'method':      f'MARS-Pareto {args.variant} ({variant_cfg["description"]})',
        'variant':     args.variant,
        'temperature': temperature,
        'blend_mode':  blend_mode,
        'dji':         {'periods': dji_results, 'wins': f'{dji_wins}/{dji_total}'},
        'sp100':       {'periods': sp100_results, 'wins': f'{sp_wins}/{sp_total}'},
        'total_wins':  f'{dji_wins + sp_wins}/{dji_total + sp_total}',
    }
    with open(os.path.join(out_dir, 'final_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ → {out_dir}/final_results.json")


if __name__ == '__main__':
    main()
