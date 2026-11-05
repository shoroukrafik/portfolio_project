"""
Evaluate MARS-Pareto v1c — cross-variant specialists.

v1c pulls defensive/balanced/aggressive from DIFFERENT training runs so each
role's strength comes from the best model for that regime:

  DJI:    defensive=v1 FULL   balanced=v2   aggressive=v6 step 600k
  SP100:  defensive=v1 FULL   balanced=v6 step 1M   aggressive=v5 aggressive

Specialists have different obs shapes, so we build per-role envs and step
them in lockstep with the same blended action.
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
from mars_pareto_controller import (
    build_controller_v1c,
    V1C_ENV_CONFIG_DJI, V1C_ENV_CONFIG_SP100,
)


def load_full_df(path):
    """Load df WITHOUT dropping any feature columns (v3 cols kept)."""
    df = pd.read_csv(f'data/{path}')
    df['date'] = pd.to_datetime(df['date'])
    return df


def build_role_envs(df, env_kwargs_base, role_configs):
    """Build one env per role, each with role-specific df cols + flags.

    All envs share the same dates, prices, and initial portfolio state, so
    when stepped with the same action they evolve identically.
    """
    envs = {}
    for role, cfg in role_configs.items():
        role_df = df.drop(columns=[c for c in cfg['drop'] if c in df.columns])
        kwargs = dict(env_kwargs_base)
        kwargs.update(cfg['flags'])
        envs[role] = PortfolioEnv(df=role_df, **kwargs)
    return envs


def run_v1c(envs, controller):
    """Step all per-role envs in lockstep using the v1c controller."""
    obs_dict = {role: env.reset()[0] for role, env in envs.items()}
    # Master env is whichever — they all carry the same regime_prob_array.
    master = envs['balanced']
    done = False
    records = []
    controller.reset_log()
    while not done:
        step = master.current_step
        date = master.dates[step]
        regime_probs = master.regime_prob_array[step].copy()
        regime_id    = int(master.regime_id_array[step])

        action, _ = controller.predict(obs_dict, regime_probs, regime_id,
                                       deterministic=True)

        # Step every env with the same blended action. They should evolve
        # identically in portfolio_value (deterministic env + same action).
        new_obs = {}
        term, trunc = False, False
        for role, env in envs.items():
            o, _r, t, tr, _info = env.step(action)
            new_obs[role] = o
            if role == 'balanced':
                term, trunc = t, tr
        obs_dict = new_obs

        records.append({
            'date': pd.Timestamp(date).isoformat()[:10],
            'value': float(master.portfolio_value),
            'regime_id': int(master.regime_id_array[step]),
            'bull_prob': float(regime_probs[0]),
            'bear_prob': float(regime_probs[1]),
            'sideways_prob': float(regime_probs[2]),
        })
        done = term or trunc
    return records


def evaluate_universe(universe, period_files, env_kwargs_base, role_configs,
                       label='MARS-Pareto v1c', temperature=1.0, blend_mode='logits'):
    print(f"\n{'='*78}")
    print(f"   {label} — {universe}  (T={temperature}, blend={blend_mode})")
    print(f"{'='*78}\n")
    controller = build_controller_v1c(universe, temperature=temperature,
                                       blend_mode=blend_mode)
    results = {}
    for period_key, csv_name in period_files.items():
        df = load_full_df(csv_name)
        envs = build_role_envs(df, env_kwargs_base, role_configs)
        records = run_v1c(envs, controller)
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
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Routing temperature (1.0 = HMM probs direct)')
    parser.add_argument('--blend_mode', default='logits',
                        choices=['logits','weights'])
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base, 'results', 'mars_pareto_v1c')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*78}")
    print(f"   MARS-Pareto v1c: cross-variant specialists")
    print(f"   DJI:   defensive=v1 FULL, balanced=v2, aggressive=v6 (600k)")
    print(f"   SP100: defensive=v1 FULL, balanced=v6 (1M), aggressive=v5 agg")
    print(f"{'='*78}")

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

    env_kwargs_base = dict(
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: 0.0, 2: 0.0},
        risk_penalties={0: 0.1, 1: 0.5, 2: 0.3},
        include_regime=True,
    )

    dji_results = evaluate_universe('DJI 30', DJI_PERIODS, env_kwargs_base,
                                     V1C_ENV_CONFIG_DJI,
                                     temperature=args.temperature,
                                     blend_mode=args.blend_mode)
    sp100_results = evaluate_universe('SP100', SP100_PERIODS, env_kwargs_base,
                                       V1C_ENV_CONFIG_SP100,
                                       temperature=args.temperature,
                                       blend_mode=args.blend_mode)

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
    SP100_V6_BEST = {'val': 27.34, 'test_bear': -10.06, 'test_2023': 15.09, 'test_bull': 20.34}

    def print_universe_comparison(name, results, bench, references):
        print(f"\n{'='*92}")
        print(f"   MARS-Pareto v1c on {name} — vs benchmarks AND reference single-checkpoint")
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
            print(f"    v1c            {mars:>+8.2f}  (Sharpe {mars_sharpe:+.3f})")
            for b, val in bench[p].items():
                better = (mars > val)
                mark = "✅" if better else "❌"
                d = mars - val
                print(f"    vs {b:<14}: {mark}  Δ {d:+.2f}pp")
                if better: wins += 1
        print(f"\n  → v1c on {name}: {wins}/{total} benchmark head-to-head wins")
        return wins, total

    dji_wins, dji_total = print_universe_comparison(
        'DJI 30', dji_results, DJI_BENCH, {'DJI FULL': DJI_FULL})
    sp_wins, sp_total = print_universe_comparison(
        'SP100', sp100_results, SP100_BENCH,
        {'v6 main (900k)': SP100_V6_BEST})

    print(f"\n{'='*78}")
    print(f"   TOTAL v1c across both universes: "
          f"{dji_wins + sp_wins} / {dji_total + sp_total} wins")
    print(f"{'='*78}")

    out = {
        'method':      'MARS-Pareto v1c (cross-variant specialists)',
        'variant':     'v1c',
        'temperature': args.temperature,
        'blend_mode':  args.blend_mode,
        'dji':         {'periods': dji_results, 'wins': f'{dji_wins}/{dji_total}'},
        'sp100':       {'periods': sp100_results, 'wins': f'{sp_wins}/{sp_total}'},
        'total_wins':  f'{dji_wins + sp_wins}/{dji_total + sp_total}',
    }
    with open(os.path.join(out_dir, 'final_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ → {out_dir}/final_results.json")


if __name__ == '__main__':
    main()
