"""
Post-training analysis for the sp100_mom_noregimeobs_2m experiment.

Two steps:
  1. Build the checkpoint sweep: load every saved checkpoint (every 25k
     steps), evaluate on all four periods, save per-checkpoint metrics.
  2. Apply the DEFENSIVE rule (step >= 20% of total = 400k for 2M).
     Pick the highest val Sharpe among eligible checkpoints.
  3. At the DEFENSIVE pick, sweep over cash floors {0, 0.3, 0.5, 0.7}.

Output:
  results/sp100_mom_noregimeobs_2m/checkpoint_sweep.json
  results/sp100_mom_noregimeobs_2m/cashfloor_overlay_DEFENSIVE.json
  results/sp100_mom_noregimeobs_2m/canonical_vs_experiment.md
"""

import os
import sys
import json
import glob

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_universal_v6_sp100 import V3_COLS_TO_DROP, V6_RISK_PENALTIES


TAG = 'sp100_mom_noregimeobs_2m'
DEFENSIVE_THRESHOLD = 400_000  # 20% of 2M
FLOORS = [0.0, 0.3, 0.5, 0.7]

CHECKPOINT_DIR = f'models/saved/checkpoints/{TAG}'
RESULTS_DIR    = f'results/{TAG}'

PERIODS = {
    'val':       'val_data_sp100.csv',
    'test_bear': 'test_bear_data_sp100.csv',
    'test_2023': 'test_2023_data_sp100.csv',
    'test_bull': 'test_bull_data_sp100.csv',
}

BENCH = {  # SP100 BH + EW (no OEX, matching canonical)
    'val':       {'SP500 BH': 28.79, 'EW SP100': 33.11},
    'test_bear': {'SP500 BH': -19.95, 'EW SP100': -9.48},
    'test_2023': {'SP500 BH': 24.73, 'EW SP100': 21.17},
    'test_bull': {'SP500 BH': 24.54, 'EW SP100': 18.16},
}


def load_df(csv):
    df = pd.read_csv(f'data/{csv}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def eval_period(model, df, floor=0.0):
    env = PortfolioEnv(
        df=df,
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: float(floor), 2: 0.0},
        risk_penalties=V6_RISK_PENALTIES,
        include_regime=False,
        include_regime_probs=False,
        include_vol_features=False,
        use_markowitz_reward=False,
        use_hyperheuristic=False,
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    h = env.get_portfolio_history()
    v = h['value'].values.astype(float)
    n = len(v)
    cum_ret = float((v[-1] / v[0] - 1) * 100)
    ann_ret = float(((v[-1] / v[0]) ** (252 / n) - 1) * 100)
    dr = np.diff(v) / v[:-1]
    ann_vol = float(np.std(dr) * np.sqrt(252) * 100)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(np.min((v - peak) / peak) * 100)
    return {
        'cumulative_return':    cum_ret,
        'sharpe_ratio':         sharpe,
        'max_drawdown':         mdd,
    }


def count_wins(period_results):
    wins, total = 0, 0
    for p, bs in BENCH.items():
        ret = period_results[p]['cumulative_return']
        for _, val in bs.items():
            total += 1
            if ret > val:
                wins += 1
    return wins, total


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    period_dfs = {p: load_df(csv) for p, csv in PERIODS.items()}

    # === STEP 1: checkpoint sweep ===
    print('=' * 80)
    print(f'   Step 1: Checkpoint sweep over {CHECKPOINT_DIR}')
    print('=' * 80)
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'step_*.zip')))
    if not checkpoints:
        print(f'  ✗ No checkpoints found in {CHECKPOINT_DIR}')
        sys.exit(1)

    sweep = []
    print(f"\n  {'step':>8} | {'val':>7} {'bear':>7} {'2023':>7} {'bull':>7} | {'v.shp':>6}")
    for ck in checkpoints:
        step = int(os.path.basename(ck).replace('step_', '').replace('.zip', ''))
        model = PPO.load(ck, device='cpu')
        per = {}
        for p, df in period_dfs.items():
            per[p] = eval_period(model, df, floor=0.0)
        sweep.append({
            'step': step,
            'val_ret': per['val']['cumulative_return'],
            'test_bear_ret': per['test_bear']['cumulative_return'],
            'test_2023_ret': per['test_2023']['cumulative_return'],
            'test_bull_ret': per['test_bull']['cumulative_return'],
            'val_sharpe': per['val']['sharpe_ratio'],
            'val_mdd': per['val']['max_drawdown'],
        })
        print(f"  {step:>8,} | "
              f"{per['val']['cumulative_return']:>+7.2f} "
              f"{per['test_bear']['cumulative_return']:>+7.2f} "
              f"{per['test_2023']['cumulative_return']:>+7.2f} "
              f"{per['test_bull']['cumulative_return']:>+7.2f} | "
              f"{per['val']['sharpe_ratio']:>+6.3f}")

    with open(os.path.join(RESULTS_DIR, 'checkpoint_sweep.json'), 'w') as f:
        json.dump(sweep, f, indent=2)

    # === STEP 2: DEFENSIVE rule ===
    print(f"\n{'='*80}")
    print(f'   Step 2: DEFENSIVE rule (step >= {DEFENSIVE_THRESHOLD:,}, max val Sharpe)')
    print('=' * 80)
    eligible = [r for r in sweep if r['step'] >= DEFENSIVE_THRESHOLD]
    if not eligible:
        print(f'  ✗ No checkpoints pass the {DEFENSIVE_THRESHOLD:,} threshold')
        sys.exit(1)
    defensive = max(eligible, key=lambda r: r['val_sharpe'])
    print(f"\n  DEFENSIVE pick: step {defensive['step']:,}  val Sharpe {defensive['val_sharpe']:+.3f}")
    print(f"    val={defensive['val_ret']:+.2f}%  bear={defensive['test_bear_ret']:+.2f}%  "
          f"2023={defensive['test_2023_ret']:+.2f}%  bull={defensive['test_bull_ret']:+.2f}%")

    # === STEP 3: floor sweep at DEFENSIVE pick ===
    print(f"\n{'='*80}")
    print(f'   Step 3: Floor sweep over {FLOORS} at the DEFENSIVE pick')
    print('=' * 80)
    defensive_ckpt = os.path.join(CHECKPOINT_DIR, f"step_{defensive['step']:07d}.zip")
    model = PPO.load(defensive_ckpt, device='cpu')

    out = {
        'checkpoint': defensive_ckpt,
        'defensive_step': defensive['step'],
        'description': 'sp100_mom_noregimeobs_2m + cash floor sweep',
        'by_floor': {},
    }
    print(f"\n  {'floor':>5} | {'val':>8} {'bear':>8} {'2023':>8} {'bull':>8} | {'val Sh':>7} {'bear Sh':>8} {'wins':>5}")
    for f in FLOORS:
        per = {}
        for p, df in period_dfs.items():
            per[p] = eval_period(model, df, floor=f)
        wins, total = count_wins(per)
        out['by_floor'][f"{f}"] = per
        print(f"  {f:>5.2f} | "
              f"{per['val']['cumulative_return']:>+8.2f} "
              f"{per['test_bear']['cumulative_return']:>+8.2f} "
              f"{per['test_2023']['cumulative_return']:>+8.2f} "
              f"{per['test_bull']['cumulative_return']:>+8.2f} | "
              f"{per['val']['sharpe_ratio']:>+7.3f} "
              f"{per['test_bear']['sharpe_ratio']:>+8.3f}  "
              f"{wins:>2}/{total}")

    with open(os.path.join(RESULTS_DIR, 'cashfloor_overlay_DEFENSIVE.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # === STEP 4: comparison vs canonical ===
    print(f"\n{'='*80}")
    print('   Step 4: Comparison vs canonical (SP100 mom_2m + 30% floor)')
    print('=' * 80)
    canonical = {
        'val':       +37.27,
        'test_bear': -12.71,
        'test_2023': +23.37,
        'test_bull': +18.17,
        'val_Sh':    +3.41,
        'wins':      5,
    }
    print(f"  Canonical:                            val=+37.27  bear=-12.71  2023=+23.37  bull=+18.17  v.Sh=+3.41  wins=5/8")
    print(f"  Experiment (DEFENSIVE pick):          val={defensive['val_ret']:+6.2f}  bear={defensive['test_bear_ret']:+6.2f}  2023={defensive['test_2023_ret']:+6.2f}  bull={defensive['test_bull_ret']:+6.2f}  v.Sh={defensive['val_sharpe']:+5.2f}")
    for f in FLOORS:
        bf = out['by_floor'][str(f)]
        wins, total = count_wins(bf)
        print(f"  Experiment + {int(f*100):2d}% floor overlay:    "
              f"val={bf['val']['cumulative_return']:+6.2f}  bear={bf['test_bear']['cumulative_return']:+6.2f}  "
              f"2023={bf['test_2023']['cumulative_return']:+6.2f}  bull={bf['test_bull']['cumulative_return']:+6.2f}  "
              f"v.Sh={bf['val']['sharpe_ratio']:+5.2f}  wins={wins}/{total}")
    print()
    print(f"  Hypothesis: this experiment + floor BEATS canonical on val_Sharpe + wins?")
    print(f"  ANSWER: see numbers above. Highest val_Sh = "
          f"{max(out['by_floor'][str(f)]['val']['sharpe_ratio'] for f in FLOORS):.3f}, "
          f"canonical = {canonical['val_Sh']:.3f}")
    print()
    print(f"✅ All outputs in {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
