"""
Quick experiment: run the DJI canonical agent on 2025 with NO cash floor.
Isolates how much the 70% bear-regime cash floor cost the agent in 2025.

Reuses the simulator infrastructure from simulator_2025.py (cache + features).
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))

from simulator_2025 import (
    CANONICAL, INDEX_TICKERS, EVAL_START, EVAL_END,
    download_indices, download_universe_stocks,
    build_hmm_features, fit_hmm, apply_hmm,
    assemble_universe_panel, run_canonical_agent, metrics,
    bh_values, ew_values,
    HMM_TRAIN_START, DOWNLOAD_START, DOWNLOAD_END,
)


def main():
    cfg = dict(CANONICAL['DJI'])   # copy so we don't mutate

    print('=' * 72)
    print('   DJI 2025 — no-floor experiment')
    print('=' * 72)

    print('\n[1/4] Loading cached indices and DJI stocks...')
    indices  = download_indices(HMM_TRAIN_START, DOWNLOAD_END)
    djs      = download_universe_stocks(cfg['stock_list_file'], DOWNLOAD_START, DOWNLOAD_END)

    print('[2/4] Refitting DJI HMM and labelling 2025...')
    hmm_feats_dji   = build_hmm_features(indices['dji'],   indices['vix'])
    hmm_feats_sp500 = build_hmm_features(indices['sp500'], indices['vix'])
    hmm_dji   = fit_hmm(hmm_feats_dji,   seed=42)
    hmm_sp500 = fit_hmm(hmm_feats_sp500, seed=42)
    reg_dji   = apply_hmm(hmm_dji,   hmm_feats_dji)
    reg_sp500 = apply_hmm(hmm_sp500, hmm_feats_sp500)

    print('[3/4] Assembling DJI 2025 panel...')
    panel = assemble_universe_panel('DJI', cfg, djs, indices, reg_dji, reg_sp500)
    print(f'  panel days: {panel["date"].nunique()}, stocks: {panel["ticker"].nunique()}')

    print('\n[4/4] Running DJI agent at each candidate cash floor...')
    floors = [0.0, 0.3, 0.5, 0.7]
    results = {}
    for f in floors:
        hist = run_canonical_agent(cfg['ckpt'], panel, f)
        m = metrics(hist['values'])
        results[f] = m
        print(f"  floor {int(f*100):>2}%   cum={m['cumulative_return']:>+7.2f}%  "
              f"Sh={m['sharpe_ratio']:>+.3f}  MDD={m['max_drawdown']:>+6.2f}%  n={m['n_days']}")

    bh_m = metrics(bh_values(indices['dji'], EVAL_START, EVAL_END))
    ew_m = metrics(ew_values(djs, EVAL_START, EVAL_END))
    print(f"\n  Reference:")
    print(f"  DJI BH    cum={bh_m['cumulative_return']:>+7.2f}%  Sh={bh_m['sharpe_ratio']:>+.3f}  MDD={bh_m['max_drawdown']:>+6.2f}%")
    print(f"  EW DJI    cum={ew_m['cumulative_return']:>+7.2f}%  Sh={ew_m['sharpe_ratio']:>+.3f}  MDD={ew_m['max_drawdown']:>+6.2f}%")

    out_path = PROJECT_ROOT / 'results' / '_canonical' / 'simulator_2025_dji_floor_sweep.json'
    with open(out_path, 'w') as f:
        json.dump({'floors': results, 'BH': bh_m, 'EW': ew_m}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
