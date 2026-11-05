# Results directory — organization guide

Reorganized 2026-05-23 to separate cross-analyses, active per-universe outputs,
and legacy work. Loose top-level files were moved into `_cross_analyses/`
subfolders; clearly-legacy directories went to `_legacy/`.

Active per-universe directories remain flat at root **only because 24 scripts
have hardcoded `results/{name}/` paths**. Per-universe physical grouping is a
future cleanup item (would require coordinated script updates).

## Structure

```
results/
├── INDEX.md                        ← this file
├── best_hyperparams.json           ← shared hyperparameter config (read by all trainers)
│
├── _cross_analyses/                ← multi-system / multi-universe analyses
│   ├── ablations/                   ablation_summary.{json,md,png}, ablation_h2.png
│   ├── bootstrap_ci/                bootstrap_ci.{json,md,png}
│   ├── cumulative_pnl/              cumulative_pnl.png, cumulative_pnl_data.json
│   ├── drawdown_profiles/           drawdown_profiles.png, drawdown_data.json
│   ├── mac_ensemble/                mac_ensemble.{json,md}
│   ├── markowitz_baseline/          markowitz_baseline.{json,md,png}
│   ├── multi_seed/                  multi_seed_dji/sp100/summary{md,png,json}
│   ├── ndx_experiments/             ndx_experiments_*.{md,png}, ndx_2017_concentration_*
│   ├── pareto_frontier/             pareto_frontier.png
│   ├── per_regime_routing/          per_regime_routing.*, routing_data.json, routing_trajectory.png
│   ├── risk_adjusted_metrics/       sortino_calmar.{json,md,png}
│   └── turnover/                    turnover.* (basic) and turnover_full.* (per-specialist)
│
├── _legacy/                        ← old / deprecated work (preserved for traceability)
│   ├── ablation_ext_*               early DJI ablations (no_attention, no_dsr, regime_lambda variants)
│   ├── ablation_no_*                early universe-less ablations
│   ├── ablation_sp100_no_*          early SP100 ablations
│   ├── attention_test_*.png         old attention test charts
│   ├── extended_nofloor             early DJI no-floor experiment
│   ├── extended_sp30, _sp30_v2      legacy SP30 (replaced by SP100 universe)
│   ├── extended_vix                 legacy VIX experiment
│   ├── mars_pareto_v1*              early MARS variants (predecessors of MAC v3)
│   ├── master_comparison            old comparison directory
│   ├── methodology_sweep            old methodology exploration
│   ├── mtl_extended, mtl_v2_*       MTL (multi-task learning) experiments — superseded by MAC
│   ├── multi_agent *, _comparison   early MAC predecessors
│   ├── regime_by_year.png, regime_detection.png  early HMM plots
│   ├── risk_personality*            early specialist methodology
│   ├── sp30_comparison              legacy SP30 vs DJI comparison
│   ├── test_2023_summary.*          superseded by sweep data
│   ├── training_curve.png, training_rewards.json  early training plots
│   ├── universal_v2_*, v3_*, v5_*   predecessors of v6 production
│   ├── v4_hh_dji, _sp100             head-to-head experiment (no longer used)
│   └── vanilla_extended, _sp100     pure-PPO baselines (superseded by v6)
│
└── (Active per-universe outputs — flat at root for now)
    ╔═════════════════════════════════════════════════════════════════════╗
    ║  DJI 30 universe                                                    ║
    ╠═════════════════════════════════════════════════════════════════════╣
    ║  Production: universal_v6_dji/                                      ║
    ║  Multi-seed: universal_v6_dji_s41/, universal_v6_dji_s43/           ║
    ║  H2 ablation: universal_v6_dji_noregime/                            ║
    ║  Legacy baseline: extended/                                          ║
    ║                                                                      ║
    ║  MAC: mac_dji/, mac_dji_v2/, mac_dji_v3/                            ║
    ║  MAC ablations: mac_dji_v3_{ent05,novol,nowarm,shuf}/                ║
    ║  MAC seeds: mac_dji_v3_s41/, mac_dji_v3_s43/                        ║
    ║  MAC variants: mac_dji_v4/, v5/, v6/, v7/                            ║
    ║                                                                      ║
    ║  Specialists: specialist_defender_dji/, specialist_steady_dji/,     ║
    ║               specialist_hunter_dji/                                ║
    ║  Defender ablation: specialist_defender_dji_f50/  (0.5 cash floor)  ║
    ╚═════════════════════════════════════════════════════════════════════╝

    ╔═════════════════════════════════════════════════════════════════════╗
    ║  SP100 universe                                                     ║
    ╠═════════════════════════════════════════════════════════════════════╣
    ║  Production: universal_v6_sp100/                                    ║
    ║  Multi-seed: universal_v6_sp100_s41/, _s43/                          ║
    ║  H2 ablation: universal_v6_sp100_noregime/                          ║
    ║  Legacy baseline: extended_sp100/                                    ║
    ║                                                                      ║
    ║  MAC: mac_sp100/, mac_sp100_v2/, mac_sp100_v3/                       ║
    ║  MAC ablations: mac_sp100_v3_{ent05,novol,nowarm,shuf}/              ║
    ║  MAC seeds: mac_sp100_v3_s41/, _s43/                                 ║
    ║  MAC variants: mac_sp100_v4/, v5/, v6/, v7/                          ║
    ║                                                                      ║
    ║  Specialists: specialist_defender_sp100/, specialist_steady_sp100/, ║
    ║               specialist_hunter_sp100/                              ║
    ║  Defender ablation: specialist_defender_sp100_f50/                  ║
    ╚═════════════════════════════════════════════════════════════════════╝

    ╔═════════════════════════════════════════════════════════════════════╗
    ║  NDX universe (experimental sandbox)                                 ║
    ╠═════════════════════════════════════════════════════════════════════╣
    ║  Production: universal_v6_ndx/                                       ║
    ║  Reward-shaping experiments:                                         ║
    ║    universal_v6_ndx_exp1/   lower bull penalty                       ║
    ║    universal_v6_ndx_exp2/   concentration bonus                       ║
    ║    universal_v6_ndx_exp3/   pure log-return reward                    ║
    ║    universal_v6_ndx_exp4/   lower ent_coef                            ║
    ║                                                                       ║
    ║  Training-data + universe experiments:                                ║
    ║    universal_v6_ndx_2017/      4-year train, 90 stocks               ║
    ║    universal_v6_ndx_2017_mom/  + long-horizon momentum features      ║
    ║                                                                       ║
    ║  Note: NDX has no MAC or specialists (experimental sandbox only)     ║
    ╚═════════════════════════════════════════════════════════════════════╝
```

## Top-level files at a glance

| What | Where to look |
|---|---|
| Production model results per universe | `universal_v6_{dji,sp100,ndx}/final_results.json` |
| Checkpoint sweeps per system | `{system}/checkpoint_sweep.json` |
| Cross-system ablation table | `_cross_analyses/ablations/ablation_summary.md` |
| Multi-seed CIs | `_cross_analyses/multi_seed/multi_seed_{dji,sp100}.md` |
| Risk-adjusted metrics (Sortino, Calmar) | `_cross_analyses/risk_adjusted_metrics/sortino_calmar.md` |
| Drawdown profiles | `_cross_analyses/drawdown_profiles/drawdown_profiles.png` |
| Routing trajectory + per-regime | `_cross_analyses/per_regime_routing/` |
| MV (Markowitz) baseline | `_cross_analyses/markowitz_baseline/` |
| Cumulative P&L curves | `_cross_analyses/cumulative_pnl/` |
| Turnover (basic + per-specialist) | `_cross_analyses/turnover/` |
| NDX experimental comparison | `_cross_analyses/ndx_experiments/` |
| Pareto frontier (bull vs bear) | `_cross_analyses/pareto_frontier/pareto_frontier.png` |
| MAC ensemble (no improvement found) | `_cross_analyses/mac_ensemble/` |
| Bootstrap CIs (deprioritized) | `_cross_analyses/bootstrap_ci/` |

## What was MOVED in this reorganization

```
NOT MOVED (still at root):
  - best_hyperparams.json  (shared config; many trainers read it)
  - all active per-universe directories (would break 24 scripts)
  - extended/, extended_sp100/  (legacy baselines used by v6 trainers' comparison print)

MOVED to _cross_analyses/:  ~30 loose top-level files
MOVED to _legacy/:           ~35 old/superseded directories + ~10 old loose files
```

## Future cleanup (optional, only when no training is active)

If you want fully per-universe grouped folders later:

1. Wait until all trainings finish.
2. Move active directories into `dji/`, `sp100/`, `ndx/` subdirectories.
3. Update 24 scripts that have hardcoded paths (use sed to find/replace).
4. Verify each script still works.

This is ~2 hours of work and best done in one session.
