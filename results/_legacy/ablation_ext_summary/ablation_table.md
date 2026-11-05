# Extended-baseline ablation summary


## Validation (2021)

| System | Cumulative Return (%) | Sharpe | Max Drawdown (%) |
|---|---:|---:|---:|
| FULL (baseline) | 27.20 | 2.3928 | -5.36 |
| VANILLA | 22.06 | 2.1068 | -4.70 |
| no_floor (eval) | 29.31 | 2.6737 | -4.22 |
| no_dsr | 19.77 | 2.1474 | -3.89 |
| no_attention | 19.14 | 2.0692 | -4.52 |
| no_regime_λ=0.1 | 31.69 | 2.8137 | -4.83 |
| no_regime_λ=0.3 | 28.22 | 2.5825 | -5.30 |
| no_regime_λ=0.5 | 32.84 | 2.5404 | -5.74 |

## Test — Bear (2022)

| System | Cumulative Return (%) | Sharpe | Max Drawdown (%) |
|---|---:|---:|---:|
| FULL (baseline) | -7.36 | -0.6104 | -15.01 |
| VANILLA | -9.78 | -0.4712 | -22.34 |
| no_floor (eval) | -13.69 | -0.6302 | -23.83 |
| no_dsr | -5.48 | -0.5072 | -12.01 |
| no_attention | -4.50 | -0.4242 | -11.71 |
| no_regime_λ=0.1 | -6.23 | -0.5270 | -13.68 |
| no_regime_λ=0.3 | -6.86 | -0.5765 | -14.22 |
| no_regime_λ=0.5 | -10.32 | -0.7335 | -19.39 |

## Test — Recovery (2023)

| System | Cumulative Return (%) | Sharpe | Max Drawdown (%) |
|---|---:|---:|---:|
| FULL (baseline) | 27.16 | 2.2862 | -9.49 |
| VANILLA | 21.57 | 2.0930 | -7.83 |
| no_floor (eval) | 29.03 | 2.2681 | -10.02 |
| no_dsr | 16.39 | 1.7367 | -8.20 |
| no_attention | 17.29 | 1.8345 | -7.66 |
| no_regime_λ=0.1 | 16.10 | 1.4589 | -9.88 |
| no_regime_λ=0.3 | 20.00 | 1.8613 | -9.17 |
| no_regime_λ=0.5 | 30.53 | 2.3502 | -10.22 |

## Test — Bull (2024)

| System | Cumulative Return (%) | Sharpe | Max Drawdown (%) |
|---|---:|---:|---:|
| FULL (baseline) | 24.35 | 2.1159 | -6.04 |
| VANILLA | 20.21 | 1.9593 | -7.53 |
| no_floor (eval) | 18.15 | 1.7849 | -6.05 |
| no_dsr | 12.09 | 1.3745 | -5.33 |
| no_attention | 15.21 | 1.7461 | -5.03 |
| no_regime_λ=0.1 | 21.53 | 2.0113 | -5.43 |
| no_regime_λ=0.3 | 19.38 | 1.9764 | -5.11 |
| no_regime_λ=0.5 | 28.94 | 2.2032 | -6.33 |

## Δ vs FULL baseline (ablation − FULL)

Positive Δ-return is better; positive Δ-Sharpe is better; positive Δ-MDD is better (closer to 0).

| System | Period | ΔReturn (pp) | ΔSharpe | ΔMDD (pp) |
|---|---|---:|---:|---:|
| VANILLA | val | -5.14 | -0.2860 | +0.66 |
| VANILLA | test_bear | -2.42 | +0.1392 | -7.33 |
| VANILLA | test_2023 | -5.59 | -0.1932 | +1.66 |
| VANILLA | test_bull | -4.14 | -0.1566 | -1.49 |
| no_floor (eval) | val | +2.11 | +0.2809 | +1.14 |
| no_floor (eval) | test_bear | -6.33 | -0.0198 | -8.82 |
| no_floor (eval) | test_2023 | +1.87 | -0.0181 | -0.53 |
| no_floor (eval) | test_bull | -6.20 | -0.3310 | -0.01 |
| no_dsr | val | -7.43 | -0.2454 | +1.47 |
| no_dsr | test_bear | +1.88 | +0.1032 | +3.00 |
| no_dsr | test_2023 | -10.77 | -0.5495 | +1.29 |
| no_dsr | test_bull | -12.26 | -0.7414 | +0.71 |
| no_attention | val | -8.06 | -0.3236 | +0.84 |
| no_attention | test_bear | +2.86 | +0.1862 | +3.30 |
| no_attention | test_2023 | -9.87 | -0.4517 | +1.83 |
| no_attention | test_bull | -9.14 | -0.3698 | +1.01 |
| no_regime_λ=0.1 | val | +4.49 | +0.4209 | +0.53 |
| no_regime_λ=0.1 | test_bear | +1.13 | +0.0834 | +1.33 |
| no_regime_λ=0.1 | test_2023 | -11.06 | -0.8273 | -0.39 |
| no_regime_λ=0.1 | test_bull | -2.82 | -0.1046 | +0.61 |
| no_regime_λ=0.3 | val | +1.02 | +0.1897 | +0.06 |
| no_regime_λ=0.3 | test_bear | +0.50 | +0.0339 | +0.79 |
| no_regime_λ=0.3 | test_2023 | -7.16 | -0.4249 | +0.32 |
| no_regime_λ=0.3 | test_bull | -4.97 | -0.1395 | +0.93 |
| no_regime_λ=0.5 | val | +5.64 | +0.1476 | -0.38 |
| no_regime_λ=0.5 | test_bear | -2.96 | -0.1231 | -4.38 |
| no_regime_λ=0.5 | test_2023 | +3.37 | +0.0640 | -0.73 |
| no_regime_λ=0.5 | test_bull | +4.59 | +0.0873 | -0.29 |
