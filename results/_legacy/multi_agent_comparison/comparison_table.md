# Multi-agent architectures vs FULL

Separate from the component-ablation comparison. Each row is a multi-agent design tested against FULL on four out-of-sample periods.


## Validation (2021)

| System | Return (%) | Sharpe | MDD (%) |
|---|---:|---:|---:|
| FULL (baseline) | 27.20 | 2.3928 | -5.36 |
| MTL v2 | 20.78 | 2.1855 | -4.50 |
| Multi-agent (soft) | 26.33 | 2.6700 | -4.02 |
| Risk-personality (soft) | 30.80 | 2.6421 | -5.38 |

## Test — Bear (2022)

| System | Return (%) | Sharpe | MDD (%) |
|---|---:|---:|---:|
| FULL (baseline) | -7.36 | -0.6104 | -15.01 |
| MTL v2 | -6.86 | -0.6521 | -13.66 |
| Multi-agent (soft) | -6.70 | -0.6027 | -13.46 |
| Risk-personality (soft) | -8.37 | -0.6748 | -15.34 |

## Test — Recovery (2023)

| System | Return (%) | Sharpe | MDD (%) |
|---|---:|---:|---:|
| FULL (baseline) | 27.16 | 2.2862 | -9.49 |
| MTL v2 | 18.82 | 1.8952 | -7.12 |
| Multi-agent (soft) | 21.66 | 2.1392 | -8.56 |
| Risk-personality (soft) | 18.18 | 1.6827 | -9.68 |

## Test — Bull (2024)

| System | Return (%) | Sharpe | MDD (%) |
|---|---:|---:|---:|
| FULL (baseline) | 24.35 | 2.1159 | -6.04 |
| MTL v2 | 10.21 | 1.0750 | -5.43 |
| Multi-agent (soft) | 16.50 | 1.7838 | -5.51 |
| Risk-personality (soft) | 20.14 | 1.8973 | -5.12 |

## Δ vs FULL (system − FULL)

Positive Δ-return / Δ-Sharpe = system better. Positive Δ-MDD = system has less-negative drawdown (better).

| System | Period | ΔReturn (pp) | ΔSharpe | ΔMDD (pp) |
|---|---|---:|---:|---:|
| MTL v2 | val | -6.42 | -0.2073 | +0.86 |
| MTL v2 | test_bear | +0.50 | -0.0417 | +1.35 |
| MTL v2 | test_2023 | -8.34 | -0.3910 | +2.37 |
| MTL v2 | test_bull | -14.14 | -1.0409 | +0.61 |
| Multi-agent (soft) | val | -0.87 | +0.2772 | +1.34 |
| Multi-agent (soft) | test_bear | +0.66 | +0.0077 | +1.55 |
| Multi-agent (soft) | test_2023 | -5.50 | -0.1470 | +0.93 |
| Multi-agent (soft) | test_bull | -7.85 | -0.3321 | +0.53 |
| Risk-personality (soft) | val | +3.60 | +0.2493 | -0.02 |
| Risk-personality (soft) | test_bear | -1.01 | -0.0644 | -0.33 |
| Risk-personality (soft) | test_2023 | -8.98 | -0.6035 | -0.19 |
| Risk-personality (soft) | test_bull | -4.21 | -0.2186 | +0.92 |
