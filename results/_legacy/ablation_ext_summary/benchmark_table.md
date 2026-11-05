# Extended systems vs benchmarks (DJI, equal-weight)


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
| BENCHMARK: DJI | 18.27 | 1.6501 | -6.62 |
| BENCHMARK: EW | 22.94 | 2.1582 | -5.42 |

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
| BENCHMARK: DJI | -6.97 | -0.3741 | -19.69 |
| BENCHMARK: EW | -9.01 | -0.4424 | -21.52 |

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
| BENCHMARK: DJI | 10.68 | 1.0593 | -9.02 |
| BENCHMARK: EW | 18.05 | 1.7475 | -8.29 |

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
| BENCHMARK: DJI | 10.52 | 0.9964 | -6.06 |
| BENCHMARK: EW | 14.99 | 1.4782 | -6.17 |
