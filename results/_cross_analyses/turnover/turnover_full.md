# Turnover per system (Deep-dive A)

Daily turnover = 0.5 × Σ|w_t − w_{t-1}|. 100% = full portfolio rebalanced.

Specialists evaluated alone through the MAC master env (hard one-hot routing).

## DJI mean daily turnover (%)

| Period | defender | steady | hunter | MAC v3 | v6 univ |
|---|---|---|---|---|---|
| val | 2.01 | 2.00 | 2.52 | 3.43 | 1.05 |
| test_bear | 1.22 | 1.82 | 2.48 | 2.08 | 1.38 |
| test_2023 | 0.89 | 1.44 | 2.19 | 2.11 | 1.04 |
| test_bull | 2.30 | 1.98 | 3.12 | 4.08 | 1.16 |

## SP100 mean daily turnover (%)

| Period | defender | steady | hunter | MAC v3 | v6 univ |
|---|---|---|---|---|---|
| val | 2.09 | 1.92 | 1.66 | 3.31 | 1.55 |
| test_bear | 1.34 | 2.34 | 1.38 | 1.95 | 1.93 |
| test_2023 | 1.84 | 1.93 | 1.72 | 3.17 | 1.77 |
| test_bull | 3.32 | 2.76 | 1.81 | 4.86 | 2.30 |