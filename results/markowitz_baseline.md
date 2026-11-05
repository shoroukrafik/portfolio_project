# Markowitz Mean-Variance baselines (60-day lookback, 20-day rebalance, 30% cap)

Long-only constraint, transaction cost 10 bps per turnover, walk-forward.

| Strategy | Universe | Period | Cum.Ret % | Sharpe | MDD % |
|---|---|---|---|---|---|
| gmv | DJI | val | +26.33 | +2.355 | -5.24 |
| max_sharpe | DJI | val | +25.66 | +2.117 | -4.16 |
| gmv | DJI | test_bear | -9.00 | -0.476 | -22.96 |
| max_sharpe | DJI | test_bear | -11.60 | -0.572 | -24.38 |
| gmv | DJI | test_2023 | +18.76 | +1.647 | -7.97 |
| max_sharpe | DJI | test_2023 | +30.22 | +2.557 | -7.20 |
| gmv | DJI | test_bull | +16.35 | +1.580 | -5.89 |
| max_sharpe | DJI | test_bull | +21.21 | +1.784 | -6.14 |
| gmv | SP100 | val | +30.08 | +2.491 | -6.31 |
| max_sharpe | SP100 | val | +33.71 | +2.682 | -4.99 |
| gmv | SP100 | test_bear | -6.33 | -0.308 | -20.06 |
| max_sharpe | SP100 | test_bear | -10.15 | -0.475 | -22.83 |
| gmv | SP100 | test_2023 | +12.98 | +1.045 | -10.36 |
| max_sharpe | SP100 | test_2023 | +17.43 | +1.383 | -10.05 |
| gmv | SP100 | test_bull | +15.70 | +1.624 | -5.22 |
| max_sharpe | SP100 | test_bull | +15.97 | +1.556 | -5.22 |