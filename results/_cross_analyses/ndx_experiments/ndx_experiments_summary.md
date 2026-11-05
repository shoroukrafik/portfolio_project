# NDX experiments — comparative summary

All variants compared at val-Sharpe-best checkpoint.
^NDX 2024 cap-weighted benchmark: +28.13%   ^NDX 2022 benchmark: -33.71%

| Variant | What changed | Best step | Val Sharpe | val | bear | 2023 | bull |
|---|---|---|---|---|---|---|---|
| production v6 | baseline (14yr train, 68 stocks, 1M) | 175,000 | 2.065 | +27.87 | -8.38 | +22.77 | +8.03 |
| exp1 lower bull λ | bull penalty 0.1 → 0.01 | 175,000 | 2.064 | +27.95 | -8.07 | +22.54 | +7.74 |
| exp2 concentration | concentration bonus top-5, λ=0.05 | 200,000 | 2.010 | +27.18 | -8.72 | +23.52 | +7.71 |
| exp3 log-return | DSR → raw return reward | 100,000 | 1.920 | +26.32 | -9.23 | +23.44 | +9.23 |
| exp4 low ent_coef | ent_coef 0.01 → 0.001 | 175,000 | 2.070 | +27.66 | -8.33 | +22.12 | +7.66 |
| 2017-variant | 4yr recent train, 90 stocks | 50,000 | 1.924 | +27.29 | -11.57 | +25.35 | +12.18 |