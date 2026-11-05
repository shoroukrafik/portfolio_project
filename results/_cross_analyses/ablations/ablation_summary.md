# Ablation summary (all single-seed, val-Sharpe-best checkpoint)

## MAC ablations (cumulative return % | Sharpe)

| Ablation | What changed | Justifies | Universe | val | bear | 2023 | bull |
|---|---|---|---|---|---|---|---|
| v3 (PRODUCTION) | Production lineup, soft warm-start | baseline | DJI | +33.03 (+2.376) | -7.95 (-0.847) | +34.56 (+2.535) | +38.61 (+3.036) |
| v3 (PRODUCTION) | Production lineup, soft warm-start | baseline | SP100 | +22.62 (+2.765) | -7.19 (-0.810) | +17.25 (+1.487) | +17.33 (+1.785) |
| Hard-switch | No learned controller, HMM argmax routing | controller | DJI | +33.59 (+2.375) | -8.43 (-1.043) | +35.22 (+2.536) | +39.48 (+3.068) |
| Hard-switch | No learned controller, HMM argmax routing | controller | SP100 | +21.98 (+2.740) | -6.54 (-0.792) | +17.63 (+1.514) | +16.21 (+1.684) |
| v4 (one-hot ws) | One-hot warm-start instead of soft | controller | DJI | +33.02 (+2.375) | -7.86 (-0.787) | +34.05 (+2.516) | +38.28 (+3.003) |
| v4 (one-hot ws) | One-hot warm-start instead of soft | controller | SP100 | +22.72 (+2.798) | -6.99 (-0.797) | +17.20 (+1.482) | +16.73 (+1.731) |
| A: no warm-start | PPO from scratch, no supervised warm-start | warm-start | DJI | +32.43 (+2.639) | -11.98 (-0.675) | +28.68 (+2.407) | +34.26 (+3.130) |
| A: no warm-start | PPO from scratch, no supervised warm-start | warm-start | SP100 | +27.48 (+2.739) | -10.19 (-0.503) | +13.94 (+1.236) | +18.18 (+1.898) |
| B: shuffled regime | Random Dirichlet replaces real HMM probs in MAC | regime signal | DJI | +24.15 (+1.852) | -16.97 (-0.794) | +25.87 (+2.126) | +32.89 (+2.641) |
| B: shuffled regime | Random Dirichlet replaces real HMM probs in MAC | regime signal | SP100 | +19.58 (+2.058) | -19.42 (-1.002) | +9.87 (+0.887) | +11.15 (+1.158) |
| C: no vol features | MAC obs = 3D regime probs only (drop vol) | vol features | DJI | +33.43 (+2.376) | -7.46 (-0.872) | +35.00 (+2.535) | +38.77 (+3.018) |
| C: no vol features | MAC obs = 3D regime probs only (drop vol) | vol features | SP100 | +22.85 (+2.792) | -7.22 (-0.796) | +17.32 (+1.488) | +16.95 (+1.749) |
| F: ent_coef 0.05 | PPO entropy coef 0.05 (default 0.01) | hyperparameter | DJI | +34.29 (+2.391) | -11.90 (-0.816) | +34.27 (+2.524) | +40.72 (+3.115) |
| F: ent_coef 0.05 | PPO entropy coef 0.05 (default 0.01) | hyperparameter | SP100 | +28.30 (+2.792) | -11.56 (-0.534) | +15.74 (+1.340) | +20.09 (+1.998) |
| v5 (drop balanced) | 2-way router, no steady specialist | specialist count | DJI | +28.50 (+2.482) | -5.21 (-0.600) | +23.32 (+2.042) | +28.35 (+2.712) |
| v5 (drop balanced) | 2-way router, no steady specialist | specialist count | SP100 | +20.85 (+2.546) | -5.51 (-0.703) | +15.48 (+1.403) | +14.43 (+1.601) |
| v6 (drop defender) | 2-way router, no defender specialist | specialist count | DJI | +34.79 (+2.307) | -13.06 (-0.595) | +37.46 (+2.651) | +42.54 (+3.085) |
| v6 (drop defender) | 2-way router, no defender specialist | specialist count | SP100 | +24.82 (+2.545) | -9.96 (-0.543) | +16.47 (+1.395) | +19.44 (+1.960) |
| v7 (drop hunter) | 2-way router, no hunter specialist | specialist count | DJI | +32.09 (+2.599) | -5.30 (-0.704) | +22.24 (+2.032) | +29.30 (+2.543) |
| v7 (drop hunter) | 2-way router, no hunter specialist | specialist count | SP100 | +23.93 (+3.022) | -4.44 (-0.585) | +14.00 (+1.320) | +12.90 (+1.420) |
| v2 (MARS legacy) | Per-role envs (legacy MARS arch) | architecture | DJI | +29.28 (+2.438) | -8.97 (-0.556) | +24.51 (+2.099) | +25.82 (+2.328) |
| v2 (MARS legacy) | Per-role envs (legacy MARS arch) | architecture | SP100 | +21.56 (+2.447) | -7.09 (-0.572) | +15.90 (+1.437) | +15.45 (+1.693) |

## H2: universal v6 single agent — regime ON vs OFF

| System | Universe | val | bear | 2023 | bull |
|---|---|---|---|---|---|
| universal v6 ORIG | DJI | +31.42 (+2.628) | -10.79 (-0.479) | +22.74 (+1.994) | +28.44 (+2.562) |
| universal v6 H2 | DJI | +27.35 (+2.495) | -9.00 (-0.423) | +20.50 (+1.977) | +22.84 (+2.236) |
| universal v6 ORIG | SP100 | +29.54 (+2.754) | -7.94 (-0.385) | +12.61 (+1.141) | +16.54 (+1.750) |
| universal v6 H2 | SP100 | +31.31 (+2.916) | -11.30 (-0.513) | +18.16 (+1.607) | +19.60 (+1.992) |