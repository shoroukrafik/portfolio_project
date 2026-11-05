# MAC v3 ensemble inference results

Ensemble averages routing logits across 3 MAC seeds (42, 41, 43) before softmax + blend.

| Universe | Period | Ensemble | Mean ± SD (3 seeds) | Best single seed | Δ Ensemble−Mean |
|---|---|---|---|---|---|
| DJI | val | +32.99 | +32.97 ± 0.37 | +33.30 | +0.02 |
| DJI | test_bear | -8.22 | -8.30 ± 0.44 | -7.95 | +0.08 |
| DJI | test_2023 | +35.11 | +35.00 ± 0.57 | +35.64 | +0.11 |
| DJI | test_bull | +39.18 | +39.20 ± 0.84 | +40.17 | -0.02 |
| SP100 | val | +24.12 | +24.65 ± 2.28 | +27.11 | -0.53 |
| SP100 | test_bear | -7.47 | -7.65 ± 0.45 | -7.19 | +0.18 |
| SP100 | test_2023 | +17.22 | +17.14 ± 0.11 | +17.25 | +0.08 |
| SP100 | test_bull | +18.83 | +18.65 ± 1.37 | +20.07 | +0.18 |