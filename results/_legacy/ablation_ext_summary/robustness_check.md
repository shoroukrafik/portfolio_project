# Robustness check — single-eval vs rolling-mean selection

For each system, compare the **saved checkpoint** (chosen at the single-eval peak val Sharpe) against the **5-eval trailing rolling mean** at that same step.

* `spike_gap`  = `saved_sharpe − rolling_mean_at_saved`. Larger gap → the saved peak was a transient spike, not a stable plateau.
* `window_peak_step`  = step that would have been chosen under a rolling-mean criterion.
* `same_choice`  = whether the rolling-mean criterion picks the same step as the single-peak criterion.
* `robust` = `spike_gap < 0.10` AND `|saved_step − window_step| ≤ 50k`.

| System | Saved step | Saved Sharpe | Rolling@saved | Spike gap | Window-peak step | Window-peak Sharpe | Same choice | Robust |
|---|---:|---:|---:|---:|---:|---:|---|---|
| FULL | 625,000 | 2.3928 | 2.0201 | +0.3727 | 700,000 | 2.2548 | ✗ | ⚠ |
| VANILLA | 675,000 | 2.1068 | 1.8356 | +0.2712 | 125,000 | 1.9864 | ✗ | ⚠ |
| no_dsr | 375,000 | 2.1474 | 2.0019 | +0.1456 | 425,000 | 2.0534 | ✗ | ⚠ |
| no_attention | 25,000 | 2.0692 | 2.0692 | +0.0000 | 125,000 | 1.9962 | ✗ | ⚠ |
| no_regime_λ=0.1 | 775,000 | 2.8137 | 2.6257 | +0.1880 | 850,000 | 2.7025 | ✗ | ⚠ |
| no_regime_λ=0.3 | 700,000 | 2.5825 | 2.5064 | +0.0760 | 700,000 | 2.5064 | ✓ | ✓ |
| no_regime_λ=0.5 | 625,000 | 2.5404 | 2.2012 | +0.3392 | 675,000 | 2.3196 | ✗ | ⚠ |
