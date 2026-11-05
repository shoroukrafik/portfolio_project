# NDX 2017+mom+CVaR Bear-Defense — Negative Result

**Date**: 2026-05-23
**Experiment**: NDX 2017 + long-momentum + CVaR penalty (cvar_lambda=0.5)
**Hypothesis**: CVaR penalty would asymmetrically improve bear (tail-risk-aware) without hurting bull (since CVaR is dormant when no tail loss occurs)
**Result**: HYPOTHESIS REJECTED. CVaR trades bull for bear at ~2:1 ratio.

## Per-checkpoint comparison vs mom-only

```
step      CVaR_bull  mom_bull   Δbull    CVaR_bear  mom_bear   Δbear
─────────────────────────────────────────────────────────────────────
25-200k   ±0.1-0.6   ±0.0-0.5   ~zero    ~zero      ~zero      ~zero  (early: no effect)
425k      +19.61     +28.90     -9.29    -14.03     -18.49     +4.46  (mid-late: lose 2 bull per 1 bear)
500k      +22.12     +46.35     -24.23   -15.24     -27.02     +11.78 (late: 2:1 trade-off)
```

## Why CVaR isn't asymmetric in practice

```
Hypothesized: CVaR penalty only fires when realized tail loss exceeds threshold
              → On bull days with no losses, term is zero → no bull impact
              → On bear days with losses, term is positive → defensive pull

Empirical: CVaR penalty depends on a 60-day ROLLING window of returns.
           During 2024 bull rally, the rolling window catches recent high-vol
           returns (NVDA +5% / -3% / +7% / -4%). Even though net is positive,
           the high VOLATILITY → high realized CVaR → penalty fires.
           
           Net: CVaR penalizes concentration on HIGH-VOL stocks regardless
                of regime direction. NVDA, TSLA — even in bull — have high CVaR.
                Penalty inhibits concentration on them during bull periods too.
```

## Headline result for thesis

```
Best NDX experimental pick UNCHANGED: NDX 2017 + LONG-MOM (no CVaR) step 425k
  val +25.12  bear -18.49  2023 +32.53  bull +28.90
  
  Bull BEATS ^NDX +28.13%
  Bear BEATS ^NDX -33.71% by 15 pp
  Only ^NDX 2023 still unbeaten (distributional shift)
```

## Thesis takeaway

CVaR penalty is a STRUCTURAL anti-concentration mechanism, not a regime-conditional
bear-defense. In dispersion-heavy markets like NDX 2024, this means CVaR reduces
both bull capture AND bear losses proportionally — there is no "free lunch" via
tail-risk-aware reward shaping.

For asymmetric bear defense, regime-conditional cash floor (defender-style) might
work better, but would require a separate experiment.
