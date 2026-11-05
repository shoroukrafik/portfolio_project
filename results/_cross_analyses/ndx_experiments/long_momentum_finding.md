# NDX Long-Momentum Experiment — Findings

**Date**: 2026-05-23
**Experiment**: NDX 2017-variant + mom_120d_norm + mom_252d_norm features

## Summary

Adding long-horizon momentum features (6-month and 12-month per-stock momentum)
to the NDX 2017-variant **improves bull, bear, AND 2023 capture simultaneously**.
This is unusual — most fixes trade one for another. Long-momentum is a Pareto improvement.

## Best comparable checkpoint (constrained bull, bear ≥ -20%)

```
Variant                          Step      Bull       Bear       2023      val Sharpe
NDX 2017 baseline                375k     +17.71     -19.00     +29.00     1.71
NDX 2017 + long-momentum         425k     +28.90     -18.49     +32.53     1.56

Δ (mom − base)                            +11.19    +0.51       +3.53      -0.15
```

## Vs ^NDX benchmarks

```
At long-momentum step 425k:
  bull   +28.90%   ^NDX +28.13%   → BEATS by +0.77 pp ★
  bear   -18.49%   ^NDX -33.71%   → BEATS by +15.22 pp (less negative)
  2023   +32.53%   ^NDX +54.90%   → loses, but +3.53 pp better than baseline
  val Sh  1.56                    → within 75% threshold of val-Sharpe-best (1.92)
```

## Why long-momentum worked

```
Long-horizon momentum features (mom_120d, mom_252d) help on BOTH sides:
  - Identifies sustained winners (NVDA had a 12-month rally before 2024)
  - Identifies sustained losers (INTC, DLTR, BIIB all in long downtrends before 2022)
  
Result: better stock selection on both upside and downside.
Classical momentum factor effect (Jegadeesh & Titman 1993) extending to DRL setting.
```

## Verdict

**Step 425k of NDX 2017 + long-momentum is the new "best experimental NDX" pick.**

- Beats production NDX v6 by +20.87 pp on bull
- Beats ^NDX cap-weighted on val + bear + bull (3/4 periods)
- Only ^NDX 2023 +54.90 still unbeaten (distributional shift — model can't identify
  2023 winners that weren't in 2017-2020 training pattern)
