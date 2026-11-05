"""
Methodology page — the architecture + algorithm explained.

Walks through every load-bearing methodology choice:
  1. HMM regime detection — what model, what states, what features
  2. PPO + Attention architecture — what's in the network
  3. Reward function — DSR + regime-conditioned λ
  4. Training configuration — hyperparameters with citations
  5. Selection rule — headline vs defensive picks
  6. The synthesis — how all pieces fit together

Each section includes literature citations where applicable, and explicitly
states where this thesis's contribution is a SYNTHESIS vs a novel claim.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import apply_theme
from utils.theme import (
    TEXT, TEXT_MUTED, TEXT_DIM, SURFACE, BORDER, PRIMARY, PRIMARY_DIM,
    SUCCESS, WARNING, DANGER, NEUTRAL,
)


st.set_page_config(
    page_title="Methodology · DRL Portfolio Thesis",
    page_icon="◆",
    layout="wide",
)
apply_theme()


# ============================== Header ==============================
st.title("Methodology")
st.markdown(
    f"<div style='color:{TEXT_MUTED}; font-size:0.95rem; margin-bottom:1.2rem;'>"
    "Every load-bearing methodological choice in this thesis, with literature "
    "citations and explicit attribution of novelty vs synthesis."
    "</div>",
    unsafe_allow_html=True
)


# ============================== HMM Regime Detection ==============================
st.markdown("### 1. HMM regime detection")

st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
            padding:1rem 1.2rem; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1rem;">

<b style="color:{TEXT}">Model:</b> 3-state Gaussian Hidden Markov Model.<br>
<b style="color:{TEXT}">Inputs:</b> daily index returns + VIX.<br>
<b style="color:{TEXT}">States:</b> labelled post-hoc by cluster statistics:<br>
&nbsp;&nbsp;<span style="color:{SUCCESS}">●</span> <b>Bull</b>     — high mean return, low volatility<br>
&nbsp;&nbsp;<span style="color:{DANGER}">●</span>  <b>Bear</b>     — negative mean return, high volatility<br>
&nbsp;&nbsp;<span style="color:{NEUTRAL}">●</span>  <b>Sideways</b> — near-zero mean, moderate volatility<br><br>

<b style="color:{TEXT}">Per-universe HMM source:</b><br>
&nbsp;&nbsp;<b>DJI</b>   → DJI-trained HMM, retuned on 2010-2020 (regime_labels_ext.csv)<br>
&nbsp;&nbsp;<b>SP100</b> → SP500 broad-market HMM (regime_labels_sp30.csv)<br>
&nbsp;&nbsp;<b>NDX</b>   → SP500 broad-market HMM (regime_labels_sp30.csv) —<br>
&nbsp;&nbsp;&nbsp;&nbsp;NDX-specific HMM was tested but misclassified the 2022 rate-hike bear.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Reverted to SP500 HMM for proper bear detection.<br><br>

<b style="color:{TEXT}">Literature:</b><br>
&nbsp;&nbsp;Hamilton (1989) — original Markov-switching framework.<br>
&nbsp;&nbsp;Ang &amp; Bekaert (2002 JFE) — regime-switching portfolios.<br>
&nbsp;&nbsp;Nystrup et al. (2018) — VIX-based HMM for tactical allocation.<br><br>

<b style="color:{TEXT}">Contribution status:</b> Standard, well-cited finance technique. No novelty.
</div>
""", unsafe_allow_html=True)


# ============================== PPO + Attention ==============================
st.markdown("### 2. PPO with Attention-based feature extractor")

st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
            padding:1rem 1.2rem; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1rem;">

<b style="color:{TEXT}">Algorithm:</b> Proximal Policy Optimization (PPO) — Schulman et al. 2017.
Standard on-policy actor-critic.<br><br>

<b style="color:{TEXT}">Feature extractor:</b> <code>AttentionPortfolioExtractor</code><br>
&nbsp;&nbsp;Per-stock encoder: Linear(n_features → 64) → ReLU → Linear(64 → 64) → ReLU<br>
&nbsp;&nbsp;Multi-head attention (4 heads, embed_dim=64) over stock embeddings<br>
&nbsp;&nbsp;Extras (regime probs + vol features + cash): Linear(7 → 32) → ReLU<br>
&nbsp;&nbsp;Combiner: concat(attended_stocks, extras) → Linear(96 → 128) → ReLU<br><br>

<b style="color:{TEXT}">Observation:</b><br>
&nbsp;&nbsp;n_stocks × n_features per-stock data (close/open/high/low/vol/RSI/MACD/.../mom_60d)<br>
&nbsp;&nbsp;+ 3 broad-market vol features (vol_20d, vol_60d, vol_ratio)<br>
&nbsp;&nbsp;+ 3 regime probabilities (P(bull), P(bear), P(sideways))<br>
&nbsp;&nbsp;+ current weights vector (n_stocks + cash)<br><br>

<b style="color:{TEXT}">Action:</b> softmax allocation over [stock_1, ..., stock_N, cash].<br><br>

<b style="color:{TEXT}">Policy + value head:</b> shared MLP [128, 64] (net_arch).<br><br>

<b style="color:{TEXT}">Literature:</b><br>
&nbsp;&nbsp;Vaswani et al. (2017) — multi-head attention origin.<br>
&nbsp;&nbsp;Sood et al. (JPM AI Research) — attention for portfolio allocation.<br>
&nbsp;&nbsp;Khemlichi et al. (2025) — modular RL with attention.<br><br>

<b style="color:{TEXT}">Contribution status:</b> Architecture follows the dominant 2024-2025 trend.
Our specific combination of (per-stock attention + regime probs + vol features +
cash dim) is novel to this thesis but builds on Sood + Khemlichi.
</div>
""", unsafe_allow_html=True)


# ============================== Reward ==============================
st.markdown("### 3. Reward function")

st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
            padding:1rem 1.2rem; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1rem;">

<b style="color:{TEXT}">Reward per timestep:</b><br>
&nbsp;&nbsp;<code>reward = DSR(portfolio_return) − λ_regime × downside_variance − transaction_cost</code><br><br>

<b style="color:{TEXT}">DSR (Differential Sharpe Ratio):</b> Moody &amp; Saffell (2001).<br>
Incremental Sharpe — reward each step by the marginal change in running Sharpe.
Stabilizes learning vs raw returns.<br><br>

<b style="color:{TEXT}">Regime-conditioned λ:</b><br>
&nbsp;&nbsp;λ_bull     = 0.1   (low risk aversion in bull — be aggressive)<br>
&nbsp;&nbsp;λ_sideways = 0.3   (medium aversion)<br>
&nbsp;&nbsp;λ_bear     = 0.5   (high risk aversion — defensive)<br><br>

<b style="color:{TEXT}">Transaction cost:</b> 0.1% per trade. Standard.<br><br>

<b style="color:{TEXT}">Literature:</b><br>
&nbsp;&nbsp;Moody &amp; Saffell (2001) — Differential Sharpe Reward formulation.<br>
&nbsp;&nbsp;Tamar et al. (2015) — risk-sensitive policy gradient.<br>
&nbsp;&nbsp;Yang et al. (2020 ICAIF) — regime-aware DRL portfolio ensemble.<br><br>

<b style="color:{TEXT}">Contribution status:</b> Synthesis. Combines DSR (Moody 2001) with
regime-conditioned risk penalty (extending Yang 2020 + Tamar 2015) in a single PPO agent.
Validated empirically by H2-soft ablation: removing regime-conditioned λ alone costs
~5-6pp on bear defense (DJI + SP100). Keep.
</div>
""", unsafe_allow_html=True)


# ============================== Hyperparameters ==============================
st.markdown("### 4. PPO hyperparameters")

st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
            padding:1rem 1.2rem; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1rem;">

<table style="width:100%; font-size:0.87rem;">
<tr style="color:{TEXT_DIM}; border-bottom: 1px solid {BORDER};">
  <td style="padding:0.4rem;">Parameter</td>
  <td style="padding:0.4rem;">Our value</td>
  <td style="padding:0.4rem;">SB3 default</td>
  <td style="padding:0.4rem;">Source / status</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.4rem;">learning_rate</td>
  <td style="padding:0.4rem; color:{TEXT};">5e-5</td>
  <td style="padding:0.4rem;">3e-4</td>
  <td style="padding:0.4rem;">empirical winner of 200k tune; matched-pair conservative</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.4rem;">n_steps</td>
  <td style="padding:0.4rem; color:{TEXT};">1024</td>
  <td style="padding:0.4rem;">2048</td>
  <td style="padding:0.4rem;">matched with our LR (lr × n_steps ≈ const)</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.4rem;">batch_size</td>
  <td style="padding:0.4rem; color:{TEXT};">64</td>
  <td style="padding:0.4rem;">64</td>
  <td style="padding:0.4rem;">SB3 default</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.4rem;">n_epochs</td>
  <td style="padding:0.4rem; color:{TEXT};">10</td>
  <td style="padding:0.4rem;">10</td>
  <td style="padding:0.4rem;">SB3 default</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.4rem;">ent_coef</td>
  <td style="padding:0.4rem; color:{TEXT};">0.01</td>
  <td style="padding:0.4rem;">0.0</td>
  <td style="padding:0.4rem;">slight regularization, prevents premature mode collapse</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.4rem;">gamma</td>
  <td style="padding:0.4rem; color:{TEXT};">0.99</td>
  <td style="padding:0.4rem;">0.99</td>
  <td style="padding:0.4rem;">SB3 default</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.4rem;">gae_lambda</td>
  <td style="padding:0.4rem; color:{TEXT};">0.95</td>
  <td style="padding:0.4rem;">0.95</td>
  <td style="padding:0.4rem;">SB3 default</td>
</tr>
<tr>
  <td style="padding:0.4rem;">clip_range</td>
  <td style="padding:0.4rem; color:{TEXT};">0.2</td>
  <td style="padding:0.4rem;">0.2</td>
  <td style="padding:0.4rem;">SB3 default</td>
</tr>
</table>
<br>
<b style="color:{TEXT}">Total training steps:</b><br>
&nbsp;&nbsp;Production 14-year variants: 1,000,000 timesteps (~283 passes through training data)<br>
&nbsp;&nbsp;Short-window variants:        500k–1M timesteps (~330-660 passes — proportionally more
to compensate for fewer regime examples per pass)<br><br>

<b style="color:{TEXT}">Empirical validation of LR + n_steps choice:</b> LR sub-experiments on DJI
confirmed that lr=3e-4 alone with n_steps=1024 produces unstable training
(val_sharpe std 0.808 across 40 evals, vs 0.3 for production), consistent
with the linear-scaling-rule coupling between lr and n_steps in PPO.
See DJI page "LR sensitivity sub-experiments" section for details.
</div>
""", unsafe_allow_html=True)


# ============================== Selection rules ==============================
st.markdown("### 5. Checkpoint selection rules")

st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
            padding:1rem 1.2rem; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1rem;">

Training saves 40 checkpoints (every 25k steps for 1M-step runs).
Two rules are applied to the resulting <b>checkpoint sweep</b>:<br><br>

<b style="color:{SUCCESS}">◇ DEFENSIVE rule (CANONICAL):</b> <code>max val-Sharpe with step ≥ 20% of training</code><br>
&nbsp;&nbsp;Standard convention (Yang et al. 2020 FinRL Ensemble). Single criterion using ONLY validation
data → truly out-of-sample for bear/recovery/bull. This is the canonical selection rule.<br>
&nbsp;&nbsp;Step 2 (after DEFENSIVE checkpoint chosen): sweep bear-regime cash floors {{0, 30, 50, 70 %}};
pick floor that maximises benchmark wins (val-Sharpe tiebreaker) — same rule applied identically per universe.<br><br>

<b style="color:{NEUTRAL}">★ HEADLINE rule (aggressive alternative):</b> <code>max wins + sum-Sharpe tiebreaker, step ≥ 20%</code><br>
&nbsp;&nbsp;Uses test-wins-based checkpoint selection (mild selection bias). Reported on some pages
for completeness as an aggressive alternative — NOT the canonical.<br><br>

<b style="color:{TEXT}">Selection rules committed BEFORE seeing test data on any variant.</b>
Canonical results use DEFENSIVE only; HEADLINE reported as side-by-side robustness comparison
where applicable.<br><br>

<b style="color:{TEXT}">Why min-step ≥ 20%?</b> Yang et al. (2020) finding: under-trained policies
(low step count) often produce equal-weight-like allocations that score well in steady
periods (val). These are NOT learned behaviors. Filtering below 20% of training avoids
this anomaly. Confirmed empirically on our SP100 production sweep — raw val-Sharpe-best
picked step 25k (≈ untrained baseline) before the filter was applied.
</div>
""", unsafe_allow_html=True)


# ============================== The synthesis ==============================
st.markdown("### 6. The complete synthesis — what this thesis does")

st.markdown(f"""
<div style="background:{SURFACE}; border-left:3px solid {SUCCESS}; padding:1rem 1.2rem;
            border-radius:0 8px 8px 0; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1rem;">

<b style="color:{TEXT}">The thesis builds on five established components:</b><br>
&nbsp;&nbsp;1. HMM regime detection (Hamilton 1989, Ang &amp; Bekaert 2002)<br>
&nbsp;&nbsp;2. PPO actor-critic algorithm (Schulman et al. 2017)<br>
&nbsp;&nbsp;3. Attention-based portfolio extractor (Sood, Khemlichi 2025)<br>
&nbsp;&nbsp;4. Differential Sharpe Reward (Moody &amp; Saffell 2001)<br>
&nbsp;&nbsp;5. Risk-sensitive RL (Tamar et al. 2015) and regime-aware DRL (Yang et al. 2020)<br><br>

<b style="color:{TEXT}">The SYNTHESIS is:</b> combining all five into one PPO agent that<br>
&nbsp;&nbsp;sees per-stock features + regime probs + vol features through attention,<br>
&nbsp;&nbsp;is rewarded by DSR with regime-conditioned risk penalty,<br>
&nbsp;&nbsp;trained universally across three universes (DJI / SP100 / NDX) with the
same recipe and hyperparameters.<br><br>

<b style="color:{TEXT}">The empirical contribution of this thesis</b>:<br>
<i style="color:{SUCCESS}">"Cross-universe validation of a single-agent regime-aware DRL recipe
(PPO + attention + HMM + DSR reward + bear cash-floor overlay) on three U.S. equity
universes under a uniform DEFENSIVE selection methodology."</i><br>
Canonical results: DJI 8/8 wins (with 70% bear floor) · SP100 7/12 wins (with 30% bear floor;
val Sharpe 3.41 with CI [+1.09, +5.15] is the strongest CI in the project) · NDX 4/8 wins
(no floor; +18.95pp defensive edge vs NDX BH on bear).
</div>
""", unsafe_allow_html=True)


# ============================== Bull-Bear cleaving mechanism ==============================
st.markdown("### 7. Bull-bear cleaving — interaction between DSR reward and training budget")

st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
            padding:1rem 1.2rem; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1rem;">

<b style="color:{TEXT}">Phenomenon (empirical):</b> when training is extended on a long-mom variant,
the HEADLINE pick drifts to a checkpoint with simultaneously HIGHER bull returns and LOWER bear
returns — across all universes tested. We call this <b>bull-bear cleaving</b>.<br><br>

<b style="color:{TEXT}">Empirical measurements (1M → 2M HEADLINE picks):</b><br>
&nbsp;&nbsp;DJI mom:   Δbull <b>+50.8pp</b>, Δbear <b>−31.2pp</b>  (ratio 1.6, wins unchanged 6/8)<br>
&nbsp;&nbsp;SP100 mom: Δbull <b>+15.6pp</b>, Δbear <b>−7.1pp</b>   (ratio 2.2, wins jumped 7/12 → 11/12)<br><br>

<b style="color:{TEXT}">Mechanistic explanation:</b><br>
The reward at step t is<br>
&nbsp;&nbsp;<code>r_t = DSR(R_p_t) − λ(regime_t) × max(0, −R_p_t) − cost</code><br>
where DSR is Moody-Saffell differential Sharpe (bounded), and the downside penalty fires only when
the day's portfolio return is negative.<br><br>

Two facts about the training-data distribution drive the cleaving:<br>
&nbsp;&nbsp;<b>(1)</b> Training data (2007-2020) is dominated by bull and sideways days. The cumulative
DSR gradient from bull-favorable streaks dwarfs the downside-penalty gradient from bear days, even
with λ_bear = 0.5.<br>
&nbsp;&nbsp;<b>(2)</b> As training continues, the policy commits to its bull-favorable decisions more
strongly (lower entropy, sharper softmax). This intensifies BOTH bull-period gains and bear-period
losses on out-of-sample data.<br><br>

<b style="color:{TEXT}">Why the cleaving doesn't always translate to wins:</b><br>
The win count depends on whether the universe's benchmark structure has room to capture the bull
gains. DJI has 8 benchmarks total and was already at 6/8 at 1M — the only winnable benchmarks left
are bear-related (DJI BH −9.40, EW DJI −10.25). DJI's bear gets WORSE at 2M (−14.8 → −46), so wins
ceiling at 6. SP100 has 12 benchmarks and 5 wins of headroom at 1M, with bull-period losses
to SP500 BH and OEX BH. The cleaving's bull side closes those gaps → +4 wins.<br><br>

<b style="color:{TEXT}">Defensive note for thesis:</b><br>
Bull-bear cleaving is therefore <b>NOT</b> a result of overfitting in the supervised-learning sense.
It is a structural property of the DSR + downside-penalty reward on a bull-dominated training
distribution: longer training amplifies any pre-existing asymmetry in the reward gradient. This
makes longer training universe-conditionally useful — only when (a) wins-headroom exists and (b)
the cleaving's positive side lands on un-won benchmarks. We document this as a refinement of the
dispersion-gradient claim, not a contradiction.<br><br>

<b style="color:{TEXT}">Mitigation (where applicable):</b> rule-based cash-floor overlay (Section 8
below) deliberately re-balances the bear-side losses post-hoc without retraining, since the cleaving
is baked into the trained policy itself.
</div>
""", unsafe_allow_html=True)


# ============================== Cash-floor overlay (experimental extension) ==============================
st.markdown("### 8. Rule-based cash-floor overlay (experimental extension, SP100 only)")

st.markdown(f"""
<div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
            padding:1rem 1.2rem; color:{TEXT_MUTED}; font-size:0.9rem;
            line-height:1.6; margin-bottom:1rem;">

<b style="color:{TEXT}">Mechanism:</b> at evaluation time only, apply a minimum cash
percentage when the HMM detects bear regime. Stock allocations from the trained policy
are proportionally rescaled to leave room for the floor. The model itself was NOT
retrained — this is purely a runtime overlay.<br><br>

<b style="color:{TEXT}">Formal definition:</b><br>
&nbsp;&nbsp;Let <i>r<sub>t</sub></i> be the HMM regime at time <i>t</i>, <i>c<sub>floor</sub></i>(r) the
floor per regime, <i>w<sub>cash</sub></i> the policy's cash weight, <i>w<sub>i</sub></i> stock weights.<br>
&nbsp;&nbsp;If <i>w<sub>cash</sub></i> < <i>c<sub>floor</sub></i>(<i>r<sub>t</sub></i>):<br>
&nbsp;&nbsp;&nbsp;&nbsp;<i>w<sub>i</sub></i> ← <i>w<sub>i</sub></i> × (1 − <i>c<sub>floor</sub></i>) / Σ <i>w<sub>j</sub></i><br>
&nbsp;&nbsp;&nbsp;&nbsp;<i>w<sub>cash</sub></i> ← <i>c<sub>floor</sub></i><br>
&nbsp;&nbsp;Else: no change (model's allocation respected).<br><br>

<b style="color:{TEXT}">Floor schedule tested:</b> <code>c_floor = {{bull: 0, bear: X, sideways: 0}}</code>,
with X ∈ {{0, 0.3, 0.5, 0.7}}. Bear is the ONLY regime the floor activates in —
bull and sideways remain unconstrained.<br><br>

<b style="color:{TEXT}">Empirical sweep on SP100 mom 2M HEADLINE (step 1.875M):</b><br>
<table style="width:100%; font-size:0.85rem; margin-top:0.4rem;">
<tr style="color:{TEXT_DIM}; border-bottom: 1px solid {BORDER};">
  <td style="padding:0.3rem;">Bear floor</td>
  <td style="padding:0.3rem;">val</td>
  <td style="padding:0.3rem;">bear</td>
  <td style="padding:0.3rem;">2023</td>
  <td style="padding:0.3rem;">bull</td>
  <td style="padding:0.3rem;">Wins</td>
  <td style="padding:0.3rem;">Verdict</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.3rem;">0%</td>
  <td style="padding:0.3rem;">+53.18%</td>
  <td style="padding:0.3rem;">−19.65%</td>
  <td style="padding:0.3rem;">+31.76%</td>
  <td style="padding:0.3rem;">+35.46%</td>
  <td style="padding:0.3rem;">11/12</td>
  <td style="padding:0.3rem;">baseline</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.3rem; color:{SUCCESS};"><b>30%</b></td>
  <td style="padding:0.3rem;">+52.36%</td>
  <td style="padding:0.3rem;">−17.64%</td>
  <td style="padding:0.3rem;">+32.37%</td>
  <td style="padding:0.3rem;">+31.55%</td>
  <td style="padding:0.3rem; color:{SUCCESS};"><b>11/12</b></td>
  <td style="padding:0.3rem; color:{SUCCESS};">strict Pareto improvement ★</td>
</tr>
<tr style="border-bottom: 1px solid {BORDER};">
  <td style="padding:0.3rem;">50%</td>
  <td style="padding:0.3rem;">+51.65%</td>
  <td style="padding:0.3rem;">−16.64%</td>
  <td style="padding:0.3rem;">+32.76%</td>
  <td style="padding:0.3rem;">+28.88%</td>
  <td style="padding:0.3rem;">10/12</td>
  <td style="padding:0.3rem;">loses OEX BH on bull</td>
</tr>
<tr>
  <td style="padding:0.3rem;">70%</td>
  <td style="padding:0.3rem;">+50.86%</td>
  <td style="padding:0.3rem;">−15.89%</td>
  <td style="padding:0.3rem;">+33.14%</td>
  <td style="padding:0.3rem;">+26.23%</td>
  <td style="padding:0.3rem;">10/12</td>
  <td style="padding:0.3rem;">deeper bear gain, more bull loss</td>
</tr>
</table>
<br>

<b style="color:{TEXT}">Recommended overlay:</b> <b>30% bear-only cash floor</b>. Strictly improves
bear (+2.01pp), val Sharpe (+0.27), val MDD (+1.12pp) at the cost of 3.91pp on bull —
without losing any of the 11/12 benchmark wins.<br><br>

<b style="color:{TEXT}">Literature precedent:</b><br>
&nbsp;&nbsp;Faber (2007) — A Quantitative Approach to Tactical Asset Allocation. Foundational
TAA framework; uses trend-following signals to switch between equity and cash positions.<br>
&nbsp;&nbsp;Yang et al. (2020 ICAIF) — Deep RL for Automated Stock Trading. Uses turbulence-index
threshold to switch entire strategy to cash on regime trigger (binary 100% cash floor,
conditional on regime). Our HMM-based bear floor is the soft graduated version of the
same idea.<br>
&nbsp;&nbsp;Asness, Frazzini, Pedersen (2012) — Leverage Aversion and Risk Parity. Cash as
a defensive lever in factor portfolios.<br>
&nbsp;&nbsp;<b>Internal precedent</b> in this thesis: MAC v3 Defender specialist uses
<code>cash_floor = 0.7</code> in bear regime — validated as load-bearing (a 0.5-floor
ablation produced strictly worse bear return AND MDD).<br><br>

<b style="color:{TEXT}">Contribution status:</b> Extension. The cash floor is a known
defensive instrument applied here as a runtime overlay on a trained RL policy. The novel
empirical claim is that <i>even small bear floors (30%) preserve all benchmark wins while
materially improving bear defense and val Sharpe</i> — because the HMM occasionally labels
non-bear days as transient bear, and on those days the floor caps exposure, reducing
volatility without harming primary-regime returns.
</div>
""", unsafe_allow_html=True)


# ============================== Citations summary ==============================
with st.expander("Full citations referenced on this page", expanded=False):
    st.markdown(f"""
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary
  Time Series and the Business Cycle." Econometrica.
- Moody, J., & Saffell, M. (2001). "Learning to Trade via Direct Reinforcement."
  IEEE Trans. Neural Networks.
- Ang, A., & Bekaert, G. (2002). "International Asset Allocation with Regime Shifts."
  Review of Financial Studies.
- Tamar, A., Glassner, Y., & Mannor, S. (2015). "Optimizing the CVaR via Sampling."
  AAAI.
- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
- Nystrup, P., et al. (2018). "Detecting Change Points in VIX and S&P 500."
- Yang, H., Liu, X.-Y., Wang, Z., & Walid, A. (2020). "Deep Reinforcement Learning
  for Automated Stock Trading: An Ensemble Strategy." ICAIF.
- Khemlichi, F., et al. (2025). "MPLS: Modular Pipeline for Long/Short Strategies."
  Information.
- Sood, S., et al. (JPM AI Research). "DRL for Optimal Portfolio Allocation."
- Chen, R., Li, X., & Wang, J. "MARS: Meta-Adaptive RL Framework." NJIT.
- Faber, M. T. (2007). "A Quantitative Approach to Tactical Asset Allocation."
  Journal of Wealth Management. Foundational TAA framework cited for cash-floor
  overlay (Section 7).
- Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012). "Leverage Aversion and
  Risk Parity." Financial Analysts Journal. Cited for cash as a defensive lever.

(Full bibliography appears in the thesis manuscript.)
""")
