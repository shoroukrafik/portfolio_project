"""
Multi-task learning components for the multi-agent MTL architecture.

Two new pieces:
  - AttentionMTLExtractor: shared attention backbone + 3 regime-specific
    feature heads + 1 auxiliary regime-prediction head. The 3 regime heads
    are soft-mixed by the HMM probabilities present in the observation.
  - MultiTaskPPO: PPO subclass that adds an auxiliary regime-classification
    cross-entropy loss to the standard PPO objective.

Observation layout (expected, 607 dims for 30 stocks × 19 features):
    [0   : 570]   stock features (n_stocks × n_features, row-major)
    [570 : 573]   regime one-hot (kept for backward-compat / used as aux target)
    [573 : 576]   HMM regime probabilities (bull, bear, sideways)
    [576 : 607]   portfolio weights (n_stocks + 1 cash)

The aux loss target is the one-hot's argmax (= the HMM's argmax regime
on that day) and gradients on it force the shared backbone to encode
regime structure even when the policy/value losses don't.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Slice positions (canonical layout)
N_STOCKS_DEFAULT      = 30
N_FEATURES_DEFAULT    = 19
ONEHOT_SLICE          = slice(570, 573)   # regime one-hot
PROBS_SLICE           = slice(573, 576)   # HMM probabilities
WEIGHTS_SLICE         = slice(576, 607)   # 30 stocks + cash


class AttentionMTLExtractor(BaseFeaturesExtractor):
    """
    Shared attention encoder, three regime-specialised projection heads,
    one auxiliary regime-prediction head.

    Output is a single features_dim vector formed by softly mixing the
    three regime heads using the HMM probabilities (extracted from obs).
    """

    def __init__(self, observation_space,
                 n_stocks=N_STOCKS_DEFAULT,
                 n_features=N_FEATURES_DEFAULT,
                 embed_dim=64, n_heads=4, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.n_stocks   = n_stocks
        self.n_features = n_features
        self.features_dim_ = features_dim

        # ---- Shared backbone (same as AttentionPortfolioExtractor) ----
        self.embed = nn.Linear(n_features, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ---- Shared trunk: stock-attended features + portfolio context ----
        # Input: n_stocks * embed_dim (from attention) + n_stocks+1 (weights)
        # We mix in portfolio weights here so all heads see them.
        trunk_in = n_stocks * embed_dim + (n_stocks + 1)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        trunk_out = 128

        # ---- Three regime-specialised projection heads ----
        # Each maps the shared trunk into the final features_dim.
        # During forward, the three head outputs are blended by HMM probs.
        self.head_bull     = nn.Sequential(nn.Linear(trunk_out, features_dim), nn.ReLU())
        self.head_bear     = nn.Sequential(nn.Linear(trunk_out, features_dim), nn.ReLU())
        self.head_sideways = nn.Sequential(nn.Linear(trunk_out, features_dim), nn.ReLU())

        # ---- Auxiliary regime-prediction head ----
        # Takes trunk features → 3-way logits (bull/bear/sideways).
        # The target is the HMM's argmax label (read from obs one-hot).
        self.aux_regime_head = nn.Linear(trunk_out, 3)

        # Buffers populated each forward — read by MultiTaskPPO.train().
        # We avoid storing the actual loss to keep this stateless w.r.t.
        # autograd graphs across calls.
        self._last_aux_logits = None
        self._last_regime_targets = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]

        # Slice observation
        stock_feat = obs[:, :self.n_stocks * self.n_features].reshape(
            batch, self.n_stocks, self.n_features
        )
        regime_onehot = obs[:, ONEHOT_SLICE]   # (batch, 3) — used for aux target
        regime_probs  = obs[:, PROBS_SLICE]    # (batch, 3) — used for soft mixing
        weights       = obs[:, WEIGHTS_SLICE]  # (batch, 31)

        # Shared backbone: per-stock embed → attention → norm
        x = self.embed(stock_feat)              # (batch, n_stocks, embed_dim)
        x_att, _ = self.attention(x, x, x)      # self-attention across stocks
        x = self.norm(x + x_att)                # residual
        x = x.reshape(batch, -1)                # (batch, n_stocks*embed_dim)

        # Shared trunk: stocks + portfolio context
        trunk_in = torch.cat([x, weights], dim=1)
        trunk = self.trunk(trunk_in)            # (batch, 128)

        # Three regime-specialised heads
        f_bull     = self.head_bull(trunk)      # (batch, features_dim)
        f_bear     = self.head_bear(trunk)
        f_sideways = self.head_sideways(trunk)

        # Soft mixing by HMM probabilities.
        # regime_probs[:,0]=bull, [:,1]=bear, [:,2]=sideways.
        # Renormalise just in case rounding errors made probs not sum to 1.
        probs = regime_probs / (regime_probs.sum(dim=1, keepdim=True) + 1e-8)
        mixed = (probs[:, 0:1] * f_bull
                 + probs[:, 1:2] * f_bear
                 + probs[:, 2:3] * f_sideways)

        # ---- Auxiliary regime-prediction (cached for the PPO subclass) ----
        self._last_aux_logits = self.aux_regime_head(trunk)
        # Target is the one-hot's argmax (HMM's argmax regime label).
        self._last_regime_targets = torch.argmax(regime_onehot, dim=1).long()

        return mixed


# ============================================
# Custom PPO with auxiliary regime-classification loss
# ============================================

class MultiTaskPPO(PPO):
    """
    PPO + auxiliary regime-classification cross-entropy loss.

    Each minibatch, after evaluate_actions runs the policy forward (which
    populates the extractor's _last_aux_logits and _last_regime_targets),
    we add  aux_loss_coef * CE(aux_logits, regime_targets)  to the total
    PPO loss. Coefficient is small (default 0.1) so the policy stays the
    primary objective and aux is a side signal.
    """

    def __init__(self, *args, aux_loss_coef: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_coef = aux_loss_coef
        # Track for logging
        self._last_aux_loss_value = 0.0

    def train(self) -> None:
        """
        Copy of stable_baselines3 PPO.train() with `aux_loss_coef * CE`
        added to the total loss. Kept structurally identical so any
        SB3 logging utilities still work.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, aux_losses = [], [], [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, type(self.action_space)) \
                   and len(actions.shape) > 1 and self.action_space.shape == ():
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # ---- AUX LOSS ----
                # The extractor populated these during the forward inside
                # evaluate_actions(). Read them now.
                aux_logits  = self.policy.features_extractor._last_aux_logits
                aux_targets = self.policy.features_extractor._last_regime_targets
                if aux_logits is None or aux_targets is None:
                    aux_loss = torch.tensor(0.0, device=values.device)
                else:
                    aux_loss = F.cross_entropy(aux_logits, aux_targets)
                aux_losses.append(aux_loss.item())

                # ---- Standard PPO losses ----
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # ---- Total loss (PPO + scaled aux) ----
                loss = (policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.aux_loss_coef * aux_loss)

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        # Log aux loss so it shows up in training output / tensorboard
        self._last_aux_loss_value = float(np.mean(aux_losses)) if aux_losses else 0.0
        if self.logger is not None:
            self.logger.record("train/aux_regime_loss", self._last_aux_loss_value)
            self.logger.record("train/aux_regime_coef", self.aux_loss_coef)
