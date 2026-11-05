"""
MTL v2 — Architectural fixes targeting the "head collapse" finding from v1.

Diagnosis of v1: the three regime heads produced nearly-identical outputs
(DD-aware switching shifted 92.6% of bear-2022 days toward the bear head
yet changed metrics by <0.2pp). The cause was four-fold:
  (a) Shallow heads (1 Linear+ReLU) couldn't differentiate enough
  (b) Aux loss baked regime info into the trunk, leaving heads with
      little to specialize on
  (c) Shared downstream policy net forced a unified mapping anyway
  (d) No explicit pressure for heads to be diverse

v2 fixes:
  1. Deep heads          — each head is now a 3-layer MLP, not 1 Linear
  2. Diversity loss      — pairwise cosine penalty between head outputs
                           to force the three heads apart
  3. No aux loss         — the regime info is supplied via observation
                           (one-hot + probs); we no longer pressure the
                           trunk to "predict" regime, freeing capacity
                           for downstream specialization
  4. Wider features_dim  — 192 dims (was 128) so heads have more room
                           to express different patterns
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Observation layout — same as v1
ONEHOT_SLICE  = slice(570, 573)
PROBS_SLICE   = slice(573, 576)
WEIGHTS_SLICE = slice(576, 607)


class AttentionMTLExtractorV2(BaseFeaturesExtractor):
    """
    Same shared backbone as v1, but with substantially deeper regime heads
    and explicit diversity pressure.

    Each head: trunk_features -> [Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU]
    instead of v1's single Linear+ReLU. This gives each head ~5x more
    parameters to learn a genuinely different transformation.

    Stores buffers used by MultiTaskPPOv2 to compute the diversity loss:
        _last_head_outputs   = list[Tensor] of 3 head outputs (pre-mix)
    """

    def __init__(self, observation_space,
                 n_stocks=30, n_features=19,
                 embed_dim=64, n_heads=4,
                 features_dim=192,
                 head_hidden_dim=256):
        super().__init__(observation_space, features_dim)
        self.n_stocks = n_stocks
        self.n_features = n_features

        # ---- Shared attention backbone (same as v1) ----
        self.embed = nn.Linear(n_features, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ---- Shared trunk ----
        trunk_in = n_stocks * embed_dim + (n_stocks + 1)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        trunk_out = 128

        # ---- Three DEEP regime-specialized heads ----
        # Each head: trunk_out -> head_hidden_dim -> head_hidden_dim ->
        #            features_dim, with ReLU between layers. ~5x more params
        # than v1.
        def deep_head():
            return nn.Sequential(
                nn.Linear(trunk_out, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, features_dim),
                nn.ReLU(),
            )
        self.head_bull     = deep_head()
        self.head_bear     = deep_head()
        self.head_sideways = deep_head()

        # Buffers for diversity loss
        self._last_head_outputs = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]

        # Slice
        stock_feat = obs[:, :self.n_stocks * self.n_features].reshape(
            batch, self.n_stocks, self.n_features
        )
        regime_probs = obs[:, PROBS_SLICE]
        weights      = obs[:, WEIGHTS_SLICE]

        # Backbone
        x = self.embed(stock_feat)
        x_att, _ = self.attention(x, x, x)
        x = self.norm(x + x_att)
        x = x.reshape(batch, -1)
        trunk_in = torch.cat([x, weights], dim=1)
        trunk = self.trunk(trunk_in)

        # Three deep heads
        f_bull     = self.head_bull(trunk)
        f_bear     = self.head_bear(trunk)
        f_sideways = self.head_sideways(trunk)

        # Cache for diversity loss
        self._last_head_outputs = [f_bull, f_bear, f_sideways]

        # Soft mixing by HMM probabilities
        probs = regime_probs / (regime_probs.sum(dim=1, keepdim=True) + 1e-8)
        mixed = (probs[:, 0:1] * f_bull
                 + probs[:, 1:2] * f_bear
                 + probs[:, 2:3] * f_sideways)
        return mixed


# ============================================
# Custom PPO with diversity loss between heads
# ============================================

class MultiTaskPPOv2(PPO):
    """
    PPO + pairwise cosine-distance diversity loss between regime-head outputs.

    Mechanism:
        For each minibatch, compute three pairwise cosine similarities
        between the head outputs (bull-bear, bull-sideways, bear-sideways)
        and add their mean as a penalty (multiplied by diversity_coef).
        High cosine similarity = heads agree = bad → loss is high.

    Target: drive heads to point in different directions so they encode
    different regime-specific transformations.
    """

    def __init__(self, *args, diversity_coef: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_coef = diversity_coef
        self._last_diversity_loss_value = 0.0

    def _compute_diversity_loss(self, head_outputs):
        """
        Mean pairwise cosine similarity. Lower = more diverse heads.
        We minimize this scaled by diversity_coef.
        """
        if head_outputs is None or len(head_outputs) < 2:
            return torch.tensor(0.0)
        h_bull, h_bear, h_sideways = head_outputs
        # Normalize for cosine similarity
        eps = 1e-8
        nb = h_bull     / (h_bull.norm(dim=1, keepdim=True) + eps)
        nr = h_bear     / (h_bear.norm(dim=1, keepdim=True) + eps)
        ns = h_sideways / (h_sideways.norm(dim=1, keepdim=True) + eps)
        sim_br = (nb * nr).sum(dim=1).mean()
        sim_bs = (nb * ns).sum(dim=1).mean()
        sim_rs = (nr * ns).sum(dim=1).mean()
        return (sim_br + sim_bs + sim_rs) / 3.0

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, div_losses = [], [], [], []
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

                # --- Diversity loss ---
                heads = self.policy.features_extractor._last_head_outputs
                diversity = self._compute_diversity_loss(heads)
                div_losses.append(diversity.item())

                # --- Standard PPO losses ---
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

                # Total = standard PPO + diversity penalty
                loss = (policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.diversity_coef * diversity)

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
        self._last_diversity_loss_value = float(np.mean(div_losses)) if div_losses else 0.0
        if self.logger is not None:
            self.logger.record("train/diversity_loss", self._last_diversity_loss_value)
            self.logger.record("train/diversity_coef", self.diversity_coef)
