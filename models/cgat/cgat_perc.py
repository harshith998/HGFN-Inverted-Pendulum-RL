"""
CGAT PERC — scalar β·M̃ + Potential Energy Residual Critic (w_H init=1).

    V(s) = value_head( critic_trunk( z_GNN ) )  +  w_H · V̂_pot(s)

Key difference from the original PERC attempt: w_H is initialised to 1.0
(not 0.0).  The critic starts from the physics prior and learns corrections,
rather than starting at zero and slowly discovering the physics signal.

V̂_pot = g·Σᵢ mᵢ·hᵢ(q) / H_scale is always exact, always positive, and
monotonically correlated with upright-ness — an ideal warm-start baseline.
"""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy, LOG_STD_MIN, LOG_STD_MAX
from ._physics import compute_inertia_coupling, compute_hamiltonian
from ._icga_base import ICGALayerBase, CGATEncoderBase


class ICGALayer(ICGALayerBase):
    """Same scalar β as base variant."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__(hidden, n_heads)
        self.physics_beta = nn.Parameter(torch.zeros(1))

    def _apply_physics_bias(self, logits, M_edge, src_idx, dst_idx):
        return logits + torch.tanh(self.physics_beta) * M_edge.unsqueeze(-1)


class CGATPercPPOPolicy(BasePPOPolicy):
    """
    Transformer + scalar β·M̃ + PERC critic with w_H initialised to 1.
    Critic = GNN value + w_H · V_pot  (physics warm-start, not cold-start).
    """

    VARIANT = "perc"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4, max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = CGATEncoderBase(hidden, n_icga_layers, n_heads,
                                       icga_cls=ICGALayer)
        # init=1 → critic starts from physics prior, learns residual correction
        self.w_H = nn.Parameter(torch.ones(1))

    def encode(self, obs: dict) -> torch.Tensor:
        return self.encoder(obs, compute_inertia_coupling(obs))

    def get_value(self, obs: dict) -> torch.Tensor:
        M_tilde = compute_inertia_coupling(obs)
        z       = self.encoder(obs, M_tilde)
        V_pot   = compute_hamiltonian(obs)
        v_gnn   = self.value_head(self.critic_trunk(z))
        return v_gnn + self.w_H * V_pot.unsqueeze(-1)

    def get_action_and_value(self, obs: dict, action=None):
        M_tilde = compute_inertia_coupling(obs)
        z       = self.encoder(obs, M_tilde)
        V_pot   = compute_hamiltonian(obs)

        actor_h  = self.actor_trunk(z)
        critic_h = self.critic_trunk(z)

        raw_mean = self.mean_head(actor_h)
        log_std  = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std      = log_std.exp().expand_as(raw_mean)
        dist     = torch.distributions.Normal(raw_mean, std)

        if action is None:
            raw_action = dist.rsample()
        else:
            a_norm     = (action / self.max_force).clamp(-1 + 1e-6, 1 - 1e-6)
            raw_action = torch.atanh(a_norm)

        squashed = torch.tanh(raw_action) * self.max_force
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(
            self.max_force * (1.0 - torch.tanh(raw_action).pow(2)) + 1e-6)
        log_prob = log_prob.squeeze(-1)
        entropy  = dist.entropy().squeeze(-1)

        value = self.value_head(critic_h) + self.w_H * V_pot.unsqueeze(-1)
        return squashed, log_prob, entropy, value
