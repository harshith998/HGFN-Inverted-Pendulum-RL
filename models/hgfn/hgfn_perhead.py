"""
HGFN Per-Head β — one learned β per attention head.

    logit_ij[h] = Q·K/√d  +  w_e·e_ij  +  tanh(β_h) · M̃ᵢⱼ

Each head independently decides how much to weight inertia coupling.
Head h might amplify coupling (β_h > 0), suppress it (β_h < 0),
or ignore it (β_h ≈ 0). More expressive than a single shared scalar.
"""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy
from ._physics import compute_inertia_coupling
from ._icga_base import ICGALayerBase, HGFNEncoderBase


class ICGALayer(ICGALayerBase):

    def __init__(self, hidden: int, n_heads: int):
        super().__init__(hidden, n_heads)
        # One β per head instead of one shared scalar
        self.physics_beta = nn.Parameter(torch.zeros(n_heads))

    def _apply_physics_bias(self, logits, M_edge, src_idx, dst_idx):
        # logits : (B, E, H)
        # M_edge : (B, E)
        # physics_beta: (H,) → broadcast to (1, 1, H)
        beta = torch.tanh(self.physics_beta).unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        return logits + beta * M_edge.unsqueeze(-1)


class HGFNPerHeadPPOPolicy(BasePPOPolicy):
    """Transformer + per-head β·M̃ — each attention head has its own physics scale."""

    VARIANT = "perhead"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4, max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = HGFNEncoderBase(hidden, n_icga_layers, n_heads,
                                       icga_cls=ICGALayer)

    def encode(self, obs: dict) -> torch.Tensor:
        return self.encoder(obs, compute_inertia_coupling(obs))
